#!/usr/bin/env python3
"""Neck/shoulder via semantic basis + neck joint (trial).

Represent neck/shoulder with {neck_thickness, neck_length} semantic coefficients
plus an adjustable neck-joint rotation, and best fit that to the current
(deform_neck) neck/shoulder shape. Ears keep their local static offset.
Reports region residual and renders current vs semantic+joint (clay, 2 panels).

This is a feasibility trial for the *pre-photometric* stage; it is NOT required
to match exactly (photometric refines later).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
REPO = next(p for p in [ROOT, *ROOT.parents] if (p / "GNM-main").is_dir()) / "GNM-main"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT))

from fit_gnm import FitState  # noqa: E402
from gnm.shape import gnm_pytorch  # noqa: E402
from nvdiffrast_render import NVDiffRenderer  # noqa: E402
import semantic_basis as sb  # noqa: E402

NOTES = Path("/root/autodl-tmp/notes/gnm_uv/测试用图片/肩颈关节")


def _tp(v):
  return torch.tensor(np.asarray(v), dtype=torch.float32)


def load_state(p):
  rot = np.asarray(p["rotations"], np.float32)
  return FitState(
      id_head=_tp(p["id_head"]), id_eyes=_tp(p["id_eyes"]),
      ex_left=_tp(p["expression_left_eye"] if "expression_left_eye" in p.files else np.zeros(100, np.float32)),
      ex_right=_tp(p["expression_right_eye"] if "expression_right_eye" in p.files else np.zeros(100, np.float32)),
      ex_lower=_tp(p["expression_lower_face"]),
      rot_neck=_tp(rot[0]), rot_head=_tp(rot[1]), rot_leye=_tp(rot[2]), rot_reye=_tp(rot[3]),
      translation=_tp(p["translation"]), cam_aa=_tp(p["cam_aa"]),
      cam_t=_tp(p["cam_t"]), focal_norm=_tp(p["focal_norm"]),
  )


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--params", type=Path, required=True)
  ap.add_argument("--params_lmk", type=Path,
                  default=Path("/root/autodl-tmp/gnm_fit_out/expression_v3/lmk_align_vhap_frame0_v4_sam_eyes_after_neck_len_more/frame_000000_params.npz"),
                  help="pre-ear params; ear-delta = off(params) - off(params_lmk) keeps the smooth anisotropic ear blend.")
  ap.add_argument("--image", type=Path, default=Path("/root/autodl-tmp/gnm_fit_out/expression_v3/images/000001.jpg"))
  ap.add_argument("--iters", type=int, default=800)
  ap.add_argument("--lr", type=float, default=5e-3)
  ap.add_argument("--joint_lr", type=float, default=0.05, help="neck-joint delta learning rate (higher = more joint).")
  ap.add_argument("--w_joint", type=float, default=0.0, help="L2 penalty on neck-joint delta (rad^2).")
  ap.add_argument("--w_sem", type=float, default=0.0, help="L2 penalty on semantic coeffs (higher = joint/semantic split).")
  ap.add_argument("--w_thick", type=float, default=0.0, help="Extra L2 penalty on neck_thickness coeff only (throttle 粗细).")
  ap.add_argument("--out_name", type=str, default="颈肩 语义基+关节 vs 原版现状 GT 四格.jpg")
  args = ap.parse_args()

  p = np.load(args.params)
  st = load_state(p)
  off_full = torch.as_tensor(np.asarray(p["static_offset"], np.float32))  # metres
  gnm = gnm_pytorch.GNM.from_local(version=gnm_pytorch.GNMMajorVersion.V3, variant=gnm_pytorch.GNMVariant.HEAD)
  faces = np.asarray(gnm.triangles.detach().cpu().numpy())

  # semantic neck dirs (mm) + groups
  g, rest, B, grp, dirs, neck = sb.build()
  dT = torch.as_tensor(dirs["neck_thickness"] / 1000.0).float()  # metres per unit
  dL = torch.as_tensor(dirs["neck_length"] / 1000.0).float()
  ears = torch.as_tensor(grp["ears"]); skin = grp["skin"]
  ys = rest[:, 1]
  # neck+shoulder region: skin, y in [0.06,0.22], not ears, not front face (z small-ish)
  reg = np.intersect1d(skin, np.where((ys >= 0.06) & (ys <= 0.22))[0])
  reg = np.setdiff1d(reg, grp["ears"])
  reg = reg[np.abs(rest[reg, 0]) > 0.0]  # keep all; jaw front mostly y>0.18 handled by band
  reg_t = torch.as_tensor(reg, dtype=torch.long)

  # keep ear motion as the smooth anisotropic ear-delta (params - lmk), NOT just the ear group
  p_lmk = np.load(args.params_lmk)
  if "static_offset" in p_lmk.files:
    off_lmk = torch.as_tensor(np.asarray(p_lmk["static_offset"], np.float32))
    ear_off = off_full - off_lmk          # smooth ear blend baked by fit_ear_mask
  else:
    ear_off = torch.zeros_like(off_full)
    ear_off[ears] = off_full[ears]
  print(f"[ear] ear-delta |.| mean on ears={float(torch.linalg.norm(ear_off[ears],dim=1).mean())*1000:.2f}mm "
        f"nonzero={int((torch.linalg.norm(ear_off,dim=1)>1e-5).sum())} verts", flush=True)

  def posed(rot_neck):
    # build rotations with modified neck
    rot = torch.stack([rot_neck, st.rot_head, st.rot_leye, st.rot_reye], 0)[None]
    return gnm(st.identity()[None], st.expression()[None], rot, st.translation[None])[0]

  with torch.no_grad():
    want = posed(st.rot_neck) + off_full  # current good mesh (with full 25mm + ear)
  base0 = st.rot_neck.detach().clone()

  dneck = torch.zeros(3, requires_grad=True)   # rot_neck delta (axis-angle)
  cT = torch.zeros(1, requires_grad=True)
  cL = torch.zeros(1, requires_grad=True)
  opt = torch.optim.Adam([
      {"params": [dneck], "lr": args.joint_lr},
      {"params": [cT, cL], "lr": args.lr},
  ])

  best = None
  for it in range(args.iters):
    opt.zero_grad()
    approx = posed(base0 + dneck) + ear_off + cT * dT + cL * dL
    loss = ((approx[reg_t] - want[reg_t]) ** 2).mean() \
        + args.w_joint * dneck.pow(2).sum() + args.w_sem * (cT.pow(2) + cL.pow(2)).sum() \
        + args.w_thick * cT.pow(2).sum()
    loss.backward()
    opt.step()
    if best is None or float(loss) < best[0]:
      with torch.no_grad():
        best = (float(loss), dneck.detach().clone(), float(cT), float(cL))
    if it == 0 or (it + 1) % 100 == 0:
      print(f"  [{it+1:3d}] regionMSE={float(loss):.3e}", flush=True)

  with torch.no_grad():
    dneck_b, cT_b, cL_b = best[1], best[2], best[3]
    approx = posed(base0 + dneck_b) + ear_off + cT_b * dT + cL_b * dL
    res = torch.linalg.norm(approx[reg_t] - want[reg_t], dim=1)
    tgt = torch.linalg.norm(want[reg_t], dim=1)
    print(f"[neck] region {reg.size}  residual mean={float(res.mean())*1000:.2f}mm  "
          f"want|.|={float(tgt.mean())*1000:.2f}mm  cT={cT_b:.3f} cL={cL_b:.3f} "
          f"drot_neck={[round(float(x),3) for x in dneck_b]}", flush=True)

  # render 4-panel: GT | 原版(no offset) | 现状(full offset) | 语义+关节
  rend = NVDiffRenderer(use_opengl=False, lighting_type="front", lighting_space="camera", flip_y_raster=False).cuda()
  faces_t = torch.as_tensor(faces, dtype=torch.int32).cuda()
  W, H = 1024, 576
  rt = st.camera_rt().detach()[None].cuda(); k4 = st.camera_k(W, H).detach()[None].cuda()
  def clay(v):
    with torch.no_grad():
      rgba = rend.render_rgba_vis(v.cuda()[None], faces_t, rt, k4, (H, W), background_color=[0.12, 0.12, 0.14])["rgba"][0].float().cpu().numpy()
    return cv2.cvtColor(np.clip(rgba[..., :3]*255,0,255).astype(np.uint8), cv2.COLOR_RGB2BGR)
  with torch.no_grad():
    base_mesh = posed(st.rot_neck)           # 原版: 无 static_offset
  gt = cv2.resize(cv2.imread(str(args.image)), (W, H), interpolation=cv2.INTER_AREA)
  c_orig = clay(base_mesh)
  c_now = clay(want)
  c_app = clay(approx.detach())
  def lab(img, t):
    cv2.putText(img, t, (12,34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0),3,cv2.LINE_AA)
    cv2.putText(img, t, (12,34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),2,cv2.LINE_AA)
    return img
  row = np.concatenate([lab(gt,"GT"), lab(c_orig,"原版(无offset)"),
                        lab(c_now,"现状(25mm offset)"), lab(c_app,"语义基+关节")], axis=1)
  NOTES.mkdir(parents=True, exist_ok=True)
  out = NOTES / args.out_name
  cv2.imwrite(str(out), row)
  print("wrote", out, flush=True)


if __name__ == "__main__":
  main()
