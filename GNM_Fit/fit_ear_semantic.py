#!/usr/bin/env python3
"""Ear morphology via explicit rigid-translation basis + two identity semantic
bases (protrusion, size), all masked by the anisotropic weight w and forced
left/right symmetric. Built on top of an eyes+neck/shoulder base (no ear).

Basis (all X w, symmetric in |x| canonical space):
  rigid translate: lat(+/-X), vert(+Y), front(+Z)   * trans_scale (slightly < 1)
  semantic:        ear_protrusion, ear_size (identity, symmetrised)
Hinge loss to the SAM ear mask, symmetric coefficients -> one ear mask gives both.

4-panel output: GT+mask | 原版+mask | 现状+mask | 新(刚性+语义+各向异性)+mask
"""
from __future__ import annotations

import argparse
import json
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
from gnm.shape import gnm_numpy, gnm_pytorch  # noqa: E402
from nvdiffrast_render.camera import project_world_to_pixels  # noqa: E402
from nvdiffrast_render import NVDiffRenderer  # noqa: E402
from fit_ear_mask import load_state, mask_quality, build_blend_weights  # noqa: E402
import semantic_basis as sb  # noqa: E402

NOTES_DIR = Path("/root/autodl-tmp/notes/gnm_uv/测试用图片")


def _unit_mm(vec):
  """Scale a (V,3) direction so mean |vec[ear]| = 1mm; return (scaled, ear_mean)."""
  return vec


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--params", type=Path, required=True,
                  help="eyes+neck/shoulder base (NO ear offset), e.g. lmk_align...after_neck_len_more")
  ap.add_argument("--params_now", type=Path, required=True,
                  help="current ear result for the '现状' panel, e.g. ear_align_v2_vertmore")
  ap.add_argument("--ear_mask_npz", type=Path, required=True)
  ap.add_argument("--image", type=Path, required=True)
  ap.add_argument("--out_dir", type=Path, required=True)
  ap.add_argument("--iters", type=int, default=300)
  ap.add_argument("--lr", type=float, default=2e-3)
  ap.add_argument("--w_reg", type=float, default=3.0)
  ap.add_argument("--trans_scale", type=float, default=0.85, help="rigid translation weight (<1 = slightly less than semantic).")
  ap.add_argument("--rigid_axes", type=str, default="lat,vert,front",
                  help="Comma subset of lat,vert,front to keep as RIGID translation (rest handled by semantic).")
  ap.add_argument("--margin_px", type=float, default=8.0)
  ap.add_argument("--margin_in", type=float, default=4.0, help="Inside margin: ear verts deeper than this inside the mask are pushed out (fill the mask).")
  ap.add_argument("--w_inside", type=float, default=1.0, help="Weight of the inside (fill-the-mask) hinge vs the outside pull-in.")
  ap.add_argument("--dt_cap", type=float, default=24.0)
  ap.add_argument("--sig_x", type=float, default=0.06)
  ap.add_argument("--sig_y_up", type=float, default=0.12)
  ap.add_argument("--sig_y_dn", type=float, default=0.035)
  ap.add_argument("--sig_z", type=float, default=0.07)
  ap.add_argument("--coef_max", type=float, default=25.0, help="cap per coefficient (mm-equivalent).")
  ap.add_argument("--vert_max", type=float, default=-1.0, help="cap (mm) for the vert(down/up) coefficient; -1 = use coef_max.")
  ap.add_argument("--prot_max", type=float, default=-1.0, help="cap (mm) for the prot(outward) coefficient; -1 = use coef_max.")
  ap.add_argument("--size_bias", type=float, default=0.0, help="Added to the size coefficient after fit (mm; +ve = bigger ears).")
  ap.add_argument("--out_name", type=str, default="GNM v4 耳朵 刚性平移+语义基+各向异性 四格.jpg")
  args = ap.parse_args()
  args.out_dir.mkdir(parents=True, exist_ok=True)

  bgr0 = cv2.imread(str(args.image))
  H0, W0 = bgr0.shape[:2]
  W, H = 1024, 576
  bgr = cv2.resize(bgr0, (W, H), interpolation=cv2.INTER_AREA) if (W0, H0) != (W, H) else bgr0

  params = np.load(args.params)
  state = load_state(params)
  for t in (state.id_head, state.id_eyes, state.ex_left, state.ex_right, state.ex_lower,
            state.rot_neck, state.rot_head, state.rot_leye, state.rot_reye,
            state.translation, state.cam_aa, state.cam_t, state.focal_norm):
    t.requires_grad_(False)

  gnm = gnm_pytorch.GNM.from_local(version=gnm_pytorch.GNMMajorVersion.V3, variant=gnm_pytorch.GNMVariant.HEAD)
  gnm_np = gnm_numpy.GNM.from_local(version=gnm_numpy.GNMMajorVersion.V3, variant=gnm_numpy.GNMVariant.HEAD)
  faces = np.asarray(gnm.triangles.detach().cpu().numpy())
  ear_vids = np.unique(np.asarray(gnm_np.vertex_group_indices("ears"), dtype=np.int64))
  rest = np.asarray(gnm_np(None, None, None, None), dtype=np.float64)
  n_verts = int(rest.shape[0])

  base_off = torch.zeros(n_verts, 3)
  if "static_offset" in params.files:
    base_off = torch.as_tensor(np.asarray(params["static_offset"], np.float32))

  # masks
  masks = np.load(args.ear_mask_npz)
  left = cv2.resize(np.asarray(masks["left"], np.float32), (W, H), interpolation=cv2.INTER_NEAREST) > 0.5
  right = cv2.resize(np.asarray(masks["right"], np.float32), (W, H), interpolation=cv2.INTER_NEAREST) > 0.5
  combined = (left | right).astype(np.uint8)
  ok_a, cx_a, why_a = mask_quality(left, W, H, 250, 8000)
  ok_b, cx_b, why_b = mask_quality(right, W, H, 250, 8000)
  opposite = (cx_a >= 0 and cx_b >= 0 and (cx_a - W * 0.5) * (cx_b - W * 0.5) < 0)
  ok = ok_a and ok_b and opposite
  print(f"[ear] maskA ok={ok_a} cx={cx_a:.0f}  maskB ok={ok_b} cx={cx_b:.0f}  opposite={opposite}", flush=True)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  rt = state.camera_rt().detach()[None].to(device)
  k4 = state.camera_k(W, H).detach()[None].to(device)
  with torch.no_grad():
    posed = gnm(state.identity()[None], state.expression()[None], state.rotations()[None], state.translation[None])[0]

  # anisotropic weight + basis
  w_np, Cr = build_blend_weights(rest.astype(np.float32), ear_vids, args.sig_x, args.sig_y_up, args.sig_y_dn, args.sig_z)
  w = torch.as_tensor(w_np, dtype=torch.float32)
  xs = np.sign(rest[:, 0]); xs[xs == 0] = 1.0

  # rigid translation (metres per unit) — only selected axes
  _rig_def = {"lat": np.stack([xs, 0 * xs, 0 * xs], 1),
              "vert": np.stack([0 * xs, np.ones_like(xs), 0 * xs], 1),
              "front": np.stack([0 * xs, 0 * xs, np.ones_like(xs)], 1)}
  rigid_axes = [a.strip() for a in args.rigid_axes.split(",") if a.strip() in _rig_def]
  rigid = [(_rig_def[a], a) for a in rigid_axes]

  # semantic identity directions (mm -> m), symmetrised across mirror, then x w
  g_sb, rest_sb, B_sb, grp_sb, dirs, neck = sb.build()
  mirror = np.asarray(g_sb.mirror_indices, np.int64)
  flip = np.array([-1.0, 1.0, 1.0])
  sem = []
  for key in ("ear_protrusion", "ear_size"):
    d = dirs[key] / 1000.0  # m
    d = 0.5 * (d + d[mirror] * flip)  # mirror-symmetrise
    sem.append(d)

  # build torch basis, each normalised to ~1mm mean on ears, then x w; rigid x trans_scale
  basis = []
  names = []
  for vec, nm in rigid:
    b = torch.as_tensor(vec, dtype=torch.float32) * w[:, None] * args.trans_scale
    basis.append(b); names.append(nm)
  for i, d in enumerate(sem):
    b = torch.as_tensor(d, dtype=torch.float32) * w[:, None]
    basis.append(b); names.append(["prot", "size"][i])

  # normalise each basis so mean |.| on ears = 1 mm (per unit coef = 1mm on ears)
  ear_idx_t = torch.as_tensor(ear_vids, dtype=torch.long)
  with torch.no_grad():
    for j in range(len(basis)):
      m = torch.linalg.norm(basis[j][ear_idx_t], dim=1).mean()
      if float(m) > 1e-9:
        basis[j] = basis[j] / m * 1e-3  # metres per unit coef -> 1mm ear motion per unit
  print(f"[ear] basis {names}  influent(w>0.05)={int((w_np>0.05).sum())}  ear={ear_vids.size}", flush=True)

  if not ok:
    print("[ear] mask bad -> skip, params copied", flush=True)
    save = {k: np.asarray(params[k]) for k in params.files}
    np.savez(args.out_dir / "frame_000000_params.npz", **save)
    return 0

  inv = (1 - combined).astype(np.uint8)
  dt = cv2.distanceTransform(inv, cv2.DIST_L2, 5).astype(np.float32)        # outside: dist to mask (0 inside)
  dt_t = torch.as_tensor(dt, dtype=torch.float32)[None, None]

  K = len(basis)
  coef = torch.zeros(K, requires_grad=True)

  def ear_delta() -> torch.Tensor:
    d = torch.zeros(n_verts, 3)
    for c, b in zip(coef, basis):
      d = d + c * b
    return d

  def full_verts() -> torch.Tensor:
    return posed + base_off + ear_delta()

  def ear_hinge_loss() -> torch.Tensor:
    v = full_verts()
    xy = project_world_to_pixels(v[ear_idx_t], state.camera_rt(), state.camera_k(W, H), W, H, flip_y=True)
    gx = 2.0 * xy[:, 0] / W - 1.0
    gy = 2.0 * xy[:, 1] / H - 1.0
    dd = torch.nn.functional.grid_sample(dt_t, torch.stack([gx, gy], -1)[None, None], align_corners=False)[0, 0, 0]
    dd = dd.clamp(max=float(args.dt_cap))
    return torch.clamp(dd - float(args.margin_px), min=0.0).mean()

  opt = torch.optim.Adam([coef], lr=args.lr)
  _def = args.coef_max
  _map = {"vert": args.vert_max, "prot": args.prot_max}
  lim_vec = torch.tensor([(_map.get(nm, -1.0) if _map.get(nm, -1.0) and _map.get(nm, -1.0) > 0 else _def) for nm in names])
  last = 0.0
  for i in range(int(args.iters)):
    opt.zero_grad()
    ear_loss = ear_hinge_loss()
    loss = ear_loss + args.w_reg * (coef / args.coef_max).pow(2).sum()
    loss.backward()
    opt.step()
    with torch.no_grad():
      coef.clamp_(-lim_vec, lim_vec)
    last = float(ear_loss)
    if i == 0 or (i + 1) % max(args.iters // 10, 1) == 0 or i + 1 == args.iters:
      print(f"  [{i+1:4d}] hinge={last:.3f}  coef={[round(float(c),2) for c in coef]}", flush=True)

  # meshes for panels
  with torch.no_grad():
    if args.size_bias != 0.0:
      coef[names.index("size")] += args.size_bias
      coef.clamp_(-lim_vec, lim_vec)
      print(f"[ear] size_bias {args.size_bias:+.1f} applied -> coef={[round(float(c),2) for c in coef]}", flush=True)
  params_now = np.load(args.params_now)
  now_off = torch.as_tensor(np.asarray(params_now["static_offset"], np.float32)) if "static_offset" in params_now.files else torch.zeros_like(base_off)
  with torch.no_grad():
    v_orig = posed.detach()
    v_now = (posed + now_off).detach()
    v_new = full_verts().detach()

  # render blue-silhouette over GT + mask overlay, 4 panels
  rend_paint = NVDiffRenderer(use_opengl=False, lighting_type="constant", lighting_space="camera", flip_y_raster=False).to(device)
  faces_t = torch.as_tensor(faces, dtype=torch.int32, device=device)
  def alpha_of(v):
    with torch.no_grad():
      p = rend_paint.render_rgba_vis(v.to(device)[None], faces_t, rt, k4, (H, W), background_color=[0, 0, 0])["rgba"][0].float().cpu().numpy()
    return p[..., 3] > 0.5
  def panel(v, title, with_mesh=True):
    img = bgr.copy()
    over = img.copy(); over[left] = (0, 255, 0); over[right] = (0, 0, 255)
    img = cv2.addWeighted(img, 0.6, over, 0.4, 0)
    cnts, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cnts, -1, (0, 255, 255), 2)
    if with_mesh and v is not None:
      a = alpha_of(v)
      bl = img.copy(); bl[a] = (255, 120, 0)
      img = cv2.addWeighted(img, 0.45, bl, 0.55, 0)
    cv2.putText(img, title, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, title, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return img
  row = np.concatenate([
      panel(None, "GT+耳朵mask", with_mesh=False),
      panel(v_orig, "原版+mask"),
      panel(v_now, "现状+mask"),
      panel(v_new, "刚性+语义+各向异性+mask"),
  ], axis=1)
  cv2.imwrite(str(args.out_dir / "ear_semantic_row.jpg"), row)
  NOTES_DIR.mkdir(parents=True, exist_ok=True)
  notes_path = NOTES_DIR / args.out_name
  cv2.imwrite(str(notes_path), row)

  with torch.no_grad():
    full_off = base_off + ear_delta()
  save = {k: np.asarray(params[k]) for k in params.files}
  save["static_offset"] = full_off.detach().cpu().numpy().astype(np.float32)
  save["ear_coef"] = coef.detach().cpu().numpy().astype(np.float32)
  save["ear_basis_names"] = np.array(names)
  np.savez(args.out_dir / "frame_000000_params.npz", **save)
  summary = {"names": names, "coef": [float(c) for c in coef.detach()], "ear_hinge_final": last,
             "influent_verts": int((w_np > 0.05).sum()), "notes_row": str(notes_path), "head_frozen": True}
  (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
  print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
