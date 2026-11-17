#!/usr/bin/env python3
"""After rigid: hard-grid over (len, rad; jaw-gated) + micro head via inverse_head_delta.

Ablation: joints-only and joints+rad(jaw-gate) look clean under the chin;
ungated rad caused the shelf. Soft face fade is in neck_boundary_vertex_ids.

The grid searches len x rad x d_head and keeps the combo with the lowest hard
shoulder chamfer subject to a face-orientation budget (``--face_max``).

Defaults reproduce the committed after_rigid result (len=1.0, hard~382).
To reproduce the len_more variant (hard~26), e.g.:
  python hard_grid_after_rigid.py \
    --lens 1.20,1.22 --face_max 8 \
    --deltas '0,0,0;0.12,0,-0.06;0.4,0,-0.08;0.4,0,-0.12;0.4,0.06,0;0.4,-0.06,0' \
    --out_dir /root/autodl-tmp/gnm_fit_out/expression_v3/neck_shoulder_after_rigid_len_more
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
REPO = next(
    (p for p in [ROOT, *ROOT.parents] if (p / "GNM-main").is_dir()), ROOT.parent
) / "GNM-main"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT))

from fit_gnm import FitState  # noqa: E402
from gnm.shape import gnm_numpy, gnm_pytorch  # noqa: E402
from nvdiffrast_render import NVDiffRenderer  # noqa: E402
from nvdiffrast_render.camera import project_world_to_pixels  # noqa: E402
from photo_fit import UV_MASKS, uv_from_triangles  # noqa: E402
import neck_shoulder_l2 as neck_l2  # noqa: E402

EV3 = Path("/root/autodl-tmp/gnm_fit_out/expression_v3")
DEFAULT_PARAMS = EV3 / "lmk_align_vhap_frame0_v4_rigid/frame_000000_params.npz"
DEFAULT_SHOULDER = EV3 / "neck_shoulder_preview/neck_shoulder_line_000001.npz"
DEFAULT_IMAGE = EV3 / "images/000001.jpg"
DEFAULT_OUT = EV3 / "neck_shoulder_after_rigid"
DEFAULT_RADS = "1.00,1.10,1.15,1.20,1.25,1.30"
DEFAULT_DELTAS = (
    "0,0,0;0.12,0,-0.06;0.12,0,0;0.12,0,-0.12;0.06,0,-0.06;"
    "0.12,0.06,0;0.12,-0.06,0;0.06,0.06,-0.06;-0.06,0,0.06;0,0.12,0;0,-0.12,0"
)


def tp(v) -> torch.Tensor:
  return torch.tensor(np.asarray(v), dtype=torch.float32)


def _floats(s: str):
  return [float(x) for x in str(s).split(",") if x.strip() != ""]


def _triples(s: str):
  out = []
  for chunk in str(s).split(";"):
    chunk = chunk.strip()
    if chunk:
      out.append(tuple(float(v) for v in chunk.split(",")))
  return out


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--params", type=Path, default=DEFAULT_PARAMS)
  ap.add_argument("--shoulder_lines", type=Path, default=DEFAULT_SHOULDER)
  ap.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
  ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
  ap.add_argument("--lens", type=str, default="1.0", help="comma list of neck length scales")
  ap.add_argument("--rads", type=str, default=DEFAULT_RADS, help="comma list of neck radius scales")
  ap.add_argument("--deltas", type=str, default=DEFAULT_DELTAS,
                  help="semicolon-separated 'x,y,z' head axis-angle deltas (rad)")
  ap.add_argument("--face_max", type=float, default=5.0,
                  help="max face(px^2) drift allowed for a grid cell to be valid")
  ap.add_argument("--mode_tag", type=str, default=None, help="summary 'mode' string override")
  ap.add_argument("--vis_name", type=str, default=None, help="notes figure filename override")
  args = ap.parse_args()

  PARAMS = args.params
  SHOULDER = args.shoulder_lines
  IMAGE = args.image
  OUT = args.out_dir
  OUT.mkdir(parents=True, exist_ok=True)
  FACE_MAX = float(args.face_max)
  lens = _floats(args.lens)
  rads = _floats(args.rads)
  deltas = _triples(args.deltas)
  is_more_len = max(lens) > 1.0 + 1e-6
  mode_tag = args.mode_tag or (
      "after_rigid_more_len_jawgate" if is_more_len else "after_rigid_hard_grid_head_jawgate"
  )
  if args.vis_name is not None:
    vis_name = args.vis_name
  elif is_more_len:
    vis_name = "GNM v4 刚性后 颈肩再拉长 肩线前后灰模.jpg"
  else:
    vis_name = "GNM v4 刚性后 颈肩关节微调 肩线前后灰模.jpg"

  bgr = cv2.imread(str(IMAGE))
  H, W = bgr.shape[:2]
  left_gt, right_gt = neck_l2.load_shoulder_gt(SHOULDER)
  left_gt_np = left_gt.detach().cpu().numpy()
  right_gt_np = right_gt.detach().cpu().numpy()

  params = np.load(PARAMS)
  rot = np.asarray(params["rotations"], dtype=np.float32)
  state = FitState(
      id_head=tp(params["id_head"]),
      id_eyes=tp(params["id_eyes"]),
      ex_left=tp(
          params["expression_left_eye"]
          if "expression_left_eye" in params.files
          else np.zeros(100, np.float32)
      ),
      ex_right=tp(
          params["expression_right_eye"]
          if "expression_right_eye" in params.files
          else np.zeros(100, np.float32)
      ),
      ex_lower=tp(params["expression_lower_face"]),
      rot_neck=tp(rot[0]),
      rot_head=tp(rot[1]),
      rot_leye=tp(rot[2]),
      rot_reye=tp(rot[3]),
      translation=tp(params["translation"]),
      cam_aa=tp(params["cam_aa"]),
      cam_t=tp(params["cam_t"]),
      focal_norm=tp(params["focal_norm"]),
  )
  for t in (
      state.id_head, state.id_eyes, state.ex_left, state.ex_right, state.ex_lower,
      state.rot_neck, state.rot_head, state.rot_leye, state.rot_reye,
      state.translation, state.cam_aa, state.cam_t, state.focal_norm,
  ):
    t.requires_grad_(False)

  gnm = gnm_pytorch.GNM.from_local(
      version=gnm_pytorch.GNMMajorVersion.V3, variant=gnm_pytorch.GNMVariant.HEAD,
  )
  gnm_np = gnm_numpy.GNM.from_local(
      version=gnm_numpy.GNMMajorVersion.V3, variant=gnm_numpy.GNMVariant.HEAD,
  )
  faces = np.asarray(gnm.triangles.detach().cpu().numpy())
  ids_np, wfade_np = neck_l2.neck_boundary_vertex_ids(gnm_np, faces, UV_MASKS)
  rest = np.asarray(gnm_np(None, None, None, None), dtype=np.float32)
  origin_np, axis_np = neck_l2.neck_axis_from_rest(rest, ids_np)
  ids = torch.as_tensor(ids_np, dtype=torch.long)
  weights = torch.as_tensor(wfade_np, dtype=torch.float32)
  origin_rest = torch.as_tensor(origin_np, dtype=torch.float32)
  axis_rest = torch.as_tensor(axis_np, dtype=torch.float32)
  print(
      f"[grid] deform_verts={ids_np.size}  wfade>0.05={int((wfade_np > 0.05).sum())}",
      flush=True,
  )

  rot_neck0 = state.rot_neck.detach().clone()
  rot_head0 = state.rot_head.detach().clone()
  trans0 = state.translation.detach().clone()
  t_head_target = neck_l2.head_joint_target(
      gnm, state.identity().detach(), state.rotations().detach(), trans0,
  )
  upper_vids = neck_l2.upper_orient_vertex_ids(gnm_np, faces, UV_MASKS, max_n=300)
  upper_idx = torch.as_tensor(upper_vids, dtype=torch.long)

  def posed_base():
    return gnm(
        state.identity()[None],
        state.expression()[None],
        state.rotations()[None],
        state.translation[None],
    )[0]

  with torch.no_grad():
    state.rot_neck = rot_neck0
    state.rot_head = rot_head0
    state.translation = trans0
    v0 = posed_base()
    upper0 = project_world_to_pixels(
        v0[upper_idx], state.camera_rt(), state.camera_k(W, H), W, H, flip_y=True,
    ).detach().clone()

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  rend_clay = NVDiffRenderer(
      use_opengl=False, lighting_type="front", lighting_space="camera",
      flip_y_raster=False,
  ).to(device)
  rend_paint = NVDiffRenderer(
      use_opengl=False, lighting_type="constant", lighting_space="camera",
      flip_y_raster=False,
  ).to(device)
  faces_t = torch.as_tensor(faces, dtype=torch.int32, device=device)
  verts_uv, faces_uv = uv_from_triangles(gnm_np)
  verts_uv_t = torch.as_tensor(verts_uv, dtype=torch.float32, device=device)
  faces_uv_t = torch.as_tensor(faces_uv, dtype=torch.int32, device=device)
  tex_solid, tex_vis = neck_l2.build_uv_paint_textures(UV_MASKS)
  tex_solid = tex_solid.to(device)
  tex_vis = tex_vis.to(device)

  def apply_d_head(d_aa):
    d = torch.tensor(d_aa, dtype=torch.float32)
    rn, rh, tr = neck_l2.inverse_head_delta(
        gnm,
        state.identity().detach(),
        rot_neck0, rot_head0,
        state.rot_leye.detach(), state.rot_reye.detach(),
        trans0, d, t_head_target,
    )
    state.rot_neck = rn
    state.rot_head = rh
    state.translation = tr
    return d

  def deform(len_s: float, rad_s: float) -> torch.Tensor:
    posed = posed_base()
    t_neck = gnm.get_posed_joint_transforms(
        state.identity()[None],
        state.rotations()[None],
        state.translation[None],
    )[0, 0]
    origin_w = t_neck[:3, :3] @ origin_rest + t_neck[:3, 3]
    axis_w = t_neck[:3, :3] @ axis_rest
    axis_w = axis_w / axis_w.norm().clamp(min=1e-8)
    return neck_l2.deform_neck(
        posed, ids, weights, origin_w, axis_w, float(len_s), float(rad_s),
    )

  def mesh_shoulder_lines(verts_w: torch.Tensor):
    v = verts_w.to(device)[None]
    rt = state.camera_rt().to(device)[None]
    k = state.camera_k(W, H).to(device)[None]
    clay = rend_clay.render_rgba_vis(
        v, faces_t, rt, k, (H, W), background_color=[0.12, 0.12, 0.14],
    )["rgba"][0].detach().float().cpu().numpy()
    mesh_alpha = clay[..., 3] > 0.5
    mesh_bgr = cv2.cvtColor(
        np.clip(clay[..., :3] * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR,
    )
    rgba_s = rend_paint.render_rgba_vis(
        v, faces_t, rt, k, (H, W), background_color=[0, 0, 0],
        verts_uv=verts_uv_t, faces_uv=faces_uv_t, tex=tex_solid, lights=None,
    )["rgba"][0].detach().float().cpu().numpy()
    mask_paint = (rgba_s[..., 1] > 0.2) & (rgba_s[..., 3] > 0.4)
    rgba_v = rend_paint.render_rgba_vis(
        v, faces_t, rt, k, (H, W), background_color=[0, 0, 0],
        verts_uv=verts_uv_t, faces_uv=faces_uv_t, tex=tex_vis, lights=None,
    )["rgba"][0].detach().float().cpu().numpy()
    r, g2, a2 = rgba_v[..., 0], rgba_v[..., 1], rgba_v[..., 3]
    mask_neck = (g2 > 0.3) & (r < 0.25) & (a2 > 0.4)
    if int(mask_neck.sum()) < 200:
      ys = np.where(mask_paint.any(1))[0]
      if ys.size:
        y_cut = np.percentile(ys, 55)
        mask_neck = mask_paint & (np.arange(H)[:, None] <= y_cut)
      else:
        mask_neck = mask_paint
    left, right, meta = neck_l2.extract_shoulder_lines(
        mask_paint, mask_neck, mesh_alpha, H, W,
    )
    return left, right, mesh_bgr, meta

  with torch.no_grad():
    apply_d_head((0.0, 0.0, 0.0))
    left0, right0, _, _ = mesh_shoulder_lines(posed_base())
    hard0 = (
        neck_l2.chamfer_lines_np(left0, left_gt_np)
        + neck_l2.chamfer_lines_np(right0, right_gt_np)
    )
  print(f"[grid] baseline hard_chamfer={hard0:.2f}", flush=True)

  best = None
  rows = []
  for len_s in lens:
    for rad in rads:
      for d_aa in deltas:
        with torch.no_grad():
          apply_d_head(d_aa)
          verts = deform(len_s, rad)
          left, right, clay_bgr, meta = mesh_shoulder_lines(verts)
          hard = (
              neck_l2.chamfer_lines_np(left, left_gt_np)
              + neck_l2.chamfer_lines_np(right, right_gt_np)
          )
          face_xy = project_world_to_pixels(
              verts[upper_idx], state.camera_rt(), state.camera_k(W, H), W, H, flip_y=True,
          )
          face_mse = float((face_xy - upper0).pow(2).mean())
        row = {
            "len": float(len_s), "rad": float(rad), "d_head": list(d_aa),
            "hard": hard, "face_mse": face_mse,
        }
        rows.append(row)
        ok = face_mse <= FACE_MAX
        if ok and (best is None or hard < best["hard"]):
          best = {
              **row,
              "verts": verts.detach().clone(),
              "clay_bgr": clay_bgr,
              "meta": meta,
              "left": left,
              "right": right,
              "rot_neck": state.rot_neck.detach().clone(),
              "rot_head": state.rot_head.detach().clone(),
              "translation": state.translation.detach().clone(),
          }
        if d_aa == (0.0, 0.0, 0.0) or (ok and hard < hard0 * 0.7):
          print(
              f"  len={len_s:.2f} rad={rad:.2f} d={d_aa}  hard={hard:.1f}  "
              f"face={face_mse:.3f}  {'OK' if ok else 'skip'}",
              flush=True,
          )

  assert best is not None
  print(
      f"[grid] BEST len={best['len']:.2f} rad={best['rad']:.2f} d_head={best['d_head']} "
      f"hard={best['hard']:.2f} (base {hard0:.2f}) face={best['face_mse']:.3f}",
      flush=True,
  )

  with torch.no_grad():
    state.rot_neck = best["rot_neck"]
    state.rot_head = best["rot_head"]
    state.translation = best["translation"]
    posed_only = posed_base()
    offset = (best["verts"] - posed_only).detach().cpu().numpy().astype(np.float32)

  def draw_poly(im, pts, col, th):
    if pts is None or len(pts) < 2:
      return im
    p = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
    return cv2.polylines(im, [p], False, col, th, cv2.LINE_AA)

  def mark4(im, meta):
    if not meta:
      return im
    out = im.copy()
    for key, col in (
        ("TL", (0, 255, 0)), ("TR", (0, 255, 0)),
        ("WL", (0, 165, 255)), ("WR", (0, 165, 255)),
    ):
      if key not in meta:
        continue
      P = meta[key]
      cv2.circle(out, (int(P[0]), int(P[1])), 7, col, -1)
    return out

  def label(im, title):
    out = im.copy()
    cv2.putText(out, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
    cv2.putText(out, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    return out

  with torch.no_grad():
    apply_d_head((0.0, 0.0, 0.0))
    left_b, right_b, _, meta_b = mesh_shoulder_lines(posed_base())

  def panel(title, left_m, right_m, meta=None):
    im = bgr.copy()
    im = draw_poly(im, left_gt_np, (0, 255, 255), 3)
    im = draw_poly(im, right_gt_np, (0, 255, 255), 3)
    im = draw_poly(im, left_m, (255, 0, 255), 2)
    im = draw_poly(im, right_m, (255, 0, 255), 2)
    im = mark4(im, meta)
    return label(im, title)

  clay = label(mark4(best["clay_bgr"], best["meta"]), "AFTER mesh (clay)")
  ba = np.concatenate(
      [
          panel("BEFORE cyan=photo magenta=mesh", left_b, right_b, meta_b),
          panel(
              f"AFTER len={best['len']:.2f} rad={best['rad']:.2f} head+jawgate",
              best["left"], best["right"], best["meta"],
          ),
          clay,
      ],
      1,
  )
  notes = Path("/root/autodl-tmp/notes/gnm_uv/测试用图片")
  notes.mkdir(parents=True, exist_ok=True)
  vis_path = notes / vis_name
  cv2.imwrite(str(vis_path), ba)
  cv2.imwrite(str(OUT / "shoulder_before_after_clay.jpg"), ba)

  rots = rot.copy()
  rots[0] = best["rot_neck"].cpu().numpy().astype(np.float32)
  rots[1] = best["rot_head"].cpu().numpy().astype(np.float32)
  save = {k: np.asarray(params[k]) for k in params.files}
  save["rotations"] = rots
  save["translation"] = best["translation"].cpu().numpy().astype(np.float32)
  save["static_offset"] = offset
  np.savez(OUT / "frame_000000_params.npz", **save)

  top = sorted(
      [r for r in rows if r["face_mse"] <= FACE_MAX],
      key=lambda r: r["hard"],
  )[:8]
  summary = {
      "mode": mode_tag,
      "lock": "upper",
      "joint_target": "head",
      "len_s": float(best["len"]),
      "rad_s": float(best["rad"]),
      "d_head": best["d_head"],
      "hard_chamfer_before": float(hard0),
      "hard_chamfer_after": float(best["hard"]),
      "face_mse": float(best["face_mse"]),
      "face_max": FACE_MAX,
      "params_in": str(PARAMS),
      "vis": str(vis_path),
      "grid_rows_top": top,
      "note": "ablation: joints + jaw-gated rad; soft face fade; no hard face drop",
  }
  (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
  print(json.dumps(summary, indent=2), flush=True)
  print("[grid] wrote", OUT / "frame_000000_params.npz", flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
