#!/usr/bin/env python3
"""Smooth localized ear alignment (head FROZEN), v2.

Changes vs v1:
  * Weight field is anisotropic and reaches UP into the temple / upper skull, so
    the head side expands with the ear (no more "big face, small head").
  * Hinge loss max(0, dt - margin): only pull the ear *toward* the mask, not a
    hard full fit; coefficient caps are smaller.
  * Mask quality gate: if either left/right mask is missing or degenerate, the
    whole supervision stage is skipped (params copied unchanged).
  * Left/right symmetry: weights and basis are built in |x| canonical space, so
    both sides transform identically.

Head pose / identity / expression are frozen (after SAM eyes the head must not
move). Deformation is baked into static_offset. Emits a one-row 4-panel grid.
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

NOTES_DIR = Path("/root/autodl-tmp/notes/gnm_uv/测试用图片")


def _tp(v):
  return torch.tensor(np.asarray(v), dtype=torch.float32)


def load_state(params):
  rot = np.asarray(params["rotations"], np.float32)
  return FitState(
      id_head=_tp(params["id_head"]), id_eyes=_tp(params["id_eyes"]),
      ex_left=_tp(params["expression_left_eye"] if "expression_left_eye" in params.files else np.zeros(100, np.float32)),
      ex_right=_tp(params["expression_right_eye"] if "expression_right_eye" in params.files else np.zeros(100, np.float32)),
      ex_lower=_tp(params["expression_lower_face"]),
      rot_neck=_tp(rot[0]), rot_head=_tp(rot[1]), rot_leye=_tp(rot[2]), rot_reye=_tp(rot[3]),
      translation=_tp(params["translation"]), cam_aa=_tp(params["cam_aa"]),
      cam_t=_tp(params["cam_t"]), focal_norm=_tp(params["focal_norm"]),
  )


def mask_quality(mask: np.ndarray, W: int, H: int, min_area: int, max_area: int):
  """Return (ok_basic, cx, reason). Does not assume which image side is 'left'."""
  n, labels, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
  if n <= 1:
    return False, -1.0, "empty"
  areas = stats[1:, cv2.CC_STAT_AREA]
  k = 1 + int(np.argmax(areas))
  area = int(stats[k, cv2.CC_STAT_AREA])
  if area < min_area:
    return False, -1.0, f"area={area}<{min_area}"
  if area > max_area:
    return False, -1.0, f"area={area}>{max_area}"
  x, y, w, h = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP], stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
  if w < 4 or h < 6:
    return False, -1.0, f"bbox {w}x{h} tiny"
  ratio = h / max(w, 1)
  if ratio < 0.5 or ratio > 5.0:
    return False, -1.0, f"aspect={ratio:.2f}"
  if area / (w * h) < 0.25:
    return False, -1.0, f"fill={area/(w*h):.2f} too sparse"
  return True, float(cents[k, 0]), "ok"


def build_blend_weights(rest: np.ndarray, ear_vids: np.ndarray,
                        sig_x: float, sig_y_up: float, sig_y_dn: float, sig_z: float):
  """Anisotropic Gaussian falloff from the right-ear centroid in |x| canonical
  space, GATED to the lateral side of the head so it never bleeds onto the
  front face / nose. Symmetric for both sides; reaches up into temple/skull."""
  x_right = rest[ear_vids][rest[ear_vids, 0] > 0.0]
  if x_right.shape[0] == 0:
    x_right = rest[ear_vids]
  C = x_right.mean(axis=0)
  Cr = np.array([abs(C[0]), C[1], C[2]], np.float64)
  xc = np.abs(rest[:, 0])
  dx = xc - Cr[0]
  dy = rest[:, 1] - Cr[1]
  dz = rest[:, 2] - Cr[2]
  sy = np.where(dy >= 0, sig_y_up, sig_y_dn)
  d2 = (dx / sig_x) ** 2 + (dy / sy) ** 2 + (dz / sig_z) ** 2
  w = np.exp(-0.5 * d2)
  # lateral gate: kill face centre (|x| tiny) so nose/cheeks are untouched
  gx = np.clip((xc - 0.040) / 0.020, 0.0, 1.0)
  # front gate: exclude front face (z large); keep ear-plane and a bit behind
  gz = np.clip((0.065 - rest[:, 2]) / 0.025, 0.0, 1.0)
  # back gate: exclude deep back-of-head
  gb = np.clip((rest[:, 2] + 0.060) / 0.020, 0.0, 1.0)
  w = (w * gx * gz * gb).astype(np.float32)
  w = np.clip(w, 0.0, 1.0)
  w[ear_vids] = 1.0
  return w, Cr


def build_basis(rest: np.ndarray, w: np.ndarray, Cr: np.ndarray):
  """Semantic directions over the whole mesh, scaled by blend weight w.
  Built in |x| canonical space -> mirror-symmetric apply on both sides."""
  n = rest.shape[0]
  xs = np.sign(rest[:, 0]); xs[xs == 0] = 1.0
  f_lat = np.zeros((n, 3), np.float32); f_lat[:, 0] = xs
  f_up = np.zeros((n, 3), np.float32); f_up[:, 1] = 1.0
  f_front = np.zeros((n, 3), np.float32); f_front[:, 2] = 1.0
  # per-ear radial scale: centroid mirrored to each side
  cent = np.zeros((n, 3), np.float32)
  cent[:, 0] = xs * Cr[0]
  cent[:, 1] = Cr[1]
  cent[:, 2] = Cr[2]
  f_scale = (rest - cent) * 0.1
  out = []
  for f in (f_lat, f_up, f_front, f_scale):
    out.append(torch.as_tensor(f * w[:, None], dtype=torch.float32))
  return out


def render_row(vw, faces, rt, k4, W, H, bgr, left, right, combined):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  faces_t = torch.as_tensor(faces, dtype=torch.int32, device=device)
  rend_clay = NVDiffRenderer(use_opengl=False, lighting_type="front", lighting_space="camera", flip_y_raster=False).to(device)
  rend_paint = NVDiffRenderer(use_opengl=False, lighting_type="constant", lighting_space="camera", flip_y_raster=False).to(device)
  with torch.no_grad():
    clay = rend_clay.render_rgba_vis(vw.to(device), faces_t, rt, k4, (H, W), background_color=[0.12, 0.12, 0.14])["rgba"][0].float().cpu().numpy()
    paint = rend_paint.render_rgba_vis(vw.to(device), faces_t, rt, k4, (H, W), background_color=[0, 0, 0])["rgba"][0].float().cpu().numpy()
  clay_bgr = cv2.cvtColor(np.clip(clay[..., :3] * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
  mesh_alpha = paint[..., 3] > 0.5
  p1 = bgr.copy()
  p2 = bgr.copy(); over = p2.copy(); over[left] = (0, 255, 0); over[right] = (0, 0, 255)
  p2 = cv2.addWeighted(p2, 0.55, over, 0.45, 0)
  cnts, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cv2.drawContours(p2, cnts, -1, (0, 255, 255), 2)
  p3 = bgr.copy(); bl = p3.copy(); bl[mesh_alpha] = (255, 120, 0)
  p3 = cv2.addWeighted(p3, 0.4, bl, 0.6, 0)
  p4 = clay_bgr.copy()
  def label(img, t):
    cv2.putText(img, t, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, t, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return img
  return np.concatenate([label(p1, "GT"), label(p2, "GT+耳朵mask"),
                         label(p3, "mesh蓝色透明"), label(p4, "mesh灰模")], axis=1)


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--params", type=Path, required=True)
  ap.add_argument("--ear_mask_npz", type=Path, required=True)
  ap.add_argument("--image", type=Path, required=True)
  ap.add_argument("--out_dir", type=Path, required=True)
  ap.add_argument("--iters", type=int, default=250)
  ap.add_argument("--lr", type=float, default=2e-3)
  ap.add_argument("--w_reg", type=float, default=3.0)
  ap.add_argument("--margin_px", type=float, default=8.0,
                  help="Hinge margin: stop pulling once ear verts are within this distance of the mask.")
  ap.add_argument("--dt_cap", type=float, default=24.0)
  ap.add_argument("--sig_x", type=float, default=0.06)
  ap.add_argument("--sig_y_up", type=float, default=0.12, help="Upward sigma (temple/skull reach).")
  ap.add_argument("--sig_y_dn", type=float, default=0.035)
  ap.add_argument("--sig_z", type=float, default=0.07)
  ap.add_argument("--lat_max", type=float, default=0.012)
  ap.add_argument("--vert_max", type=float, default=0.012)
  ap.add_argument("--front_max", type=float, default=0.010)
  ap.add_argument("--scale_max", type=float, default=0.12)
  ap.add_argument("--out_name", type=str, default="GNM v4 耳朵平滑对称局部变形 四格.jpg")
  args = ap.parse_args()
  args.out_dir.mkdir(parents=True, exist_ok=True)

  bgr0 = cv2.imread(str(args.image))
  if bgr0 is None:
    raise FileNotFoundError(args.image)
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
  rest = np.asarray(gnm_np(None, None, None, None), dtype=np.float32)
  n_verts = int(rest.shape[0])

  offset_np = np.asarray(params["static_offset"], np.float32) if "static_offset" in params.files else None
  base_off = torch.zeros(n_verts, 3)
  if offset_np is not None:
    base_off = torch.as_tensor(offset_np, dtype=torch.float32)

  masks = np.load(args.ear_mask_npz)
  left = cv2.resize(np.asarray(masks["left"], np.float32), (W, H), interpolation=cv2.INTER_NEAREST) > 0.5
  right = cv2.resize(np.asarray(masks["right"], np.float32), (W, H), interpolation=cv2.INTER_NEAREST) > 0.5
  combined = (left | right).astype(np.uint8)
  ok_a, cx_a, why_a = mask_quality(left, W, H, 250, 8000)
  ok_b, cx_b, why_b = mask_quality(right, W, H, 250, 8000)
  # require the two masks to sit on opposite image halves (mirror pair); naming not assumed.
  opposite = (cx_a >= 0 and cx_b >= 0 and (cx_a - W * 0.5) * (cx_b - W * 0.5) < 0)
  ok = ok_a and ok_b and opposite
  print(f"[ear] maskA ok={ok_a}({why_a}) cx={cx_a:.0f}  maskB ok={ok_b}({why_b}) cx={cx_b:.0f}  "
        f"opposite={opposite}", flush=True)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  rt = state.camera_rt().detach()[None].to(device)
  k4 = state.camera_k(W, H).detach()[None].to(device)
  with torch.no_grad():
    posed = gnm(state.identity()[None], state.expression()[None], state.rotations()[None], state.translation[None])[0]

  skip = not ok
  if skip:
    print("[ear] mask bad/missing on one side -> SKIP ear supervision stage", flush=True)
    with torch.no_grad():
      vw = (posed + base_off).to(device)[None]
    row = render_row(vw, faces, rt, k4, W, H, bgr, left, right, combined)
    cv2.putText(row, "SKIP (mask bad)", (12, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    notes_path = NOTES_DIR / args.out_name
    cv2.imwrite(str(notes_path), row)
    save = {k: np.asarray(params[k]) for k in params.files}
    np.savez(args.out_dir / "frame_000000_params.npz", **save)
    (args.out_dir / "summary.json").write_text(json.dumps(
        {"skipped": True, "maskA": why_a, "maskB": why_b, "opposite": opposite, "notes_row": str(notes_path)},
        indent=2, ensure_ascii=False))
    print("skipped; params unchanged ->", args.out_dir / "frame_000000_params.npz", flush=True)
    return 0

  inv = (1 - combined).astype(np.uint8)
  dt = cv2.distanceTransform(inv, cv2.DIST_L2, 5).astype(np.float32)
  dt_t = torch.as_tensor(dt, dtype=torch.float32)[None, None]

  w_np, Cr = build_blend_weights(rest, ear_vids, args.sig_x, args.sig_y_up, args.sig_y_dn, args.sig_z)
  print(f"[ear] blend influent verts(w>0.05)={int((w_np>0.05).sum())}  ear={ear_vids.size}  Crabs=[{Cr[0]:.3f},{Cr[1]:.3f},{Cr[2]:.3f}]", flush=True)
  basis = build_basis(rest, w_np, Cr)
  coef = torch.zeros(4, requires_grad=True)

  def full_verts() -> torch.Tensor:
    d = torch.zeros(n_verts, 3)
    for c, f in zip(coef, basis):
      d = d + c * f
    return posed + base_off + d

  ear_idx_t = torch.as_tensor(ear_vids, dtype=torch.long)

  def ear_hinge_loss() -> torch.Tensor:
    v = full_verts()
    xy = project_world_to_pixels(v[ear_idx_t], state.camera_rt(), state.camera_k(W, H), W, H, flip_y=True)
    gx = 2.0 * xy[:, 0] / W - 1.0
    gy = 2.0 * xy[:, 1] / H - 1.0
    d = torch.nn.functional.grid_sample(dt_t, torch.stack([gx, gy], -1)[None, None], align_corners=False)[0, 0, 0]
    d = d.clamp(max=float(args.dt_cap))
    return torch.clamp(d - float(args.margin_px), min=0.0).mean()

  opt = torch.optim.Adam([coef], lr=args.lr)
  lim = torch.tensor([args.lat_max, args.vert_max, args.front_max, args.scale_max])
  last = 0.0
  for i in range(int(args.iters)):
    opt.zero_grad()
    ear_loss = ear_hinge_loss()
    loss = ear_loss + args.w_reg * (coef / lim).pow(2).sum()
    loss.backward()
    opt.step()
    with torch.no_grad():
      coef.clamp_(-lim, lim)
    last = float(ear_loss)
    if i == 0 or (i + 1) % max(args.iters // 10, 1) == 0 or i + 1 == args.iters:
      print(f"  [{i+1:4d}/{args.iters}] ear_hinge={last:.3f}  coef={[round(float(c),4) for c in coef]}", flush=True)

  with torch.no_grad():
    vw = full_verts().to(device)[None]
  row = render_row(vw, faces, rt, k4, W, H, bgr, left, right, combined)
  row_path = args.out_dir / "ear_align_row.jpg"
  cv2.imwrite(str(row_path), row)
  NOTES_DIR.mkdir(parents=True, exist_ok=True)
  notes_path = NOTES_DIR / args.out_name
  cv2.imwrite(str(notes_path), row)

  with torch.no_grad():
    full_off = base_off + sum(c * f for c, f in zip(coef, basis))
  save = {k: np.asarray(params[k]) for k in params.files}
  save["static_offset"] = full_off.detach().cpu().numpy().astype(np.float32)
  np.savez(args.out_dir / "frame_000000_params.npz", **save)
  summary = {
      "skipped": False,
      "ear_verts": int(ear_vids.size),
      "influenced_verts": int((w_np > 0.05).sum()),
      "coef": {n: float(c) for n, c in zip(("lat", "vert", "front", "scale"), coef.detach().tolist())},
      "ear_hinge_final": last,
      "margin_px": args.margin_px,
      "head_frozen": True,
      "notes_row": str(notes_path),
  }
  (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
  print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
