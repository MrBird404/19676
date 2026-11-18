#!/usr/bin/env python3
"""Joint PCA-style fit: global (len, rad, d_head) + smooth neck-shoulder modes.

Search space (start at the after_rigid / hard~382 solution):
  len in [1.0,1.5], rad in [1.0,1.6], d_head in [-0.5,0.5]^3,
  plus K smooth modes (trap/trans/widen, built with semantic_basis.solve_measure).

Render-based coordinate descent on HARD shoulder chamfer -- so the optimiser may
either walk the global params toward the len_more (~26) corner OR use the local
smooth modes; we just observe what it does from the 382 start.
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
REPO = next((p for p in [ROOT, *ROOT.parents] if (p / "GNM-main").is_dir()), ROOT.parent) / "GNM-main"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT))

from fit_gnm import FitState  # noqa: E402
from gnm.shape import gnm_numpy, gnm_pytorch  # noqa: E402
from nvdiffrast_render import NVDiffRenderer  # noqa: E402
from nvdiffrast_render.camera import project_world_to_pixels  # noqa: E402
from photo_fit import UV_MASKS, uv_from_triangles  # noqa: E402
import neck_shoulder_l2 as neck_l2  # noqa: E402
import semantic_basis as sb  # noqa: E402

EV3 = Path("/root/autodl-tmp/gnm_fit_out/expression_v3")


def tp(v):
  return torch.tensor(np.asarray(v), dtype=torch.float32)


def build_modes():
  g, rest, B, grp, dirs, neck = sb.build()
  V = rest.shape[0]
  ys = rest[:, 1]

  def radial(region):
    c = rest[region].mean(0)
    r = rest[region] - c
    r[:, 1] = 0.0
    n = np.linalg.norm(r, axis=1, keepdims=True)
    n[n < 1e-9] = 1.0
    return r / n

  lo = neck[ys[neck] <= 0.145]
  mid = neck[(ys[neck] >= 0.130) & (ys[neck] <= 0.175)]
  modes = {}
  gt = np.zeros((V, 3))
  gt[lo] = (radial(lo) * 0.8 + np.array([0.0, -0.4, 0.0])) / max(lo.size, 1)
  modes["trap"] = sb.solve_measure(B, rest, gt, lo, y_band=[0.10, 0.15], lam_out=45.0, falloff_mm=10.0, target_mm=4.0)[0]
  gm = np.zeros((V, 3))
  gm[mid] = radial(mid) / max(mid.size, 1)
  modes["trans"] = sb.solve_measure(B, rest, gm, mid, y_band=[0.12, 0.18], lam_out=45.0, falloff_mm=9.0, target_mm=3.0)[0]
  gw = np.zeros((V, 3))
  sx = np.sign(rest[lo, 0]); sx[sx == 0] = 1.0
  gw[lo, 0] = sx / max(lo.size, 1)
  modes["widen"] = sb.solve_measure(B, rest, gw, lo, y_band=[0.10, 0.15], lam_out=45.0, falloff_mm=10.0, target_mm=3.5)[0]

  lo2 = neck[ys[neck] <= 0.125]
  hi = neck[ys[neck] >= 0.160]
  gd = np.zeros((V, 3)); gd[lo, 1] = -1.0 / max(lo.size, 1)
  modes["drop"] = sb.solve_measure(B, rest, gd, lo, y_band=[0.10, 0.15], lam_out=45.0, falloff_mm=10.0, target_mm=4.0)[0]
  ghi = np.zeros((V, 3)); ghi[hi] = radial(hi) / max(hi.size, 1)
  modes["trans_hi"] = sb.solve_measure(B, rest, ghi, hi, y_band=[0.15, 0.20], lam_out=45.0, falloff_mm=9.0, target_mm=3.0)[0]
  gl = np.zeros((V, 3)); gl[lo2] = radial(lo2) / max(lo2.size, 1)
  modes["low_flare"] = sb.solve_measure(B, rest, gl, lo2, y_band=[0.10, 0.14], lam_out=45.0, falloff_mm=8.0, target_mm=3.5)[0]
  gs = np.zeros((V, 3)); sxx = np.sign(rest[lo, 0]); sxx[sxx == 0] = 1.0
  gs[lo, 0] = sxx / max(lo.size, 1); gs[lo, 1] = -1.0 / max(lo.size, 1)
  modes["slope"] = sb.solve_measure(B, rest, gs, lo, y_band=[0.10, 0.15], lam_out=45.0, falloff_mm=10.0, target_mm=3.5)[0]
  ga = np.zeros((V, 3)); ga[lo, 2] = 1.0 / max(lo.size, 1)
  modes["ant"] = sb.solve_measure(B, rest, ga, lo, y_band=[0.10, 0.15], lam_out=45.0, falloff_mm=10.0, target_mm=3.0)[0]
  gp = np.zeros((V, 3)); gp[lo, 2] = -1.0 / max(lo.size, 1)
  modes["post"] = sb.solve_measure(B, rest, gp, lo, y_band=[0.10, 0.15], lam_out=45.0, falloff_mm=10.0, target_mm=3.0)[0]
  lm = neck[(ys[neck] >= 0.130) & (ys[neck] <= 0.150)]
  glt = np.zeros((V, 3)); sxl = np.sign(rest[lm, 0]); sxl[sxl == 0] = 1.0
  glt[lm, 0] = sxl / max(lm.size, 1)
  modes["lat"] = sb.solve_measure(B, rest, glt, lm, y_band=[0.12, 0.16], lam_out=45.0, falloff_mm=9.0, target_mm=3.0)[0]
  hc = neck[(ys[neck] >= 0.160) & (ys[neck] <= 0.195)]
  gc = np.zeros((V, 3)); gc[hc] = (radial(hc) * 0.7 + np.array([0.0, -0.3, 0.0])) / max(hc.size, 1)
  modes["cap"] = sb.solve_measure(B, rest, gc, hc, y_band=[0.15, 0.20], lam_out=45.0, falloff_mm=9.0, target_mm=3.5)[0]
  return g, rest, neck, modes


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--params", type=Path, default=EV3 / "lmk_align_vhap_frame0_v4_rigid/frame_000000_params.npz")
  ap.add_argument("--shoulder_lines", type=Path, default=EV3 / "neck_shoulder_preview/neck_shoulder_line_000001.npz")
  ap.add_argument("--image", type=Path, default=EV3 / "images/000001.jpg")
  ap.add_argument("--out_dir", type=Path, default=EV3 / "neck_pca_global")
  ap.add_argument("--start_len", type=float, default=1.0)
  ap.add_argument("--start_rad", type=float, default=1.2)
  ap.add_argument("--start_d_head", type=str, default="0.12,0,-0.06")
  ap.add_argument("--rounds", type=int, default=10)
  ap.add_argument("--mode_set", choices=["k3", "k7", "k11"], default="k3")
  args = ap.parse_args()
  args.out_dir.mkdir(parents=True, exist_ok=True)
  start_dh = [float(x) for x in args.start_d_head.split(",")]

  g_sb, rest, neck_idx, modes = build_modes()
  _sets = {
      "k3": ["trap", "trans", "widen"],
      "k7": ["trap", "trans", "widen", "drop", "trans_hi", "low_flare", "slope"],
      "k11": ["trap", "trans", "widen", "drop", "trans_hi", "low_flare", "slope",
              "ant", "post", "lat", "cap"],
  }
  names = _sets[args.mode_set]
  M = np.stack([modes[n] for n in names], 0).astype(np.float32)  # [K,V,3] mm
  K = M.shape[0]
  print(f"[pca] modes={names} neck_verts={neck_idx.size}", flush=True)

  bgr = cv2.imread(str(args.image)); H, W = bgr.shape[:2]
  left_gt, right_gt = neck_l2.load_shoulder_gt(args.shoulder_lines)
  left_gt_np = left_gt.detach().cpu().numpy(); right_gt_np = right_gt.detach().cpu().numpy()

  params = np.load(args.params); rot = np.asarray(params["rotations"], dtype=np.float32)
  state = FitState(
      id_head=tp(params["id_head"]), id_eyes=tp(params["id_eyes"]),
      ex_left=tp(params["expression_left_eye"] if "expression_left_eye" in params.files else np.zeros(100, np.float32)),
      ex_right=tp(params["expression_right_eye"] if "expression_right_eye" in params.files else np.zeros(100, np.float32)),
      ex_lower=tp(params["expression_lower_face"]),
      rot_neck=tp(rot[0]), rot_head=tp(rot[1]), rot_leye=tp(rot[2]), rot_reye=tp(rot[3]),
      translation=tp(params["translation"]), cam_aa=tp(params["cam_aa"]),
      cam_t=tp(params["cam_t"]), focal_norm=tp(params["focal_norm"]))
  for t in (state.id_head, state.id_eyes, state.ex_left, state.ex_right, state.ex_lower,
            state.rot_neck, state.rot_head, state.rot_leye, state.rot_reye,
            state.translation, state.cam_aa, state.cam_t, state.focal_norm):
    t.requires_grad_(False)

  gnm = gnm_pytorch.GNM.from_local(version=gnm_pytorch.GNMMajorVersion.V3, variant=gnm_pytorch.GNMVariant.HEAD)
  gnm_np = gnm_numpy.GNM.from_local(version=gnm_numpy.GNMMajorVersion.V3, variant=gnm_numpy.GNMVariant.HEAD)
  faces = np.asarray(gnm.triangles.detach().cpu().numpy())
  ids_np, wfade_np = neck_l2.neck_boundary_vertex_ids(gnm_np, faces, UV_MASKS)
  rest_gnm = np.asarray(gnm_np(None, None, None, None), dtype=np.float32)
  origin_np, axis_np = neck_l2.neck_axis_from_rest(rest_gnm, ids_np)
  ids = torch.as_tensor(ids_np, dtype=torch.long)
  weights = torch.as_tensor(wfade_np, dtype=torch.float32)
  origin_rest = torch.as_tensor(origin_np, dtype=torch.float32)
  axis_rest = torch.as_tensor(axis_np, dtype=torch.float32)

  rot_neck0 = state.rot_neck.detach().clone(); rot_head0 = state.rot_head.detach().clone()
  trans0 = state.translation.detach().clone()
  t_head_target = neck_l2.head_joint_target(gnm, state.identity().detach(), state.rotations().detach(), trans0)

  def posed_base():
    return gnm(state.identity()[None], state.expression()[None], state.rotations()[None], state.translation[None])[0]

  assert rest.shape[0] == rest_gnm.shape[0], (rest.shape[0], rest_gnm.shape[0])
  M_t = torch.as_tensor(M / 1000.0, dtype=torch.float32)  # mm->m
  neck_t = torch.as_tensor(neck_idx, dtype=torch.long)

  # Rad-only jaw guard: thickness must NOT reach the jaw. Gate the radial delta
  # to ~0 near the jaw (high y) and full near the shoulders (low y). Length (len)
  # is left completely untouched (that part was already jaw-safe).
  y_id = rest_gnm[ids_np, 1]
  y_lo = float(np.quantile(y_id, 0.20)); y_hi = float(np.quantile(y_id, 0.55))
  rg = np.clip((y_hi - y_id) / max(y_hi - y_lo, 1e-4), 0.0, 1.0)
  rad_gate = torch.as_tensor((rg * rg * (3.0 - 2.0 * rg)).astype(np.float32), dtype=torch.float32)
  # Same under-chin guard for the smooth modes: chin rigid, shoulders free.
  y_mode = rest_gnm[neck_idx, 1]
  mg = np.clip((y_hi - y_mode) / max(y_hi - y_lo, 1e-4), 0.0, 1.0)
  mode_gate = torch.as_tensor((mg * mg * (3.0 - 2.0 * mg)).astype(np.float32), dtype=torch.float32)

  def build_verts(P):
    len_s = float(P[0]); rad_s = float(P[1]); dh = [float(P[2]), float(P[3]), float(P[4])]
    cvec = torch.as_tensor(P[5:5 + K], dtype=torch.float32)
    d = torch.tensor(dh, dtype=torch.float32)
    rn, rh, tr = neck_l2.inverse_head_delta(gnm, state.identity().detach(), rot_neck0, rot_head0,
                                            state.rot_leye.detach(), state.rot_reye.detach(), trans0, d, t_head_target)
    state.rot_neck = rn; state.rot_head = rh; state.translation = tr
    posed = posed_base()
    t_neck = gnm.get_posed_joint_transforms(state.identity()[None], state.rotations()[None], state.translation[None])[0, 0]
    origin_w = t_neck[:3, :3] @ origin_rest + t_neck[:3, 3]
    axis_w = t_neck[:3, :3] @ axis_rest; axis_w = axis_w / axis_w.norm().clamp(min=1e-8)
    v_len = neck_l2.deform_neck(posed, ids, weights, origin_w, axis_w, len_s, 1.0)
    v_full = neck_l2.deform_neck(posed, ids, weights, origin_w, axis_w, len_s, rad_s)
    v = v_len.clone()
    v[ids] = v_len[ids] + rad_gate[:, None] * (v_full[ids] - v_len[ids])
    if cvec is not None:
      disp = torch.einsum("k,kvd->vd", cvec, M_t[:, neck_t, :]) * mode_gate[:, None]
      v = v.clone(); v[neck_t] = v[neck_t] + disp
    return v

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  rc = NVDiffRenderer(use_opengl=False, lighting_type="front", lighting_space="camera", flip_y_raster=False).to(device)
  rp = NVDiffRenderer(use_opengl=False, lighting_type="constant", lighting_space="camera", flip_y_raster=False).to(device)
  faces_t = torch.as_tensor(faces, dtype=torch.int32, device=device)
  verts_uv, faces_uv = uv_from_triangles(gnm_np)
  verts_uv_t = torch.as_tensor(verts_uv, dtype=torch.float32, device=device)
  faces_uv_t = torch.as_tensor(faces_uv, dtype=torch.int32, device=device)
  tex_solid, tex_vis = neck_l2.build_uv_paint_textures(UV_MASKS)
  tex_solid = tex_solid.to(device); tex_vis = tex_vis.to(device)

  def mesh_lines(vw):
    v = vw.to(device)[None]; rt = state.camera_rt().to(device)[None]; k = state.camera_k(W, H).to(device)[None]
    clay = rc.render_rgba_vis(v, faces_t, rt, k, (H, W), background_color=[0.12, 0.12, 0.14])["rgba"][0].detach().float().cpu().numpy()
    ma = clay[..., 3] > 0.5
    cb = cv2.cvtColor(np.clip(clay[..., :3] * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    rs = rp.render_rgba_vis(v, faces_t, rt, k, (H, W), background_color=[0, 0, 0], verts_uv=verts_uv_t, faces_uv=faces_uv_t, tex=tex_solid, lights=None)["rgba"][0].detach().float().cpu().numpy()
    mp = (rs[..., 1] > 0.2) & (rs[..., 3] > 0.4)
    rv = rp.render_rgba_vis(v, faces_t, rt, k, (H, W), background_color=[0, 0, 0], verts_uv=verts_uv_t, faces_uv=faces_uv_t, tex=tex_vis, lights=None)["rgba"][0].detach().float().cpu().numpy()
    r, g2, a2 = rv[..., 0], rv[..., 1], rv[..., 3]
    mn = (g2 > 0.3) & (r < 0.25) & (a2 > 0.4)
    if int(mn.sum()) < 200:
      ys_ = np.where(mp.any(1))[0]; mn = mp & (np.arange(H)[:, None] <= np.percentile(ys_, 55)) if ys_.size else mp
    l, rr, meta = neck_l2.extract_shoulder_lines(mp, mn, ma, H, W)
    return l, rr, cb, meta

  def hard(P):
    with torch.no_grad():
      l, rr, _, _ = mesh_lines(build_verts(P))
    return neck_l2.chamfer_lines_np(l, left_gt_np) + neck_l2.chamfer_lines_np(rr, right_gt_np)

  # param bounds/steps: [len, rad, dhx, dhy, dhz, cK...]
  lo = [1.0, 1.0, -0.5, -0.5, -0.5] + [-1.5] * K
  hi = [1.5, 1.6, 0.5, 0.5, 0.5] + [1.5] * K
  steps = [0.05, 0.05, 0.04, 0.04, 0.04] + [0.3] * K
  P = [args.start_len, args.start_rad, start_dh[0], start_dh[1], start_dh[2]] + [0.0] * K
  best = hard(P)
  print(f"[pca] start hard={best:.2f}  P={[round(x,3) for x in P]}", flush=True)

  for rnd in range(int(args.rounds)):
    improved = False
    for i in range(len(P)):
      for sgn in (1.0, -1.0):
        cand = list(P); cand[i] = max(lo[i], min(hi[i], cand[i] + sgn * steps[i]))
        if abs(cand[i] - P[i]) < 1e-9:
          continue
        h = hard(cand)
        if h < best - 1e-3:
          best = h; P = cand; improved = True
    print(f"  [round {rnd+1}/{args.rounds}] hard={best:.2f} step0={steps[0]:.3f} "
          f"len={P[0]:.3f} rad={P[1]:.3f} dh=[{P[2]:+.3f},{P[3]:+.3f},{P[4]:+.3f}] "
          f"c={[round(x,2) for x in P[5:]]}", flush=True)
    if not improved:
      steps = [s * 0.5 for s in steps]
      if max(steps) < 0.03:
        break

  verts = build_verts(P)
  hard_final = hard(P)
  with torch.no_grad():
    cvec = torch.as_tensor(P[5:5 + K], dtype=torch.float32)
    disp_mm = (torch.einsum("k,kvd->vd", cvec, M_t[:, neck_t, :]) * 1000.0).detach().cpu().numpy()
    max_mm = float(np.abs(disp_mm).max()); rms_mm = float(np.sqrt((disp_mm ** 2).mean()))
    posed_only = posed_base()
    offset_final = (verts - posed_only).detach().cpu().numpy().astype(np.float32)
    # offset_final is defined relative to THIS (final) pose; capture the pose now so
    # save() writes the matching rotations/translation (build_verts below resets
    # state.rot_* to the START dh, which would otherwise be saved by mistake).
    final_rot_neck = state.rot_neck.detach().cpu().numpy().astype(np.float32)
    final_rot_head = state.rot_head.detach().cpu().numpy().astype(np.float32)
    final_trans = state.translation.detach().cpu().numpy().astype(np.float32)
  start_verts = build_verts([args.start_len, args.start_rad, *start_dh] + [0.0] * K)
  print(f"[pca] hard {best if False else 'start'}-> {hard_final:.2f}  final len={P[0]:.3f} rad={P[1]:.3f} "
        f"dh=[{P[2]:+.3f},{P[3]:+.3f},{P[4]:+.3f}] coef={[round(x,2) for x in P[5:]]} "
        f"max|disp|={max_mm:.2f}mm rms={rms_mm:.2f}mm", flush=True)

  with torch.no_grad():
    ls_, rs_, _, _ = mesh_lines(start_verts); lf, rf, cb, _ = mesh_lines(verts)
    start_hard = hard([args.start_len, args.start_rad, *start_dh] + [0.0] * K)

  def draw(im, p, col, th):
    if p is None or len(p) < 2:
      return im
    return cv2.polylines(im, [np.round(p).astype(np.int32).reshape(-1, 1, 2)], False, col, th, cv2.LINE_AA)

  def lab(im, s):
    cv2.putText(im, s, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (0, 0, 0), 3)
    cv2.putText(im, s, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (255, 255, 255), 1); return im

  def panel(t, lm, rm):
    im = bgr.copy()
    im = draw(im, left_gt_np, (0, 255, 255), 3); im = draw(im, right_gt_np, (0, 255, 255), 3)
    im = draw(im, lm, (255, 0, 255), 2); im = draw(im, rm, (255, 0, 255), 2); return lab(im, t)

  ba = np.concatenate([panel(f"START hard={start_hard:.1f}", ls_, rs_),
                       panel(f"AFTER joint PCA hard={hard_final:.1f}", lf, rf),
                       lab(cb.copy(), "AFTER clay")], 1)
  cv2.imwrite(str(args.out_dir / "shoulder_before_after_clay.jpg"), ba)
  _notes = Path("/root/autodl-tmp/notes/gnm_uv/测试用图片")
  _notes.mkdir(parents=True, exist_ok=True)
  cv2.imwrite(str(_notes / f"GNM v4 刚性后 颈肩PCA_K{K} 肩线前后灰模.jpg"), ba)

  def legend(im):
    items = [("GT", (0, 255, 255)), ("L1 (before)", (255, 160, 0)), ("L2 (after)", (0, 0, 255))]
    for i, (t, col) in enumerate(items):
      y = 30 + i * 26
      cv2.putText(im, t, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
      cv2.putText(im, t, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 1, cv2.LINE_AA)
    return im

  pA = bgr.copy()
  for p in (left_gt_np, right_gt_np):
    pA = draw(pA, p, (0, 255, 255), 3)
  for p in (ls_, rs_):
    pA = draw(pA, p, (255, 160, 0), 2)
  for p in (lf, rf):
    pA = draw(pA, p, (0, 0, 255), 2)
  pA = legend(pA)
  pB = cb.copy()
  for p in (left_gt_np, right_gt_np):
    pB = draw(pB, p, (0, 255, 255), 2)
  for p in (lf, rf):
    pB = draw(pB, p, (0, 0, 255), 2)
  cv2.putText(pB, "mesh (L2)", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (0, 0, 0), 3, cv2.LINE_AA)
  cv2.putText(pB, "mesh (L2)", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (255, 255, 255), 1, cv2.LINE_AA)
  gl = np.concatenate([pA, pB], 1)
  cv2.imwrite(str(args.out_dir / "neck_gt_l1_l2_mesh.jpg"), gl)
  cv2.imwrite(str(_notes / "GNM v4 颈肩 GT_L1_L2_mesh.jpg"), gl)

  rots = rot.copy(); rots[0] = final_rot_neck
  rots[1] = final_rot_head
  save = {k: np.asarray(params[k]) for k in params.files}
  save["rotations"] = rots
  save["translation"] = final_trans
  save["static_offset"] = offset_final
  np.savez(args.out_dir / "frame_000000_params.npz", **save)
  json.dump({"mode": "neck_pca_global",
             "start": {"len": args.start_len, "rad": args.start_rad, "d_head": start_dh, "hard": float(start_hard)},
             "final": {"len": P[0], "rad": P[1], "d_head": [P[2], P[3], P[4]],
                       "coef": list(P[5:]), "hard": float(hard_final),
                       "max_disp_mm": max_mm, "rms_disp_mm": rms_mm},
             "modes": names}, open(args.out_dir / "summary.json", "w"), indent=2)
  print("[pca] wrote", args.out_dir / "frame_000000_params.npz", flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
