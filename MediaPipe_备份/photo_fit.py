#!/usr/bin/env python3
"""VHAP-style photometric stages on a finished GNM landmark fit.

Color is compared to the white-background photo, only on pixels the mesh
covers. There is no silhouette / leak / cover loss. Jaw landmarks are off.
Scalp under the cap, the back of the head, and the ears are occluded: they
do not paint the texture and do not pull the surface. The neck cut is kept
off the image border.

Stages, in order:
  1. texture + spherical harmonics, geometry frozen
  2. color pulls identity, joints, expression, pose, texture, and lights
  3. a shared static vertex offset, with teeth, eyes, scalp, and ears pinned
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent / "GNM-main"
if str(REPO) not in sys.path:
  sys.path.insert(0, str(REPO))
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))


def scale_for_photo(height: int, width: int, max_side: int) -> tuple[int, int, float]:
  long_side = max(height, width)
  if long_side <= max_side:
    return height, width, 1.0
  s = max_side / float(long_side)
  return max(int(round(height * s)), 1), max(int(round(width * s)), 1), s


def uv_from_triangles(gnm_np) -> tuple[np.ndarray, np.ndarray]:
  tuv = np.asarray(gnm_np.triangle_uvs, dtype=np.float32)
  faces_uv = np.arange(tuv.shape[0] * 3, dtype=np.int32).reshape(-1, 3)
  verts_uv = tuv.reshape(-1, 2)
  return verts_uv, faces_uv


def _tv(tex: torch.Tensor) -> torch.Tensor:
  y = (tex[:, :, :-1, :] - tex[:, :, 1:, :]).pow(2).mean()
  x = (tex[:, :, :, :-1] - tex[:, :, :, 1:]).pow(2).mean()
  return x + y


def _mean_face_color(rgb: np.ndarray, cover: np.ndarray) -> np.ndarray:
  w = cover[..., None]
  s = float(cover.sum()) + 1e-6
  return (rgb * w).sum(axis=(0, 1)) / s


def _group_ids(gnm_np, name: str) -> np.ndarray:
  names = set(gnm_np.vertex_group_names)
  if name not in names:
    return np.zeros(0, dtype=np.int64)
  return np.asarray(gnm_np.vertex_group_indices(name), dtype=np.int64)


def _faces_with(faces: np.ndarray, verts: np.ndarray, n_verts: int, need: int = 2) -> np.ndarray:
  if verts.size == 0:
    return np.zeros(0, dtype=np.int64)
  flag = np.zeros(n_verts, dtype=np.uint8)
  flag[verts] = 1
  return np.nonzero(flag[faces].sum(axis=1) >= need)[0].astype(np.int64)


def _boundary_vertices(faces: np.ndarray, n_verts: int) -> np.ndarray:
  edges: dict[tuple[int, int], int] = {}
  for tri in faces:
    a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
    for i, j in ((a, b), (b, c), (c, a)):
      e = (i, j) if i < j else (j, i)
      edges[e] = edges.get(e, 0) + 1
  mark = np.zeros(n_verts, dtype=bool)
  for (i, j), count in edges.items():
    if count == 1:
      mark[i] = True
      mark[j] = True
  return np.nonzero(mark)[0].astype(np.int64)


def _edge_index(faces: np.ndarray) -> np.ndarray:
  pairs = []
  seen = set()
  for tri in faces:
    a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
    for i, j in ((a, b), (b, c), (c, a)):
      e = (i, j) if i < j else (j, i)
      if e not in seen:
        seen.add(e)
        pairs.append(e)
  return np.asarray(pairs, dtype=np.int64)


def build_photo_regions(gnm_np, faces: np.ndarray) -> dict:
  """Scalp under the cap, back of the head, and ears stay out of the color term."""
  rest = np.asarray(gnm_np(None, None, None, None), dtype=np.float32)
  n_verts = int(rest.shape[0])
  skin = _group_ids(gnm_np, "skin")
  hockey = set(_group_ids(gnm_np, "hockey_mask").tolist())
  ears = _group_ids(gnm_np, "ears")
  forehead = _group_ids(gnm_np, "forehead_region")
  y_cap = float(np.percentile(rest[forehead, 1], 80)) if forehead.size else 0.35
  neck_y = 0.18
  occluded = np.zeros(n_verts, dtype=bool)
  occluded[ears] = True
  for i in skin.tolist():
    y = float(rest[i, 1])
    z = float(rest[i, 2])
    if y < neck_y:
      continue
    if i in hockey and y < y_cap:
      continue
    if y >= y_cap or z < 0.04:
      occluded[i] = True
  boundary = _boundary_vertices(faces, n_verts)
  bottom = boundary[rest[boundary, 1] < 0.12]
  neck = skin[rest[skin, 1] < neck_y]
  eyes = _group_ids(gnm_np, "eyes")
  teeth = np.unique(np.concatenate([
      _group_ids(gnm_np, "teeth"),
      _group_ids(gnm_np, "gums"),
      _group_ids(gnm_np, "tongue"),
  ]))
  lips = np.unique(np.concatenate([
      _group_ids(gnm_np, "upper_lip"),
      _group_ids(gnm_np, "lower_lip"),
  ]))
  pin = np.unique(np.concatenate([
      np.nonzero(occluded)[0],
      bottom,
      eyes,
      teeth,
  ])).astype(np.int64)
  occ_f = _faces_with(faces, np.nonzero(occluded)[0], n_verts)
  neck_f = _faces_with(faces, neck, n_verts)
  bottom_f = _faces_with(faces, bottom, n_verts)
  color_except = np.unique(np.concatenate([occ_f, bottom_f])).astype(np.int64)
  texture_except = np.unique(np.concatenate([occ_f, neck_f, bottom_f])).astype(np.int64)
  bound_except = np.unique(np.concatenate([np.nonzero(occluded)[0], bottom])).astype(np.int64)

  cid = np.ones(faces.shape[0], dtype=np.int64)
  def paint(vert_ids: np.ndarray, cluster: int) -> None:
    fid = _faces_with(faces, vert_ids, n_verts)
    cid[fid] = cluster
  paint(skin, 2)
  paint(neck, 3)
  paint(np.nonzero(occluded)[0], 4)
  paint(ears, 5)
  paint(lips, 6)
  paint(teeth, 7)
  paint(eyes, 8)
  print(
      f"[photo] regions  verts={n_verts}  occluded={int(occluded.sum())}  "
      f"bottom={bottom.size}  neck={neck.size}  pin={pin.size}  "
      f"color_except_faces={color_except.size}  y_cap={y_cap:.3f}",
      flush=True,
  )
  return {
      "n_verts": n_verts,
      "pin": pin,
      "color_except": color_except,
      "texture_except": texture_except,
      "bound_except": bound_except,
      "fid2cid": cid,
      "edges": _edge_index(faces),
  }


def dump_photo_preview(out_dir: Path, tag: str, gt_rgb: np.ndarray, pred_rgba: np.ndarray) -> None:
  pred = np.clip(pred_rgba[..., :3], 0, 1)
  alpha = np.clip(pred_rgba[..., 3], 0, 1)
  overlay = gt_rgb * (1.0 - alpha[..., None]) + pred * alpha[..., None]
  err = np.abs(gt_rgb - pred).mean(-1)
  err_u8 = cv2.applyColorMap(np.clip(err * 4.0 * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
  row = np.concatenate([
      np.clip(gt_rgb * 255, 0, 255).astype(np.uint8),
      np.clip(overlay * 255, 0, 255).astype(np.uint8),
      np.clip(pred * 255, 0, 255).astype(np.uint8),
      cv2.cvtColor(err_u8, cv2.COLOR_BGR2RGB),
  ], axis=1)
  path = out_dir / f"stage_{tag}_photo.jpg"
  cv2.imwrite(str(path), cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
  render_path = out_dir / f"stage_{tag}_render.jpg"
  cv2.imwrite(str(render_path), cv2.cvtColor(np.clip(pred * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
  print(f"  [dump] {path.name}  {render_path.name}", flush=True)


def run_photometric(
    *,
    args,
    state,
    gnm,
    gnm_np,
    posed_verts_world,
    sample_lmk3d,
    project_xy,
    dump_stage,
    stage_bgr: np.ndarray,
    gt_xy: np.ndarray,
    gt_t: torch.Tensor,
    width: int,
    height: int,
    mouth_locked: bool,
    metric_reg,
    weighted_landmark_loss,
    w_lmk: np.ndarray,
    faces: np.ndarray,
) -> dict:
  del gnm, mouth_locked
  if stage_bgr is None:
    print("[photo] skip: no frame image", flush=True)
    return {}
  if not torch.cuda.is_available():
    print("[photo] skip: CUDA required for nvdiffrast photometric", flush=True)
    return {}

  from nvdiffrast_render import NVDiffRenderer

  device = torch.device("cuda")
  ph, pw, scale = scale_for_photo(height, width, int(args.photo_max_side))
  bgr = stage_bgr
  if scale != 1.0:
    bgr = cv2.resize(stage_bgr, (pw, ph), interpolation=cv2.INTER_AREA)
  rgb_np = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
  cover = np.zeros(bgr.shape[:2], np.float32)
  hull = cv2.convexHull((gt_xy * scale).astype(np.int32))
  cv2.fillConvexPoly(cover, hull, 1.0)
  gt_rgb = torch.as_tensor(rgb_np, dtype=torch.float32, device=device)[None]
  if scale == 1.0:
    k4 = state.camera_k(width, height).detach()
  else:
    k4 = state.camera_k(width, height).detach() * scale
    # cx, cy are the last two; focal scales, and so do the principals.
  k_t = k4.to(device)[None]
  faces_t = torch.as_tensor(faces, dtype=torch.int32, device=device)
  verts_uv, faces_uv = uv_from_triangles(gnm_np)
  verts_uv_t = torch.as_tensor(verts_uv, dtype=torch.float32, device=device)
  faces_uv_t = torch.as_tensor(faces_uv, dtype=torch.int32, device=device)

  regions = build_photo_regions(gnm_np, faces)
  pin = torch.as_tensor(regions["pin"], dtype=torch.long)
  edges = torch.as_tensor(regions["edges"], dtype=torch.long)
  tex_except = torch.as_tensor(regions["texture_except"], dtype=torch.long, device=device)
  bound_except = torch.as_tensor(regions["bound_except"], dtype=torch.long, device=device)
  color_lut = torch.zeros(faces.shape[0] + 1, dtype=torch.bool, device=device)
  color_except = torch.as_tensor(regions["color_except"], dtype=torch.long, device=device)
  color_lut[color_except + 1] = True

  mean_c = np.clip(_mean_face_color(rgb_np, cover), 1e-3, 1.0 - 1e-3)
  tex_res = int(args.photo_tex_res)
  logit = np.log(mean_c / (1.0 - mean_c)).astype(np.float32)
  tex = torch.nn.Parameter(torch.zeros(1, 3, tex_res, tex_res, dtype=torch.float32))
  with torch.no_grad():
    tex[:, :, :, :] = torch.as_tensor(logit)[None, :, None, None]
  lights = torch.nn.Parameter(torch.zeros(1, 9, 3, dtype=torch.float32))
  n_verts = int(regions["n_verts"])
  offset = torch.nn.Parameter(torch.zeros(n_verts, 3, dtype=torch.float32))

  w_photo = np.array(w_lmk, dtype=np.float32, copy=True)
  if w_photo.shape[0] >= 17:
    w_photo[:17] = 0.0
  conf = torch.as_tensor(w_photo, dtype=torch.float32)
  print(
      f"[photo] white bg  {pw}x{ph}  jaw weight 0  "
      f"nose mean w={float(w_photo[27:36].mean()):.1f}  no silhouette loss",
      flush=True,
  )

  renderer = NVDiffRenderer(
      use_opengl=False,
      lighting_type="SH",
      lighting_space="world",
      disturb_rate_fg=0.5,
      disturb_rate_bg=0.3,
      fid2cid=torch.as_tensor(regions["fid2cid"], dtype=torch.long),
      shade_smooth=True,
      flip_y_raster=False,
  ).to(device)
  with torch.no_grad():
    lights[:, 0, :] = 1.0 / renderer.sh_const[0].cpu()
    init_appearance = getattr(args, "init_appearance", None)
    if init_appearance is not None:
      saved = np.load(init_appearance)
      albedo = np.clip(saved["albedo"].astype(np.float32), 1e-4, 1.0 - 1e-4)
      if albedo.shape[-2:] != (tex_res, tex_res):
        raise ValueError(f"saved albedo {albedo.shape} does not match tex_res {tex_res}")
      tex.copy_(torch.as_tensor(np.log(albedo / (1.0 - albedo))))
      lights.copy_(torch.as_tensor(saved["lights"].astype(np.float32)))
      print(f"[photo] loaded texture+SH from {init_appearance}", flush=True)

  stats: dict = {"photo_size": [pw, ph], "photo_scale": scale}

  def world_verts() -> torch.Tensor:
    return posed_verts_world() + offset

  def render_pack(disturb: bool):
    verts = world_verts()
    verts_g = verts.to(device)[None].contiguous()
    rt = state.camera_rt().to(device)[None]
    rast = renderer.rasterize(
        verts_g, faces_t, rt, k_t, (ph, pw), use_cache=False, require_grad=True
    )
    out = renderer.render_rgba(
        rast,
        verts_g[0],
        faces_t,
        verts_uv_t,
        faces_uv_t,
        torch.sigmoid(tex).to(device),
        lights.to(device),
        background_color=[1.0, 1.0, 1.0],
        align_texture_except_fid=tex_except,
        align_boundary_except_vid=bound_except,
        enable_disturbance=disturb,
    )
    return verts, rast, out

  def photo_loss(rast, out, w_rgb: float, w_lmk_term: float, use_offset_reg: bool):
    rgba = out["rgba"]
    pred = rgba[..., :3]
    fid = renderer._to_image_space(rast["rast_out"][..., 3:4].long())
    covered = rgba[..., 3:] > 0
    blocked = color_lut[fid.clamp(0, color_lut.shape[0] - 1)]
    vis = (covered & ~blocked).detach()
    rgb_loss = ((pred - gt_rgb).abs() * vis).sum() / (vis.sum() * 3).clamp(min=1.0)
    pred_xy = project_xy(sample_lmk3d(world_verts()))
    lmk_loss = weighted_landmark_loss(pred_xy, conf)
    extra = metric_reg(True, True, True)
    albedo = torch.sigmoid(tex)
    tv = _tv(albedo)
    diffuse = out["diffuse_detach_normal"]
    reg_diffuse = torch.relu(diffuse.max() - 1.0) + diffuse.var()
    sh_reg = (lights[:, 1:] ** 2).mean()
    loss = w_rgb * rgb_loss
    loss = loss + w_lmk_term * lmk_loss.to(loss.device)
    loss = loss + 0.05 * extra.to(loss.device)
    loss = loss + 1.0e4 * tv.to(loss.device)
    loss = loss + 10.0 * reg_diffuse
    loss = loss + 0.01 * sh_reg.to(loss.device)
    lap = tex.new_zeros(())
    off = tex.new_zeros(())
    if use_offset_reg:
      delta = offset[edges[:, 0]] - offset[edges[:, 1]]
      lap = delta.pow(2).mean()
      free = torch.ones(offset.shape[0], dtype=torch.bool)
      free[pin] = False
      off = offset[free].abs().mean() if bool(free.any()) else offset.new_zeros(())
      loss = loss + 1.0e5 * lap.to(loss.device) + 3.0e2 * off.to(loss.device)
    with torch.no_grad():
      err = torch.linalg.norm(pred_xy.detach() - gt_t, dim=-1)
    return loss, {
        "loss": float(loss.detach()),
        "rgb": float(rgb_loss.detach()),
        "lmk": float(err.mean()),
        "nose": float(err[27:36].mean()) if err.numel() >= 36 else float(err.mean()),
        "tv": float(tv.detach()),
        "lap": float(lap.detach()),
        "off": float(off.detach()) if torch.is_tensor(off) else float(off),
    }

  def snapshot():
    snap = state.snapshot()
    snap["tex"] = tex.detach().clone()
    snap["lights"] = lights.detach().clone()
    snap["offset"] = offset.detach().clone()
    return snap

  def restore(snap) -> None:
    state.restore({k: v for k, v in snap.items() if k not in ("tex", "lights", "offset")})
    with torch.no_grad():
      tex.copy_(snap["tex"])
      lights.copy_(snap["lights"])
      offset.copy_(snap["offset"])

  def run_loop(name: str, iters: int, groups: list[dict], w_rgb: float, w_lmk_term: float,
               disturb: bool, use_offset_reg: bool) -> dict:
    groups = [g for g in groups if g["params"]]
    if iters <= 0 or not groups:
      return {}
    opt = torch.optim.Adam(groups)
    for g in opt.param_groups:
      g["base_lr"] = g["lr"]
    best_score = float("inf")
    best = snapshot()
    best_i = 0
    last_log: dict = {}
    print(f"\n==> {name}  iters={iters}  {ph}x{pw}  disturb={disturb}", flush=True)
    for i in range(iters):
      t = i / max(iters - 1, 1)
      lr_scale = 0.15 + 0.85 * (0.5 * (1.0 + np.cos(np.pi * t)))
      for g in opt.param_groups:
        g["lr"] = g["base_lr"] * lr_scale
      opt.zero_grad()
      _, rast, out = render_pack(disturb)
      loss, last_log = photo_loss(rast, out, w_rgb, w_lmk_term, use_offset_reg)
      loss.backward()
      if offset.grad is not None:
        offset.grad[pin] = 0
      torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
      score = last_log["loss"]
      if score < best_score:
        best_score = score
        best = snapshot()
        best_i = i + 1
      opt.step()
      with torch.no_grad():
        offset[pin] = 0
        offset.clamp_(-0.008, 0.008)
        state.focal_norm.clamp_(0.3, 5.0)
        if state.id_head.requires_grad:
          state.id_head.clamp_(-5.0, 5.0)
        if state.ex_lower.requires_grad:
          state.ex_lower.clamp_(-4.0, 4.0)
      if i == 0 or (i + 1) % max(iters // 8, 1) == 0 or i + 1 == iters:
        print(
            f"  [{name}] {i+1:4d}/{iters}  rgb={last_log['rgb']:.4f}  "
            f"lmk={last_log['lmk']:.2f}  nose={last_log['nose']:.2f}  "
            f"tv={last_log['tv']:.6f}  off={last_log['off']:.5f}  "
            f"best={best_score:.4f}@{best_i}",
            flush=True,
        )
    restore(best)
    with torch.no_grad():
      _, _, out = render_pack(False)
      dump_photo_preview(
          args.out_dir,
          name.replace("gnm_", ""),
          rgb_np,
          out["rgba"][0].detach().float().cpu().numpy(),
      )
    last_log["best_score"] = best_score
    last_log["best_iter"] = best_i
    return last_log

  for t in (
      state.id_head, state.id_eyes, state.ex_lower, state.rot_neck, state.rot_head,
      state.rot_leye, state.rot_reye, state.translation, state.cam_aa, state.cam_t,
      state.focal_norm, offset,
  ):
    t.requires_grad_(False)

  tex_iters = int(args.photo_tex_iters)
  rgb_iters = int(args.photo_iters)
  off_iters = int(getattr(args, "photo_offset_iters", 250))
  stop_after = getattr(args, "stop_after", "tex")
  lmk_w = float(getattr(args, "w_landmark", 10.0))

  print("[photo] stage texture+lights, geometry frozen", flush=True)
  log_tex = run_loop(
      "gnm_photo_tex",
      tex_iters,
      [{"params": [tex], "lr": 2e-2}, {"params": [lights], "lr": 5e-3}],
      w_rgb=30.0,
      w_lmk_term=0.0,
      disturb=False,
      use_offset_reg=False,
  )
  if dump_stage is not None:
    dump_stage("4_photo_tex", "stage4 texture+SH, mesh frozen", stage_bgr)
  stats["photo_tex"] = log_tex
  stats["stop_after"] = stop_after
  if stop_after == "tex":
    print("[photo] stop after texture+SH, mesh unchanged", flush=True)

  if stop_after in ("rgb", "offset"):
    print("[photo] stage color pulls identity, joints, expression, pose", flush=True)
    for t in (
        state.id_head, state.id_eyes, state.ex_lower, state.rot_neck, state.rot_head,
        state.rot_leye, state.rot_reye, state.translation, state.cam_aa, state.cam_t,
        state.focal_norm,
    ):
      t.requires_grad_(True)
    log_rgb = run_loop(
        "gnm_photo_all",
        rgb_iters,
        [
            {"params": [tex], "lr": 5e-3},
            {"params": [lights], "lr": 5e-3},
            {"params": [state.id_head, state.id_eyes], "lr": 1e-3},
            {"params": [state.ex_lower], "lr": 5e-3},
            {"params": [state.rot_neck, state.rot_head, state.rot_leye, state.rot_reye], "lr": 1e-3},
            {"params": [state.translation, state.cam_t], "lr": 3e-4},
            {"params": [state.cam_aa], "lr": 1e-3},
            {"params": [state.focal_norm], "lr": 5e-5},
        ],
        w_rgb=30.0,
        w_lmk_term=lmk_w,
        disturb=True,
        use_offset_reg=False,
    )
    if dump_stage is not None:
      dump_stage("5_photo", "stage5 photometric, jaw landmarks off", stage_bgr)
    stats["photo_all"] = log_rgb

  if stop_after == "offset":
    print("[photo] stage static offset, scalp and ears pinned", flush=True)
    offset.requires_grad_(True)
    log_off = run_loop(
        "gnm_photo_offset",
        off_iters,
        [
            {"params": [offset], "lr": 1e-4},
            {"params": [tex], "lr": 2e-3},
            {"params": [lights], "lr": 2e-3},
            {"params": [state.id_head], "lr": 5e-4},
            {"params": [state.ex_lower], "lr": 2e-3},
            {"params": [state.rot_neck, state.rot_head], "lr": 5e-4},
            {"params": [state.translation, state.cam_t, state.cam_aa], "lr": 2e-4},
        ],
        w_rgb=30.0,
        w_lmk_term=lmk_w,
        disturb=True,
        use_offset_reg=True,
    )
    if dump_stage is not None:
      dump_stage("6_offset", "stage6 static offset, scalp/ears pinned", stage_bgr)
    stats["photo_offset"] = log_off

  with torch.no_grad():
    verts = world_verts().detach().cpu().numpy()
    pred = project_xy(sample_lmk3d(world_verts())).detach().cpu().numpy()
  err = np.linalg.norm(pred - gt_xy, axis=1)
  stats["rmse_all"] = float(err.mean())
  stats["rmse_nose"] = float(err[27:36].mean())
  stats["rmse_jaw"] = float(err[:17].mean())
  stats["offset_mm_p95"] = float(np.percentile(np.linalg.norm(offset.detach().cpu().numpy(), axis=1), 95) * 1000.0)
  np.savez(
      args.out_dir / "photo_appearance.npz",
      albedo=torch.sigmoid(tex.detach()).cpu().numpy(),
      lights=lights.detach().cpu().numpy(),
      static_offset=offset.detach().cpu().numpy(),
      pin=regions["pin"],
  )
  trimesh.Trimesh(vertices=verts, faces=faces, process=False).export(args.out_dir / "frame_photo.obj")
  vis = stage_bgr.copy()
  for i in range(pred.shape[0]):
    gx, gy = int(round(gt_xy[i, 0])), int(round(gt_xy[i, 1]))
    px, py = int(round(pred[i, 0])), int(round(pred[i, 1]))
    cv2.circle(vis, (gx, gy), 3, (0, 255, 0), -1)
    cv2.line(vis, (gx, gy), (px, py), (0, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(vis, (px, py), 3, (0, 0, 255), -1)
  cv2.imwrite(str(args.out_dir / "frame_photo_overlay.jpg"), vis)
  print(
      f"[photo] done  lmk={stats['rmse_all']:.2f}px  nose={stats['rmse_nose']:.2f}  "
      f"jaw={stats['rmse_jaw']:.2f}  offset_p95={stats['offset_mm_p95']:.2f}mm",
      flush=True,
  )
  for t in (
      state.id_head, state.id_eyes, state.ex_lower, state.rot_neck, state.rot_head,
      state.rot_leye, state.rot_reye, state.translation, state.cam_aa, state.cam_t,
      state.focal_norm, offset,
  ):
    t.requires_grad_(False)
  return stats


def _build_and_run(args) -> int:
  from gnm.shape import gnm_numpy, gnm_pytorch, gnm_utils
  import gnm_mediapipe_landmarks as mp_lmk
  from nvdiffrast_render.camera import project_world_to_pixels
  from fit_gnm import FitState, ibug68_vhap_weights

  params = np.load(args.params)
  lmk_npz = np.load(args.landmarks, allow_pickle=True)
  gt_xy = np.asarray(lmk_npz["landmarks"][args.frame_index], dtype=np.float32)
  width, height = [int(x) for x in lmk_npz["image_size"]]
  if args.image is None:
    image = Path(str(lmk_npz["image_paths"][args.frame_index]))
  else:
    image = args.image
  stage_bgr = cv2.imread(str(image))
  if stage_bgr is None:
    raise FileNotFoundError(image)
  if stage_bgr.shape[1] != width or stage_bgr.shape[0] != height:
    stage_bgr = cv2.resize(stage_bgr, (width, height), interpolation=cv2.INTER_AREA)

  gnm = gnm_pytorch.GNM.from_local(
      version=gnm_pytorch.GNMMajorVersion.V3,
      variant=gnm_pytorch.GNMVariant.HEAD,
  )
  gnm_np = gnm_numpy.GNM.from_local(
      version=gnm_numpy.GNMMajorVersion.V3,
      variant=gnm_numpy.GNMVariant.HEAD,
  )
  id_scale = torch.as_tensor(gnm_utils.scaling_for_identity_loss(gnm_np), dtype=torch.float32)
  ex_scale = torch.as_tensor(
      gnm_utils.scaling_for_expression_loss(gnm_np)[200:350, 200:350], dtype=torch.float32
  )
  faces = np.asarray(gnm.triangles.detach().cpu().numpy())
  lmk_idx, lmk_w = mp_lmk.load_landmarks(args.landmark_file)

  def p(v):
    return torch.tensor(np.asarray(v), dtype=torch.float32).requires_grad_(True)

  rot = np.asarray(params["rotations"], dtype=np.float32)
  state = FitState(
      id_head=p(params["id_head"]),
      id_eyes=p(params["id_eyes"]),
      ex_lower=p(params["expression_lower_face"]),
      rot_neck=p(rot[0]),
      rot_head=p(rot[1]),
      rot_leye=p(rot[2]),
      rot_reye=p(rot[3]),
      translation=p(params["translation"]),
      cam_aa=p(params["cam_aa"]),
      cam_t=p(params["cam_t"]),
      focal_norm=p(params["focal_norm"]),
  )
  idx_t = torch.as_tensor(lmk_idx, dtype=torch.long)
  bary_t = torch.as_tensor(lmk_w, dtype=torch.float32)
  gt_t = torch.as_tensor(gt_xy)

  def posed_verts_world() -> torch.Tensor:
    return gnm(
        state.identity()[None],
        state.expression()[None],
        state.rotations()[None],
        state.translation[None],
    )[0]

  def sample_lmk3d(verts_world: torch.Tensor) -> torch.Tensor:
    return (verts_world[idx_t] * bary_t.unsqueeze(-1)).sum(dim=1)

  def project_xy(lmk3d: torch.Tensor) -> torch.Tensor:
    return project_world_to_pixels(
        lmk3d, state.camera_rt(), state.camera_k(width, height), width, height, flip_y=True,
    )

  def metric_reg(reg_id: bool, reg_ex: bool, reg_joints: bool) -> torch.Tensor:
    reg = torch.tensor(0.0)
    if reg_id:
      reg = reg + args.reg_id * ((id_scale @ state.identity()) ** 2).mean()
    if reg_ex:
      reg = reg + args.reg_ex * ((ex_scale @ state.ex_lower) ** 2).mean()
    if reg_joints:
      for t in (state.rot_neck, state.rot_head, state.rot_leye, state.rot_reye, state.cam_aa):
        reg = reg + args.reg_joint * (t ** 2).mean()
      reg = reg + args.reg_jaw_yz * (state.rot_head[1:] ** 2).mean()
    return reg

  def weighted_landmark_loss(pred_xy: torch.Tensor, conf: torch.Tensor) -> torch.Tensor:
    err = torch.linalg.norm(gt_t - pred_xy, dim=-1)
    return (err * conf).sum() / conf.sum().clamp(min=1.0)

  w = ibug68_vhap_weights(nose_boost=args.nose_weight, mouth_boost=1.0, eye_boost=1.0, brow_boost=1.0)
  args.out_dir.mkdir(parents=True, exist_ok=True)
  photo = run_photometric(
      args=args,
      state=state,
      gnm=gnm,
      gnm_np=gnm_np,
      posed_verts_world=posed_verts_world,
      sample_lmk3d=sample_lmk3d,
      project_xy=project_xy,
      dump_stage=lambda *_a, **_k: None,
      stage_bgr=stage_bgr,
      gt_xy=gt_xy,
      gt_t=gt_t,
      width=width,
      height=height,
      mouth_locked=False,
      metric_reg=metric_reg,
      weighted_landmark_loss=weighted_landmark_loss,
      w_lmk=w,
      faces=faces,
  )
  np.savez(
      args.out_dir / "frame_photo_params.npz",
      id_head=state.id_head.detach().numpy(),
      id_eyes=state.id_eyes.detach().numpy(),
      expression_lower_face=state.ex_lower.detach().numpy(),
      rotations=state.rotations().detach().numpy(),
      translation=state.translation.detach().numpy(),
      cam_aa=state.cam_aa.detach().numpy(),
      cam_t=state.cam_t.detach().numpy(),
      focal_norm=float(state.focal_norm.detach()),
      image_size=np.array([width, height], dtype=np.int32),
      rmse_all=photo.get("rmse_all", 0.0),
      rmse_nose=photo.get("rmse_nose", 0.0),
  )
  summary = {k: v for k, v in photo.items() if k not in ("photo_tex", "photo_all", "photo_offset")}
  summary["photo_tex"] = photo.get("photo_tex", {})
  summary["photo_all"] = photo.get("photo_all", {})
  summary["photo_offset"] = photo.get("photo_offset", {})
  summary["image"] = str(image)
  summary["params_in"] = str(args.params)
  (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
  print(json.dumps(summary, indent=2), flush=True)
  return 0


def main() -> int:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--params", type=Path, required=True)
  p.add_argument("--landmarks", type=Path, required=True)
  p.add_argument("--landmark_file", type=Path, required=True)
  p.add_argument("--out_dir", type=Path, required=True)
  p.add_argument("--image", type=Path, default=None)
  p.add_argument("--frame_index", type=int, default=0)
  p.add_argument("--photo_max_side", type=int, default=1024)
  p.add_argument("--photo_tex_res", type=int, default=256)
  p.add_argument("--photo_tex_iters", type=int, default=400)
  p.add_argument("--photo_iters", type=int, default=400)
  p.add_argument("--photo_offset_iters", type=int, default=250)
  p.add_argument("--stop_after", choices=("tex", "rgb", "offset"), default="tex")
  p.add_argument("--init_appearance", type=Path, default=None,
                 help="Load albedo and SH from a previous texture stage, then continue.")
  p.add_argument("--w_landmark", type=float, default=10.0)
  p.add_argument("--nose_weight", type=float, default=10.0)
  p.add_argument("--reg_id", type=float, default=0.3)
  p.add_argument("--reg_ex", type=float, default=0.03)
  p.add_argument("--reg_joint", type=float, default=0.3)
  p.add_argument("--reg_jaw_yz", type=float, default=3.0)
  args = p.parse_args()
  return _build_and_run(args)


if __name__ == "__main__":
  raise SystemExit(main())
