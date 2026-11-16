#!/usr/bin/env python3
"""Neck/shoulder DOF toward photo shoulder lines.

Preferred slot: **after rigid, before lmk** (``--lock upper``). Then only a
rough head orientation is locked — lower face contour may move; lmk fixes it.

Legacy slot after SAM eyes uses ``--lock full`` (face+lips). Does not touch
photo_fit. Writes params for the next stage.
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
from photo_fit import UV_MASKS, uv_from_triangles  # noqa: E402
from nvdiffrast_render import NVDiffRenderer  # noqa: E402
import neck_shoulder_l2 as neck_l2  # noqa: E402


def _tp(v) -> torch.Tensor:
  return torch.tensor(np.asarray(v), dtype=torch.float32)


def main() -> int:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--params", type=Path, required=True)
  p.add_argument("--shoulder_lines", type=Path, required=True)
  p.add_argument("--image", type=Path, required=True)
  p.add_argument("--out_dir", type=Path, required=True)
  p.add_argument("--iters", type=int, default=300)
  p.add_argument(
      "--mode",
      choices=("length_only", "full", "joints"),
      default="length_only",
      help="length_only: neck_len. full: len+rad+yawY. "
           "joints: freeze len/rad, micro 3-axis joints (see --joint_target).",
  )
  p.add_argument(
      "--joint_target",
      choices=("root", "head"),
      default="head",
      help="head: micro rot_head via inverse_head_delta (ablation-preferred). "
           "root: micro rot_neck + translation polish.",
  )
  p.add_argument("--w_outer", type=float, default=5.0,
                 help="Weight: mesh must not exceed photo shoulder lines.")
  p.add_argument("--w_reach", type=float, default=2.0,
                 help="Weight: lengthen until mesh reaches shoulder-line bottom y.")
  p.add_argument("--w_line", type=float, default=1.0,
                 help="Weight: soft mesh side-lines ↔ photo shoulder lines (px^2).")
  p.add_argument("--w_inside", type=float, default=0.0,
                 help="Weight: close gap from inside (full mode).")
  p.add_argument("--w_shoulder", type=float, default=None,
                 help="Deprecated alias: if set, used as w_outer.")
  p.add_argument("--shoulder_slack", type=float, default=2.0)
  p.add_argument("--lr", type=float, default=2e-2)
  p.add_argument(
      "--freeze_len",
      type=float,
      default=None,
      help="Lock neck length scale (e.g. 1.47 from length_only).",
  )
  p.add_argument(
      "--freeze_rad",
      type=float,
      default=None,
      help="Lock neck radius scale (e.g. 1.30 from hard grid). Required for joints.",
  )
  p.add_argument(
      "--joint_max",
      type=float,
      default=0.35,
      help="Max |Δaa| rad per axis for joints mode (tanh-bounded).",
  )
  p.add_argument(
      "--lock",
      choices=("upper", "full"),
      default="upper",
      help="upper: forehead/nose/eyes only (after rigid). "
           "full: face+lips (legacy after SAM eyes).",
  )
  args = p.parse_args()
  if args.w_shoulder is not None:
    args.w_outer = float(args.w_shoulder)
  if args.mode == "joints":
    if args.freeze_len is None or args.freeze_rad is None:
      raise SystemExit("joints mode needs --freeze_len and --freeze_rad")
  args.out_dir.mkdir(parents=True, exist_ok=True)

  bgr = cv2.imread(str(args.image))
  if bgr is None:
    raise FileNotFoundError(args.image)
  H, W = bgr.shape[:2]
  left_gt, right_gt = neck_l2.load_shoulder_gt(args.shoulder_lines)

  params = np.load(args.params)
  rot = np.asarray(params["rotations"], dtype=np.float32)
  state = FitState(
      id_head=_tp(params["id_head"]),
      id_eyes=_tp(params["id_eyes"]),
      ex_left=_tp(
          params["expression_left_eye"]
          if "expression_left_eye" in params.files
          else np.zeros(100, np.float32)
      ),
      ex_right=_tp(
          params["expression_right_eye"]
          if "expression_right_eye" in params.files
          else np.zeros(100, np.float32)
      ),
      ex_lower=_tp(params["expression_lower_face"]),
      rot_neck=_tp(rot[0]),
      rot_head=_tp(rot[1]),
      rot_leye=_tp(rot[2]),
      rot_reye=_tp(rot[3]),
      translation=_tp(params["translation"]),
      cam_aa=_tp(params["cam_aa"]),
      cam_t=_tp(params["cam_t"]),
      focal_norm=_tp(params["focal_norm"]),
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
  # Soft shoulder losses: only high-fade (shoulder-end) verts — full neck set
  # makes soft_x look "outside" while clay silhouette is still inside the photo.
  shoulder_sel = wfade_np >= 0.45
  if int(shoulder_sel.sum()) < 32:
    shoulder_sel = wfade_np >= float(np.median(wfade_np))
  shoulder_ids = torch.as_tensor(ids_np[shoulder_sel], dtype=torch.long)
  print(
      f"[prephoto] deform_verts={ids_np.size}  soft_shoulder_verts={int(shoulder_sel.sum())}",
      flush=True,
  )

  rot_neck0 = state.rot_neck.detach().clone()
  rot_head0 = state.rot_head.detach().clone()
  trans0 = state.translation.detach().clone()
  t_head_target = neck_l2.head_joint_target(
      gnm, state.identity().detach(), state.rotations().detach(), trans0,
  )

  # Head lock samples.
  if args.lock == "upper":
    face_vids = neck_l2.upper_orient_vertex_ids(gnm_np, faces, UV_MASKS, max_n=300)
    print(f"[prephoto] lock=upper  orient_verts={face_vids.size}", flush=True)
  else:
    masks = np.load(UV_MASKS)
    face_uv = np.asarray(masks["face"], bool)
    if "lips" in masks.files:
      face_uv = face_uv | np.asarray(masks["lips"], bool)
    tuv = np.asarray(gnm_np.triangle_uvs, dtype=np.float32).mean(1)
    h, w = face_uv.shape
    ui = np.clip(np.rint(tuv[:, 0] * (w - 1)).astype(int), 0, w - 1)
    vi = np.clip(np.rint((1.0 - tuv[:, 1]) * (h - 1)).astype(int), 0, h - 1)
    face_vids = np.unique(faces[face_uv[vi, ui]].ravel())
    rng = np.random.default_rng(0)
    if face_vids.size > 400:
      face_vids = np.sort(rng.choice(face_vids, 400, replace=False))
    print(f"[prephoto] lock=full  face+lips_verts={face_vids.size}", flush=True)
  face_idx = torch.as_tensor(face_vids, dtype=torch.long)

  def posed_base() -> torch.Tensor:
    return gnm(
        state.identity()[None],
        state.expression()[None],
        state.rotations()[None],
        state.translation[None],
    )[0]

  with torch.no_grad():
    face0 = project_world_to_pixels(
        posed_base()[face_idx],
        state.camera_rt(),
        state.camera_k(W, H),
        W, H, flip_y=True,
    ).detach().clone()

  length_only = args.mode == "length_only"
  joints_mode = args.mode == "joints"

  def _logit_from_len_scale(s: float) -> float:
    # len_s = 1 + 0.5 * sigmoid(p)  =>  sigmoid = 2*(s-1)
    u = float(np.clip(2.0 * (s - 1.0), 1e-4, 1.0 - 1e-4))
    return float(np.log(u / (1.0 - u)))

  def _logit_from_rad_scale(s: float) -> float:
    # rad_s = 1 + 0.5 * sigmoid(p)
    u = float(np.clip(2.0 * (s - 1.0), 1e-4, 1.0 - 1e-4))
    return float(np.log(u / (1.0 - u)))

  # Start near scale 1.0. Full: rad mid-range so thicken can move.
  if args.freeze_len is not None:
    neck_len = torch.nn.Parameter(
        torch.tensor(_logit_from_len_scale(float(args.freeze_len)))
    )
    neck_len.requires_grad_(False)
  else:
    neck_len = torch.nn.Parameter(torch.tensor(-1.5))  # ~1.09
  if args.freeze_rad is not None:
    neck_rad = torch.nn.Parameter(
        torch.tensor(_logit_from_rad_scale(float(args.freeze_rad)))
    )
    neck_rad.requires_grad_(False)
  else:
    neck_rad = torch.nn.Parameter(
        torch.tensor(-4.0 if length_only else -1.0)
    )  # length_only ~1.00; full ~1.13
  d_yaw = torch.nn.Parameter(torch.zeros(()))  # full mode Y-only
  d_head_raw = torch.nn.Parameter(torch.zeros(3))  # joints mode 3-axis
  d_trans = torch.nn.Parameter(torch.zeros(3))  # root joints: translation polish
  if length_only:
    neck_rad.requires_grad_(False)
    d_yaw.requires_grad_(False)
    d_head_raw.requires_grad_(False)
    d_trans.requires_grad_(False)
    opt = torch.optim.Adam([{"params": [neck_len], "lr": args.lr}])
    print("[prephoto] mode=length_only  (rad=1, yaw=0 fixed)", flush=True)
  elif joints_mode:
    d_yaw.requires_grad_(False)
    if args.joint_target == "root":
      opt = torch.optim.Adam(
          [
              {"params": [d_head_raw], "lr": args.lr},
              {"params": [d_trans], "lr": args.lr * 2.0},
          ]
      )
      print(
          f"[prephoto] mode=joints  target=root  "
          f"freeze_len={float(args.freeze_len):.3f}  "
          f"freeze_rad={float(args.freeze_rad):.3f}  "
          f"joint_max={float(args.joint_max):.3f}rad  lock={args.lock}",
          flush=True,
      )
    else:
      d_trans.requires_grad_(False)
      opt = torch.optim.Adam([{"params": [d_head_raw], "lr": args.lr}])
      print(
          f"[prephoto] mode=joints  target=head  "
          f"freeze_len={float(args.freeze_len):.3f}  "
          f"freeze_rad={float(args.freeze_rad):.3f}  "
          f"joint_max={float(args.joint_max):.3f}rad  lock={args.lock}",
          flush=True,
      )
  elif args.freeze_len is not None:
    d_head_raw.requires_grad_(False)
    d_trans.requires_grad_(False)
    opt = torch.optim.Adam(
        [
            {"params": [neck_rad], "lr": args.lr * 2.0},
            {"params": [d_yaw], "lr": args.lr * 2.0},
        ]
    )
    print(
        f"[prephoto] mode=full  freeze_len={float(args.freeze_len):.3f}  "
        f"(opt rad+yawY only)",
        flush=True,
    )
  else:
    d_head_raw.requires_grad_(False)
    d_trans.requires_grad_(False)
    opt = torch.optim.Adam(
        [
            {"params": [neck_len], "lr": args.lr},
            {"params": [neck_rad], "lr": args.lr * 2.0},
            {"params": [d_yaw], "lr": args.lr * 2.0},
        ]
    )
    print("[prephoto] mode=full  (len+rad+yawY)", flush=True)

  # Orientation lock weight: upper = rough head pose; full = face+lips.
  w_face = 80.0 if args.lock == "upper" else 300.0
  w_chin = 20.0 if args.lock == "upper" else 80.0

  def forward_mesh():
    if length_only:
      state.rot_neck = rot_neck0
      state.rot_head = rot_head0
      state.translation = trans0
      d_head = neck_len.new_zeros(3)
    elif joints_mode and args.joint_target == "root":
      # Micro root twist; keep rot_head; translation free for upper lock.
      d_head = neck_l2.micro_head_delta(d_head_raw, max_rad=float(args.joint_max))
      state.rot_neck = rot_neck0 + d_head
      state.rot_head = rot_head0
      state.translation = trans0 + d_trans
    elif joints_mode:
      d_head = neck_l2.micro_head_delta(d_head_raw, max_rad=float(args.joint_max))
      rn, rh, tr = neck_l2.inverse_head_delta(
          gnm,
          state.identity().detach(),
          rot_neck0,
          rot_head0,
          state.rot_leye.detach(),
          state.rot_reye.detach(),
          trans0,
          d_head,
          t_head_target,
      )
      state.rot_neck = rn
      state.rot_head = rh
      state.translation = tr
    else:
      d_head = torch.stack(
          [d_yaw.new_zeros(()), d_yaw, d_yaw.new_zeros(())], dim=0,
      )
      rn, rh, tr = neck_l2.inverse_head_delta(
          gnm,
          state.identity().detach(),
          rot_neck0,
          rot_head0,
          state.rot_leye.detach(),
          state.rot_reye.detach(),
          trans0,
          d_head,
          t_head_target,
      )
      state.rot_neck = rn
      state.rot_head = rh
      state.translation = tr
    posed = posed_base()
    t_neck = gnm.get_posed_joint_transforms(
        state.identity()[None],
        state.rotations()[None],
        state.translation[None],
    )[0, 0]
    origin_w = t_neck[:3, :3] @ origin_rest + t_neck[:3, 3]
    axis_w = t_neck[:3, :3] @ axis_rest
    axis_w = axis_w / axis_w.norm().clamp(min=1e-8)
    if length_only:
      len_s = 1.0 + 0.5 * torch.sigmoid(neck_len)
      rad_s = neck_len.new_tensor(1.0)
    elif args.freeze_rad is not None and (joints_mode or not neck_rad.requires_grad):
      len_s = 1.0 + 0.5 * torch.sigmoid(neck_len)
      rad_s = neck_len.new_tensor(float(args.freeze_rad))
    else:
      len_s, rad_s = neck_l2.scales_from_params(neck_len, neck_rad)
    deformed = neck_l2.deform_neck(
        posed, ids, weights, origin_w, axis_w, len_s, rad_s,
    )
    return deformed, len_s, rad_s, d_head

  def soft_lines(verts: torch.Tensor):
    xy = project_world_to_pixels(
        verts[shoulder_ids],
        state.camera_rt().detach(), state.camera_k(W, H).detach(),
        W, H, flip_y=True,
    )
    y0 = min(float(left_gt[:, 1].min()), float(right_gt[:, 1].min())) - 8
    y1 = max(float(left_gt[:, 1].max()), float(right_gt[:, 1].max())) + 8
    band = (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
    pts = xy[band] if int(band.sum()) >= 8 else xy
    xl = neck_l2.soft_x_at_ys(pts, left_gt[:, 1], left=True)
    xr = neck_l2.soft_x_at_ys(pts, right_gt[:, 1], left=False)
    left = torch.stack([xl, left_gt[:, 1]], 1)
    right = torch.stack([xr, right_gt[:, 1]], 1)
    return left, right

  def line_dist(left_m: torch.Tensor, right_m: torch.Tensor) -> torch.Tensor:
    """Mean px^2 between soft mesh side-lines and photo lines (both sides)."""
    d_l = (left_m - left_gt).pow(2).sum(dim=-1).mean()
    d_r = (right_m - right_gt).pow(2).sum(dim=-1).mean()
    return d_l + d_r

  slack = float(args.shoulder_slack)

  def shoulder_terms(xy: torch.Tensor, left_m: torch.Tensor, right_m: torch.Tensor):
    outer = neck_l2.shoulder_outer_loss(xy, left_gt, right_gt, slack=slack)
    reach = neck_l2.shoulder_reach_down_loss(xy, left_gt, right_gt, slack=slack)
    inside = neck_l2.shoulder_approach_inside_loss(xy, left_gt, right_gt, slack=slack)
    lined = line_dist(left_m, right_m)
    return outer, reach, inside, lined

  # Near-jaw neck verts: penalize residual stretch (chin pull).
  jaw_sel = wfade_np <= 0.15
  if int(jaw_sel.sum()) < 16:
    jaw_sel = wfade_np <= float(np.percentile(wfade_np, 20))
  jaw_ids = torch.as_tensor(ids_np[jaw_sel], dtype=torch.long)

  # BEFORE
  with torch.no_grad():
    v0, _, _, _ = forward_mesh()
    xy0 = project_world_to_pixels(
        v0[shoulder_ids], state.camera_rt(), state.camera_k(W, H), W, H, flip_y=True,
    )
    left0, right0 = soft_lines(v0)
    o0, r0, i0, l0 = shoulder_terms(xy0, left0, right0)
    print(
        f"[prephoto] before outer={float(o0):.3f} reach={float(r0):.3f} "
        f"line={float(l0):.3f} inside={float(i0):.3f}  verts={ids_np.size}",
        flush=True,
    )

  last = {}
  for i in range(int(args.iters)):
    opt.zero_grad()
    verts, len_s, rad_s, d_head = forward_mesh()
    xy = project_world_to_pixels(
        verts[shoulder_ids],
        state.camera_rt().detach(),
        state.camera_k(W, H).detach(),
        W, H, flip_y=True,
    )
    left_m, right_m = soft_lines(verts)
    outer, reach, inside, lined = shoulder_terms(xy, left_m, right_m)
    face_xy = project_world_to_pixels(
        verts[face_idx],
        state.camera_rt().detach(),
        state.camera_k(W, H).detach(),
        W, H, flip_y=True,
    )
    face_lock = (face_xy - face0).pow(2).mean()
    # Chin band: keep near-jaw neck verts close to undeformed pose (after joints).
    posed_now = posed_base()
    chin_pull = (verts[jaw_ids] - posed_now[jaw_ids]).pow(2).mean()
    if length_only:
      # 只下拉：靠 reach + 肩线距离；outer 防止拉长后外轮廓越界。
      reg = 0.005 * neck_len.pow(2)
      loss = (
          float(args.w_reach) * reach
          + float(args.w_line) * lined
          + float(args.w_outer) * outer
          + 50.0 * face_lock
          + 20.0 * chin_pull
          + reg
      )
    elif joints_mode:
      # Micro 3-axis; orientation lock weight depends on --lock.
      reg = 0.5 * d_head.pow(2).sum() + 0.05 * d_trans.pow(2).sum()
      loss = (
          float(args.w_line) * lined
          + float(args.w_inside) * inside
          + float(args.w_outer) * outer
          + w_face * face_lock
          + w_chin * chin_pull
          + reg
      )
    else:
      # Light rad reg; stronger yaw+face lock so thicken does not drift the face.
      reg = (
          0.08 * d_yaw.pow(2)
          + 0.002 * neck_rad.pow(2)
          + 0.005 * neck_len.pow(2)
      )
      face_w = 80.0 if args.lock == "upper" else 200.0
      loss = (
          float(args.w_outer) * outer
          + float(args.w_reach) * reach
          + float(args.w_inside) * inside
          + float(args.w_line) * lined
          + face_w * face_lock
          + 20.0 * chin_pull
          + reg
      )
    loss.backward()
    opt.step()
    last = {
        "loss": float(loss.detach()),
        "outer": float(outer.detach()),
        "reach": float(reach.detach()),
        "inside": float(inside.detach()),
        "line": float(lined.detach()),
        "face": float(face_lock.detach()),
        "chin": float(chin_pull.detach()),
        "len": float(len_s.detach()),
        "rad": float(rad_s.detach()),
        "d_yaw": float(d_yaw.detach()),
        "d_head": [float(x) for x in d_head.detach().cpu().tolist()],
    }
    if i == 0 or (i + 1) % max(args.iters // 8, 1) == 0 or i + 1 == args.iters:
      dh = last["d_head"]
      print(
          f"  [{i+1:4d}/{args.iters}]  line={last['line']:.3f}  "
          f"chin={last['chin']:.4f}  face={last['face']:.4f}  "
          f"len={last['len']:.3f}  rad={last['rad']:.3f}  "
          f"d_head=[{dh[0]:+.3f},{dh[1]:+.3f},{dh[2]:+.3f}]",
          flush=True,
      )

  with torch.no_grad():
    verts, len_s, rad_s, d_head = forward_mesh()
    face_xy = project_world_to_pixels(
        verts[face_idx], state.camera_rt(), state.camera_k(W, H), W, H, flip_y=True,
    )
    face_mse = float((face_xy - face0).pow(2).mean())
    # Bake deform into static_offset for photo entry (pose has joints applied).
    posed_only = posed_base()
    offset = (verts - posed_only).detach().cpu().numpy().astype(np.float32)

  # --- viz: step1-style UV-mask outer contours (not soft_x zigzags) ---
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  # front = shaded 灰模; constant = flat UV paint for masks
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
  left_gt_np = left_gt.detach().cpu().numpy()
  right_gt_np = right_gt.detach().cpu().numpy()

  def mesh_shoulder_lines(verts_w: torch.Tensor):
    """Same pipeline as /tmp/fit_rot_neck_uvmask.lines_for_current_pose."""
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

  def draw_poly(img, pts, color, thick=2):
    out = img.copy()
    pts_np = pts.detach().cpu().numpy() if torch.is_tensor(pts) else np.asarray(pts)
    if pts_np is None or pts_np.size == 0 or pts_np.shape[0] < 2:
      return out
    cv2.polylines(
        out, [pts_np.astype(np.int32).reshape(-1, 1, 2)], False, color, thick, cv2.LINE_AA,
    )
    return out

  def label(img, text):
    out = img.copy()
    cv2.putText(out, text, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return out

  def mark4(img, meta):
    out = img.copy()
    if not meta:
      return out
    for key, col in (
        ("TL", (0, 255, 0)), ("TR", (0, 255, 0)),
        ("WL", (0, 165, 255)), ("WR", (0, 165, 255)),
    ):
      if key not in meta:
        continue
      P = meta[key]
      cv2.circle(out, (int(P[0]), int(P[1])), 7, col, -1)
      cv2.circle(out, (int(P[0]), int(P[1])), 8, (0, 0, 0), 1)
    return out

  def panel(title, left_m, right_m, meta=None):
    im = bgr.copy()
    im = draw_poly(im, left_gt_np, (0, 255, 255), 3)
    im = draw_poly(im, right_gt_np, (0, 255, 255), 3)
    im = draw_poly(im, left_m, (255, 0, 255), 2)
    im = draw_poly(im, right_m, (255, 0, 255), 2)
    im = mark4(im, meta)
    return label(im, title)

  with torch.no_grad():
    if joints_mode:
      # BEFORE = frozen len/rad, zero joint delta (post hard-grid).
      d_head_raw.zero_()
      v_before, _, _, _ = forward_mesh()
    else:
      # BEFORE: undeformed pose (len=1)
      state.rot_neck = rot_neck0
      state.rot_head = rot_head0
      state.translation = trans0
      v_before = posed_base()
    left0_h, right0_h, _, meta0 = mesh_shoulder_lines(v_before)
    left1_h, right1_h, clay_bgr, meta1 = mesh_shoulder_lines(verts)
    hard_chamfer0 = (
        neck_l2.chamfer_lines_np(left0_h, left_gt_np)
        + neck_l2.chamfer_lines_np(right0_h, right_gt_np)
    )
    hard_chamfer1 = (
        neck_l2.chamfer_lines_np(left1_h, left_gt_np)
        + neck_l2.chamfer_lines_np(right1_h, right_gt_np)
    )
    print(
        f"[prephoto] hard-contour chamfer before={hard_chamfer0:.2f} after={hard_chamfer1:.2f}",
        flush=True,
    )

  clay = label(mark4(clay_bgr, meta1), "AFTER mesh (clay)")
  if length_only:
    mode_tag = "只下拉"
  elif joints_mode:
    mode_tag = "joints"
  else:
    mode_tag = "full"
  ba = np.concatenate(
      [
          panel("BEFORE cyan=photo magenta=mesh", left0_h, right0_h, meta0),
          panel(f"AFTER {mode_tag} cyan=photo magenta=mesh", left1_h, right1_h, meta1),
          clay,
      ],
      1,
  )
  notes = Path("/root/autodl-tmp/notes/gnm_uv/测试用图片")
  notes.mkdir(parents=True, exist_ok=True)
  after_rigid = "rigid" in str(args.params).lower() or args.lock == "upper"
  if after_rigid:
    if length_only:
      vis_name = "GNM v4 刚性后 颈肩只下拉 肩线前后灰模.jpg"
    elif joints_mode:
      vis_name = "GNM v4 刚性后 颈肩关节微调 肩线前后灰模.jpg"
    else:
      vis_name = "GNM v4 刚性后 颈肩DOF 肩线前后灰模.jpg"
  elif length_only:
    vis_name = "GNM v4 SAM眼后 光度前只下拉 肩线前后灰模.jpg"
  elif joints_mode:
    vis_name = "GNM v4 SAM眼后 光度前关节微调 肩线前后灰模.jpg"
  else:
    vis_name = "GNM v4 SAM眼后 光度前颈DOF 肩线前后灰模.jpg"
  vis_path = notes / vis_name
  cv2.imwrite(str(vis_path), ba)
  cv2.imwrite(str(args.out_dir / "shoulder_before_after_clay.jpg"), ba)

  rots = np.asarray(params["rotations"], dtype=np.float32).copy()
  rots[0] = state.rot_neck.detach().cpu().numpy().astype(np.float32)
  rots[1] = state.rot_head.detach().cpu().numpy().astype(np.float32)
  save = {k: np.asarray(params[k]) for k in params.files}
  save["rotations"] = rots
  save["translation"] = state.translation.detach().cpu().numpy().astype(np.float32)
  save["static_offset"] = offset
  np.savez(args.out_dir / "frame_000000_params.npz", **save)
  summary = {
      "mode": args.mode,
      "lock": args.lock,
      "joint_target": args.joint_target if joints_mode else None,
      "outer_before": float(o0),
      "reach_before": float(r0),
      "line_before": float(l0),
      "inside_before": float(i0),
      "outer_after": last.get("outer"),
      "reach_after": last.get("reach"),
      "line_after": last.get("line"),
      "inside_after": last.get("inside"),
      "len_s": float(len_s),
      "rad_s": float(rad_s),
      "d_yaw": float(d_yaw.detach()),
      "d_head": [float(x) for x in d_head.detach().cpu().tolist()],
      "d_trans": [float(x) for x in d_trans.detach().cpu().tolist()],
      "face_mse": face_mse,
      "hard_chamfer_before": float(hard_chamfer0),
      "hard_chamfer_after": float(hard_chamfer1),
      "w_outer": float(args.w_outer),
      "w_reach": float(args.w_reach),
      "w_line": float(args.w_line),
      "w_inside": float(args.w_inside),
      "shoulder_slack": float(args.shoulder_slack),
      "iters": int(args.iters),
      "params_in": str(args.params),
      "vis": str(vis_path),
  }
  (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
  print(json.dumps(summary, indent=2), flush=True)
  print("[prephoto] wrote", args.out_dir / "frame_000000_params.npz", flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
