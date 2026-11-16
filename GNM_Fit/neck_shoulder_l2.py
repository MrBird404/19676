"""Pre-photo neck DOFs: lengthen/thicken + inverse yaw-Y; bilateral shoulder lines.

Used by ``fit_neck_shoulder_prephoto.py`` (before photometric). Scales are
constrained to ``>= 1`` (lengthen/thicken only). Shoulder loss mirrors jaw
outer: two-sided error with a pixel slack deadzone.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch


def matrix_to_axis_angle(r: torch.Tensor) -> torch.Tensor:
  aa_sin = torch.stack(
      [r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]],
      dim=0,
  ) * 0.5
  cos_t = ((torch.trace(r) - 1.0) * 0.5).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
  sin_t = torch.linalg.norm(aa_sin).clamp(min=1e-8)
  theta = torch.atan2(sin_t, cos_t)
  return aa_sin / sin_t * theta


def resample_poly(poly: np.ndarray, n: int = 64) -> np.ndarray:
  poly = np.asarray(poly, dtype=np.float32)
  if poly.ndim != 2 or poly.shape[0] < 2:
    return np.zeros((0, 2), dtype=np.float32)
  d = np.linalg.norm(poly[1:] - poly[:-1], axis=-1)
  s = np.concatenate([[0.0], np.cumsum(d)])
  total = max(float(s[-1]), 1e-6)
  t = np.linspace(0.0, total, n)
  out, j = [], 0
  for ti in t:
    while j + 1 < len(s) and s[j + 1] < ti:
      j += 1
    if j + 1 >= len(s):
      out.append(poly[-1])
    else:
      u = (ti - s[j]) / max(s[j + 1] - s[j], 1e-6)
      out.append(poly[j] * (1.0 - u) + poly[j + 1] * u)
  return np.asarray(out, dtype=np.float32)


def load_shoulder_gt(path: Path, n: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
  data = np.load(path)
  left = resample_poly(np.asarray(data["left_line"], dtype=np.float32), n)
  right = resample_poly(np.asarray(data["right_line"], dtype=np.float32), n)
  if left.shape[0] < 4 or right.shape[0] < 4:
    raise ValueError(f"shoulder lines too short in {path}")
  return (
      torch.as_tensor(left, dtype=torch.float32),
      torch.as_tensor(right, dtype=torch.float32),
  )


def neck_boundary_vertex_ids(
    gnm_np,
    faces: np.ndarray,
    uv_masks: Path,
    *,
    dilate: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
  """UV neck|boundary verts + fade weight (1 at shoulder cut, 0 toward face).

  Do **not** hard-drop face/lips verts (that made a cliff at the jaw). Keep them
  in the set with weight driven to 0 by a smooth rest-Y fade plus a soft UV
  face proximity gate. Squared falloff keeps the chin band near weight 0.
  """
  masks = np.load(uv_masks)
  neck = np.asarray(masks["neck"], dtype=bool)
  bound = np.asarray(masks["boundary"], dtype=bool)
  face = np.asarray(masks["face"], dtype=bool)
  lips = np.asarray(masks["lips"], dtype=bool) if "lips" in masks.files else np.zeros_like(face)
  if dilate > 0:
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
    neck = cv2.dilate(neck.astype(np.uint8), ker).astype(bool)
    bound = cv2.dilate(bound.astype(np.uint8), ker).astype(bool)
  union = neck | bound
  tuv = np.asarray(gnm_np.triangle_uvs, dtype=np.float32).mean(axis=1)
  h, w = union.shape
  x = np.clip(np.rint(tuv[:, 0] * (w - 1)).astype(np.int32), 0, w - 1)
  y = np.clip(np.rint((1.0 - tuv[:, 1]) * (h - 1)).astype(np.int32), 0, h - 1)
  hit = union[y, x]
  if not bool(hit.any()):
    raise RuntimeError("no faces hit UV neck|boundary")
  ids = np.unique(faces[hit].reshape(-1)).astype(np.int64)
  if ids.size < 32:
    raise RuntimeError("too few neck|boundary verts")
  rest = np.asarray(gnm_np(None, None, None, None), dtype=np.float32)
  ys = rest[ids, 1]
  # Shoulder (low y) -> 1; wide near-zero band under the jaw.
  y_lo, y_hi = float(np.percentile(ys, 8)), float(np.percentile(ys, 62))
  if y_hi <= y_lo + 1e-6:
    w_y = np.ones(ids.shape[0], dtype=np.float32)
  else:
    w_y = np.clip((y_hi - ys) / (y_hi - y_lo), 0.0, 1.0).astype(np.float32)
    w_y = (w_y * w_y).astype(np.float32)

  # Soft face gate: 0 on face/lips UV, ramps to 1 over a dilated ring.
  face_uv = (face | lips).astype(np.uint8)
  dist = cv2.distanceTransform((1 - face_uv).astype(np.uint8), cv2.DIST_L2, 5)
  # ~12 px in UV mask space → full weight; inside face → 0.
  gate_uv = np.clip(dist / 12.0, 0.0, 1.0).astype(np.float32)
  # Per-vert gate = min over incident face samples (conservative near lips).
  gate = np.ones(ids.shape[0], dtype=np.float32)
  id_to_i = {int(v): i for i, v in enumerate(ids.tolist())}
  for fi, face_row in enumerate(faces):
    if not bool(hit[fi]):
      continue
    g = float(gate_uv[y[fi], x[fi]])
    for vid in face_row:
      j = id_to_i.get(int(vid))
      if j is not None:
        gate[j] = min(gate[j], g)
  wfade = (w_y * gate * gate).astype(np.float32)
  return ids, wfade


def neck_axis_from_rest(
    rest: np.ndarray,
    ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  pts = rest[ids]
  ys = pts[:, 1]
  hi = ys >= np.percentile(ys, 70)
  lo = ys <= np.percentile(ys, 30)
  origin = pts[hi].mean(axis=0).astype(np.float32)
  bottom = pts[lo].mean(axis=0).astype(np.float32)
  axis = bottom - origin
  n = float(np.linalg.norm(axis))
  if n < 1e-6:
    axis = np.array([0.0, -1.0, 0.0], dtype=np.float32)
  else:
    axis = (axis / n).astype(np.float32)
  return origin, axis


def deform_neck(
    verts: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    origin: torch.Tensor,
    axis: torch.Tensor,
    len_s: torch.Tensor,
    rad_s: torch.Tensor,
    *,
    jaw_t: float | None = None,
) -> torch.Tensor:
  """Lengthen/thicken neck verts with fade weights.

  Both axial stretch and radial thicken use a soft jaw gate (``t`` along the
  neck axis). A hard ``clamp`` on length alone previously left ``rad_s`` full
  strength under the chin → shelf at the jaw. Gate is a smoothstep from the
  jaw plane down toward the shoulders.
  """
  out = verts.clone()
  v = verts[ids]
  rel = v - origin.unsqueeze(0)
  t = (rel * axis.unsqueeze(0)).sum(dim=-1, keepdim=True)
  radial = rel - t * axis.unsqueeze(0)
  t_flat = t.detach().reshape(-1)
  if jaw_t is None:
    jaw = float(torch.quantile(t_flat, 0.45))
  else:
    jaw = float(jaw_t)
  t_hi = float(torch.quantile(t_flat, 0.90))
  span = max(t_hi - jaw, 1e-4)
  # Smoothstep gate in [0,1]: 0 at/above jaw, 1 well below toward shoulders.
  g = ((t - jaw) / span).clamp(0.0, 1.0)
  gate = g * g * (3.0 - 2.0 * g)
  # Length: only the axial excess below the jaw plane (unchanged idea).
  excess = (t - t.new_tensor(jaw)).clamp(min=0.0)
  t_new = t + excess * (len_s - 1.0)
  # Radius: soft-gated so chin/jaw is not thickened into a shelf.
  rad_eff = 1.0 + (rad_s - 1.0) * gate
  v_new = origin.unsqueeze(0) + axis.unsqueeze(0) * t_new + radial * rad_eff
  w = weights.unsqueeze(-1).to(dtype=v.dtype)
  out[ids] = v * (1.0 - w) + v_new * w
  return out


def scales_from_params(len_p: torch.Tensor, rad_p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """Unconstrained params -> scales >=1 (lengthen/thicken only).

  Length up to 1.5; radius up to 1.5 (outer loss still caps overshoot).
  """
  len_s = 1.0 + 0.5 * torch.sigmoid(len_p)
  rad_s = 1.0 + 0.5 * torch.sigmoid(rad_p)
  return len_s, rad_s


def head_joint_target(
    gnm,
    identity: torch.Tensor,
    rotations: torch.Tensor,
    translation: torch.Tensor,
) -> torch.Tensor:
  t = gnm.get_posed_joint_transforms(
      identity[None], rotations[None], translation[None],
  )[0, 1]
  return t.detach().clone()


def inverse_head_delta(
    gnm,
    identity: torch.Tensor,
    rot_neck0: torch.Tensor,
    rot_head0: torch.Tensor,
    rot_leye: torch.Tensor,
    rot_reye: torch.Tensor,
    trans0: torch.Tensor,
    d_head: torch.Tensor,
    t_head_target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Apply rot_head Δ (any axis); closed-form root+translation locks head world pose.

  Same IK idea as step1 ``fit_rot_neck_uvmask``: change head local AA, compensate
  ``rot_neck`` + ``translation`` so the head joint stays at ``t_head_target``.
  """
  d = d_head.reshape(3)
  rot_head = rot_head0 + d
  rots = torch.stack([rot_neck0, rot_head, rot_leye, rot_reye], dim=0)
  joints = gnm.get_posed_joint_transforms(
      identity[None], rots[None], trans0[None],
  )[0]
  t_neck = joints[0]
  t_head = joints[1]
  t_local = torch.linalg.inv(t_neck) @ t_head
  t_neck_new = t_head_target @ torch.linalg.inv(t_local)
  d_r = t_neck_new[:3, :3] @ t_neck[:3, :3].transpose(0, 1)
  aa_delta = matrix_to_axis_angle(d_r)
  t_delta = t_neck_new[:3, 3] - t_neck[:3, 3]
  return rot_neck0 + aa_delta, rot_head, trans0 + t_delta


def inverse_yaw_y(
    gnm,
    identity: torch.Tensor,
    rot_neck0: torch.Tensor,
    rot_head0: torch.Tensor,
    rot_leye: torch.Tensor,
    rot_reye: torch.Tensor,
    trans0: torch.Tensor,
    d_yaw: torch.Tensor,
    t_head_target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Backward-compatible: only rot_head Y via ``inverse_head_delta``."""
  dy = d_yaw.reshape(())
  d_head = torch.stack(
      [dy.new_zeros(()), dy, dy.new_zeros(())],
      dim=0,
  )
  return inverse_head_delta(
      gnm, identity, rot_neck0, rot_head0, rot_leye, rot_reye, trans0,
      d_head, t_head_target,
  )


def micro_head_delta(raw: torch.Tensor, max_rad: float = 0.12) -> torch.Tensor:
  """Map unconstrained 3-vector to a micro AA delta in (-max_rad, max_rad)."""
  return float(max_rad) * torch.tanh(raw.reshape(3))


def upper_orient_vertex_ids(
    gnm_np,
    faces: np.ndarray,
    uv_masks: Path,
    *,
    max_n: int = 300,
    seed: int = 0,
) -> np.ndarray:
  """Verts for rough head-orientation lock (no lips/chin/lower face).

  Uses forehead + nose + eyelids/iris when present. Intended after rigid,
  before lmk — lower contour may still move.
  """
  masks = np.load(uv_masks)
  upper = np.asarray(masks["forehead"], bool) | np.asarray(masks["nose"], bool)
  for key in ("left_eyelid", "right_eyelid", "left_iris", "right_iris"):
    if key in masks.files:
      upper = upper | np.asarray(masks[key], bool)
  tuv = np.asarray(gnm_np.triangle_uvs, dtype=np.float32).mean(axis=1)
  h, w = upper.shape
  x = np.clip(np.rint(tuv[:, 0] * (w - 1)).astype(np.int32), 0, w - 1)
  y = np.clip(np.rint((1.0 - tuv[:, 1]) * (h - 1)).astype(np.int32), 0, h - 1)
  hit = upper[y, x]
  if not bool(hit.any()):
    raise RuntimeError("no faces hit upper-orient UV masks")
  vids = np.unique(faces[hit].reshape(-1)).astype(np.int64)
  rng = np.random.default_rng(seed)
  if vids.size > max_n:
    vids = np.sort(rng.choice(vids, max_n, replace=False))
  return vids


def polish_translation_to_xy(
    *,
    posed_fn,
    project_fn,
    translation0: torch.Tensor,
    set_translation,
    lock_idx: torch.Tensor,
    lock0_xy: torch.Tensor,
    n_iters: int = 40,
    lr: float = 5e-2,
) -> torch.Tensor:
  """Adam-polish translation so ``lock_idx`` verts stay near ``lock0_xy``.

  Used after a root (``rot_neck``) micro-delta so upper face orientation is
  held without changing ``rot_head`` (avoids jaw LBS twist).
  """
  tr = translation0.detach().clone().requires_grad_(True)
  opt = torch.optim.Adam([tr], lr=float(lr))
  for _ in range(int(n_iters)):
    opt.zero_grad()
    set_translation(tr)
    verts = posed_fn()
    xy = project_fn(verts[lock_idx])
    loss = (xy - lock0_xy).pow(2).mean()
    loss.backward()
    opt.step()
  out = tr.detach()
  set_translation(out)
  return out


def soft_x_at_ys(
    proj_xy: torch.Tensor,
    ys: torch.Tensor,
    *,
    left: bool,
    tau_x: float = 2.5,
    sigma_y: float = 5.0,
) -> torch.Tensor:
  w_y = torch.exp(-((proj_xy[None, :, 1] - ys[:, None]) / sigma_y) ** 2)
  x = proj_xy[:, 0]
  if left:
    score = -x[None, :] / tau_x + torch.log(w_y + 1e-8)
  else:
    score = x[None, :] / tau_x + torch.log(w_y + 1e-8)
  w = torch.softmax(score, dim=1)
  return (w * x[None, :]).sum(dim=1)


def _shoulder_band_pts(
    proj_xy: torch.Tensor,
    left_gt: torch.Tensor,
    right_gt: torch.Tensor,
) -> torch.Tensor:
  y0 = torch.minimum(left_gt[:, 1].min(), right_gt[:, 1].min()) - 8.0
  y1 = torch.maximum(left_gt[:, 1].max(), right_gt[:, 1].max()) + 8.0
  band = (proj_xy[:, 1] >= y0) & (proj_xy[:, 1] <= y1)
  pts = proj_xy[band]
  return pts if pts.shape[0] >= 8 else proj_xy


def shoulder_outer_loss(
    proj_xy: torch.Tensor,
    left_gt: torch.Tensor,
    right_gt: torch.Tensor,
    *,
    slack: float = 2.0,
) -> torch.Tensor:
  """One-sided: mesh silhouette must not go *outside* photo shoulder lines.

  Left line: punish mesh_x < gt_x - slack. Right: punish mesh_x > gt_x + slack.
  Used to clamp thicken/outward expand.
  """
  if proj_xy.numel() == 0:
    return proj_xy.new_zeros(())
  pts = _shoulder_band_pts(proj_xy, left_gt, right_gt)
  xs_l = soft_x_at_ys(pts, left_gt[:, 1], left=True)
  xs_r = soft_x_at_ys(pts, right_gt[:, 1], left=False)
  # Outside left = smaller x; outside right = larger x.
  out_l = torch.relu((left_gt[:, 0] - slack) - xs_l).pow(2).mean()
  out_r = torch.relu(xs_r - (right_gt[:, 0] + slack)).pow(2).mean()
  return out_l + out_r


def shoulder_reach_down_loss(
    proj_xy: torch.Tensor,
    left_gt: torch.Tensor,
    right_gt: torch.Tensor,
    *,
    slack: float = 2.0,
    tau: float = 4.0,
) -> torch.Tensor:
  """One-sided: lengthen until mesh reaches photo shoulder-line bottom y."""
  if proj_xy.numel() == 0:
    return proj_xy.new_zeros(())
  gt_ymax = torch.maximum(left_gt[:, 1].max(), right_gt[:, 1].max())
  # Soft max y of projected neck verts.
  w = torch.softmax(proj_xy[:, 1] / tau, dim=0)
  mesh_ymax = (w * proj_xy[:, 1]).sum()
  return torch.relu((gt_ymax - slack) - mesh_ymax).pow(2)


def shoulder_approach_inside_loss(
    proj_xy: torch.Tensor,
    left_gt: torch.Tensor,
    right_gt: torch.Tensor,
    *,
    slack: float = 2.0,
) -> torch.Tensor:
  """One-sided from *inside*: gently close horizontal gap, never rewards overshoot.

  Left: punish mesh_x > gt_x + slack (too far toward center).
  Right: punish mesh_x < gt_x - slack.
  """
  if proj_xy.numel() == 0:
    return proj_xy.new_zeros(())
  pts = _shoulder_band_pts(proj_xy, left_gt, right_gt)
  xs_l = soft_x_at_ys(pts, left_gt[:, 1], left=True)
  xs_r = soft_x_at_ys(pts, right_gt[:, 1], left=False)
  in_l = torch.relu(xs_l - (left_gt[:, 0] + slack)).pow(2).mean()
  in_r = torch.relu((right_gt[:, 0] - slack) - xs_r).pow(2).mean()
  return in_l + in_r


def shoulder_bilateral_loss(
    proj_xy: torch.Tensor,
    left_gt: torch.Tensor,
    right_gt: torch.Tensor,
    *,
    slack: float = 2.0,
) -> torch.Tensor:
  """Legacy name: outer clamp + inside approach (asymmetric bilateral)."""
  return shoulder_outer_loss(
      proj_xy, left_gt, right_gt, slack=slack,
  ) + shoulder_approach_inside_loss(
      proj_xy, left_gt, right_gt, slack=slack,
  )

# --- step1 mesh shoulder contour extract (from fit_rot_neck_uvmask) ---

def stick_contacts(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  m_u8 = (mask.astype(np.uint8) * 255)
  cnts, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  if not cnts:
    return np.zeros(2, np.float32), np.zeros(2, np.float32)
  cnt = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
  hull_idx = cv2.convexHull(cnt, returnPoints=False).flatten()
  hull_pts = cnt[hull_idx]
  n = len(hull_pts)
  cx, cy = hull_pts.mean(axis=0)
  best = None
  for i in range(n):
    p, q = hull_pts[i], hull_pts[(i + 1) % n]
    mid = 0.5 * (p + q)
    vx, vy = q[0] - p[0], q[1] - p[1]
    n1 = np.array([-vy, vx], np.float32)
    n2 = -n1
    to_out = mid - np.array([cx, cy], np.float32)
    normal = n1 if np.dot(n1, to_out) > 0 else n2
    normal = normal / (np.linalg.norm(normal) + 1e-8)
    up_align = float(np.dot(normal, np.array([0.0, -1.0])))
    width = abs(q[0] - p[0])
    if up_align > 0.15 and width > 20:
      score = (up_align, -float(mid[1]))
      if best is None or score > best[0]:
        best = (score, p, q)
  if best is None:
    ys, xs = np.where(mask)
    thr = np.percentile(ys, 5)
    sel = ys <= thr
    xs_s, ys_s = xs[sel], ys[sel]
    return (
        np.array([xs_s[xs_s.argmin()], ys_s[xs_s.argmin()]], np.float32),
        np.array([xs_s[xs_s.argmax()], ys_s[xs_s.argmax()]], np.float32),
    )
  _, p, q = best
  TL, TR = (p, q) if p[0] <= q[0] else (q, p)
  return TL.astype(np.float32), TR.astype(np.float32)


def widest_lr(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  ys, xs = np.where(mask)
  iL, iR = int(xs.argmin()), int(xs.argmax())
  return (
      np.array([xs[iL], ys[iL]], np.float32),
      np.array([xs[iR], ys[iR]], np.float32),
  )


def halfplane_keep(mask, a, b, ref_xy, keep_same_side_as_ref=True):
  vx, vy = float(b[0] - a[0]), float(b[1] - a[1])
  def side(x, y):
    return vx * (y - float(a[1])) - vy * (x - float(a[0]))
  ref = side(ref_xy[0], ref_xy[1])
  sign = 1.0 if ref >= 0 else -1.0
  if not keep_same_side_as_ref:
    sign = -sign
  ys, xs = np.where(mask)
  out = np.zeros_like(mask)
  if ys.size == 0:
    return out
  s = vx * (ys.astype(np.float32) - float(a[1])) - vy * (xs.astype(np.float32) - float(a[0]))
  out[ys, xs] = (s * sign) >= -0.5
  return out


def fill_holes_only(mask: np.ndarray, k: int = 5) -> np.ndarray:
  ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
  closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, ker)
  dil1 = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8))
  return ((closed > 0) & (dil1 > 0)).astype(bool)


def keep_ccs_min_area(mask: np.ndarray, min_area: int = 80) -> np.ndarray:
  m = mask.astype(np.uint8)
  n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
  if n <= 1:
    return mask
  out = np.zeros_like(mask)
  for i in range(1, n):
    if int(stats[i, cv2.CC_STAT_AREA]) >= min_area:
      out[lab == i] = True
  return out if int(out.sum()) > 0 else mask


def side_contours_four_points(mask, TL, TR, WL, WR):
  """Band outer L/R with fixed endpoints: left TL→WL, right TR→WR.

  Use per-row outer edges of the cut band (scanline). Contour walks can wrap
  around and invent 折线; scanline stays on the silhouette and is y-monotonic.
  """
  z = np.zeros((0, 2), np.float32)
  TL = np.asarray(TL, np.float32)
  TR = np.asarray(TR, np.float32)
  WL = np.asarray(WL, np.float32)
  WR = np.asarray(WR, np.float32)

  def chain(top, bot, want_left: bool):
    y0 = int(round(min(float(top[1]), float(bot[1]))))
    y1 = int(round(max(float(top[1]), float(bot[1]))))
    pts = []
    for y in range(y0, y1 + 1):
      if y < 0 or y >= mask.shape[0]:
        continue
      xs = np.where(mask[y])[0]
      if xs.size == 0:
        continue
      x = float(xs[0] if want_left else xs[-1])
      pts.append((x, float(y)))
    if len(pts) < 2:
      return z
    poly = np.asarray(pts, np.float32)
    # Light median on x to kill single-pixel stair noise (keeps no big kinks).
    if poly.shape[0] >= 5:
      xs = poly[:, 0].copy()
      for i in range(len(poly)):
        a, b = max(0, i - 2), min(len(poly), i + 3)
        poly[i, 0] = float(np.median(xs[a:b]))
    poly[0] = top
    poly[-1] = bot
    return poly

  left = chain(TL, WL, want_left=True)
  right = chain(TR, WR, want_left=False)
  return left, right


def extract_shoulder_lines(mask_paint, mask_neck, mesh_alpha, H, W):
  """Strict step1:
  1) neck mask → stick contacts TL, TR
  2) paint∩alpha below stick → widest WL, WR
  3) cut band between stick & widest → side contours
  4) endpoints are exactly TL/WL and TR/WR (no 折线 point-skip)
  """
  if int(mask_paint.sum()) < 50:
    z = np.zeros((0, 2), np.float32)
    return z, z, None
  solid = keep_ccs_min_area(mask_paint.astype(bool) & mesh_alpha.astype(bool), 80)
  solid = fill_holes_only(solid, 5)
  neck = keep_ccs_min_area(mask_neck.astype(bool) & mesh_alpha.astype(bool), 50)
  neck = fill_holes_only(neck, 3) if int(neck.sum()) > 50 else solid
  TL, TR = stick_contacts(neck)
  below = halfplane_keep(solid, TL, TR, (W * 0.5, H - 1.0), True)
  below = keep_ccs_min_area(below, 80)
  if int(below.sum()) < 50:
    z = np.zeros((0, 2), np.float32)
    return z, z, {"TL": TL, "TR": TR, "mask_cc": solid}
  WL, WR = widest_lr(below)
  band = halfplane_keep(below, WL, WR, (W * 0.5, 0.0), True) & below
  band = fill_holes_only(band, 5)
  left, right = side_contours_four_points(band, TL, TR, WL, WR)
  return left, right, {
      "TL": TL, "TR": TR, "WL": WL, "WR": WR,
      "band": band, "mask_cc": solid, "mask_neck": neck,
  }


def chamfer_lines_np(pred: np.ndarray, gt: np.ndarray) -> float:
  if pred.shape[0] < 2 or gt.shape[0] < 2:
    return 1e4
  dy = pred[:, None, 1] - gt[None, :, 1]
  dx = pred[:, None, 0] - gt[None, :, 0]
  dist2 = dx * dx + 0.25 * dy * dy
  return float(0.5 * (dist2.min(0).mean() + dist2.min(1).mean()))


def build_uv_paint_textures(uv_masks: Path, dilate: int = 3):
  """Solid green union + vis (green neck / magenta boundary), nvdiffrast CHW."""
  masks = np.load(uv_masks)
  neck = np.asarray(masks["neck"], bool)
  bound = np.asarray(masks["boundary"], bool)
  if dilate > 0:
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
    neck = cv2.dilate(neck.astype(np.uint8), ker).astype(bool)
    bound = cv2.dilate(bound.astype(np.uint8), ker).astype(bool)
  union = neck | bound
  solid = np.zeros((1024, 1024, 3), np.float32)
  solid[union] = (0.0, 1.0, 0.0)
  vis = np.zeros((1024, 1024, 3), np.float32)
  vis[union] = (0.0, 1.0, 0.0)
  vis[bound] = (1.0, 0.0, 1.0)
  vis[neck & ~bound] = (0.0, 1.0, 0.0)
  tex_solid = torch.as_tensor(np.flipud(solid).transpose(2, 0, 1)[None].copy())
  tex_vis = torch.as_tensor(np.flipud(vis).transpose(2, 0, 1)[None].copy())
  return tex_solid, tex_vis
