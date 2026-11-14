#!/usr/bin/env python3
"""Single-frame GNM teeth optimization (route A, static stage only).

Pipeline (face already fitted; do NOT re-fit face landmarks here):
  0. Load frozen face params (identity head/eyes, expression, pose, camera).
  1. Rigid arch align: optimize M_up / M_low (axis-angle + translation).
  2. Teeth shape: optimize id_teeth (80-D) with L_sem + L_semXY + prior.
  3. Optional light refine of M_low only (shape frozen).

Supervision is teeth instance masks only (MediaPipe landmarks are unused).
Occlusion (doc §5.4 teeth-hiding):
  Render lips-only and teeth-only separately; penalize teeth that land in the
  lip screen region (joint z-buffer would hide the lip and kill the gradient).
  Also penalize pred teeth outside dilated GT, and translation toward camera.

Use ``--shape_only`` to freeze M_up/M_low (recommended when free rigid pushes
teeth through the lips).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
REPO = next(p for p in [ROOT, *ROOT.parents] if (p / "GNM-main").is_dir()) / "GNM-main"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT))

from gnm.shape import gnm_pytorch, gnm_numpy  # pylint: disable=wrong-import-position
from nvdiffrast_render import NVDiffRenderer  # pylint: disable=wrong-import-position
from nvdiffrast_render.camera import project_world_to_pixels  # pylint: disable=wrong-import-position
import nvdiffrast.torch as dr  # pylint: disable=wrong-import-position


def axis_angle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
  angle = torch.linalg.norm(aa, dim=-1, keepdim=True).clamp(min=1e-8)
  axis = aa / angle
  x, y, z = axis.unbind(-1)
  zeros = torch.zeros_like(x)
  k = torch.stack([zeros, -z, y, z, zeros, -x, -y, x, zeros], dim=-1)
  k = k.reshape(aa.shape[:-1] + (3, 3))
  eye = torch.eye(3, device=aa.device, dtype=aa.dtype).expand(k.shape)
  return eye + torch.sin(angle)[..., None] * k + (1.0 - torch.cos(angle)[..., None]) * (k @ k)


def parse_k(raw: np.ndarray) -> torch.Tensor:
  raw = np.asarray(raw, dtype=np.float32).reshape(-1)
  if raw.size == 4:
    return torch.tensor(raw, dtype=torch.float32)
  if raw.size == 9:
    km = raw.reshape(3, 3)
    return torch.tensor([km[0, 0], km[1, 1], km[0, 2], km[1, 2]], dtype=torch.float32)
  raise ValueError(f"bad K shape {raw.shape}")


def load_fdi_vertex_map(
    path: Path,
    num_vertices: int,
    visible_fdi: list[int],
) -> tuple[dict[int, np.ndarray], torch.Tensor, list[int]]:
  """Return {fdi: idx}, vertex→channel (-1 none), ordered visible channel list."""
  z = np.load(path)
  per: dict[int, np.ndarray] = {}
  for f in visible_fdi:
    key = f"fdi_{int(f)}"
    if key not in z.files:
      raise KeyError(f"missing {key} in {path}")
    per[int(f)] = np.asarray(z[key], dtype=np.int64)
  channels = sorted(per.keys())
  v2c = torch.full((num_vertices,), -1, dtype=torch.long)
  for c, f in enumerate(channels):
    v2c[torch.as_tensor(per[f], dtype=torch.long)] = c
  return per, v2c, channels


def load_gt_masks(
    path: Path,
    channels: list[int],
    height: int,
    width: int,
    use_soft: bool = True,
) -> torch.Tensor:
  """(C,H,W) float masks in [0,1]."""
  z = np.load(path)
  out = np.zeros((len(channels), height, width), dtype=np.float32)
  for i, f in enumerate(channels):
    soft_k, hard_k = f"soft_{f}", f"mask_{f}"
    if use_soft and soft_k in z.files:
      m = np.asarray(z[soft_k], dtype=np.float32)
    else:
      m = (np.asarray(z[hard_k]) > 0).astype(np.float32)
    if m.shape != (height, width):
      raise ValueError(f"mask {f} shape {m.shape} != {(height, width)}")
    out[i] = np.clip(m, 0.0, 1.0)
  return torch.from_numpy(out)


def mouth_roi(gt: torch.Tensor, pad: int = 48) -> tuple[int, int, int, int]:
  """y0,y1,x0,x1 from union of GT masks."""
  union = gt.sum(0) > 1e-4
  ys, xs = torch.where(union)
  if ys.numel() == 0:
    h, w = gt.shape[-2:]
    return 0, h, 0, w
  y0 = int(max(0, ys.min().item() - pad))
  y1 = int(min(gt.shape[-2], ys.max().item() + 1 + pad))
  x0 = int(max(0, xs.min().item() - pad))
  x1 = int(min(gt.shape[-1], xs.max().item() + 1 + pad))
  # even sizes help some raster paths
  if (y1 - y0) % 2:
    y1 = min(gt.shape[-2], y1 + 1)
  if (x1 - x0) % 2:
    x1 = min(gt.shape[-1], x1 + 1)
  return y0, y1, x0, x1


def soft_centroid(mask: torch.Tensor, ys: torch.Tensor, xs: torch.Tensor, eps: float = 1e-6):
  """mask (H,W) → (2,) xy in full-image coords if ys/xs are absolute."""
  m = mask.clamp(min=0)
  mass = m.sum().clamp(min=eps)
  return torch.stack([(m * xs).sum() / mass, (m * ys).sum() / mass])


def soft_bottom_y(
    mask: torch.Tensor,
    ys: torch.Tensor,
    tau: float = 0.1,
    eps: float = 1e-6,
) -> torch.Tensor:
  """Soft incisal/bottom edge y (OpenCV y-down = larger y is lower on image)."""
  m = mask.clamp(min=0)
  w = m * torch.exp(tau * ys)
  col_mass = w.sum(dim=0).clamp(min=eps)
  y_col = (w * ys).sum(dim=0) / col_mass
  active = m.sum(dim=0) > 1e-3
  if not bool(active.any()):
    return mask.new_zeros(())
  return y_col[active].mean()


def soft_edge_y_from_points(ys: torch.Tensor, *, bottom: bool, tau: float = 0.15) -> torch.Tensor:
  """Stable soft min/max y from a 1D set of pixel ys (no exp overflow)."""
  if ys.numel() == 0:
    return ys.new_zeros(())
  logits = tau * ys if bottom else -tau * ys
  logits = logits - logits.max().detach()
  w = torch.softmax(logits, dim=0)
  return (ys * w).sum()


def soft_bottom_y_stable(
    mask: torch.Tensor,
    ys: torch.Tensor,
    tau: float = 0.1,
    eps: float = 1e-6,
) -> torch.Tensor:
  """Soft bottom y; amax only over masked pixels (avoid empty-row bias)."""
  m = mask.clamp(min=0)
  active = m > 1e-4
  if not bool(active.any()):
    return mask.new_zeros(())
  neg = torch.finfo(ys.dtype).min / 4
  logits = torch.where(active, tau * ys, torch.full_like(ys, neg))
  logits = logits - logits.amax(dim=0, keepdim=True).detach()
  w = torch.where(active, torch.exp(logits) * m, torch.zeros_like(m))
  col_mass = w.sum(dim=0).clamp(min=eps)
  y_col = (w * ys).sum(dim=0) / col_mass
  col_ok = m.sum(dim=0) > 1e-3
  return y_col[col_ok].mean()


def soft_top_y(
    mask: torch.Tensor,
    ys: torch.Tensor,
    tau: float = 0.1,
    eps: float = 1e-6,
) -> torch.Tensor:
  """Soft top y; amax only over masked pixels."""
  m = mask.clamp(min=0)
  active = m > 1e-4
  if not bool(active.any()):
    return mask.new_zeros(())
  neg = torch.finfo(ys.dtype).min / 4
  logits = torch.where(active, -tau * ys, torch.full_like(ys, neg))
  logits = logits - logits.amax(dim=0, keepdim=True).detach()
  w = torch.where(active, torch.exp(logits) * m, torch.zeros_like(m))
  col_mass = w.sum(dim=0).clamp(min=eps)
  y_col = (w * ys).sum(dim=0) / col_mass
  col_ok = m.sum(dim=0) > 1e-3
  return y_col[col_ok].mean()


def faces_excluding_groups(gnm_np, exclude: set[str]) -> np.ndarray:
  """All triangles except those belonging to excluded vertex groups."""
  all_faces = np.asarray(gnm_np.triangles, dtype=np.int32)
  if not exclude:
    return all_faces
  ban = set()
  for g in exclude:
    if g in gnm_np.vertex_group_names:
      ban.update(np.asarray(gnm_np.vertex_group_indices(g)).tolist())
  if not ban:
    return all_faces
  ban_arr = np.fromiter(ban, dtype=np.int64)
  # drop faces that touch any banned vertex
  keep = ~np.isin(all_faces, ban_arr).any(axis=1)
  return all_faces[keep]


def apply_arch_rigid(
    verts: torch.Tensor,
    idx: torch.Tensor,
    aa: torch.Tensor,
    t: torch.Tensor,
    pivot: torch.Tensor,
) -> torch.Tensor:
  """Rotate about fixed pivot then translate; writes into a clone of verts."""
  out = verts.clone()
  r = axis_angle_to_matrix(aa)
  v = verts[idx]
  out[idx] = (v - pivot) @ r.T + pivot + t
  return out


def teeth_faces_from_groups(gnm_np, groups: list[str]) -> np.ndarray:
  faces = []
  for g in groups:
    faces.append(np.asarray(gnm_np.triangles_group(g), dtype=np.int32))
  return np.concatenate(faces, axis=0)


def render_semantic_and_lips(
    renderer: NVDiffRenderer,
    verts: torch.Tensor,
    faces: torch.Tensor,
    v_teeth_attr: torch.Tensor,
    v_lip_attr: torch.Tensor,
    rt: torch.Tensor,
    k: torch.Tensor,
    image_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
  """Rasterize teeth+lips together (z-buffer occlusion).

  Returns:
    teeth_sem: (C,H,W) visible teeth instance channels (lips can occlude)
    lip_mask: (H,W) soft lip occupancy in image space
  """
  h, w = image_size
  # pack attrs: [teeth_C | lip_1]
  v_attr = torch.cat([v_teeth_attr, v_lip_attr[:, None]], dim=-1)
  rast = renderer.rasterize(
      verts[None], faces, rt[None], k[None], (h, w),
      use_cache=False, require_grad=True,
  )
  rast_out = rast["rast_out"]
  verts_clip = rast["verts_clip"]
  attr, _ = dr.interpolate(v_attr[None].contiguous(), rast_out, faces.int())
  alpha = (rast_out[..., -1:] > 0).float()
  packed = attr[0] * alpha[0]
  rgba = torch.cat([packed, alpha[0]], dim=-1)[None]
  rgba_aa = dr.antialias(rgba, rast_out, verts_clip, faces.int())[0]
  if not renderer.flip_y_raster:
    rgba_aa = rgba_aa.flip(0)
  teeth_sem = rgba_aa[..., :-2].permute(2, 0, 1).contiguous()
  lip_mask = rgba_aa[..., -2].contiguous()
  return teeth_sem, lip_mask


def render_semantic_masks(
    renderer: NVDiffRenderer,
    verts: torch.Tensor,
    faces: torch.Tensor,
    v_attr: torch.Tensor,
    rt: torch.Tensor,
    k: torch.Tensor,
    image_size: tuple[int, int],
) -> torch.Tensor:
  """Differentiable (C,H,W) soft semantic channels via nvdiffrast."""
  h, w = image_size
  rast = renderer.rasterize(
      verts[None], faces, rt[None], k[None], (h, w),
      use_cache=False, require_grad=True,
  )
  rast_out = rast["rast_out"]
  verts_clip = rast["verts_clip"]
  attr, _ = dr.interpolate(v_attr[None].contiguous(), rast_out, faces.int())
  alpha = (rast_out[..., -1:] > 0).float()
  sem = attr[0] * alpha[0]
  rgba = torch.cat([sem, alpha[0]], dim=-1)[None]
  rgba_aa = dr.antialias(rgba, rast_out, verts_clip, faces.int())[0]
  if not renderer.flip_y_raster:
    rgba_aa = rgba_aa.flip(0)
  return rgba_aa[..., :-1].permute(2, 0, 1).contiguous()


def colorize_masks(masks: np.ndarray) -> np.ndarray:
  """(C,H,W) → RGB uint8 overlay colors."""
  rng = np.random.RandomState(0)
  h, w = masks.shape[-2:]
  rgb = np.zeros((h, w, 3), dtype=np.float32)
  for i in range(masks.shape[0]):
    col = rng.rand(3).astype(np.float32)
    col = 0.35 + 0.65 * col
    rgb += masks[i][..., None] * col
  rgb = np.clip(rgb, 0, 1)
  return (rgb * 255).astype(np.uint8)


def overlay_pred_gt(image: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
  """Green=GT, Red=pred, Yellow=overlap."""
  out = image.astype(np.float32).copy()
  g = (gt.max(0) > 0.2).astype(np.float32)
  p = (pred.max(0) > 0.2).astype(np.float32)
  out = out * (1.0 - 0.45 * np.maximum(g, p)[..., None])
  out[..., 1] += 180 * g
  out[..., 2] += 180 * p
  return np.clip(out, 0, 255).astype(np.uint8)


def centroid_diag_image(
    image: np.ndarray,
    verts: torch.Tensor,
    per_fdi: dict[int, np.ndarray],
    channels: list[int],
    gt: torch.Tensor,
    rt: torch.Tensor,
    k: torch.Tensor,
    width: int,
    height: int,
) -> np.ndarray:
  vis = image.copy()
  device = rt.device
  verts = verts.to(device)
  ys = torch.arange(height, device=device, dtype=torch.float32)[:, None].expand(height, width)
  xs = torch.arange(width, device=device, dtype=torch.float32)[None, :].expand(height, width)
  for i, f in enumerate(channels):
    idx = torch.as_tensor(per_fdi[f], device=device)
    xy = project_world_to_pixels(verts[idx], rt, k, width, height, flip_y=True)
    gc = xy.mean(0).detach().cpu().numpy()
    sc = soft_centroid(gt[i].to(device), ys, xs).detach().cpu().numpy()
    c1 = tuple(np.round(gc).astype(int))
    c2 = tuple(np.round(sc).astype(int))
    arch_col = (255, 128, 0) if f < 30 else (255, 0, 255)
    cv2.circle(vis, c1, 4, (0, 0, 255), -1)
    cv2.circle(vis, c2, 4, arch_col, -1)
    cv2.line(vis, c1, c2, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, str(f), c2, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
  return vis


def build_argparser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument(
      "--params",
      type=Path,
      default=ROOT.parent / "gnm_fit_out/fit_frame_0027_hybrid468/frame_000000_params.npz",
  )
  p.add_argument(
      "--masks",
      type=Path,
      default=ROOT.parent / "sam3_teeth_out/gnm_bind/masks_fdi.npz",
  )
  p.add_argument(
      "--fdi_verts",
      type=Path,
      default=ROOT.parent / "sam3_teeth_out/gnm_bind/gnm_fdi_vertex_indices.npz",
  )
  p.add_argument(
      "--image",
      type=Path,
      default=ROOT.parent / "sam3_teeth_out/frame_0027.png",
  )
  p.add_argument(
      "--out_dir",
      type=Path,
      default=ROOT.parent / "gnm_fit_out/fit_teeth_frame_0027",
  )
  p.add_argument("--rigid_iters", type=int, default=200,
                 help="Stage1b iters (after centroid-only warm-start).")
  p.add_argument("--rigid_centroid_iters", type=int, default=150,
                 help="Stage1a: centroid-only rigid warm-start (no raster).")
  p.add_argument("--shape_iters", type=int, default=300)
  p.add_argument("--refine_iters", type=int, default=80)
  p.add_argument("--lr_rigid", type=float, default=0.015)
  p.add_argument("--lr_shape", type=float, default=0.02)
  p.add_argument("--lr_refine", type=float, default=0.005)
  p.add_argument("--w_centroid", type=float, default=1.0)
  p.add_argument("--w_sem", type=float, default=20.0)
  p.add_argument("--w_semxy", type=float, default=1.0)
  p.add_argument("--w_prior", type=float, default=0.05)
  p.add_argument("--w_rigid_reg", type=float, default=1e-3)
  p.add_argument("--w_hide", type=float, default=8.0,
                 help="Teeth-hiding: penalize teeth rasterized on lip surface.")
  p.add_argument("--w_outside", type=float, default=4.0,
                 help="Penalize pred teeth outside dilated GT teeth region.")
  p.add_argument("--w_forward", type=float, default=2.0,
                 help="Penalize rigid translation toward camera (poking out).")
  p.add_argument("--w_upper_bottom", type=float, default=25.0,
                 help="Align upper-teeth soft bottom edge to GT mask bottom.")
  p.add_argument(
      "--shape_only",
      action="store_true",
      help="Freeze ALL rigid (upper+lower); only id_teeth.",
  )
  p.add_argument(
      "--lower_rigid_only",
      action="store_true",
      default=True,
      help="Upper: no free 6-DoF. Only lower arch gets full rigid 6-DoF.",
  )
  p.add_argument("--no_lower_rigid_only", action="store_false", dest="lower_rigid_only")
  p.add_argument(
      "--upper_calib",
      action="store_true",
      default=True,
      help="Before lower rigid: 2-DoF upper calib (face-down + into-mouth), then freeze.",
  )
  p.add_argument("--no_upper_calib", action="store_false", dest="upper_calib")
  p.add_argument("--upper_calib_iters", type=int, default=120)
  p.add_argument("--lr_upper_calib", type=float, default=0.002)
  p.add_argument(
      "--w_upper_lock",
      type=float,
      default=50.0,
      help="Keep upper teeth near calibrated maxilla pose (via id_teeth).",
  )
  p.add_argument(
      "--visible_only",
      action="store_true",
      default=True,
      help="L_sem only on GT-visible pixels; block occluded/empty regions.",
  )
  p.add_argument("--no_visible_only", action="store_false", dest="visible_only")
  p.add_argument("--roi_pad", type=int, default=64)
  p.add_argument(
      "--render_scale",
      type=float,
      default=0.5,
      help="Rasterize at this scale during optim (1.0=full). Viz dumps always full-res.",
  )
  p.add_argument("--device", default="cuda")
  return p


def main() -> int:
  args = build_argparser().parse_args()
  device = torch.device(args.device if torch.cuda.is_available() else "cpu")
  if device.type != "cuda":
    raise SystemExit("nvdiffrast teeth fit requires CUDA")

  args.out_dir.mkdir(parents=True, exist_ok=True)
  params = np.load(args.params)
  image = cv2.cvtColor(cv2.imread(str(args.image), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
  width, height = map(int, params["image_size"])
  if image.shape[1] != width or image.shape[0] != height:
    raise SystemExit(f"image {image.shape} vs params size {(height, width)}")

  gnm = gnm_pytorch.GNM.from_local(
      version=gnm_pytorch.GNMMajorVersion.V3,
      variant=gnm_pytorch.GNMVariant.HEAD,
  )
  gnm_np = gnm_numpy.GNM.from_local(
      version=gnm_numpy.GNMMajorVersion.V3,
      variant=gnm_numpy.GNMVariant.HEAD,
  )

  # GNM buffers stay on CPU (index tensors not all moved by .to); raster on CUDA.
  id_head = torch.as_tensor(params["id_head"], dtype=torch.float32)
  id_eyes = torch.as_tensor(params["id_eyes"], dtype=torch.float32)
  expression = torch.as_tensor(params["expression"], dtype=torch.float32)
  rotations = torch.as_tensor(params["rotations"], dtype=torch.float32)
  translation = torch.as_tensor(params["translation"], dtype=torch.float32)
  rt = torch.as_tensor(params["RT"], dtype=torch.float32, device=device)
  k = parse_k(params["K"]).to(device)

  masks_meta = np.load(args.masks)
  visible = [int(x) for x in masks_meta["fdi"]]
  n_verts = int(np.asarray(gnm_np.skinning_weights).shape[-1])
  per_fdi, v2c, channels = load_fdi_vertex_map(args.fdi_verts, n_verts, visible)
  gt_full = load_gt_masks(args.masks, channels, height, width).to(device)
  n_c = len(channels)

  up_idx = torch.as_tensor(
      np.asarray(gnm_np.vertex_group_indices("upper_teeth_and_gums"), dtype=np.int64),
  )
  low_idx = torch.as_tensor(
      np.asarray(gnm_np.vertex_group_indices("lower_teeth_and_gums"), dtype=np.int64),
  )
  faces_np = teeth_faces_from_groups(
      gnm_np,
      [
          "upper_teeth_and_gums",
          "lower_teeth_and_gums",
          "upper_lip",
          "lower_lip",
      ],
  )
  faces = torch.as_tensor(faces_np, dtype=torch.int32, device=device)
  faces_teeth_only = torch.as_tensor(
      teeth_faces_from_groups(
          gnm_np, ["upper_teeth_and_gums", "lower_teeth_and_gums"],
      ),
      dtype=torch.int32,
      device=device,
  )
  faces_lips_only = torch.as_tensor(
      teeth_faces_from_groups(gnm_np, ["upper_lip", "lower_lip"]),
      dtype=torch.int32,
      device=device,
  )
  per_fdi_t = {
      f: torch.as_tensor(idx, dtype=torch.long) for f, idx in per_fdi.items()
  }
  lip_idx = torch.as_tensor(
      np.unique(
          np.concatenate([
              np.asarray(gnm_np.vertex_group_indices("upper_lip")),
              np.asarray(gnm_np.vertex_group_indices("lower_lip")),
          ])
      ),
      dtype=torch.long,
  )

  def posed_verts(id_teeth: torch.Tensor) -> torch.Tensor:
    identity = torch.cat([id_head, id_eyes, id_teeth], dim=0)
    return gnm(
        identity[None], expression[None], rotations[None], translation[None],
    )[0]

  id_teeth0 = torch.zeros(80, dtype=torch.float32)
  with torch.no_grad():
    base0 = posed_verts(id_teeth0).detach()
  pivot_up = base0[up_idx].mean(0).detach()
  pivot_low = base0[low_idx].mean(0).detach()

  aa_up = torch.zeros(3, requires_grad=False)  # no upper rotation (not free 6-DoF)
  t_up = torch.zeros(3, requires_grad=False)
  aa_low = torch.zeros(3, requires_grad=True)
  t_low = torch.zeros(3, requires_grad=True)

  # Camera axes in world: X_cam = R @ X + t; larger z_cam = toward camera.
  R_cw = rt[:3, :3].detach().cpu()
  toward_cam = R_cw[2].clone()
  toward_cam = toward_cam / toward_cam.norm().clamp(min=1e-8)
  into_mouth = (-toward_cam).clone()  # away from camera / behind lips
  cam_right = R_cw[0].clone()
  cam_right = cam_right / cam_right.norm().clamp(min=1e-8)
  # Face-down on image (y↓): with flip_y projection, +cam_y tends to image-up;
  # use -cam_y as image-down so positive s_vert moves crowns toward GT bottom.
  face_down = (-R_cw[1]).clone()
  face_down = face_down / face_down.norm().clamp(min=1e-8)
  # 2-DoF upper calib params (world translation only).
  s_vert = torch.zeros((), requires_grad=False)  # along face_down
  s_back = torch.zeros((), requires_grad=False)  # >=0 along into_mouth (softplus)

  def sync_t_up_from_2dof() -> None:
    """t_up = s_vert * face_down + relu(s_back) * into_mouth (no forward)."""
    with torch.no_grad():
      t_up.copy_(s_vert.detach() * face_down + F.relu(s_back.detach()) * into_mouth)

  def t_up_from_2dof() -> torch.Tensor:
    return s_vert * face_down + F.relu(s_back) * into_mouth

  # After calib, shape stage locks upper verts to this reference (not raw base0).
  base_upper_ref = base0[up_idx].detach().clone()
  supervise_upper = False  # toggled True only during upper calib

  def apply_m(verts: torch.Tensor) -> torch.Tensor:
    # Upper: translation-only (2-DoF calib then frozen). Lower: full 6-DoF.
    v = apply_arch_rigid(verts, up_idx, aa_up, t_up, pivot_up)
    return apply_arch_rigid(v, low_idx, aa_low, t_low, pivot_low)

  def apply_m_calib(verts: torch.Tensor) -> torch.Tensor:
    """Like apply_m but t_up from live 2-DoF params (for Stage0 grads)."""
    tu = t_up_from_2dof()
    v = apply_arch_rigid(verts, up_idx, aa_up, tu, pivot_up)
    return apply_arch_rigid(v, low_idx, aa_low, t_low, pivot_low)

  def upper_lock_loss(verts_m: torch.Tensor) -> torch.Tensor:
    """Upper teeth stay at calibrated maxilla pose (block id_teeth drift)."""
    return ((verts_m[up_idx] - base_upper_ref) ** 2).mean()

  def lower_centroid_loss(verts_m: torch.Tensor) -> torch.Tensor:
    """Centroid loss on lower visible FDI only."""
    v = verts_m.to(device)
    loss = v.new_zeros(())
    n = 0
    for i, f in enumerate(channels):
      if f < 30:
        continue
      idx = per_fdi_t[f].to(device)
      xy = project_world_to_pixels(v[idx], rt, k, width, height, flip_y=True)
      loss = loss + ((xy.mean(0) - gt_cent[i]) ** 2).sum()
      n += 1
    return loss / max(n, 1)

  def upper_centroid_loss(verts_m: torch.Tensor) -> torch.Tensor:
    v = verts_m.to(device)
    loss = v.new_zeros(())
    n = 0
    for i, f in enumerate(channels):
      if f >= 30:
        continue
      idx = per_fdi_t[f].to(device)
      xy = project_world_to_pixels(v[idx], rt, k, width, height, flip_y=True)
      loss = loss + ((xy.mean(0) - gt_cent[i]) ** 2).sum()
      n += 1
    return loss / max(n, 1)

  ys_full = torch.arange(height, device=device, dtype=torch.float32)[:, None].expand(height, width)
  xs_full = torch.arange(width, device=device, dtype=torch.float32)[None, :].expand(height, width)
  gt_cent = torch.stack([soft_centroid(gt_full[i], ys_full, xs_full) for i in range(n_c)], dim=0)

  y0, y1, x0, x1 = mouth_roi(gt_full, pad=args.roi_pad)
  gt_roi = gt_full[:, y0:y1, x0:x1]
  rh, rw = y1 - y0, x1 - x0
  print(f"ROI mouth=({x0},{y0})-({x1},{y1}) size={rw}x{rh}  channels={channels}")
  if args.shape_only:
    print("MODE: --shape_only (freeze ALL rigid; only id_teeth)")
    aa_low.requires_grad_(False)
    t_low.requires_grad_(False)
    args.upper_calib = False
  elif args.lower_rigid_only:
    calib_txt = " + upper 2-DoF calib then freeze" if args.upper_calib else ""
    print(f"MODE: upper no-6DoF{calib_txt}; ONLY lower 6-DoF")
  else:
    print("MODE: free upper+lower rigid (legacy)")

  gt_union = (gt_full.sum(0) > 0.2).float().cpu().numpy()
  dil = cv2.dilate(gt_union.astype(np.uint8), np.ones((9, 9), np.uint8), iterations=2)
  allowed_full = torch.as_tensor(dil.astype(np.float32), device=device)

  v_attr = torch.zeros(n_verts, n_c, device=device)
  for c, f in enumerate(channels):
    v_attr[per_fdi_t[f].to(device), c] = 1.0
  v_lip = torch.zeros(n_verts, device=device)
  v_lip[lip_idx.to(device)] = 1.0

  renderer = NVDiffRenderer(
      use_opengl=False,
      lighting_type="constant",
      lighting_space="camera",
      flip_y_raster=False,
  ).to(device)

  def centroid_loss(verts_m: torch.Tensor) -> torch.Tensor:
    v = verts_m.to(device)
    loss = v.new_zeros(())
    for i, f in enumerate(channels):
      idx = per_fdi_t[f].to(device)
      xy = project_world_to_pixels(v[idx], rt, k, width, height, flip_y=True)
      loss = loss + ((xy.mean(0) - gt_cent[i]) ** 2).sum()
    return loss / max(n_c, 1)

  def mean_centroid_errs(verts_m: torch.Tensor) -> tuple[float, float, float]:
    v = verts_m.to(device)
    up_e, lo_e = [], []
    with torch.no_grad():
      for i, f in enumerate(channels):
        idx = per_fdi_t[f].to(device)
        xy = project_world_to_pixels(v[idx], rt, k, width, height, flip_y=True)
        e = float(torch.linalg.norm(xy.mean(0) - gt_cent[i]).detach())
        (up_e if f < 30 else lo_e).append(e)
    return float(np.mean(up_e + lo_e)), float(np.mean(up_e)), float(np.mean(lo_e))

  def forward_poke_loss(verts_m: torch.Tensor) -> torch.Tensor:
    """Penalize teeth moving toward camera vs base (lip poke-through)."""
    v = verts_m.to(device)
    teeth_i = torch.cat([up_idx.to(device), low_idx.to(device)])
    z = (v[teeth_i] @ rt[:3, :3].T + rt[:3, 3])[:, 2]
    z0 = (base0.to(device)[teeth_i] @ rt[:3, :3].T + rt[:3, 3])[:, 2]
    return F.relu(z.mean() - z0.mean())

  def sem_losses(
      verts_m: torch.Tensor,
      *,
      scale: float | None = None,
      full_res: bool = False,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """L_sem, L_semxy, L_hide, L_outside, L_upper_bottom, pred_full."""
    v = verts_m.to(device)
    s = 1.0 if full_res else float(args.render_scale if scale is None else scale)
    s = float(np.clip(s, 0.25, 1.0))
    if abs(s - 1.0) < 1e-6:
      h_r, w_r = height, width
      k_r = k
      y0r, y1r, x0r, x1r = y0, y1, x0, x1
      gt = gt_roi
      gt_c_all = gt_cent
      allowed = allowed_full[y0:y1, x0:x1]
    else:
      h_r = int(round(height * s)) // 2 * 2
      w_r = int(round(width * s)) // 2 * 2
      sx = w_r / float(width)
      sy = h_r / float(height)
      fx, fy, cx, cy = k.unbind(0)
      k_r = torch.stack([fx * sx, fy * sy, cx * sx, cy * sy])
      y0r, y1r = int(round(y0 * sy)), int(round(y1 * sy))
      x0r, x1r = int(round(x0 * sx)), int(round(x1 * sx))
      if (y1r - y0r) % 2:
        y1r = min(h_r, y1r + 1)
      if (x1r - x0r) % 2:
        x1r = min(w_r, x1r + 1)
      gt = F.interpolate(
          gt_full[None], size=(h_r, w_r), mode="bilinear", align_corners=False,
      )[0][:, y0r:y1r, x0r:x1r]
      allowed = F.interpolate(
          allowed_full[None, None], size=(h_r, w_r), mode="bilinear",
          align_corners=False,
      )[0, 0, y0r:y1r, x0r:x1r]
      gt_c_all = gt_cent * torch.tensor([sx, sy], device=device)

    teeth_only = render_semantic_masks(
        renderer, v, faces_teeth_only, v_attr, rt, k_r, (h_r, w_r),
    )
    pred_full = teeth_only
    _, lip_full = render_semantic_and_lips(
        renderer, v, faces_lips_only, v_attr * 0, v_lip, rt, k_r, (h_r, w_r),
    )
    pred = pred_full[:, y0r:y1r, x0r:x1r]
    lip = lip_full[y0r:y1r, x0r:x1r]
    teeth_u = pred.sum(0).clamp(0, 1)

    hh, ww = pred.shape[-2:]
    ys = torch.arange(hh, device=device, dtype=torch.float32)[:, None].expand(hh, ww)
    xs = torch.arange(ww, device=device, dtype=torch.float32)[None, :].expand(hh, ww)

    # Block gradients through lip-covered pred (occluded) for shape matching.
    if args.visible_only:
      pred_sem = torch.where(lip[None] > 0.4, pred.detach(), pred)
      vis = (gt > 0.15).float()
      # Default lower_rigid_only: mask drives lower only, unless upper calib on.
      if args.lower_rigid_only and not supervise_upper:
        for i, f in enumerate(channels):
          if f < 30:
            vis[i] = 0.0
      elif supervise_upper:
        # Calib: supervise upper visible only.
        for i, f in enumerate(channels):
          if f >= 30:
            vis[i] = 0.0
      denom = vis.mean().clamp(min=1e-4)
      if float(vis.sum()) < 1e-4:
        l_sem = pred.new_zeros(())
      else:
        l_sem = ((pred_sem - gt).abs() * vis).sum() / vis.sum().clamp(min=1e-4)
    else:
      denom = gt.abs().mean().clamp(min=1e-4)
      l_sem = (pred - gt).abs().mean() / denom

    l_hide = (teeth_u * lip).mean()
    l_outside = (teeth_u * (1.0 - allowed)).mean()

    # Upper bottom: during calib, or when upper rigid is free.
    if args.lower_rigid_only and not supervise_upper:
      l_ub = v.new_zeros(())
    else:
      up_ch = [i for i, f in enumerate(channels) if f < 30]
      if up_ch:
        gt_up = gt[up_ch].sum(0).clamp(0, 1)
        pr_up = pred[up_ch].sum(0).clamp(0, 1)
        l_ub = (soft_bottom_y(pr_up, ys) - soft_bottom_y(gt_up, ys)).pow(2)
      else:
        l_ub = v.new_zeros(())

    l_xy = v.new_zeros(())
    n_xy = 0
    for i, f in enumerate(channels):
      if args.lower_rigid_only and not supervise_upper and f < 30:
        continue
      if supervise_upper and f >= 30:
        continue
      mass = pred[i].sum().clamp(min=0.0)
      gt_c = gt_c_all[i]
      if args.visible_only and float(gt[i].sum()) < 5.0:
        continue
      if mass < 1.0:
        idx = per_fdi_t[f].to(device)
        xy = project_world_to_pixels(v[idx], rt, k_r, w_r, h_r, flip_y=True)
        pc = xy.mean(0)
      else:
        pc_local = soft_centroid(pred[i], ys, xs)
        pc = pc_local + torch.tensor([float(x0r), float(y0r)], device=device)
      l_xy = l_xy + ((pc - gt_c) ** 2).sum()
      n_xy += 1
    l_xy = l_xy / max(n_xy, 1)
    return l_sem, l_xy, l_hide, l_outside, l_ub, pred_full.detach()

  def occ_pack(l_hide, l_outside, l_fwd):
    return (
        args.w_hide * l_hide
        + args.w_outside * l_outside
        + args.w_forward * l_fwd
    )
  def snapshot_rigid():
    return {
        "aa_up": aa_up.detach().clone(),
        "t_up": t_up.detach().clone(),
        "aa_low": aa_low.detach().clone(),
        "t_low": t_low.detach().clone(),
    }

  def restore_rigid(snap: dict):
    with torch.no_grad():
      aa_up.copy_(snap["aa_up"])
      t_up.copy_(snap["t_up"])
      aa_low.copy_(snap["aa_low"])
      t_low.copy_(snap["t_low"])

  history: dict[str, list] = {
      "upper_calib": [], "rigid_cent": [], "rigid": [], "shape": [], "refine": [],
  }
  base_cached = base0

  if not args.shape_only and args.upper_calib and args.lower_rigid_only:
    # ---------- Stage 0: upper 2-DoF calib (face-down + into-mouth), then freeze ----------
    print("==> Stage0 UPPER 2-DoF calib (vert + into-mouth; no rotation / no forward)")
    supervise_upper = True
    s_vert = torch.zeros((), requires_grad=True)
    s_back = torch.zeros((), requires_grad=True)
    opt_u = torch.optim.Adam([s_vert, s_back], lr=args.lr_upper_calib)
    with torch.no_grad():
      sync_t_up_from_2dof()
      uu0 = mean_centroid_errs(apply_m(base_cached))[1]
      *_, pred0 = None, None, None, None, None, None
      l_sem0, l_xy0, l_hide0, l_out0, l_ub0, _ = sem_losses(apply_m(base_cached))
      l_fwd0 = forward_poke_loss(apply_m(base_cached))
    best_u = {
        "score": uu0 + 40.0 * float(l_hide0) + 20.0 * float(l_fwd0),
        "s_vert": s_vert.detach().clone(),
        "s_back": s_back.detach().clone(),
    }
    print(f"  init up|c|={uu0:.1f}px hide={float(l_hide0):.4f} (identity)")
    for it in range(args.upper_calib_iters):
      opt_u.zero_grad()
      verts_m = apply_m_calib(base_cached)
      l_cent = upper_centroid_loss(verts_m)
      l_sem, l_xy, l_hide, l_out, l_ub, _ = sem_losses(verts_m)
      l_fwd = forward_poke_loss(verts_m)
      l_reg = s_vert.pow(2) + F.relu(s_back).pow(2)
      loss = (
          args.w_centroid * l_cent
          + 0.5 * args.w_sem * l_sem
          + args.w_semxy * l_xy
          + args.w_upper_bottom * l_ub
          + 2.0 * l_reg
          + args.w_hide * 1.5 * l_hide
          + args.w_outside * l_out
          + args.w_forward * 2.0 * l_fwd
      )
      loss.backward()
      opt_u.step()
      with torch.no_grad():
        # Small calib only: ≤1.5cm vert, ≤1.5cm into mouth.
        s_vert.clamp_(-0.015, 0.015)
        s_back.clamp_(0.0, 0.015)
        sync_t_up_from_2dof()
        mu, uu, ll = mean_centroid_errs(apply_m(base_cached))
      score = uu + 40.0 * float(l_hide.detach()) + 20.0 * float(l_fwd.detach())
      history["upper_calib"].append({
          "iter": it,
          "loss": float(loss.detach()),
          "up": uu, "hide": float(l_hide.detach()),
          "s_vert": float(s_vert.detach()),
          "s_back": float(F.relu(s_back.detach())),
          "ubot": float(l_ub.detach()),
      })
      if score < best_u["score"]:
        best_u = {
            "score": score,
            "s_vert": s_vert.detach().clone(),
            "s_back": s_back.detach().clone(),
        }
      if it % 20 == 0 or it == args.upper_calib_iters - 1:
        print(
            f"  [{it:04d}] up|c|={uu:.1f} ubot={l_ub.item():.2f} hide={l_hide.item():.4f} "
            f"fwd={l_fwd.item():.4f} vert={float(s_vert.detach()):+.5f} "
            f"back={float(F.relu(s_back.detach())):.5f}"
        )
    with torch.no_grad():
      s_vert.copy_(best_u["s_vert"])
      s_back.copy_(best_u["s_back"])
      sync_t_up_from_2dof()
      base_upper_ref.copy_(apply_m(base_cached)[up_idx].detach())
    s_vert.requires_grad_(False)
    s_back.requires_grad_(False)
    supervise_upper = False
    mu, uu, ll = mean_centroid_errs(apply_m(base_cached))
    print(
        f"  restored best Stage0 up|c|={uu:.1f}px  t_up={t_up.detach().tolist()}  "
        f"(vert={float(s_vert.detach()):+.5f} back={float(F.relu(s_back.detach())):.5f})"
    )
    with torch.no_grad():
      v_uc = apply_m(base_cached)
    cv2.imwrite(
        str(args.out_dir / "stage0_upper_calib_centroids.jpg"),
        cv2.cvtColor(
            centroid_diag_image(
                image, v_uc, per_fdi, channels, gt_full, rt, k, width, height,
            ),
            cv2.COLOR_RGB2BGR,
        ),
    )
    *_, pred_uc = sem_losses(v_uc, full_res=True)
    cv2.imwrite(
        str(args.out_dir / "stage0_upper_calib_overlay.jpg"),
        cv2.cvtColor(
            overlay_pred_gt(image, pred_uc.cpu().numpy(), gt_full.cpu().numpy()),
            cv2.COLOR_RGB2BGR,
        ),
    )

  if not args.shape_only:
    # ---------- Stage 1a: lower-arch centroid-only rigid ----------
    print("==> Stage1a LOWER 6-DoF only (centroid; upper frozen after calib)")
    opt_r = torch.optim.Adam([aa_low, t_low], lr=args.lr_rigid)
    best_r = {"score": float("inf"), "snap": snapshot_rigid()}
    for it in range(args.rigid_centroid_iters):
      opt_r.zero_grad()
      verts_m = apply_m(base_cached)
      l_cent = lower_centroid_loss(verts_m)
      l_fwd = forward_poke_loss(verts_m)
      l_reg = aa_low.pow(2).sum() + t_low.pow(2).sum()
      loss = (
          args.w_centroid * l_cent
          + args.w_rigid_reg * l_reg
          + args.w_forward * l_fwd
      )
      loss.backward()
      opt_r.step()
      with torch.no_grad():
        mu, uu, ll = mean_centroid_errs(apply_m(base_cached))
      history["rigid_cent"].append(
          {"iter": it, "loss": float(loss.detach()), "mean": mu, "up": uu, "low": ll}
      )
      if ll < best_r["score"]:
        best_r = {"score": ll, "snap": snapshot_rigid()}
      if it % 25 == 0 or it == args.rigid_centroid_iters - 1:
        print(
            f"  [{it:04d}] low|c|={ll:.1f} up|c|={uu:.1f} (up frozen) "
            f"fwd={l_fwd.item():.4f}"
        )
    restore_rigid(best_r["snap"])
    print(f"  restored best Stage1a low|c|={best_r['score']:.1f}px")

    # ---------- Stage 1b: lower rigid + semantic ----------
    print("==> Stage1b LOWER 6-DoF + mask (upper frozen)")
    opt_r = torch.optim.Adam([aa_low, t_low], lr=args.lr_rigid * 0.7)
    best_r = {"score": float("inf"), "snap": snapshot_rigid()}
    for it in range(args.rigid_iters):
      opt_r.zero_grad()
      verts_m = apply_m(base_cached)
      l_cent = lower_centroid_loss(verts_m)
      l_sem, l_xy, l_hide, l_out, l_ub, _ = sem_losses(verts_m)
      l_fwd = forward_poke_loss(verts_m)
      l_reg = aa_low.pow(2).sum() + t_low.pow(2).sum()
      loss = (
          args.w_centroid * l_cent
          + args.w_sem * l_sem
          + args.w_semxy * l_xy
          + args.w_rigid_reg * l_reg
          + args.w_upper_bottom * l_ub
          + occ_pack(l_hide, l_out, l_fwd)
      )
      loss.backward()
      opt_r.step()
      with torch.no_grad():
        mu, uu, ll = mean_centroid_errs(apply_m(base_cached))
      history["rigid"].append({
          "iter": it,
          "loss": float(loss.detach()),
          "sem": float(l_sem.detach()),
          "hide": float(l_hide.detach()),
          "mean": mu, "up": uu, "low": ll,
      })
      score = ll + 50.0 * float(l_hide.detach())
      if score < best_r["score"]:
        best_r = {"score": score, "snap": snapshot_rigid()}
      if it % 25 == 0 or it == args.rigid_iters - 1:
        print(
            f"  [{it:04d}] loss={loss.item():.3f} sem={l_sem.item():.3f} "
            f"hide={l_hide.item():.4f} low|c|={ll:.1f} up|c|={uu:.1f}"
        )
    restore_rigid(best_r["snap"])
    mu, uu, ll = mean_centroid_errs(apply_m(base_cached))
    print(f"  restored best Stage1b (mean={mu:.1f} up={uu:.1f} low={ll:.1f})")

    with torch.no_grad():
      v_rigid = apply_m(base_cached)
    cv2.imwrite(
        str(args.out_dir / "stage1_rigid_centroids.jpg"),
        cv2.cvtColor(
            centroid_diag_image(
                image, v_rigid, per_fdi, channels, gt_full, rt, k, width, height,
            ),
            cv2.COLOR_RGB2BGR,
        ),
    )
    *_, pred_r = sem_losses(v_rigid, full_res=True)
    cv2.imwrite(
        str(args.out_dir / "stage1_rigid_overlay.jpg"),
        cv2.cvtColor(
            overlay_pred_gt(image, pred_r.cpu().numpy(), gt_full.cpu().numpy()),
            cv2.COLOR_RGB2BGR,
        ),
    )
  else:
    print("==> Skip Stage1 rigid (--shape_only)")

  # ---------- Stage 2: shape id_teeth (upper verts locked to maxilla init) ----------
  print("==> Stage2 id_teeth (upper locked to maxilla; lower free via prior M_low)")
  for p_ in (aa_up, t_up, aa_low, t_low):
    p_.requires_grad_(False)
  id_teeth = torch.zeros(80, requires_grad=True)
  opt_s = torch.optim.Adam([id_teeth], lr=args.lr_shape)
  best_s = {"score": float("inf"), "teeth": id_teeth.detach().clone()}
  for it in range(args.shape_iters):
    opt_s.zero_grad()
    verts = posed_verts(id_teeth)
    verts_m = apply_m(verts)
    l_sem, l_xy, l_hide, l_out, l_ub, _ = sem_losses(verts_m)
    l_cent = lower_centroid_loss(verts_m)
    l_fwd = forward_poke_loss(verts_m)
    l_prior = id_teeth.pow(2).mean()
    l_lock = upper_lock_loss(verts_m)
    w_cent = 0.0 if args.shape_only else 0.5 * args.w_centroid
    w_xy = 0.25 * args.w_semxy if args.shape_only else args.w_semxy
    loss = (
        args.w_sem * l_sem
        + w_xy * l_xy
        + w_cent * l_cent
        + args.w_prior * l_prior
        + args.w_upper_bottom * l_ub
        + args.w_upper_lock * l_lock
        + occ_pack(l_hide, l_out, l_fwd)
    )
    loss.backward()
    opt_s.step()
    with torch.no_grad():
      id_teeth.clamp_(-25.0, 25.0)
      mu, uu, ll = mean_centroid_errs(apply_m(posed_verts(id_teeth.detach())))
    cur = (
        float(loss.detach())
        + 0.01 * float(id_teeth.detach().norm())
        + 20.0 * float(l_hide.detach())
    )
    history["shape"].append({
        "iter": it,
        "loss": float(loss.detach()),
        "sem": float(l_sem.detach()),
        "hide": float(l_hide.detach()),
        "lock": float(l_lock.detach()),
        "teeth_norm": float(id_teeth.detach().norm()),
        "mean": mu, "up": uu, "low": ll,
    })
    if cur < best_s["score"]:
      best_s = {"score": cur, "teeth": id_teeth.detach().clone()}
    if it % 25 == 0 or it == args.shape_iters - 1:
      print(
          f"  [{it:04d}] loss={loss.item():.3f} sem={l_sem.item():.3f} "
          f"lock={l_lock.item():.5f} hide={l_hide.item():.4f} "
          f"|teeth|={id_teeth.detach().norm().item():.2f} "
          f"low|c|={ll:.1f} up|c|={uu:.1f}"
      )
  with torch.no_grad():
    id_teeth.copy_(best_s["teeth"])
  print(f"  restored best Stage2 |teeth|={float(id_teeth.detach().norm()):.2f}")

  with torch.no_grad():
    v_shape = apply_m(posed_verts(id_teeth).detach())
  cv2.imwrite(
      str(args.out_dir / "stage2_shape_centroids.jpg"),
      cv2.cvtColor(
          centroid_diag_image(
              image, v_shape, per_fdi, channels, gt_full, rt, k, width, height,
          ),
          cv2.COLOR_RGB2BGR,
      ),
  )
  *_, pred_s = sem_losses(v_shape, full_res=True)
  cv2.imwrite(
      str(args.out_dir / "stage2_shape_overlay.jpg"),
      cv2.cvtColor(
          overlay_pred_gt(image, pred_s.cpu().numpy(), gt_full.cpu().numpy()),
          cv2.COLOR_RGB2BGR,
      ),
  )
  cv2.imwrite(
      str(args.out_dir / "stage2_pred_masks_color.jpg"),
      cv2.cvtColor(colorize_masks(pred_s.cpu().numpy()), cv2.COLOR_RGB2BGR),
  )
  cv2.imwrite(
      str(args.out_dir / "gt_masks_color.jpg"),
      cv2.cvtColor(colorize_masks(gt_full.cpu().numpy()), cv2.COLOR_RGB2BGR),
  )

  # ---------- Stage 3: light M_low refine (skipped in shape_only) ----------
  if args.refine_iters > 0 and not args.shape_only:
    print("==> Stage3 light M_low refine (shape frozen)")
    id_teeth.requires_grad_(False)
    aa_low.requires_grad_(True)
    t_low.requires_grad_(True)
    opt_f = torch.optim.Adam([aa_low, t_low], lr=args.lr_refine)
    with torch.no_grad():
      base_shape = posed_verts(id_teeth).detach()
    best = {
        "score": float("inf"),
        "aa": aa_low.detach().clone(),
        "t": t_low.detach().clone(),
    }
    for it in range(args.refine_iters):
      opt_f.zero_grad()
      verts_m = apply_m(base_shape)
      l_sem, l_xy, l_hide, l_out, l_ub, _ = sem_losses(verts_m)
      l_cent = lower_centroid_loss(verts_m)
      l_fwd = forward_poke_loss(verts_m)
      l_lock = upper_lock_loss(verts_m)
      loss = (
          args.w_sem * l_sem
          + args.w_semxy * l_xy
          + args.w_centroid * l_cent
          + args.w_upper_bottom * l_ub
          + args.w_upper_lock * l_lock
          + args.w_rigid_reg * (aa_low.pow(2).sum() + t_low.pow(2).sum())
          + occ_pack(l_hide, l_out, l_fwd)
      )
      loss.backward()
      opt_f.step()
      with torch.no_grad():
        mu, uu, ll = mean_centroid_errs(apply_m(base_shape))
      # Upper is locked: select by lower + hide only.
      score = ll + 50.0 * float(l_hide.detach())
      history["refine"].append({
          "iter": it, "loss": float(loss.detach()),
          "hide": float(l_hide.detach()),
          "lock": float(l_lock.detach()),
          "mean": mu, "up": uu, "low": ll,
      })
      if score < best["score"]:
        best = {
            "score": score,
            "aa": aa_low.detach().clone(),
            "t": t_low.detach().clone(),
        }
      if it % 20 == 0 or it == args.refine_iters - 1:
        print(
            f"  [{it:04d}] loss={loss.item():.3f} hide={l_hide.item():.4f} "
            f"lock={l_lock.item():.5f} low|c|={ll:.1f} up|c|={uu:.1f}"
        )
    with torch.no_grad():
      aa_low.copy_(best["aa"])
      t_low.copy_(best["t"])
    print(f"  restored best Stage3 score={best['score']:.1f}")
  elif args.shape_only:
    print("==> Skip Stage3 refine (--shape_only)")

  with torch.no_grad():
    verts_final = apply_m(posed_verts(id_teeth).detach())
    l_sem_f, _, l_hide_f, l_out_f, l_ub_f, pred_f = sem_losses(verts_final, full_res=True)
    mu, uu, ll = mean_centroid_errs(verts_final)
  print(
      f"==> FINAL mean|c|={mu:.1f} up={uu:.1f} low={ll:.1f}px  "
      f"hide={float(l_hide_f):.4f} out={float(l_out_f):.4f} ubot={float(l_ub_f):.2f}"
  )
  print(
      f"==> UPPER LOCK check  aa_up={aa_up.detach().cpu().tolist()} "
      f"t_up={t_up.detach().cpu().tolist()}  "
      f"upper_lock={float(upper_lock_loss(verts_final)):.6f}"
  )
  cv2.imwrite(
      str(args.out_dir / "final_overlay.jpg"),
      cv2.cvtColor(
          overlay_pred_gt(image, pred_f.cpu().numpy(), gt_full.cpu().numpy()),
          cv2.COLOR_RGB2BGR,
      ),
  )
  cv2.imwrite(
      str(args.out_dir / "final_centroids.jpg"),
      cv2.cvtColor(
          centroid_diag_image(
              image, verts_final, per_fdi, channels, gt_full, rt, k, width, height,
          ),
          cv2.COLOR_RGB2BGR,
      ),
  )

  # export mesh + params
  faces_all = faces_excluding_groups(gnm_np, {"tongue"})
  try:
    import trimesh
    trimesh.Trimesh(
        vertices=verts_final.detach().cpu().numpy(),
        faces=faces_all,
        process=False,
    ).export(args.out_dir / "teeth_aligned_head.obj")
  except Exception as e:  # noqa: BLE001
    print("mesh export skipped:", e)

  # Full-head shaded render without tongue
  try:
    from nvdiffrast_render import NVDiffRenderer as _R
    _rend = _R(
        use_opengl=False, lighting_type="front", lighting_space="camera",
        flip_y_raster=False,
    ).to(device)
    with torch.no_grad():
      out_r = _rend.render_rgba_vis(
          verts_final.to(device)[None],
          torch.as_tensor(faces_all, dtype=torch.int32, device=device),
          rt[None], k[None], (height, width),
          background_color=[0.08, 0.08, 0.10],
      )
      rgba = out_r["rgba"][0].detach().float().cpu().numpy()
    rgb = np.clip(rgba[..., :3] * 255, 0, 255).astype(np.uint8)
    alpha = rgba[..., 3:4]
    comp = np.clip(
        image.astype(np.float32) * (1 - 0.55 * alpha)
        + rgb.astype(np.float32) * (0.55 * alpha),
        0, 255,
    ).astype(np.uint8)
    cv2.imwrite(str(args.out_dir / "final_head_render_no_tongue.jpg"),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(args.out_dir / "final_head_on_photo_no_tongue.jpg"),
                cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
    x0c, y0c, x1c, y1c = 800, 620, 1250, 900
    col = np.concatenate(
        [image[y0c:y1c, x0c:x1c], rgb[y0c:y1c, x0c:x1c], comp[y0c:y1c, x0c:x1c]], 1
    )
    for i, name in enumerate(["photo", "mesh(no tongue)", "blend"]):
      cv2.putText(col, name, (i * (x1c - x0c) + 8, 26),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imwrite(str(args.out_dir / "final_face_teeth_collage.jpg"),
                cv2.cvtColor(col, cv2.COLOR_RGB2BGR))
  except Exception as e:  # noqa: BLE001
    print("no-tongue render skipped:", e)

  # per-tooth final errors
  tooth_err = {}
  with torch.no_grad():
    vf = verts_final.to(device)
    for i, f in enumerate(channels):
      idx = per_fdi_t[f].to(device)
      xy = project_world_to_pixels(vf[idx], rt, k, width, height, flip_y=True)
      tooth_err[str(f)] = float(torch.linalg.norm(xy.mean(0) - gt_cent[i]))

  out_npz = {
      "id_teeth": id_teeth.detach().cpu().numpy(),
      "aa_up": aa_up.detach().cpu().numpy(),
      "t_up": t_up.detach().cpu().numpy(),
      "aa_low": aa_low.detach().cpu().numpy(),
      "t_low": t_low.detach().cpu().numpy(),
      "pivot_up": pivot_up.detach().cpu().numpy(),
      "pivot_low": pivot_low.detach().cpu().numpy(),
      "channels_fdi": np.asarray(channels, dtype=np.int32),
      "source_params": str(args.params),
      "roi": np.asarray([y0, y1, x0, x1], dtype=np.int32),
      "tooth_centroid_err_px": np.asarray(
          [tooth_err[str(f)] for f in channels], dtype=np.float32,
      ),
  }
  # keep frozen face for convenience
  for key in (
      "identity", "expression", "id_head", "id_eyes", "rotations", "translation",
      "K", "RT", "image_size", "cam_aa", "cam_t", "focal_norm",
  ):
    if key in params.files:
      out_npz[key] = params[key]
  # patched full identity with teeth
  id_full = np.concatenate(
      [params["id_head"], params["id_eyes"], out_npz["id_teeth"]], axis=0,
  )
  out_npz["identity"] = id_full
  np.savez(args.out_dir / "teeth_fit_params.npz", **out_npz)

  summary = {
      "channels_fdi": channels,
      "tooth_centroid_err_px": tooth_err,
      "mean_centroid_err_px": float(np.mean(list(tooth_err.values()))),
      "up_centroid_err_px": float(np.mean([tooth_err[str(f)] for f in channels if f < 30])),
      "low_centroid_err_px": float(np.mean([tooth_err[str(f)] for f in channels if f >= 30])),
      "aa_up": aa_up.detach().cpu().tolist(),
      "t_up": t_up.detach().cpu().tolist(),
      "aa_low": aa_low.detach().cpu().tolist(),
      "t_low": t_low.detach().cpu().tolist(),
      "upper_calib": bool(args.upper_calib),
      "s_vert": float(s_vert.detach()) if torch.is_tensor(s_vert) else 0.0,
      "s_back": float(F.relu(s_back.detach())) if torch.is_tensor(s_back) else 0.0,
      "id_teeth_norm": float(id_teeth.detach().norm()),
      "shape_only": bool(args.shape_only),
      "hide": float(l_hide_f),
      "outside": float(l_out_f),
      "upper_bottom": float(l_ub_f),
      "visible_only": bool(args.visible_only),
      "roi_xyxy": [x0, y0, x1, y1],
      "render_scale": args.render_scale,
      "history_tail": {
          "upper_calib": history["upper_calib"][-1] if history["upper_calib"] else None,
          "rigid_cent": history["rigid_cent"][-1] if history["rigid_cent"] else None,
          "rigid": history["rigid"][-1] if history["rigid"] else None,
          "shape": history["shape"][-1] if history["shape"] else None,
          "refine": history["refine"][-1] if history["refine"] else None,
      },
  }
  (args.out_dir / "summary.json").write_text(
      json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
  )
  print(f"done → {args.out_dir}")
  print(f"mean centroid err = {summary['mean_centroid_err_px']:.2f} px")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
