"""True-perspective camera helpers (VHAP / nvdiffrast style).

Camera space: X right, Y up, look along -Z (OpenGL-like), matching
``NVDiffRenderer.projection_from_intrinsics``.
World: GNM native (Y-up, face roughly +Z). No vertex [1,-1,-1] bake.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def look_at_y_up(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray | None = None,
) -> np.ndarray:
  """World-to-camera 4x4 (float32)."""
  if up is None:
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
  eye = np.asarray(eye, dtype=np.float64)
  target = np.asarray(target, dtype=np.float64)
  up = np.asarray(up, dtype=np.float64)
  z = eye - target
  z = z / max(np.linalg.norm(z), 1e-8)
  x = np.cross(up, z)
  x = x / max(np.linalg.norm(x), 1e-8)
  y = np.cross(z, x)
  w2c = np.eye(4, dtype=np.float64)
  w2c[0, :3] = x
  w2c[1, :3] = y
  w2c[2, :3] = z
  w2c[:3, 3] = -w2c[:3, :3] @ eye
  return w2c.astype(np.float32)


def rotation_matrix_to_axis_angle(r: np.ndarray) -> np.ndarray:
  r = np.asarray(r, dtype=np.float64)
  cos_theta = (np.trace(r) - 1.0) * 0.5
  cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
  theta = np.arccos(cos_theta)
  if theta < 1e-8:
    return np.zeros(3, dtype=np.float32)
  if abs(theta - np.pi) < 1e-4:
    # 180° — pick a stable axis from diagonal
    axis = np.sqrt(np.maximum(np.diag(r) + 1.0, 0.0))
    if axis[0] > 1e-4:
      axis[1] = np.copysign(axis[1], r[0, 1])
      axis[2] = np.copysign(axis[2], r[0, 2])
    elif axis[1] > 1e-4:
      axis[2] = np.copysign(axis[2], r[1, 2])
    axis = axis / max(np.linalg.norm(axis), 1e-8)
    return (axis * theta).astype(np.float32)
  axis = np.array([
      r[2, 1] - r[1, 2],
      r[0, 2] - r[2, 0],
      r[1, 0] - r[0, 1],
  ], dtype=np.float64)
  axis = axis / (2.0 * np.sin(theta))
  return (axis * theta).astype(np.float32)


def axis_angle_to_matrix_np(aa: np.ndarray) -> np.ndarray:
  aa = np.asarray(aa, dtype=np.float64)
  angle = float(np.linalg.norm(aa))
  if angle < 1e-8:
    return np.eye(3, dtype=np.float64)
  axis = aa / angle
  x, y, z = axis
  k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
  return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def rt_from_aa_t(cam_aa: np.ndarray, cam_t: np.ndarray) -> np.ndarray:
  rt = np.eye(4, dtype=np.float32)
  rt[:3, :3] = axis_angle_to_matrix_np(cam_aa).astype(np.float32)
  rt[:3, 3] = np.asarray(cam_t, dtype=np.float32)
  return rt


def k_from_focal(focal_norm: float, width: int, height: int) -> np.ndarray:
  """Return (4,) fx,fy,cx,cy with fx=fy=focal_norm*max(H,W)."""
  f = float(focal_norm) * float(max(width, height))
  return np.array([f, f, width * 0.5, height * 0.5], dtype=np.float32)


def projection_from_intrinsics(
    k: torch.Tensor,
    image_size: Tuple[int, int],
    near: float = 0.01,
    far: float = 100.0,
) -> torch.Tensor:
  """VHAP OpenGL-style proj. K is (B,3,3) or (B,4); image_size=(H,W)."""
  b = k.shape[0]
  h, w = image_size
  if k.shape[-2:] == (3, 3):
    fx, fy = k[..., 0, 0], k[..., 1, 1]
    cx, cy = k[..., 0, 2], k[..., 1, 2]
  elif k.shape[-1] == 4:
    fx, fy, cx, cy = k[..., 0], k[..., 1], k[..., 2], k[..., 3]
  else:
    raise ValueError(f"bad K shape {k.shape}")
  proj = torch.zeros(b, 4, 4, device=k.device, dtype=k.dtype)
  proj[:, 0, 0] = fx * 2 / w
  proj[:, 1, 1] = fy * 2 / h
  proj[:, 0, 2] = (w - 2 * cx) / w
  proj[:, 1, 2] = (h - 2 * cy) / h
  proj[:, 2, 2] = -(far + near) / (far - near)
  proj[:, 2, 3] = -2 * far * near / (far - near)
  proj[:, 3, 2] = -1
  return proj


def world_to_ndc(
    verts: torch.Tensor,
    rt: torch.Tensor,
    k: torch.Tensor,
    image_size: Tuple[int, int],
    flip_y: bool = True,
) -> torch.Tensor:
  """verts (B,V,3), rt (B,4,4), k (B,4)|(B,3,3) → ndc (B,V,3)."""
  if rt.shape[-2] == 3:
    mv = torch.nn.functional.pad(rt, [0, 0, 0, 1])
    mv[..., 3, 3] = 1
  else:
    mv = rt
  proj = projection_from_intrinsics(k, image_size)
  if proj.shape[0] < mv.shape[0]:
    proj = proj.expand(mv.shape[0], -1, -1)
  mvp = torch.bmm(proj, mv)
  ones = torch.ones(*verts.shape[:-1], 1, device=verts.device, dtype=verts.dtype)
  posw = torch.cat([verts, ones], dim=-1)
  clip = torch.bmm(posw, mvp.transpose(-1, -2))
  ndc = clip[..., :3] / clip[..., 3:].clamp(min=1e-8)
  if flip_y:
    ndc = ndc.clone()
    ndc[..., 1] = -ndc[..., 1]
  return ndc


def ndc_to_pixels(ndc_xy: torch.Tensor, width: int, height: int) -> torch.Tensor:
  u = (ndc_xy[..., 0] * 0.5 + 0.5) * width
  v = (ndc_xy[..., 1] * 0.5 + 0.5) * height
  return torch.stack([u, v], dim=-1)


def project_world_to_pixels(
    verts: torch.Tensor,
    rt: torch.Tensor,
    k: torch.Tensor,
    width: int,
    height: int,
    flip_y: bool = True,
) -> torch.Tensor:
  """Project world points to pixel xy (OpenCV image coords, Y down)."""
  single = verts.dim() == 2
  if single:
    verts = verts[None]
  if rt.dim() == 2:
    rt = rt[None]
  if k.dim() == 1:
    k = k[None]
  elif k.dim() == 2 and k.shape[-2:] == (3, 3) and k.shape[0] == 3:
    k = k[None]
  ndc = world_to_ndc(verts, rt, k, (height, width), flip_y=flip_y)
  pix = ndc_to_pixels(ndc[..., :2], width, height)
  return pix[0] if single else pix


def perspective_init_from_iod(
    gt_xy: np.ndarray,
    lmk3d_world: np.ndarray,
    width: int,
    height: int,
    iod_idx: tuple[int, int],
    focal_norm: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
  """Frontal look-at + IOD depth. Returns cam_aa, cam_t, focal_norm, pred_xy."""
  gt = np.asarray(gt_xy, dtype=np.float64)
  lmk = np.asarray(lmk3d_world, dtype=np.float64)
  ia, ib = iod_idx
  iod_img = float(np.linalg.norm(gt[ia] - gt[ib]))
  iod_3d = float(np.linalg.norm(lmk[ia] - lmk[ib]))
  f = float(focal_norm) * float(max(width, height))
  # At frontal depth Z: iod_px ≈ f * iod_3d / Z
  z_dist = f * iod_3d / max(iod_img, 1e-6)
  z_dist = float(np.clip(z_dist, 0.15, 5.0))
  center = lmk.mean(axis=0)
  # Face looks +Z in GNM native → put camera further along +Z
  eye = center + np.array([0.0, 0.0, z_dist], dtype=np.float64)
  rt = look_at_y_up(eye, center)
  cam_aa = rotation_matrix_to_axis_angle(rt[:3, :3])
  cam_t = rt[:3, 3].astype(np.float32)

  # Project and shift cam_t so mean lands on GT mean (approx, in camera XY)
  k = k_from_focal(focal_norm, width, height)
  pred = _project_np(lmk, rt, k, width, height)
  # Refine translation in image plane via camera-space XY shift
  # Δu ≈ f * ΔX / Z  →  ΔX ≈ Δu * Z / f
  residual = gt.mean(0) - pred.mean(0)
  # camera looks -Z; points in front have negative cam Z
  lmk_h = np.concatenate([lmk, np.ones((len(lmk), 1))], axis=1)
  cam = (rt @ lmk_h.T).T
  z_mean = float(np.abs(cam[:, 2].mean()))
  dx = residual[0] * z_mean / f
  dy = -residual[1] * z_mean / f  # image Y down vs camera Y up
  cam_t = cam_t + np.array([dx, dy, 0.0], dtype=np.float32)
  rt2 = rt_from_aa_t(cam_aa, cam_t)
  pred2 = _project_np(lmk, rt2, k, width, height)
  return cam_aa, cam_t, float(focal_norm), pred2.astype(np.float32)


def _project_np(
    verts: np.ndarray,
    rt: np.ndarray,
    k: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
  v = torch.as_tensor(verts, dtype=torch.float32)
  rt_t = torch.as_tensor(rt, dtype=torch.float32)
  k_t = torch.as_tensor(k, dtype=torch.float32)
  return project_world_to_pixels(v, rt_t, k_t, width, height).numpy()
