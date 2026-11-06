"""渲染 GNM 中性头 + 官方 68（正脸 / 左脸）。正左脸共用同一套相机与画线。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from .gnm_io import load_gnm_template, lookat_center, sample_official68
from .paths import MEDIAPIPE, OFFICIAL_68
from .style import crop_around_xy, draw_ibug68

# yaw=0 相机在 +Z 正脸；yaw=-90 相机在 -X，鼻子朝右（与旧 official_shipped_left 一致）
VIEWS = {
    "front": 0.0,
    "left": -90.0,
}


def orbit_rt(center: np.ndarray, radius: float, yaw_deg: float) -> np.ndarray:
  if str(MEDIAPIPE) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(MEDIAPIPE))
  from nvdiffrast_render.camera import look_at_y_up
  yaw = np.deg2rad(float(yaw_deg))
  offset = np.array(
      [radius * np.sin(yaw), 0.0, radius * np.cos(yaw)], dtype=np.float64)
  return look_at_y_up(center + offset, center)


def _renderer(device: torch.device):
  if str(MEDIAPIPE) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(MEDIAPIPE))
  from nvdiffrast_render import NVDiffRenderer
  return NVDiffRenderer(
      use_opengl=False,
      lighting_type="front",
      lighting_space="camera",
      flip_y_raster=False,
  ).to(device)


def render_official68_view(
    view: str = "front",
    out_path: Path | None = None,
    size: int = 1280,
    radius: float = 0.45,
    landmark_file: Path | None = None,
    verts: np.ndarray | None = None,
    faces: np.ndarray | None = None,
    gnm=None,
    pts68: np.ndarray | None = None,
    renderer=None,
    label: bool = True,
) -> dict[str, Path | np.ndarray]:
  """渲染一张视角。返回 bgr / 带点图路径 / crop 路径。"""
  if view not in VIEWS:
    raise ValueError(f"unknown view {view!r}, expected {sorted(VIEWS)}")
  if not torch.cuda.is_available():
    raise RuntimeError("CUDA required for nvdiffrast 3D render")

  if verts is None or faces is None or gnm is None:
    verts, faces, gnm = load_gnm_template()
  if pts68 is None:
    pts68 = sample_official68(verts, landmark_file or OFFICIAL_68)

  device = torch.device("cuda")
  own_renderer = renderer is None
  if own_renderer:
    renderer = _renderer(device)

  center = lookat_center(verts, gnm)
  rt = orbit_rt(center, radius, VIEWS[view])
  f = 1.5 * size
  k4 = np.array([f, f, size * 0.5, size * 0.5], dtype=np.float32)

  if str(MEDIAPIPE) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(MEDIAPIPE))
  from nvdiffrast_render.camera import project_world_to_pixels

  with torch.no_grad():
    rgba = renderer.render_rgba_vis(
        torch.as_tensor(verts, dtype=torch.float32, device=device)[None],
        torch.as_tensor(faces, dtype=torch.int32, device=device),
        torch.as_tensor(rt, dtype=torch.float32, device=device)[None],
        torch.as_tensor(k4, device=device)[None],
        (size, size),
        background_color=[0.04, 0.04, 0.05],
    )["rgba"][0].detach().float().cpu().numpy()
    xy = project_world_to_pixels(
        torch.as_tensor(pts68, dtype=torch.float32, device=device),
        torch.as_tensor(rt, device=device),
        torch.as_tensor(k4, device=device),
        size, size, flip_y=True,
    ).cpu().numpy()

  bgr = cv2.cvtColor(
      np.clip(rgba[..., :3] * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
  title = f"official GNM 68  {view}"
  vis = draw_ibug68(bgr, xy, title=title, radius=5, label=label)
  crop = crop_around_xy(vis, xy, pad=110)

  result: dict[str, Path | np.ndarray | object] = {
      "bgr": vis, "xy": xy, "crop": crop, "renderer": renderer,
  }
  if out_path is not None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)
    crop_path = out_path.with_name(out_path.stem + "_crop.jpg")
    cv2.imwrite(str(crop_path), crop)
    result["path"] = out_path
    result["crop_path"] = crop_path
  return result
