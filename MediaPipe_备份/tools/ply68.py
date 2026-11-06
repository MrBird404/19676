"""MeshLab 可开的灰色头网 + 68 彩色小球。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from .style import color_rgb_68


def _icosphere(radius: float, color: tuple[int, int, int]) -> trimesh.Trimesh:
  ball = trimesh.creation.icosphere(subdivisions=2, radius=float(radius))
  rgba = np.array([*color, 255], dtype=np.uint8)
  ball.visual.face_colors = np.tile(rgba, (len(ball.faces), 1))
  ball.visual.vertex_colors = np.tile(rgba, (len(ball.vertices), 1))
  return ball


def export_lmk68_ply(
    verts: np.ndarray,
    faces: np.ndarray,
    pts68: np.ndarray,
    path: Path,
    radius: float = 1.8e-3,
) -> Path:
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  head = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
  head.visual.face_colors = np.tile(
      np.array([200, 200, 205, 255], np.uint8), (len(head.faces), 1))
  parts = [head]
  for i, p in enumerate(np.asarray(pts68)):
    ball = _icosphere(radius, color_rgb_68(i))
    ball.apply_translation(np.asarray(p, dtype=np.float64))
    parts.append(ball)
  trimesh.util.concatenate(parts).export(path)
  return path
