"""加载 GNM 中性头 + 官方 68 三维点。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .paths import GNM_MAIN, MEDIAPIPE, OFFICIAL_68


def _ensure_paths() -> None:
  for p in (str(GNM_MAIN), str(MEDIAPIPE)):
    if p not in sys.path:
      sys.path.insert(0, p)


def load_gnm_template() -> tuple[np.ndarray, np.ndarray, object]:
  """中性 HEAD 网格。(verts, faces, gnm)。"""
  _ensure_paths()
  from gnm.shape import gnm_numpy
  gnm = gnm_numpy.GNM.from_local(
      version=gnm_numpy.GNMMajorVersion.V3, variant=gnm_numpy.GNMVariant.HEAD)
  verts = np.asarray(gnm(None, None, None, None), dtype=np.float64)
  faces = np.asarray(gnm.triangles, dtype=np.int32)
  return verts, faces, gnm


def sample_barycentric(verts: np.ndarray, landmark_file: Path) -> np.ndarray:
  _ensure_paths()
  import gnm_mediapipe_landmarks as mp_lmk
  idx, wt = mp_lmk.load_landmarks(landmark_file)
  return mp_lmk.sample(verts, idx, wt)


def sample_official68(verts: np.ndarray, landmark_file: Path | None = None) -> np.ndarray:
  path = Path(landmark_file) if landmark_file is not None else OFFICIAL_68
  return sample_barycentric(verts, path)


def lookat_center(verts: np.ndarray, gnm) -> np.ndarray:
  names = getattr(gnm, "vertex_group_names", ())
  if "hockey_mask" in names:
    idx = np.asarray(gnm.vertex_group_indices("hockey_mask"))
    return verts[idx].mean(axis=0)
  return verts.mean(axis=0)
