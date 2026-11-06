"""MediaPipe 468 检测，以及 468→68（mp2dlib / 教科书下标）。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from mediapipe import Image as MPImage
from mediapipe import ImageFormat
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from .paths import MEDIAPIPE, MP_MODEL

NUM_MP_FACE = 468


def _ensure_mp_on_path() -> None:
  import sys
  if str(MEDIAPIPE) not in sys.path:
    sys.path.insert(0, str(MEDIAPIPE))


def make_landmarker(model: Path | None = None):
  path = Path(model) if model is not None else MP_MODEL
  if not path.is_file():
    raise FileNotFoundError(f"MediaPipe model not found: {path}")
  opts = vision.FaceLandmarkerOptions(
      base_options=BaseOptions(model_asset_path=str(path)),
      running_mode=vision.RunningMode.IMAGE,
      num_faces=1,
  )
  return vision.FaceLandmarker.create_from_options(opts)


def detect_mp468(bgr: np.ndarray, landmarker=None, model: Path | None = None) -> np.ndarray:
  """返回 (468, 2) 像素坐标。"""
  own = landmarker is None
  if own:
    landmarker = make_landmarker(model)
  try:
    height, width = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = landmarker.detect(MPImage(image_format=ImageFormat.SRGB, data=rgb))
    if not res.face_landmarks:
      raise RuntimeError("no face detected")
    lms = res.face_landmarks[0][:NUM_MP_FACE]
    return np.array([[p.x * width, p.y * height] for p in lms], dtype=np.float64)
  finally:
    if own:
      landmarker.close()


def mp468_to_68(pts468: np.ndarray, gt68_map: str = "mp2dlib") -> np.ndarray:
  """用 Mediapipe_2_Dlib 或教科书下标把 468 收成 68。"""
  _ensure_mp_on_path()
  import gnm_landmark_sets as lmk_sets
  return lmk_sets.sparse68_gt_xy(pts468, gt68_map)


def read_bgr(path: Path) -> np.ndarray:
  bgr = cv2.imread(str(path))
  if bgr is None:
    raise FileNotFoundError(f"cannot read image {path}")
  return bgr
