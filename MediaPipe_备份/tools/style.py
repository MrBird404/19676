"""2D 画点 / iBUG-68 连线。68 的折线与 fit_gnm.py 一致。"""

from __future__ import annotations

import cv2
import numpy as np

# (chain, BGR, thickness)
IBUG68_CHAINS = (
    (list(range(0, 17)), (0, 180, 255), 2),
    (list(range(17, 22)), (0, 165, 255), 2),
    (list(range(22, 27)), (0, 255, 255), 2),
    (list(range(27, 31)), (0, 220, 0), 2),
    (list(range(31, 36)), (0, 220, 0), 2),
    (list(range(36, 42)) + [36], (255, 0, 255), 2),
    (list(range(42, 48)) + [42], (255, 128, 0), 2),
    (list(range(48, 60)) + [48], (0, 0, 220), 2),
    ([60, 61, 62, 63, 64, 65, 66, 67, 60], (0, 220, 220), 2),
)

# MeshLab / PLY 用 RGB（与 export_lmk68_meshlab 分组一致）
PLY_GROUPS = (
    (range(0, 17), (255, 200, 0)),
    (range(17, 22), (255, 165, 0)),
    (range(22, 27), (255, 255, 0)),
    (range(27, 36), (0, 220, 0)),
    (range(36, 42), (255, 0, 255)),
    (range(42, 48), (0, 128, 255)),
    (range(48, 60), (220, 0, 0)),
    (range(60, 68), (0, 220, 220)),
)


def color_bgr_68(i: int) -> tuple[int, int, int]:
  for chain, col, _ in IBUG68_CHAINS:
    if i in chain:
      return col
  return (0, 0, 255)


def color_rgb_68(i: int) -> tuple[int, int, int]:
  for rng, rgb in PLY_GROUPS:
    if i in rng:
      return rgb
  return (255, 255, 255)


def crop_around_xy(img: np.ndarray, xy: np.ndarray, pad: int = 90) -> np.ndarray:
  h, w = img.shape[:2]
  xs, ys = xy[:, 0], xy[:, 1]
  x0 = max(int(np.floor(xs.min())) - pad, 0)
  y0 = max(int(np.floor(ys.min())) - pad, 0)
  x1 = min(int(np.ceil(xs.max())) + pad, w)
  y1 = min(int(np.ceil(ys.max())) + pad, h)
  if x1 <= x0 or y1 <= y0:
    return img
  return img[y0:y1, x0:x1]


def _caption(vis: np.ndarray, text: str) -> None:
  if not text:
    return
  cv2.putText(vis, text, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
  cv2.putText(vis, text, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)


def draw_mp468_points(
    bgr: np.ndarray,
    pts: np.ndarray,
    title: str = "",
    radius: int = 2,
    color: tuple[int, int, int] = (0, 255, 0),
    label: bool = False,
) -> np.ndarray:
  """468 只画点，不连线。"""
  vis = bgr.copy()
  for i, (x, y) in enumerate(np.asarray(pts)):
    p = (int(round(float(x))), int(round(float(y))))
    cv2.circle(vis, p, radius, color, -1, cv2.LINE_AA)
    if label:
      cv2.putText(
          vis, str(i), (p[0] + 3, p[1] - 3),
          cv2.FONT_HERSHEY_SIMPLEX, 0.22, color, 1, cv2.LINE_AA,
      )
  _caption(vis, title or f"MP468  n={len(pts)}  points only")
  return vis


def draw_ibug68(
    bgr: np.ndarray,
    xy: np.ndarray,
    title: str = "",
    radius: int = 5,
    label: bool = True,
) -> np.ndarray:
  """iBUG 68：分组折线 + 点。"""
  vis = bgr.copy()
  xy = np.asarray(xy, dtype=np.float64)
  if xy.shape[0] != 68:
    raise ValueError(f"expected 68 points, got {xy.shape}")
  for chain, col, thick in IBUG68_CHAINS:
    for a, b in zip(chain[:-1], chain[1:]):
      pa = (int(round(xy[a, 0])), int(round(xy[a, 1])))
      pb = (int(round(xy[b, 0])), int(round(xy[b, 1])))
      cv2.line(vis, pa, pb, (0, 0, 0), thick + 2, cv2.LINE_AA)
      cv2.line(vis, pa, pb, col, thick, cv2.LINE_AA)
  for i, (x, y) in enumerate(xy):
    p = (int(round(float(x))), int(round(float(y))))
    c = color_bgr_68(i)
    cv2.circle(vis, p, radius + 2, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(vis, p, radius, c, -1, cv2.LINE_AA)
    if label:
      cv2.putText(
          vis, str(i), (p[0] + 5, p[1] - 4),
          cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 2, cv2.LINE_AA,
      )
      cv2.putText(
          vis, str(i), (p[0] + 5, p[1] - 4),
          cv2.FONT_HERSHEY_SIMPLEX, 0.32, c, 1, cv2.LINE_AA,
      )
  _caption(vis, title or "iBUG 68")
  return vis
