#!/usr/bin/env python3
"""对白底序列的每一帧跑 facepoints，保存 iBUG 68 点（键名 300w）。"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np

FACEPOINTS_ROOT = Path("/root/autodl-tmp/facepoints")
sys.path.insert(0, str(FACEPOINTS_ROOT.parent))
from facepoints import Detector, square_box  # noqa: E402

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)


def yunet_path() -> str:
  dest = os.path.join(tempfile.gettempdir(), "yunet_2023mar.onnx")
  if not os.path.exists(dest) or os.path.getsize(dest) < 1000:
    print(f"downloading YuNet -> {dest}", flush=True)
    urllib.request.urlretrieve(YUNET_URL, dest)
  return dest


def main() -> int:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--images", type=Path, required=True, help="white-background jpg folder")
  p.add_argument("--out", type=Path, required=True)
  p.add_argument("--video", type=Path, default=Path("/root/autodl-tmp/expression.mp4"))
  p.add_argument("--model", type=Path, default=FACEPOINTS_ROOT / "model")
  p.add_argument("--preview", type=int, default=0, help="0-based frame to draw")
  args = p.parse_args()

  paths = sorted(args.images.glob("*.jpg"))
  if not paths:
    raise SystemExit(f"no jpg in {args.images}")

  finder = cv2.FaceDetectorYN.create(yunet_path(), "", (320, 320), score_threshold=0.5)
  det = Detector(str(args.model))
  landmarks = []
  stds = []
  missed = []
  preview = None
  width = height = 0
  for i, path in enumerate(paths):
    bgr = cv2.imread(str(path))
    if bgr is None:
      raise SystemExit(f"cannot read {path}")
    height, width = bgr.shape[:2]
    finder.setInputSize((width, height))
    _, faces = finder.detect(bgr)
    if faces is None or len(faces) == 0:
      missed.append(path.name)
      landmarks.append(np.full((68, 2), -1, np.float32))
      stds.append(np.full((68,), 1e3, np.float32))
      print(f"{path.name}: no face", flush=True)
      continue
    face = max(faces, key=lambda f: float(f[2] * f[3]))
    out = det.detect(bgr, square_box(*face[:4]))
    pts = np.asarray(out["300w"], dtype=np.float32)
    if pts.shape != (68, 2):
      raise SystemExit(f"{path.name}: expected (68,2), got {pts.shape}")
    landmarks.append(pts)
    stds.append(np.full((68,), float(np.median(out["std"])), np.float32))
    if i == args.preview:
      preview = bgr
      preview_pts = pts
    if (i + 1) % 20 == 0 or i == 0:
      print(f"{path.name}: 68 ok  median std {float(np.median(out['std'])):.2f}px", flush=True)

  args.out.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
      args.out,
      frame_indices=np.arange(len(paths), dtype=np.int32),
      landmarks=np.stack(landmarks).astype(np.float32),
      point_std_median=np.stack(stds).astype(np.float32),
      image_size=np.array([width, height], dtype=np.int32),
      image_paths=np.array([str(p) for p in paths]),
      fps=np.float64(30.0),
      video=np.array(str(args.video)),
      detector=np.array("facepoints300w"),
  )
  if preview is not None:
    vis = preview.copy()
    for x, y in preview_pts:
      cv2.circle(vis, (int(round(x)), int(round(y))), 2, (0, 0, 255), -1, cv2.LINE_AA)
    prev_path = args.out.with_suffix(".jpg")
    cv2.imwrite(str(prev_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"preview {prev_path}", flush=True)
  print(f"wrote {args.out}  frames={len(paths)}  missed={len(missed)}", flush=True)
  if missed:
    print("MISSED:", ", ".join(missed), flush=True)
    return 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
