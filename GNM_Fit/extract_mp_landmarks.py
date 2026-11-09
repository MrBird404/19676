#!/usr/bin/env python3
"""Stage A: extract MediaPipe 468 landmarks from a video or a still image.

Runs WITHOUT importing torch/GNM to avoid mediapipe+torch segfaults.

Green GT preview is written next to --out:
  <out>.jpg              all 468 points as green dots
  <out>_gt_labeled.jpg   same + index labels (optional, --label)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from mediapipe import Image as MPImage
from mediapipe import ImageFormat
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

NUM_MP_FACE = 468
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def detect_one(landmarker, bgr: np.ndarray) -> np.ndarray | None:
  height, width = bgr.shape[:2]
  rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
  res = landmarker.detect(MPImage(image_format=ImageFormat.SRGB, data=rgb))
  if not res.face_landmarks:
    return None
  lms = res.face_landmarks[0][:NUM_MP_FACE]
  return np.array([[p.x * width, p.y * height] for p in lms], dtype=np.float64)


def draw_green_gt(
    bgr: np.ndarray,
    pts: np.ndarray,
    label: bool = False,
    radius: int = 2,
) -> np.ndarray:
  vis = bgr.copy()
  for i, (x, y) in enumerate(pts):
    xi, yi = int(round(x)), int(round(y))
    cv2.circle(vis, (xi, yi), radius, (0, 255, 0), -1, cv2.LINE_AA)
    if label:
      cv2.putText(
          vis, str(i), (xi + 3, yi - 3),
          cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 200, 0), 1, cv2.LINE_AA,
      )
  h, w = vis.shape[:2]
  caption = f"MediaPipe GT  n={len(pts)}  green=GT  {w}x{h}"
  cv2.putText(vis, caption, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
  cv2.putText(vis, caption, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
  return vis


def main() -> int:
  p = argparse.ArgumentParser()
  src = p.add_mutually_exclusive_group(required=True)
  src.add_argument("--video", type=Path)
  src.add_argument("--image", type=Path)
  p.add_argument("--out", type=Path, required=True)
  p.add_argument(
      "--model",
      type=Path,
      default=Path(__file__).resolve().parent.parent / "detectors" / "mediapipe" / "face_landmarker.task",
  )
  p.add_argument("--max_frames", type=int, default=1)
  p.add_argument("--stride", type=int, default=1)
  p.add_argument("--label", action="store_true", help="Also write indexed green GT overlay.")
  args = p.parse_args()

  opts = vision.FaceLandmarkerOptions(
      base_options=BaseOptions(model_asset_path=str(args.model)),
      running_mode=vision.RunningMode.IMAGE,
      num_faces=1,
  )
  landmarker = vision.FaceLandmarker.create_from_options(opts)

  frame_indices = []
  landmarks = []
  preview = None
  labeled = None
  source_path = args.image if args.image is not None else args.video
  fps = 0.0

  if args.image is not None or (args.video is not None and args.video.suffix.lower() in IMAGE_EXTS):
    bgr = cv2.imread(str(source_path))
    if bgr is None:
      raise SystemExit(f"cannot read image {source_path}")
    height, width = bgr.shape[:2]
    pts = detect_one(landmarker, bgr)
    if pts is None:
      landmarker.close()
      raise SystemExit(f"no face in {source_path}")
    frame_indices.append(0)
    landmarks.append(pts)
    preview = draw_green_gt(bgr, pts, label=False)
    if args.label:
      labeled = draw_green_gt(bgr, pts, label=True)
    print(f"image {source_path.name}: face ok  {width}x{height}", flush=True)
  else:
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
      raise SystemExit(f"cannot open {args.video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    idx = 0
    while True:
      ok, bgr = cap.read()
      if not ok:
        break
      if idx % args.stride == 0:
        pts = detect_one(landmarker, bgr)
        if pts is not None:
          frame_indices.append(idx)
          landmarks.append(pts)
          if preview is None:
            preview = draw_green_gt(bgr, pts, label=False)
            if args.label:
              labeled = draw_green_gt(bgr, pts, label=True)
          print(f"frame {idx}: face ok", flush=True)
          if len(landmarks) >= args.max_frames:
            break
        else:
          print(f"frame {idx}: no face", flush=True)
      idx += 1
    cap.release()

  landmarker.close()

  if not landmarks:
    raise SystemExit("no faces detected")

  args.out.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
      args.out,
      frame_indices=np.asarray(frame_indices, dtype=np.int32),
      landmarks=np.stack(landmarks),
      image_size=np.array([width, height], dtype=np.int32),
      fps=np.array(fps, dtype=np.float64),
      video=np.array(str(source_path)),
  )
  if preview is not None:
    cv2.imwrite(str(args.out.with_suffix(".jpg")), preview, [cv2.IMWRITE_JPEG_QUALITY, 95])
  if labeled is not None:
    labeled_path = args.out.with_name(args.out.stem + "_gt_labeled.jpg")
    cv2.imwrite(str(labeled_path), labeled, [cv2.IMWRITE_JPEG_QUALITY, 95])
  meta = {
      "num_frames": len(landmarks),
      "image_size": [width, height],
      "fps": fps,
      "source": str(source_path),
      "out": str(args.out),
      "preview": str(args.out.with_suffix(".jpg")),
  }
  print(json.dumps(meta, indent=2), flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())


if __name__ == "__main__":
  raise SystemExit(main())
