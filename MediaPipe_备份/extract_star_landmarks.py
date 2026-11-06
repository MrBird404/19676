#!/usr/bin/env python3
"""从图片/视频抽 STAR 300W 的 68 点（iBUG 顺序），供官方 GNM 68 拟合。

输出 npz 字段与 extract_mp_landmarks.py 对齐，但 landmarks 形状是 (N, 68, 2)。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from star_landmarks import detect_star68
from tools.style import crop_around_xy, draw_ibug68

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def main() -> int:
  p = argparse.ArgumentParser(description=__doc__)
  src = p.add_mutually_exclusive_group(required=True)
  src.add_argument("--video", type=Path)
  src.add_argument("--image", type=Path)
  p.add_argument("--out", type=Path, required=True)
  p.add_argument("--max_frames", type=int, default=1)
  p.add_argument("--stride", type=int, default=1)
  args = p.parse_args()

  source = args.image if args.image is not None else args.video
  frame_indices = []
  landmarks = []
  preview = None
  fps = 0.0

  if args.image is not None or source.suffix.lower() in IMAGE_EXTS:
    bgr = cv2.imread(str(source))
    if bgr is None:
      raise SystemExit(f"cannot read {source}")
    height, width = bgr.shape[:2]
    pts = detect_star68(bgr, image_path=source)
    frame_indices.append(0)
    landmarks.append(pts)
    preview = draw_ibug68(bgr, pts, title=f"STAR 300W 68  {source.name}", radius=4)
    print(f"image {source.name}: STAR 68 ok  {width}x{height}", flush=True)
  else:
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
      raise SystemExit(f"cannot open {args.video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tmp_frame = args.out.parent / "_star_crop_frame.jpg"
    idx = 0
    while True:
      ok, bgr = cap.read()
      if not ok:
        break
      if idx % args.stride == 0:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(tmp_frame), bgr)
        pts = detect_star68(bgr, image_path=tmp_frame)
        frame_indices.append(idx)
        landmarks.append(pts)
        if preview is None:
          preview = draw_ibug68(bgr, pts, title=f"STAR 300W 68  frame {idx}", radius=4)
        print(f"frame {idx}: STAR 68 ok", flush=True)
        if len(landmarks) >= args.max_frames:
          break
      idx += 1
    cap.release()
    if tmp_frame.is_file():
      tmp_frame.unlink()

  if not landmarks:
    raise SystemExit("no STAR landmarks")

  # 视频无 dlib 时：先落盘当前帧再检（保证 MP 子进程有图）
  args.out.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
      args.out,
      frame_indices=np.asarray(frame_indices, dtype=np.int32),
      landmarks=np.stack(landmarks).astype(np.float32),
      image_size=np.array([width, height], dtype=np.int32),
      fps=np.array(fps, dtype=np.float64),
      video=np.array(str(source)),
      detector=np.array("star300w"),
  )
  if preview is not None:
    cv2.imwrite(str(args.out.with_suffix(".jpg")), preview, [cv2.IMWRITE_JPEG_QUALITY, 95])
    crop = crop_around_xy(preview, landmarks[0], pad=80)
    cv2.imwrite(str(args.out.with_name(args.out.stem + "_crop.jpg")), crop)
  print(json.dumps({
      "num_frames": len(landmarks),
      "image_size": [width, height],
      "source": str(source),
      "out": str(args.out),
      "detector": "star300w",
  }, indent=2), flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
