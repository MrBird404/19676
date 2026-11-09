#!/usr/bin/env python3
"""抽帧（约 30fps，长边约 1024）并用 SAM3 把人贴到白底。

蒙版只用于合成，不作为轮廓损失。提示词与 colmap 白底脚本相同：
身体 ``person with neck and shoulders``，帽子 ``blue surgical cap``。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, "/root/autodl-tmp/colmap_video_jamin_static")
from run_sam3_whitebg_all import best_mask, postprocess_mask  # noqa: E402


def resize_long_side(bgr: np.ndarray, long_side: int) -> np.ndarray:
  h, w = bgr.shape[:2]
  m = max(h, w)
  if m <= long_side:
    return bgr
  scale = long_side / float(m)
  nw = max(1, int(round(w * scale)))
  nh = max(1, int(round(h * scale)))
  return cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)


def main() -> int:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--video", type=Path, default=Path("/root/autodl-tmp/expression.mp4"))
  p.add_argument("--out", type=Path, default=Path("/root/autodl-tmp/gnm_fit_out/expression_v3"))
  p.add_argument("--fps", type=float, default=30.0)
  p.add_argument("--long_side", type=int, default=1024)
  p.add_argument("--checkpoint", default="/root/autodl-tmp/sam3-main/checkpoints/sam3.pt")
  p.add_argument("--threshold", type=float, default=0.3)
  p.add_argument("--body-prompt", default="person with neck and shoulders")
  p.add_argument("--cap-prompt", default="blue surgical cap")
  args = p.parse_args()

  src_dir = args.out / "images_src"
  img_dir = args.out / "images"
  alpha_dir = args.out / "alpha_maps"
  for d in (src_dir, img_dir, alpha_dir):
    d.mkdir(parents=True, exist_ok=True)

  cap = cv2.VideoCapture(str(args.video))
  if not cap.isOpened():
    raise SystemExit(f"cannot open {args.video}")
  src_fps = float(cap.get(cv2.CAP_PROP_FPS) or args.fps)
  step = max(src_fps / args.fps, 1e-6)
  frames = []
  next_t = 0.0
  idx = 0
  while True:
    ok, bgr = cap.read()
    if not ok:
      break
    if idx + 0.5 >= next_t:
      bgr = resize_long_side(bgr, args.long_side)
      name = f"{len(frames) + 1:06d}.jpg"
      cv2.imwrite(str(src_dir / name), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
      frames.append(src_dir / name)
      next_t += step
    idx += 1
  cap.release()
  if not frames:
    raise SystemExit("no frames")
  print(f"extracted {len(frames)} frames from {idx}  src_fps={src_fps:.3f}  -> {src_dir}", flush=True)

  from sam3.model_builder import build_sam3_image_model
  from sam3.model.sam3_image_processor import Sam3Processor

  torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
  model = build_sam3_image_model(
      checkpoint_path=args.checkpoint, load_from_HF=False, device="cuda"
  )
  processor = Sam3Processor(model, device="cuda", confidence_threshold=args.threshold)

  failed = []
  for path in tqdm(frames, desc="SAM3 white-bg"):
    image = Image.open(path).convert("RGB")
    arr = np.array(image)
    state = processor.set_image(image)
    body = best_mask(processor, state, args.body_prompt)
    cap_m = best_mask(processor, state, args.cap_prompt)
    if body is None and cap_m is None:
      body = best_mask(processor, state, "person")
    if body is None and cap_m is None:
      failed.append(path.name)
      continue
    mask = np.zeros(arr.shape[:2], dtype=bool)
    if body is not None:
      mask |= body
    if cap_m is not None:
      mask |= cap_m
    mask = postprocess_mask(mask)
    white = np.full_like(arr, 255)
    white[mask] = arr[mask]
    Image.fromarray(white).save(img_dir / path.name, quality=95)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(alpha_dir / f"{path.stem}.png")

  print(f"white {len(frames) - len(failed)}/{len(frames)} -> {img_dir}", flush=True)
  if failed:
    print("FAILED:", ", ".join(failed), flush=True)
    return 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
