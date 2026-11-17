#!/usr/bin/env python3
"""SAM3 ear segmentation with multiple text prompts (single image).

For each prompt we run text-only grounding over the whole image, then keep the
top masks that land in plausible ear zones (left/right side bands). This tells
which prompt best isolates ears on this dataset/photo.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/root/autodl-tmp/sam3-main")
if "pkg_resources" not in sys.modules:
  stub = types.ModuleType("pkg_resources")

  def resource_filename(package, resource):
    return str(Path("/root/autodl-tmp/sam3-main") / package.replace(".", "/") / resource)

  stub.resource_filename = resource_filename
  sys.modules["pkg_resources"] = stub

from sam3.model_builder import build_sam3_image_model  # noqa: E402
from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402

CKPT = "/root/autodl-tmp/sam3-main/checkpoints/sam3.pt"
DEFAULT_IMAGE = "/root/autodl-tmp/gnm_fit_out/expression_v3/images/000001.jpg"

PROMPTS = [
    "ear",
    "left ear",
    "right ear",
    "human ear",
    "auricle",
    "ear lobe",
    "sideburn",
    "ear area beside face",
]

TOP_K = 6  # masks kept per prompt


def top_masks(masks: np.ndarray, scores: np.ndarray, k: int = TOP_K):
  order = np.argsort(-scores.reshape(-1))[:k]
  return [((masks[i] > 0.5).astype(np.uint8), float(scores.reshape(-1)[i])) for i in order]


def ear_zone_filter(mask: np.ndarray, h: int, w: int) -> tuple[float, str | None]:
  """Heuristic: ears sit on left/right lateral bands, middle vertical band."""
  ys, xs = np.nonzero(mask)
  if xs.size == 0:
    return 0.0, None
  cx = xs.mean() / w
  cy = ys.mean() / h
  side = "left" if cx < 0.5 else "right"
  # lateral-ish + mid-height
    # ears roughly y in [0.25, 0.80]
  lateral = 0.06 < cx < 0.44 or 0.56 < cx < 0.94
  midy = 0.18 < cy < 0.85
  score = float(lateral and midy)
  return score, side


def overlay(img_bgr: np.ndarray, mask: np.ndarray, color=(0, 255, 255), alpha=0.45):
  out = img_bgr.copy()
  color_mask = np.zeros_like(out, dtype=np.uint8)
  color_mask[mask.astype(bool)] = color
  out = cv2.addWeighted(out, 1.0 - alpha, color_mask, alpha, 0)
  return out


def main(image_path: str = DEFAULT_IMAGE, out_dir: str | None = None) -> None:
  img = cv2.imread(image_path)
  if img is None:
    raise FileNotFoundError(image_path)
  h, w = img.shape[:2]
  pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

  if out_dir is None:
    out_dir = Path("/root/autodl-tmp/notes/gnm_uv/sam_ears")
  else:
    out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
  model = build_sam3_image_model(checkpoint_path=CKPT, load_from_HF=False, device="cuda")
  processor = Sam3Processor(model, device="cuda", confidence_threshold=0.05)
  state = processor.set_image(pil)

  panels = []
  for prompt in PROMPTS:
    processor.reset_all_prompts(state)
    state = processor.set_text_prompt(prompt=prompt, state=state)
    masks = state.get("masks")
    scores = state.get("scores")
    if masks is None or scores is None or int(masks.shape[0]) == 0:
      panels.append((prompt, None, 0.0))
      print(f"[ear] {prompt!r}: no masks", flush=True)
      continue
    m = masks.detach().float().cpu().numpy()
    if m.ndim == 4:
      m = m[:, 0]
    s = scores.detach().float().cpu().numpy().reshape(-1)
    best = None
    best_score = -1.0
    for i in np.argsort(-s):
      mask = m[i] > 0.5
      zone, side = ear_zone_filter(mask, h, w)
      if zone and s[i] > best_score:
        best_score = float(s[i])
        best = mask
    panels.append((prompt, best, best_score))
    tag = f"{best_score:.2f}" if best is not None else "0.00"
    print(f"[ear] {prompt!r}: zone_score={tag} side={side if best is not None else '-'}",
          flush=True)

    if best is not None:
      vis = overlay(img, best)
      cv2.putText(vis, f"{prompt}  s={best_score:.2f}", (12, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3, cv2.LINE_AA)
      cv2.putText(vis, f"{prompt}  s={best_score:.2f}", (12, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
      cv2.imwrite(str(out_dir / f"ear_{prompt.replace(' ', '_')}.jpg"), vis)

  # Composite grid: original + per-prompt best overlay
  cell_h = max(h, 320)
  cell_w = max(w, 540)
  cols = 4
  rows = int(np.ceil(len(panels) / cols)) + 1  # +1 for original row start
  grid = np.zeros((cell_h * rows, cell_w * cols, 3), dtype=np.uint8)

  def place(panel_img, r, c):
    y0, x0 = r * cell_h, c * cell_w
    ph, pw = panel_img.shape[:2]
    scale = min(cell_h / ph, cell_w / pw)
    small = cv2.resize(panel_img, (max(1, round(pw * scale)), max(1, round(ph * scale))))
    sh, sw = small.shape[:2]
    grid[y0:y0 + sh, x0:x0 + sw] = small

  place(img, 0, 0)
  cv2.putText(grid, "ORIGINAL", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3, cv2.LINE_AA)
  cv2.putText(grid, "ORIGINAL", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

  idx = 1
  for prompt, mask, score in panels:
    r, c = divmod(idx, cols)
    if r >= rows:
      break
    if mask is None:
      vis = img.copy()
      cv2.putText(vis, f"{prompt}: NO MASK", (12, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
    else:
      vis = overlay(img, mask)
      cv2.putText(vis, f"{prompt}  s={score:.2f}", (12, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3, cv2.LINE_AA)
      cv2.putText(vis, f"{prompt}  s={score:.2f}", (12, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    place(vis, r, c)
    idx += 1

  grid_path = out_dir / "ear_prompts_grid.jpg"
  cv2.imwrite(str(grid_path), grid)
  print(f"wrote {grid_path}  size={grid.shape}", flush=True)

  del model, processor
  torch.cuda.empty_cache()


if __name__ == "__main__":
  import argparse
  ap = argparse.ArgumentParser()
  ap.add_argument("--image", default=DEFAULT_IMAGE)
  ap.add_argument("--out_dir", default=None)
  args = ap.parse_args()
  main(args.image, args.out_dir)
