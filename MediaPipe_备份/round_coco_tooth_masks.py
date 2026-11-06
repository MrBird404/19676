#!/usr/bin/env python3
"""Round COCO tooth-mask raster edges (anti-alias + small open/close).

Keeps Chaikin on the annotation polygon, then rasterizes at 4×, blurs,
opens/closes with a tooth-sized ellipse, and downsamples. Writes a new npz;
does not overwrite coco_smooth_as_mask.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/root/autodl-tmp")
COCO = ROOT / "labels_my-project-name_2026-09-18-11-53-16.json"
IMG = ROOT / "sam3_teeth_out" / "frame_0027.png"
OLD_NPZ = ROOT / "sam3_teeth_out" / "gnm_bind" / "masks_fdi.npz"
PREV = ROOT / "gnm_fit_out" / "coco_smooth_as_mask" / "masks_fdi_coco_smooth.npz"
OUT = ROOT / "gnm_fit_out" / "coco_smooth_rounded"
SCALE = 4


def chaikin(pts: np.ndarray, rounds: int = 5, n_samples: int = 96) -> np.ndarray:
  p = pts.astype(np.float64)
  if not np.allclose(p[0], p[-1]):
    p = np.vstack([p, p[0]])
  for _ in range(rounds):
    q = []
    for i in range(len(p) - 1):
      p0, p1 = p[i], p[i + 1]
      q.append(0.75 * p0 + 0.25 * p1)
      q.append(0.25 * p0 + 0.75 * p1)
    p = np.asarray(q)
    p = np.vstack([p, p[0]])
  seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
  s = np.concatenate([[0.0], np.cumsum(seg)])
  if s[-1] < 1e-6:
    return pts.astype(np.float32)
  s /= s[-1]
  t = np.linspace(0, 1, n_samples, endpoint=False)
  return np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], 1).astype(np.float32)


def raster_round(poly_xy: np.ndarray, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
  """4× fill + round, area-downsample. Returns (hard 0/1, occupancy float)."""
  hs, ws = h * SCALE, w * SCALE
  poly = poly_xy.astype(np.float64) * SCALE
  hard4 = np.zeros((hs, ws), np.uint8)
  cv2.fillPoly(hard4, [np.round(poly).astype(np.int32)], 1)
  if hard4.sum() == 0:
    z = np.zeros((h, w), np.uint8)
    return z, z.astype(np.float32)

  ys, xs = np.where(hard4 > 0)
  y0, y1 = max(0, int(ys.min()) - 24), min(hs, int(ys.max()) + 25)
  x0, x1 = max(0, int(xs.min()) - 24), min(ws, int(xs.max()) + 25)
  roi = hard4[y0:y1, x0:x1].copy()
  bw = float(xs.max() - xs.min() + 1)
  bh = float(ys.max() - ys.min() + 1)
  min_orig = min(bw, bh) / SCALE
  k = int(np.clip(round(min(bw, bh) * 0.14), 5, 17))
  if k % 2 == 0:
    k += 1
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
  roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel, iterations=1)
  if min_orig >= 12:
    k_open = max(5, k - 4)
    if k_open % 2 == 0:
      k_open += 1
    kernel_o = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
    roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel_o, iterations=1)

  occ4 = np.zeros((hs, ws), np.float32)
  occ4[y0:y1, x0:x1] = cv2.GaussianBlur(roi.astype(np.float32), (0, 0), sigmaX=2.8)
  occ = np.clip(cv2.resize(occ4, (w, h), interpolation=cv2.INTER_AREA), 0, 1)
  hard = (occ >= 0.45).astype(np.uint8)
  return hard, occ


def to_soft(hard: np.ndarray, occ: np.ndarray | None = None, sigma: float = 1.6) -> np.ndarray:
  hard_u8 = (hard > 0).astype(np.uint8)
  if hard_u8.sum() == 0:
    return hard.astype(np.float32)
  dist_out = cv2.distanceTransform(1 - hard_u8, cv2.DIST_L2, 3)
  fade = np.exp(-dist_out / max(sigma, 1e-3)).astype(np.float32)
  if occ is None:
    soft = np.where(hard_u8 > 0, 1.0, fade)
  else:
    soft = np.maximum(occ.astype(np.float32), fade)
  soft = cv2.GaussianBlur(np.clip(soft, 0, 1), (0, 0), sigmaX=0.7)
  return np.clip(soft, 0, 1).astype(np.float32)


def stair_score(mask: np.ndarray) -> dict:
  m = (mask > 0.5).astype(np.uint8)
  cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  if not cnts:
    return {"n": 0, "axis": 0.0, "turn90": 0}
  c = max(cnts, key=len)
  pts = c.reshape(-1, 2).astype(np.float64)
  d = np.diff(pts, axis=0)
  axis = float(((np.abs(d[:, 0]) < 1e-9) | (np.abs(d[:, 1]) < 1e-9)).mean())
  step = np.sign(d)
  turn90 = 0
  for i in range(len(step) - 1):
    a, b = step[i], step[i + 1]
    if np.dot(a, b) == 0 and np.linalg.norm(a) and np.linalg.norm(b):
      turn90 += 1
  peri = float(cv2.arcLength(c, True))
  area = float(cv2.contourArea(c))
  return {
      "n": int(len(pts)),
      "axis": axis,
      "turn90": turn90,
      "isoper": float(4 * np.pi * area / max(peri ** 2, 1e-6)),
      "area": area,
  }


def crop_mouth(img, y0, y1, x0, x1):
  return img[y0:y1, x0:x1]


def zoom_hard(hard: np.ndarray, scale: int = 8) -> np.ndarray:
  ys, xs = np.where(hard > 0)
  if len(ys) == 0:
    return np.zeros((40, 40, 3), np.uint8)
  y0, y1 = max(0, int(ys.min()) - 4), min(hard.shape[0], int(ys.max()) + 5)
  x0, x1 = max(0, int(xs.min()) - 4), min(hard.shape[1], int(xs.max()) + 5)
  c = (hard[y0:y1, x0:x1] * 255).astype(np.uint8)
  z = cv2.resize(c, (c.shape[1] * scale, c.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
  return cv2.cvtColor(z, cv2.COLOR_GRAY2BGR)


def zoom_occ(occ: np.ndarray, hard: np.ndarray, scale: int = 8) -> np.ndarray:
  ys, xs = np.where(hard > 0)
  if len(ys) == 0:
    return np.zeros((40, 40, 3), np.uint8)
  y0, y1 = max(0, int(ys.min()) - 4), min(hard.shape[0], int(ys.max()) + 5)
  x0, x1 = max(0, int(xs.min()) - 4), min(hard.shape[1], int(xs.max()) + 5)
  c = np.clip(occ[y0:y1, x0:x1] * 255, 0, 255).astype(np.uint8)
  z = cv2.resize(c, (c.shape[1] * scale, c.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
  return cv2.applyColorMap(z, cv2.COLORMAP_JET)


def main() -> int:
  OUT.mkdir(parents=True, exist_ok=True)
  d = json.loads(COCO.read_text())
  cats = {c["id"]: int(c["name"]) for c in d["categories"]}
  img_bgr = cv2.imread(str(IMG))
  h, w = img_bgr.shape[:2]
  old = np.load(OLD_NPZ)
  prev = np.load(PREV)
  fdi_order = [int(x) for x in prev["fdi"]]

  polys = {}
  for a in d["annotations"]:
    fdi = cats[a["category_id"]]
    pts = np.array(a["segmentation"][0], dtype=np.float32).reshape(-1, 2)
    polys[fdi] = chaikin(pts)

  pack = {
      "fdi": np.asarray(fdi_order, dtype=np.int32),
      "sam_ids": np.asarray(old["sam_ids"], dtype=np.int32) if "sam_ids" in old.files else np.arange(1, len(fdi_order) + 1, dtype=np.int32),
      "dist_px": np.asarray(old["dist_px"], dtype=np.float32) if "dist_px" in old.files else np.zeros(len(fdi_order), np.float32),
      "source": np.asarray("coco_chaikin_round4x"),
  }
  hard_maps, soft_maps, occ_maps = {}, {}, {}
  print(f"{'FDI':>4} {'area0':>6} {'area1':>6} {'axis0':>6} {'axis1':>6}")
  for fdi in fdi_order:
    hard, occ = raster_round(polys[fdi], h, w)
    soft = to_soft(hard, occ, sigma=1.6)
    hard_maps[fdi] = hard
    occ_maps[fdi] = occ
    soft_maps[fdi] = soft
    pack[f"mask_{fdi}"] = hard
    pack[f"soft_{fdi}"] = soft
    s0 = stair_score(prev[f"mask_{fdi}"])
    s1 = stair_score(hard)
    print(
        f"{fdi:4d} {s0['area']:6.0f} {s1['area']:6.0f} "
        f"{s0['axis']:6.3f} {s1['axis']:6.3f}"
    )

  for i, a in enumerate(fdi_order):
    for b in fdi_order[i + 1:]:
      both = (hard_maps[a] > 0) & (hard_maps[b] > 0)
      if not both.any():
        continue
      loser = a if hard_maps[a].sum() <= hard_maps[b].sum() else b
      hard_maps[loser] = (hard_maps[loser] & (~both)).astype(np.uint8)
      occ_maps[loser] = occ_maps[loser] * (hard_maps[loser] > 0)
      pack[f"mask_{loser}"] = hard_maps[loser]
      pack[f"soft_{loser}"] = to_soft(hard_maps[loser], occ_maps[loser], sigma=1.6)

  npz_path = OUT / "masks_fdi_coco_rounded.npz"
  np.savez_compressed(npz_path, **pack)
  print("wrote", npz_path)

  union = np.zeros((h, w), np.uint8)
  for fdi in fdi_order:
    union |= hard_maps[fdi]
  ys, xs = np.where(union > 0)
  y0, y1 = max(0, int(ys.min()) - 40), min(h, int(ys.max()) + 41)
  x0, x1 = max(0, int(xs.min()) - 40), min(w, int(xs.max()) + 41)
  np.savez(OUT / "roi.npz", y0=y0, y1=y1, x0=x0, x1=x1)

  colors = {}
  for i, fdi in enumerate(sorted(fdi_order)):
    hue = int(180 * i / max(len(fdi_order), 1))
    rgb = cv2.cvtColor(np.uint8([[[hue, 190, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    colors[fdi] = tuple(int(x) for x in rgb)

  def draw_edges(base, maps, title):
    vis = base.copy()
    for fdi in fdi_order:
      cnts, _ = cv2.findContours(maps[fdi], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
      cv2.drawContours(vis, cnts, -1, colors[fdi], 1, cv2.LINE_AA)
      m = maps[fdi]
      if m.sum() == 0:
        continue
      ys_, xs_ = np.where(m > 0)
      cx, cy = int(xs_.mean()), int(ys_.mean())
      cv2.putText(vis, str(fdi), (cx - 8, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 2, cv2.LINE_AA)
      cv2.putText(vis, str(fdi), (cx - 8, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.32, colors[fdi], 1, cv2.LINE_AA)
    cv2.putText(vis, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return vis

  prev_hard = {fdi: np.asarray(prev[f"mask_{fdi}"]) for fdi in fdi_order}
  e_prev = draw_edges(img_bgr, prev_hard, "Chaikin fillPoly (pixel stairs)")
  e_new = draw_edges(img_bgr, hard_maps, "rounded 4x + open/close")
  cv2.imwrite(str(OUT / "03_edges_prev_full.jpg"), e_prev)
  cv2.imwrite(str(OUT / "03_edges_rounded_full.jpg"), e_new)
  ca, cb = crop_mouth(e_prev, y0, y1, x0, x1), crop_mouth(e_new, y0, y1, x0, x1)
  cv2.imwrite(str(OUT / "03_edges_prev_vs_rounded_crop.jpg"), np.concatenate([ca, cb], 1))

  # zoom: old binary | new binary | new occupancy (what fitting sees via soft)
  zooms = []
  for fdi in (11, 31, 36, 13):
    a = zoom_hard(prev_hard[fdi])
    b = zoom_hard(hard_maps[fdi])
    c = zoom_occ(soft_maps[fdi], hard_maps[fdi])
    hh = min(a.shape[0], b.shape[0], c.shape[0])
    ww = min(a.shape[1], b.shape[1], c.shape[1])
    pair = np.concatenate([a[:hh, :ww], b[:hh, :ww], c[:hh, :ww]], 1)
    cv2.putText(pair, f"FDI {fdi}  old hard | rounded hard | rounded soft", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 180, 255), 1, cv2.LINE_AA)
    zooms.append(pair)
    cv2.imwrite(str(OUT / f"05_zoom_fdi{fdi}.jpg"), pair)
  mw = max(z.shape[1] for z in zooms)
  padded = []
  for z in zooms:
    if z.shape[1] < mw:
      pad = np.zeros((z.shape[0], mw - z.shape[1], 3), np.uint8)
      z = np.concatenate([z, pad], 1)
    padded.append(z)
  cv2.imwrite(str(OUT / "05_zoom_stack.jpg"), np.concatenate(padded, 0))

  overlay = img_bgr.copy()
  tint = np.zeros_like(overlay)
  for fdi in fdi_order:
    tint[hard_maps[fdi] > 0] = colors[fdi]
  vis_ov = cv2.addWeighted(overlay, 0.65, tint, 0.35, 0)
  cv2.imwrite(str(OUT / "02_overlay_rounded_crop.jpg"), crop_mouth(vis_ov, y0, y1, x0, x1))

  axis0 = np.mean([stair_score(prev_hard[f])["axis"] for f in fdi_order])
  axis1 = np.mean([stair_score(hard_maps[f])["axis"] for f in fdi_order])
  print(f"mean axis-aligned contour steps  prev {axis0:.3f}  rounded {axis1:.3f}")
  print("wrote", OUT)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
