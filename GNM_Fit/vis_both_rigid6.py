#!/usr/bin/env python3
"""Combine independent upper/lower 6-DoF and add arch–arch collision.

Upper is frozen at its clip6 pose (follows the head). Lower 6-DoF is refined
so crown vertices cannot sink into the opposite arch. Bite-edge mask losses
on the lower arch keep the visible alignment from drifting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent / "GNM-main"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT))

from gnm.shape import gnm_pytorch, gnm_numpy  # pylint: disable=wrong-import-position
from nvdiffrast_render import NVDiffRenderer  # pylint: disable=wrong-import-position
from nvdiffrast_render.camera import project_world_to_pixels  # pylint: disable=wrong-import-position

from fit_gnm_teeth_single import (  # pylint: disable=wrong-import-position
    apply_arch_rigid,
    load_gt_masks,
    mouth_roi,
    overlay_pred_gt,
    parse_k,
    render_semantic_masks,
    teeth_faces_from_groups,
)
from vis_lower_rigid6 import col_edge, draw_col_polyline  # pylint: disable=wrong-import-position


# FDI antagonists (upper → lower). Used only for reporting, not as an attractor.
OPP_FDI = {
    11: 41, 12: 42, 13: 43, 14: 44, 15: 45, 16: 46, 17: 47, 18: 48,
    21: 31, 22: 32, 23: 33, 24: 34, 25: 35, 26: 36, 27: 37, 28: 38,
}


def vertex_normals(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
  """Area-weighted vertex normals from triangle winding."""
  fn = torch.cross(
      verts[faces[:, 1]] - verts[faces[:, 0]],
      verts[faces[:, 2]] - verts[faces[:, 0]],
      dim=-1,
  )
  vn = torch.zeros_like(verts)
  vn.index_add_(0, faces[:, 0], fn)
  vn.index_add_(0, faces[:, 1], fn)
  vn.index_add_(0, faces[:, 2], fn)
  return F.normalize(vn, dim=-1, eps=1e-8)


def nearest_signed_sdf(
    src: torch.Tensor,
    dst: torch.Tensor,
    dst_n: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Vertex-NN SDF of `src` points against `dst` samples with outward normals.

  sdf > 0 is outside. Hard NN assignment; distance to the chosen sample is
  differentiable.
  """
  d2 = ((src[:, None, :] - dst[None, :, :]) ** 2).sum(-1)
  nn = d2.argmin(dim=1)
  rel = src - dst[nn]
  dist = rel.norm(dim=-1)
  sdf = (rel * dst_n[nn]).sum(-1)
  return sdf, dist


def arch_collision(
    verts: torch.Tensor,
    crown_up: torch.Tensor,
    crown_lo: torch.Tensor,
    n_up: torch.Tensor,
    n_lo: torch.Tensor,
    *,
    margin: float,
    gate: float,
) -> dict:
  """Bidirectional crown collision. No attraction — only penalize penetration.

  `gate` ignores far NN hits whose vertex normals would otherwise look inside
  (concave fossae). Units of verts / margin / gate are metres.
  """
  sdf_l, dist_l = nearest_signed_sdf(verts[crown_lo], verts[crown_up], n_up[crown_up])
  sdf_u, dist_u = nearest_signed_sdf(verts[crown_up], verts[crown_lo], n_lo[crown_lo])
  close_l = dist_l < gate
  close_u = dist_u < gate
  pen_l = F.relu(margin - sdf_l) * close_l.float()
  pen_u = F.relu(margin - sdf_u) * close_u.float()
  # mm^2, scaled by ~n opposing pairs so magnitude matches mask losses.
  loss = (pen_l.pow(2).sum() + pen_u.pow(2).sum()) * (1000.0 ** 2) / 16.0
  hit_l = close_l & (sdf_l < 0)
  hit_u = close_u & (sdf_u < 0)
  return {
      "loss": loss,
      "sdf_l": sdf_l,
      "sdf_u": sdf_u,
      "dist_l": dist_l,
      "dist_u": dist_u,
      "pen_l": pen_l,
      "pen_u": pen_u,
      "hit_l": hit_l,
      "hit_u": hit_u,
      "n_pen_l": int(hit_l.sum().item()),
      "n_pen_u": int(hit_u.sum().item()),
      "min_sdf": float(torch.minimum(sdf_l.min(), sdf_u.min()).detach()),
      "min_dist": float(torch.minimum(dist_l.min(), dist_u.min()).detach()),
      "max_pen": float(torch.maximum(pen_l.max(), pen_u.max()).detach()),
  }


def fdi_crown_indices(fdi_npz, is_upper: bool) -> tuple[list[int], np.ndarray]:
  all_fdi = sorted(int(k.split("_")[1]) for k in fdi_npz.files if k.startswith("fdi_"))
  arch = [f for f in all_fdi if (f < 30) == is_upper]
  idx = np.unique(np.concatenate([np.asarray(fdi_npz[f"fdi_{f}"], np.int64) for f in arch]))
  return arch, idx


def pair_min_dist(verts: torch.Tensor, fdi_npz, per_t: dict[int, torch.Tensor]) -> dict[str, float]:
  out = {}
  for u, l in OPP_FDI.items():
    if u not in per_t or l not in per_t:
      continue
    d = torch.cdist(verts[per_t[l]], verts[per_t[u]]).min()
    out[f"{u}-{l}"] = float(d.detach())
  return out


def draw_hits(
    img: np.ndarray,
    xy: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    y0: int,
    x0: int,
) -> None:
  for (x, y), hit in zip(xy, mask):
    if not hit:
      continue
    cv2.circle(img, (int(round(x - x0)), int(round(y - y0))), 3, color, -1, cv2.LINE_AA)


def build_argparser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument(
      "--params", type=Path,
      default=Path("gnm_fit_out/fit_frame_0027_mp68_official_eyes/frame_000000_params.npz"),
  )
  p.add_argument(
      "--upper_params", type=Path,
      default=Path("gnm_fit_out/vis_upper_rigid6_clip6/upper_rigid6_params.npz"),
  )
  p.add_argument(
      "--lower_params", type=Path,
      default=Path("gnm_fit_out/vis_lower_rigid6_coco/lower_rigid6_params.npz"),
  )
  p.add_argument(
      "--both_params", type=Path, default=None,
      help="If set, load aa/t for both arches from this npz instead.",
  )
  p.add_argument(
      "--masks", type=Path,
      default=Path("gnm_fit_out/coco_smooth_as_mask/masks_fdi_coco_smooth.npz"),
  )
  p.add_argument(
      "--fdi_verts", type=Path,
      default=Path("sam3_teeth_out/gnm_bind/gnm_fdi_vertex_indices.npz"),
  )
  p.add_argument("--image", type=Path, default=Path("sam3_teeth_out/frame_0027.png"))
  p.add_argument(
      "--out_dir", type=Path,
      default=Path("gnm_fit_out/vis_both_rigid6_nocollide"),
  )
  p.add_argument("--iters", type=int, default=250)
  p.add_argument("--lr", type=float, default=0.008)
  p.add_argument("--margin_mm", type=float, default=0.25, help="Keep at least this crown gap.")
  p.add_argument("--gate_mm", type=float, default=3.0, help="Ignore NN farther than this.")
  p.add_argument("--w_coll", type=float, default=80.0)
  p.add_argument("--w_over", type=float, default=250.0)
  p.add_argument("--w_over_col", type=float, default=180.0)
  p.add_argument("--w_gap", type=float, default=90.0)
  p.add_argument("--w_fp", type=float, default=160.0)
  p.add_argument("--w_fill", type=float, default=35.0)
  p.add_argument("--w_fill_top", type=float, default=80.0)
  p.add_argument("--top_band", type=int, default=14)
  p.add_argument("--w_cent", type=float, default=15.0)
  p.add_argument("--w_reg", type=float, default=1e-3)
  p.add_argument("--freeze_upper", action="store_true", default=True)
  p.add_argument("--no_freeze_upper", action="store_false", dest="freeze_upper")
  p.add_argument("--device", default="cuda")
  return p


def main() -> int:
  args = build_argparser().parse_args()
  device = torch.device(args.device if torch.cuda.is_available() else "cpu")
  if device.type != "cuda":
    raise SystemExit("CUDA required")
  args.out_dir.mkdir(parents=True, exist_ok=True)
  margin = float(args.margin_mm) * 1e-3
  gate = float(args.gate_mm) * 1e-3

  params = np.load(args.params)
  image = cv2.cvtColor(cv2.imread(str(args.image), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
  width, height = map(int, params["image_size"])

  gnm = gnm_pytorch.GNM.from_local(
      version=gnm_pytorch.GNMMajorVersion.V3, variant=gnm_pytorch.GNMVariant.HEAD,
  )
  gnm_np = gnm_numpy.GNM.from_local(
      version=gnm_numpy.GNMMajorVersion.V3, variant=gnm_numpy.GNMVariant.HEAD,
  )

  id_head = torch.as_tensor(params["id_head"], dtype=torch.float32)
  id_eyes = torch.as_tensor(params["id_eyes"], dtype=torch.float32)
  expression = torch.as_tensor(params["expression"], dtype=torch.float32)
  rotations = torch.as_tensor(params["rotations"], dtype=torch.float32)
  translation = torch.as_tensor(params["translation"], dtype=torch.float32)
  rt = torch.as_tensor(params["RT"], dtype=torch.float32, device=device)
  k = parse_k(params["K"]).to(device)

  src = np.load(args.both_params) if args.both_params is not None else None
  up_p = src if src is not None else np.load(args.upper_params)
  lo_p = src if src is not None else np.load(args.lower_params)

  n_verts = int(np.asarray(gnm_np.skinning_weights).shape[-1])
  fdi_npz = np.load(args.fdi_verts)
  up_all_fdi, crown_up_np = fdi_crown_indices(fdi_npz, True)
  lo_all_fdi, crown_lo_np = fdi_crown_indices(fdi_npz, False)
  crown_up = torch.as_tensor(crown_up_np, dtype=torch.long)
  crown_lo = torch.as_tensor(crown_lo_np, dtype=torch.long)
  per_t = {}
  for f in up_all_fdi + lo_all_fdi:
    per_t[f] = torch.as_tensor(np.asarray(fdi_npz[f"fdi_{f}"], np.int64), dtype=torch.long)

  masks_meta = np.load(args.masks)
  visible = [int(x) for x in masks_meta["fdi"]]
  vis_up = [f for f in visible if f < 30]
  vis_lo = [f for f in visible if f >= 30]
  gt_up = load_gt_masks(args.masks, vis_up, height, width).to(device)
  gt_lo = load_gt_masks(args.masks, vis_lo, height, width).to(device)
  gt_all = torch.cat([gt_up, gt_lo], 0)
  y0, y1, x0, x1 = mouth_roi(gt_all, pad=40)

  up_idx = torch.as_tensor(
      np.asarray(gnm_np.vertex_group_indices("upper_teeth_and_gums"), dtype=np.int64),
      dtype=torch.long,
  )
  lo_idx = torch.as_tensor(
      np.asarray(gnm_np.vertex_group_indices("lower_teeth_and_gums"), dtype=np.int64),
      dtype=torch.long,
  )
  faces_up = torch.as_tensor(
      teeth_faces_from_groups(gnm_np, ["upper_teeth_and_gums"]),
      dtype=torch.int32, device=device,
  )
  faces_lo = torch.as_tensor(
      teeth_faces_from_groups(gnm_np, ["lower_teeth_and_gums"]),
      dtype=torch.int32, device=device,
  )
  faces_both = torch.as_tensor(
      teeth_faces_from_groups(
          gnm_np, ["upper_teeth_and_gums", "lower_teeth_and_gums"],
      ),
      dtype=torch.int32, device=device,
  )
  faces_up_n = faces_up.detach().cpu().long()
  faces_lo_n = faces_lo.detach().cpu().long()

  identity0 = torch.cat([id_head, id_eyes, torch.zeros(80)], dim=0)
  with torch.no_grad():
    base0 = gnm(
        identity0[None], expression[None], rotations[None], translation[None],
    )[0].detach()
  pivot_up = base0[up_idx].mean(0).detach()
  pivot_lo = base0[lo_idx].mean(0).detach()

  aa_up = torch.tensor(np.asarray(up_p["aa_up"], dtype=np.float32), requires_grad=not args.freeze_upper)
  t_up = torch.tensor(np.asarray(up_p["t_up"], dtype=np.float32), requires_grad=not args.freeze_upper)
  aa_lo = torch.tensor(np.asarray(lo_p["aa_low"], dtype=np.float32), requires_grad=True)
  t_lo = torch.tensor(np.asarray(lo_p["t_low"], dtype=np.float32), requires_grad=True)

  def apply_m(verts: torch.Tensor) -> torch.Tensor:
    v = apply_arch_rigid(verts, up_idx, aa_up, t_up, pivot_up)
    return apply_arch_rigid(v, lo_idx, aa_lo, t_lo, pivot_lo)

  ys = torch.arange(height, device=device, dtype=torch.float32)[:, None].expand(height, width)
  xs = torch.arange(width, device=device, dtype=torch.float32)[None, :].expand(height, width)
  gt_lo_u = gt_lo.sum(0).clamp(0, 1)
  gt_up_u = gt_up.sum(0).clamp(0, 1)

  v_lo = torch.zeros(n_verts, len(vis_lo), device=device)
  for c, f in enumerate(vis_lo):
    v_lo[per_t[f].to(device), c] = 1.0
  v_up = torch.zeros(n_verts, len(vis_up), device=device)
  for c, f in enumerate(vis_up):
    v_up[per_t[f].to(device), c] = 1.0
  v_edge_lo = torch.zeros(n_verts, 1, device=device)
  v_edge_up = torch.zeros(n_verts, 1, device=device)
  v_edge_lo[crown_lo.to(device), 0] = 1.0
  v_edge_up[crown_up.to(device), 0] = 1.0

  gt_cent_lo = []
  for i, f in enumerate(vis_lo):
    m = gt_lo[i]
    mass = m.sum().clamp(min=1e-6)
    gt_cent_lo.append(torch.stack([(m * xs).sum() / mass, (m * ys).sum() / mass]))
  gt_cent_lo = torch.stack(gt_cent_lo, 0)

  renderer = NVDiffRenderer(
      use_opengl=False, lighting_type="constant", lighting_space="camera",
      flip_y_raster=False,
  ).to(device)
  rend_mesh = NVDiffRenderer(
      use_opengl=False, lighting_type="front", lighting_space="camera",
      flip_y_raster=False,
  ).to(device)

  def collision_pack(verts_m: torch.Tensor) -> dict:
    n_u = vertex_normals(verts_m, faces_up_n)
    n_l = vertex_normals(verts_m, faces_lo_n)
    return arch_collision(
        verts_m, crown_up, crown_lo, n_u, n_l, margin=margin, gate=gate,
    )

  def lower_edge_losses(verts_m: torch.Tensor) -> dict:
    pred = render_semantic_masks(
        renderer, verts_m.to(device), faces_lo, v_lo, rt, k, (height, width),
    )
    pr = pred.sum(0).clamp(0, 1)
    edge_full = render_semantic_masks(
        renderer, verts_m.to(device), faces_lo, v_edge_lo, rt, k, (height, width),
    )[0].clamp(0, 1)
    gy, gok = col_edge(gt_lo_u, ys, bottom=False)
    py, pok = col_edge(edge_full, ys, bottom=False)
    pok = pok & gok
    both = gok & pok
    l_gap = pr.new_zeros(())
    l_over_col = pr.new_zeros(())
    if bool(both.any()):
      l_gap = F.relu(py[both] - gy[both]).pow(2).mean()
      l_over_col = F.relu(gy[both] - py[both]).pow(2).mean()
    past = (ys < gy[None, :]) & gok[None, :]
    band = gok[None, :] & (ys >= gy[None, :]) & (ys < gy[None, :] + float(args.top_band))
    l_over_px = (pr * past.float()).sum() / gt_lo_u.sum().clamp(min=1.0)
    l_fp = (pr * (gt_lo_u < 0.12).float()).sum() / gt_lo_u.sum().clamp(min=1.0)
    miss = (gt_lo_u - pr).clamp(min=0.0) * (gt_lo_u > 0.15).float()
    l_fill = miss.sum() / gt_lo_u.sum().clamp(min=1.0)
    l_fill_top = (
        ((gt_lo_u - pr).clamp(min=0.0) * band.float()).sum()
        / band.float().sum().clamp(min=1.0)
    )
    l_cent = pr.new_zeros(())
    for i, f in enumerate(vis_lo):
      xy = project_world_to_pixels(
          verts_m.to(device)[per_t[f].to(device)], rt, k, width, height, flip_y=True,
      )
      l_cent = l_cent + ((xy.mean(0) - gt_cent_lo[i]) ** 2).sum()
    l_cent = l_cent / max(len(vis_lo), 1)
    pred_e = py[gok].mean() if bool(gok.any()) else py.new_zeros(())
    gt_e = gy[gok].mean() if bool(gok.any()) else gy.new_zeros(())
    return {
        "pred": pred, "pr": pr, "gy": gy, "gok": gok, "py": py, "pok": pok,
        "l_gap": l_gap, "l_over_col": l_over_col, "l_over_px": l_over_px,
        "l_fp": l_fp, "l_fill": l_fill, "l_fill_top": l_fill_top, "l_cent": l_cent,
        "pred_e": pred_e, "gt_e": gt_e,
    }

  def dump_vis(stem: str, verts_m: torch.Tensor, tag: str, coll: dict) -> None:
    with torch.no_grad():
      pred_up = render_semantic_masks(
          renderer, verts_m.to(device), faces_up, v_up, rt, k, (height, width),
      )
      pred_lo = render_semantic_masks(
          renderer, verts_m.to(device), faces_lo, v_lo, rt, k, (height, width),
      )
      pred = torch.cat([pred_up, pred_lo], 0)
      ov = overlay_pred_gt(
          image, pred.detach().cpu().numpy(), gt_all.detach().cpu().numpy(),
      )
      edge_up = render_semantic_masks(
          renderer, verts_m.to(device), faces_up, v_edge_up, rt, k, (height, width),
      )[0]
      edge_lo = render_semantic_masks(
          renderer, verts_m.to(device), faces_lo, v_edge_lo, rt, k, (height, width),
      )[0]
      gy_u, gok_u = col_edge(gt_up_u, ys, bottom=True)
      py_u, pok_u = col_edge(edge_up, ys, bottom=True)
      pok_u = pok_u & gok_u
      gy_l, gok_l = col_edge(gt_lo_u, ys, bottom=False)
      py_l, pok_l = col_edge(edge_lo, ys, bottom=False)
      pok_l = pok_l & gok_l
      crop = ov[y0:y1, x0:x1].copy()
      edge = crop.copy()
      draw_col_polyline(edge, gy_u.cpu().numpy()[x0:x1], gok_u.cpu().numpy()[x0:x1], 0, (0, 220, 0), y0=y0)
      draw_col_polyline(edge, py_u.cpu().numpy()[x0:x1], pok_u.cpu().numpy()[x0:x1], 0, (220, 40, 40), y0=y0)
      draw_col_polyline(edge, gy_l.cpu().numpy()[x0:x1], gok_l.cpu().numpy()[x0:x1], 0, (0, 200, 200), y0=y0)
      draw_col_polyline(edge, py_l.cpu().numpy()[x0:x1], pok_l.cpu().numpy()[x0:x1], 0, (220, 40, 200), y0=y0)
      cv2.putText(crop, tag, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
      cv2.putText(edge, "GT up bot / low top", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 2)
      cv2.putText(edge, "pred up / low", (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 40, 40), 2)

      rgba = rend_mesh.render_rgba_vis(
          verts_m.to(device)[None], faces_both, rt[None], k[None], (height, width),
          background_color=[0, 0, 0],
      )["rgba"][0].detach().float().cpu().numpy()
      rgb = np.clip(rgba[..., :3] * 255, 0, 255).astype(np.uint8)
      a = rgba[..., 3]
      blend = np.clip(
          image.astype(np.float32) * (1 - 0.65 * a[..., None])
          + rgb.astype(np.float32) * (0.65 * a[..., None]),
          0, 255,
      ).astype(np.uint8)
      hit = blend[y0:y1, x0:x1].copy()
      xy_l = project_world_to_pixels(
          verts_m.to(device)[crown_lo.to(device)], rt, k, width, height, flip_y=True,
      ).cpu().numpy()
      xy_u = project_world_to_pixels(
          verts_m.to(device)[crown_up.to(device)], rt, k, width, height, flip_y=True,
      ).cpu().numpy()
      draw_hits(hit, xy_l, coll["hit_l"].cpu().numpy(), (0, 0, 255), y0, x0)
      draw_hits(hit, xy_u, coll["hit_u"].cpu().numpy(), (255, 80, 0), y0, x0)
      cv2.putText(
          hit,
          f"pen L{coll['n_pen_l']} U{coll['n_pen_u']}  minSDF {coll['min_sdf']*1000:.2f}mm",
          (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2,
      )

    cv2.imwrite(str(args.out_dir / f"{stem}_overlay.jpg"), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(args.out_dir / f"{stem}_overlay_crop.jpg"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(args.out_dir / f"{stem}_edge_crop.jpg"), cv2.cvtColor(edge, cv2.COLOR_RGB2BGR))
    cv2.imwrite(
        str(args.out_dir / f"{stem}_mesh_on_photo_crop.jpg"),
        cv2.cvtColor(blend[y0:y1, x0:x1], cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        str(args.out_dir / f"{stem}_penetration_crop.jpg"),
        cv2.cvtColor(hit, cv2.COLOR_RGB2BGR),
    )

  def report(tag: str, verts_m: torch.Tensor, coll: dict) -> None:
    pairs = pair_min_dist(verts_m, fdi_npz, per_t)
    close_pairs = {k: v for k, v in pairs.items() if v < 0.005}
    print(
        f"{tag}  n_pen L/U={coll['n_pen_l']}/{coll['n_pen_u']}  "
        f"minSDF={coll['min_sdf']*1000:.2f}mm  minDist={coll['min_dist']*1000:.2f}mm  "
        f"maxPen={coll['max_pen']*1000:.2f}mm"
    )
    if close_pairs:
      shown = ", ".join(f"{k}:{v*1000:.2f}" for k, v in close_pairs.items())
      print(f"  pairs <5mm: {shown}")
    else:
      dmin_pair = min(pairs.values()) if pairs else float("nan")
      print(f"  antagonist min={dmin_pair*1000:.2f}mm (overjet; not used as attractor)")

  print(
      f"BOTH + collision  freeze_upper={int(args.freeze_upper)}  "
      f"margin={args.margin_mm}mm gate={args.gate_mm}mm  "
      f"crown {len(crown_up)}/{len(crown_lo)}  vis up={vis_up} lo={vis_lo}"
  )
  verts0 = apply_m(base0).detach()
  coll0 = collision_pack(verts0)
  dump_vis("00_init", verts0, "INIT independent 6-DoF", coll0)
  report("INIT", verts0, coll0)

  opt_params = [aa_lo, t_lo]
  if not args.freeze_upper:
    opt_params += [aa_up, t_up]
  opt = torch.optim.Adam(opt_params, lr=args.lr)
  best = {
      "score": float("inf"),
      "aa_up": aa_up.detach().clone(),
      "t_up": t_up.detach().clone(),
      "aa_lo": aa_lo.detach().clone(),
      "t_lo": t_lo.detach().clone(),
  }

  for it in range(args.iters):
    opt.zero_grad()
    verts = apply_m(base0)
    sm = lower_edge_losses(verts)
    coll = collision_pack(verts)
    l_reg = aa_lo.pow(2).sum() + t_lo.pow(2).sum()
    if not args.freeze_upper:
      l_reg = l_reg + aa_up.pow(2).sum() + t_up.pow(2).sum()
    loss = (
        args.w_over * sm["l_over_px"]
        + args.w_over_col * sm["l_over_col"]
        + args.w_gap * sm["l_gap"]
        + args.w_fp * sm["l_fp"]
        + args.w_fill * sm["l_fill"]
        + args.w_fill_top * sm["l_fill_top"]
        + args.w_cent * sm["l_cent"]
        + args.w_coll * coll["loss"]
        + args.w_reg * l_reg
    )
    loss.backward()
    opt.step()
    with torch.no_grad():
      aa_lo.clamp_(-0.55, 0.55)
      t_lo.clamp_(-0.035, 0.035)
      if not args.freeze_upper:
        aa_up.clamp_(-0.55, 0.55)
        t_up.clamp_(-0.035, 0.035)
    score = (
        20.0 * float(coll["loss"].detach())
        + 8.0 * float(sm["l_over_px"].detach())
        + 4.0 * float(sm["l_over_col"].detach())
        + 3.0 * float(sm["l_fp"].detach())
        + 1.2 * float(sm["l_gap"].detach())
    )
    if np.isfinite(score) and score < best["score"]:
      best = {
          "score": score,
          "aa_up": aa_up.detach().clone(),
          "t_up": t_up.detach().clone(),
          "aa_lo": aa_lo.detach().clone(),
          "t_lo": t_lo.detach().clone(),
      }
    if it % 25 == 0 or it == args.iters - 1:
      print(
          f"  [{it:03d}] loss={float(loss.detach()):.2f} "
          f"coll={float(coll['loss'].detach()):.3f} "
          f"nPen={coll['n_pen_l']+coll['n_pen_u']} "
          f"minSDF={coll['min_sdf']*1000:.2f} "
          f"over={float(sm['l_over_px'].detach()):.3f} "
          f"gap={float(sm['l_gap'].detach()):.2f} "
          f"top {float(sm['pred_e'].detach()):.1f}->{float(sm['gt_e'].detach()):.1f}"
      )

  with torch.no_grad():
    aa_up.copy_(best["aa_up"])
    t_up.copy_(best["t_up"])
    aa_lo.copy_(best["aa_lo"])
    t_lo.copy_(best["t_lo"])
  verts_f = apply_m(base0).detach()
  coll_f = collision_pack(verts_f)
  dump_vis("01_nocollide", verts_f, "BOTH + no-penetrate", coll_f)
  report("FINAL", verts_f, coll_f)
  print(f"  aa_up={aa_up.detach().tolist()}")
  print(f"  t_up={t_up.detach().tolist()}")
  print(f"  aa_low={aa_lo.detach().tolist()}")
  print(f"  t_low={t_lo.detach().tolist()}")

  init_ov = cv2.imread(str(args.out_dir / "00_init_overlay_crop.jpg"))
  aln_ov = cv2.imread(str(args.out_dir / "01_nocollide_overlay_crop.jpg"))
  init_ed = cv2.imread(str(args.out_dir / "00_init_edge_crop.jpg"))
  aln_ed = cv2.imread(str(args.out_dir / "01_nocollide_edge_crop.jpg"))
  init_m = cv2.imread(str(args.out_dir / "00_init_mesh_on_photo_crop.jpg"))
  aln_m = cv2.imread(str(args.out_dir / "01_nocollide_mesh_on_photo_crop.jpg"))
  init_p = cv2.imread(str(args.out_dir / "00_init_penetration_crop.jpg"))
  aln_p = cv2.imread(str(args.out_dir / "01_nocollide_penetration_crop.jpg"))
  if init_ov is not None and aln_ov is not None:
    pair = np.concatenate([init_ov, aln_ov], 1)
    cv2.putText(pair, "init 6-DoF", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(pair, "no-penetrate", (init_ov.shape[1] + 10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imwrite(str(args.out_dir / "02_init_vs_nocollide_overlay.jpg"), pair)
  if init_ed is not None and aln_ed is not None:
    cv2.imwrite(
        str(args.out_dir / "03_init_vs_nocollide_edge.jpg"),
        np.concatenate([init_ed, aln_ed], 1),
    )
  if init_m is not None and aln_m is not None:
    cv2.imwrite(
        str(args.out_dir / "04_init_vs_nocollide_mesh.jpg"),
        np.concatenate([init_m, aln_m], 1),
    )
  if init_p is not None and aln_p is not None:
    cv2.imwrite(
        str(args.out_dir / "05_init_vs_nocollide_penetration.jpg"),
        np.concatenate([init_p, aln_p], 1),
    )

  np.savez(
      args.out_dir / "both_rigid6_params.npz",
      aa_up=aa_up.detach().cpu().numpy(),
      t_up=t_up.detach().cpu().numpy(),
      aa_low=aa_lo.detach().cpu().numpy(),
      t_low=t_lo.detach().cpu().numpy(),
      id_teeth=np.zeros(80, dtype=np.float32),
  )
  summary = {
      "mode": "both_rigid6_freeze_upper_lower_nocollide",
      "freeze_upper": bool(args.freeze_upper),
      "margin_mm": args.margin_mm,
      "gate_mm": args.gate_mm,
      "w_coll": args.w_coll,
      "aa_up": aa_up.detach().cpu().tolist(),
      "t_up": t_up.detach().cpu().tolist(),
      "aa_low": aa_lo.detach().cpu().tolist(),
      "t_low": t_lo.detach().cpu().tolist(),
      "init_n_pen_l": coll0["n_pen_l"],
      "init_n_pen_u": coll0["n_pen_u"],
      "init_min_sdf_mm": coll0["min_sdf"] * 1000,
      "init_min_dist_mm": coll0["min_dist"] * 1000,
      "final_n_pen_l": coll_f["n_pen_l"],
      "final_n_pen_u": coll_f["n_pen_u"],
      "final_min_sdf_mm": coll_f["min_sdf"] * 1000,
      "final_min_dist_mm": coll_f["min_dist"] * 1000,
  }
  (args.out_dir / "summary.json").write_text(
      json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
  )
  print("done →", args.out_dir)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
