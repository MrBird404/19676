#!/usr/bin/env python3
"""Full teeth+gums deformation + crown-only visible losses.

- Rigid 6-DoF on full arch groups (upper/lower_teeth_and_gums).
- Upper: crown soft-bottom → GT upper-mask bottom.
- Lower: crown soft-top → GT lower-mask top.
- Shape id_teeth on full mesh; supervision only on visible crown masks
  (lip-occluded pred detached). Gum verts locked to post-rigid pose so the
  arch structure does not fly apart.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
REPO = next(p for p in [ROOT, *ROOT.parents] if (p / "GNM-main").is_dir()) / "GNM-main"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT))

from gnm.shape import gnm_pytorch, gnm_numpy  # pylint: disable=wrong-import-position
from nvdiffrast_render import NVDiffRenderer  # pylint: disable=wrong-import-position
from nvdiffrast_render.camera import project_world_to_pixels  # pylint: disable=wrong-import-position

from fit_gnm_teeth_single import (  # pylint: disable=wrong-import-position
    apply_arch_rigid,
    colorize_masks,
    load_fdi_vertex_map,
    load_gt_masks,
    mouth_roi,
    overlay_pred_gt,
    parse_k,
    render_semantic_and_lips,
    render_semantic_masks,
    soft_bottom_y_stable,
    soft_centroid,
    soft_edge_y_from_points,
    soft_top_y,
    teeth_faces_from_groups,
)


def soft_edge_y_per_col(
    mask: torch.Tensor,
    ys: torch.Tensor,
    *,
    bottom: bool,
    tau: float = 0.1,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Per-column soft bottom/top y. Returns (y_col[W], col_ok[W] bool)."""
  m = mask.clamp(min=0)
  active = m > 1e-4
  neg = torch.finfo(ys.dtype).min / 4
  signed = tau * ys if bottom else -tau * ys
  logits = torch.where(active, signed, torch.full_like(ys, neg))
  logits = logits - logits.amax(dim=0, keepdim=True).detach()
  w = torch.where(active, torch.exp(logits) * m, torch.zeros_like(m))
  col_mass = w.sum(dim=0).clamp(min=eps)
  y_col = (w * ys).sum(dim=0) / col_mass
  col_ok = m.sum(dim=0) > 1e-3
  return y_col, col_ok


def build_argparser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument(
      "--params", type=Path,
      default=Path("gnm_fit_out/fit_frame_0027_hybrid468/frame_000000_params.npz"),
  )
  p.add_argument("--masks", type=Path, default=Path("sam3_teeth_out/gnm_bind/masks_fdi.npz"))
  p.add_argument(
      "--fdi_verts", type=Path,
      default=Path("sam3_teeth_out/gnm_bind/gnm_fdi_vertex_indices.npz"),
  )
  p.add_argument("--image", type=Path, default=Path("sam3_teeth_out/frame_0027.png"))
  p.add_argument(
      "--out_dir", type=Path,
      default=Path("gnm_fit_out/fit_teeth_frame_0027_fullstruct_crownloss"),
  )
  p.add_argument("--rigid_iters", type=int, default=280)
  p.add_argument("--shape_iters", type=int, default=300)
  p.add_argument("--lr_rigid", type=float, default=0.008)
  p.add_argument("--lr_shape", type=float, default=0.015)
  p.add_argument("--w_edge", type=float, default=100.0)
  p.add_argument("--w_centroid", type=float, default=40.0)
  p.add_argument(
      "--w_contact", type=float, default=80.0,
      help="Penalize adjacent FDI crowns spreading farther than the rigid template.",
  )
  p.add_argument(
      "--freeze_occluded",
      action=argparse.BooleanOptionalAction,
      default=True,
      help="Hard-freeze crown verts whose projection is outside the GT mask "
           "(only visible mask region is deformed).",
  )
  p.add_argument(
      "--visible_centroid",
      action=argparse.BooleanOptionalAction,
      default=True,
      help="Rigid/centroid: average only verts that land in that tooth's GT mask.",
  )
  p.add_argument("--w_sem", type=float, default=25.0)
  p.add_argument(
      "--w_fp", type=float, default=40.0,
      help="Legacy FP weight inside sem_losses (rigid stage).",
  )
  p.add_argument(
      "--w_out", type=float, default=500.0,
      help="Shape: huge penalty for pred outside GT arch mask (non-lip).",
  )
  p.add_argument(
      "--w_bite", type=float, default=300.0,
      help="Shape: penalty for pixels past upper-bottom / lower-top bite edges.",
  )
  p.add_argument(
      "--w_edge_over", type=float, default=200.0,
      help="Shape: one-sided soft-edge overshoot (upper bottom / lower top).",
  )
  p.add_argument("--w_fill", type=float, default=40.0, help="Per-tooth underfill inside GT mask.")
  p.add_argument(
      "--w_fill_low", type=float, default=1.15,
      help="Mild extra multiplier on lower-FDI underfill (do not overpower out/bite).",
  )
  p.add_argument(
      "--w_col_edge", type=float, default=80.0,
      help="Per-column one-sided bite edge (same-x: no overshoot).",
  )
  p.add_argument(
      "--w_col_low", type=float, default=1.0,
      help="Scale lower-arch col/bite terms (keep 1.0; <1 lets lower overflow).",
  )
  p.add_argument(
      "--w_fp_low", type=float, default=60.0,
      help="Penalize lower pred outside lower GT union (stop lower overflow).",
  )
  p.add_argument(
      "--w_fp_up", type=float, default=200.0,
      help="Penalize upper pred outside upper GT / past bottom edge.",
  )
  p.add_argument(
      "--w_fp_tooth", type=float, default=100.0,
      help="Per-FDI pred outside that tooth GT (fill without overflow).",
  )
  p.add_argument(
      "--w_col_up", type=float, default=4.0,
      help="Extra scale on upper per-column / per-tooth bottom overshoot.",
  )
  p.add_argument(
      "--w_tooth_reg", type=float, default=20.0,
      help="L2 on per-tooth translations (keep local deltas small).",
  )
  p.add_argument(
      "--w_adj", type=float, default=50.0,
      help="Smooth adjacent FDI tooth_t (stops single teeth floating).",
  )
  p.add_argument(
      "--w_float", type=float, default=120.0,
      help="Extra L2 on float-prone FDI (34/35/44/45).",
  )
  p.add_argument(
      "--w_lock34", type=float, default=100.0,
      help="Pull FDI34 (and --lock_fdi) crown verts toward post-rigid pose.",
  )
  p.add_argument(
      "--lock_fdi", type=str, default="12,23,34",
      help="Comma-separated FDI ids to soft-lock to rigid crown pose.",
  )
  p.add_argument(
      "--tooth_t_clamp", type=float, default=0.008,
      help="Abs clamp on per-FDI crown translation (model units).",
  )
  p.add_argument(
      "--tooth_t_clamp_float", type=float, default=0.0025,
      help="Tighter clamp for float-prone FDI (34/35/44/45).",
  )
  p.add_argument(
      "--no_per_tooth", action="store_true",
      help="Disable per-tooth translations (id_teeth only).",
  )
  p.add_argument(
      "--pertooth_iters", type=int, default=180,
      help="Stage2B per-tooth micro-translation iterations.",
  )
  p.add_argument("--w_semxy", type=float, default=1.0)
  p.add_argument("--w_prior", type=float, default=0.9)
  p.add_argument(
      "--w_shape_lock", type=float, default=80.0,
      help="Keep crown verts near the frozen rigid pose during shape (stops flying teeth).",
  )
  p.add_argument(
      "--snap_only", action="store_true",
      help="Stage2B: centroid snap only, skip per-tooth Adam.",
  )
  p.add_argument("--w_rigid_reg", type=float, default=1e-3)
  p.add_argument(
      "--w_gum_lock", type=float, default=40.0,
      help="Soft gum lock weight (ignored if --hard_gum_freeze).",
  )
  p.add_argument(
      "--hard_gum_freeze",
      action="store_true",
      default=True,
      help="After rigid: hard-replace gum verts with gum_ref each step (no gum drift).",
  )
  p.add_argument("--soft_gum", action="store_false", dest="hard_gum_freeze")
  p.add_argument(
      "--init_rigid", type=Path, default=None,
      help="Load aa/t from rigid_only_params.npz (or fullstruct npz).",
  )
  p.add_argument(
      "--init_teeth", type=Path, default=None,
      help="Warm-start id_teeth (+ optional tooth_t) from fullstruct_crownloss_params.npz.",
  )
  p.add_argument(
      "--skip_rigid", action="store_true",
      help="Skip Stage1; freeze loaded/zero rigid and go straight to shape.",
  )
  p.add_argument(
      "--id_teeth_clamp", type=float, default=1.5,
      help="Abs clamp on id_teeth after each step (weak shape: try 5).",
  )
  p.add_argument("--roi_pad", type=int, default=64)
  p.add_argument("--render_scale", type=float, default=0.5)
  p.add_argument("--device", default="cuda")
  return p


def main() -> int:
  args = build_argparser().parse_args()
  device = torch.device(args.device if torch.cuda.is_available() else "cpu")
  if device.type != "cuda":
    raise SystemExit("CUDA required")
  args.out_dir.mkdir(parents=True, exist_ok=True)

  params = np.load(args.params)
  image = cv2.cvtColor(cv2.imread(str(args.image), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
  width, height = map(int, params["image_size"])
  if image.shape[1] != width or image.shape[0] != height:
    raise SystemExit(f"image {image.shape} vs params {(height, width)}")

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

  masks_meta = np.load(args.masks)
  visible = [int(x) for x in masks_meta["fdi"]]
  n_verts = int(np.asarray(gnm_np.skinning_weights).shape[-1])
  per_fdi, _, channels = load_fdi_vertex_map(args.fdi_verts, n_verts, visible)
  gt_full = load_gt_masks(args.masks, channels, height, width).to(device)
  n_c = len(channels)
  up_ch = [i for i, f in enumerate(channels) if f < 30]
  low_ch = [i for i, f in enumerate(channels) if f >= 30]
  per_fdi_t = {f: torch.as_tensor(idx, dtype=torch.long) for f, idx in per_fdi.items()}

  # Full arch groups for deformation / rigid.
  arch_up = np.asarray(gnm_np.vertex_group_indices("upper_teeth_and_gums"), dtype=np.int64)
  arch_low = np.asarray(gnm_np.vertex_group_indices("lower_teeth_and_gums"), dtype=np.int64)
  arch_up_t = torch.as_tensor(arch_up, dtype=torch.long)
  arch_low_t = torch.as_tensor(arch_low, dtype=torch.long)

  # Crowns for supervision; gums for structure lock.
  crown_up = np.unique(np.concatenate([per_fdi[f] for f in channels if f < 30]))
  crown_low = np.unique(np.concatenate([per_fdi[f] for f in channels if f >= 30]))
  crown_all = np.unique(np.concatenate([crown_up, crown_low]))
  crown_up_t = torch.as_tensor(crown_up, dtype=torch.long)
  crown_low_t = torch.as_tensor(crown_low, dtype=torch.long)
  gum = np.setdiff1d(np.unique(np.concatenate([arch_up, arch_low])), crown_all)
  gum_t = torch.as_tensor(gum, dtype=torch.long)

  faces_full_np = teeth_faces_from_groups(
      gnm_np, ["upper_teeth_and_gums", "lower_teeth_and_gums"],
  )
  faces_full = torch.as_tensor(faces_full_np, dtype=torch.int32, device=device)
  faces_lips = torch.as_tensor(
      teeth_faces_from_groups(gnm_np, ["upper_lip", "lower_lip"]),
      dtype=torch.int32, device=device,
  )
  lip_idx = torch.as_tensor(
      np.unique(np.concatenate([
          np.asarray(gnm_np.vertex_group_indices("upper_lip")),
          np.asarray(gnm_np.vertex_group_indices("lower_lip")),
      ])),
      dtype=torch.long,
  )

  def posed_verts(id_teeth: torch.Tensor) -> torch.Tensor:
    identity = torch.cat([id_head, id_eyes, id_teeth], dim=0)
    return gnm(
        identity[None], expression[None], rotations[None], translation[None],
    )[0]

  with torch.no_grad():
    base0 = posed_verts(torch.zeros(80)).detach()
  pivot_up = base0[arch_up_t].mean(0).detach()
  pivot_low = base0[arch_low_t].mean(0).detach()

  aa_up = torch.zeros(3, requires_grad=True)
  t_up = torch.zeros(3, requires_grad=True)
  aa_low = torch.zeros(3, requires_grad=True)
  t_low = torch.zeros(3, requires_grad=True)
  if args.init_rigid is not None:
    z = np.load(args.init_rigid)
    with torch.no_grad():
      aa_up.copy_(torch.as_tensor(z["aa_up"], dtype=torch.float32))
      t_up.copy_(torch.as_tensor(z["t_up"], dtype=torch.float32))
      aa_low.copy_(torch.as_tensor(z["aa_low"], dtype=torch.float32))
      t_low.copy_(torch.as_tensor(z["t_low"], dtype=torch.float32))
    print(f"loaded rigid from {args.init_rigid}")

  def apply_m(verts: torch.Tensor) -> torch.Tensor:
    v = apply_arch_rigid(verts, arch_up_t, aa_up, t_up, pivot_up)
    return apply_arch_rigid(v, arch_low_t, aa_low, t_low, pivot_low)

  tooth_t = torch.zeros(n_c, 3, requires_grad=False)
  float_fdi = {34, 35, 44, 45}
  float_idx = [i for i, f in enumerate(channels) if f in float_fdi]
  adj_chains = (
      [11, 12, 13],
      [21, 22, 23],
      [31, 32, 33, 34, 35, 36],
      [41, 42, 43, 44, 45, 46],
  )
  fdi_to_i = {int(f): i for i, f in enumerate(channels)}
  lock_fdis = []
  for tok in str(args.lock_fdi).split(","):
    tok = tok.strip()
    if tok:
      lock_fdis.append(int(tok))
  lock_vert_list = []
  for f in lock_fdis:
    if f in per_fdi_t:
      lock_vert_list.append(per_fdi_t[f])
  if lock_vert_list:
    lock_verts_t = torch.unique(torch.cat(lock_vert_list))
  else:
    lock_verts_t = torch.zeros(0, dtype=torch.long)
  occ_crown_t = torch.zeros(0, dtype=torch.long)
  crown_all_t = torch.as_tensor(crown_all, dtype=torch.long)

  def tooth_reg_terms(deltas: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    l_t = deltas.pow(2).mean()
    l_adj = deltas.new_zeros(())
    n_adj = 0
    for chain in adj_chains:
      idxs = [fdi_to_i[f] for f in chain if f in fdi_to_i]
      for a, b in zip(idxs, idxs[1:]):
        l_adj = l_adj + ((deltas[a] - deltas[b]) ** 2).sum()
        n_adj += 1
    if n_adj > 0:
      l_adj = l_adj / n_adj
    if float_idx:
      l_fl = deltas[float_idx].pow(2).mean()
    else:
      l_fl = deltas.new_zeros(())
    return l_t, l_adj, l_fl

  def clamp_tooth_t_(deltas: torch.Tensor) -> None:
    lo = -float(args.tooth_t_clamp)
    hi = float(args.tooth_t_clamp)
    deltas.clamp_(lo, hi)
    flo = -float(args.tooth_t_clamp_float)
    fhi = float(args.tooth_t_clamp_float)
    for i in float_idx:
      deltas[i].clamp_(flo, fhi)

  def apply_tooth_deltas(verts: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """Local crown translations per FDI channel (gums untouched here)."""
    v = verts.clone()
    for i, f in enumerate(channels):
      idx = per_fdi_t[f]
      v[idx] = v[idx] + deltas[i]
    return v

  def apply_shape(
      id_teeth: torch.Tensor,
      deltas: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """PCA deform + rigid + optional per-tooth t; freeze gums and occluded crowns."""
    v = apply_m(posed_verts(id_teeth))
    d = tooth_t if deltas is None else deltas
    v = apply_tooth_deltas(v, d)
    v_rigid = apply_m(base0)
    v = v.clone()
    if args.hard_gum_freeze:
      v[gum_t] = v_rigid[gum_t]
    if args.freeze_occluded and occ_crown_t.numel() > 0:
      v[occ_crown_t] = v_rigid[occ_crown_t]
    if lock_verts_t.numel() > 0:
      v[lock_verts_t] = v_rigid[lock_verts_t]
    return v

  def lock_fdi_loss(verts: torch.Tensor) -> torch.Tensor:
    if lock_verts_t.numel() == 0:
      return verts.new_zeros(())
    v_rigid = apply_m(base0)
    return ((verts[lock_verts_t] - v_rigid[lock_verts_t]) ** 2).mean()

  # Gum lock target updated after rigid.
  gum_ref = base0[gum_t].detach().clone()

  ys_full = torch.arange(height, device=device, dtype=torch.float32)[:, None].expand(height, width)
  xs_full = torch.arange(width, device=device, dtype=torch.float32)[None, :].expand(height, width)
  gt_cent = torch.stack([soft_centroid(gt_full[i], ys_full, xs_full) for i in range(n_c)], 0)
  y0, y1, x0, x1 = mouth_roi(gt_full, pad=args.roi_pad)
  gt_roi = gt_full[:, y0:y1, x0:x1]
  # Dilated GT union: outside this = protrusion (paper-style L_outside).
  gt_union_np = gt_full.sum(0).detach().cpu().numpy()
  dil = cv2.dilate(
      (gt_union_np > 0.15).astype(np.uint8), np.ones((7, 7), np.uint8), iterations=2,
  )
  allowed_full = torch.as_tensor(dil.astype(np.float32), device=device)
  # Generous keep-region: verts near annotated masks stay free (can fill or shrink).
  freeze_dil = cv2.dilate(
      (gt_union_np > 0.15).astype(np.uint8), np.ones((15, 15), np.uint8), iterations=2,
  )
  freeze_keep_full = torch.as_tensor(freeze_dil.astype(np.float32), device=device)

  print(
      f"FULL-STRUCT + CROWN-LOSS  ROI=({x0},{y0})-({x1},{y1})  "
      f"arch_up={len(arch_up)} arch_low={len(arch_low)}  "
      f"crown={len(crown_all)} gum={len(gum)}  faces={len(faces_full_np)}"
  )

  # Semantic attrs ONLY on crown verts (gums rasterize but contribute 0 to channels).
  v_attr = torch.zeros(n_verts, n_c, device=device)
  for c, f in enumerate(channels):
    v_attr[per_fdi_t[f].to(device), c] = 1.0
  v_lip = torch.zeros(n_verts, device=device)
  v_lip[lip_idx.to(device)] = 1.0

  renderer = NVDiffRenderer(
      use_opengl=False, lighting_type="constant", lighting_space="camera",
      flip_y_raster=False,
  ).to(device)

  def fdi_pred_centroid(verts_m: torch.Tensor, i: int) -> torch.Tensor:
    """2D centroid of FDI i. Optionally weight by GT mask at projected pixels."""
    v = verts_m.to(device)
    idx = per_fdi_t[channels[i]].to(device)
    xy = project_world_to_pixels(v[idx], rt, k, width, height, flip_y=True)
    if not args.visible_centroid:
      return xy.mean(0)
    xs = xy[:, 0].round().long().clamp(0, width - 1)
    ys = xy[:, 1].round().long().clamp(0, height - 1)
    w = gt_full[i][ys, xs]
    if float(w.sum()) > 1e-3:
      return (xy * w.unsqueeze(-1)).sum(0) / w.sum().clamp(min=1e-6)
    return xy.mean(0)

  def refresh_occluded(verts_m: torch.Tensor) -> None:
    """Freeze hidden/tiny crowns to the rigid template.

    Annotated teeth that land near the GT stay fully free (so overflow can
    shrink and holes can fill). Only lip-hidden verts far from GT, and FDI
    with almost no mask, are pinned. Per-pixel GT clipping is avoided so a
    tooth is never torn in half.
    """
    nonlocal occ_crown_t
    if not args.freeze_occluded:
      occ_crown_t = torch.zeros(0, dtype=torch.long)
      return
    v = verts_m.to(device)
    _, lip_full = render_semantic_and_lips(
        renderer, v, faces_lips, v_attr * 0, v_lip, rt, k, (height, width),
    )
    occ_parts = []
    n_tiny = 0
    n_hid = 0
    for i, f in enumerate(channels):
      idx = per_fdi_t[f].to(device)
      if float(gt_full[i].sum()) < 8.0:
        occ_parts.append(idx)
        n_tiny += 1
        continue
      xy = project_world_to_pixels(v[idx], rt, k, width, height, flip_y=True)
      xs = xy[:, 0].round().long().clamp(0, width - 1)
      ys = xy[:, 1].round().long().clamp(0, height - 1)
      near = freeze_keep_full[ys, xs] > 0.5
      on_lip = lip_full[ys, xs] > 0.45
      hid = (~near) & (on_lip > 0)
      if bool(hid.any()):
        occ_parts.append(idx[hid])
        n_hid += int(hid.sum())
    if occ_parts:
      occ_crown_t = torch.unique(torch.cat(occ_parts)).detach().cpu().long()
    else:
      occ_crown_t = torch.zeros(0, dtype=torch.long)
    print(
        f"  occluded freeze: tiny_fdi={n_tiny} hid_verts={n_hid} "
        f"occ={int(occ_crown_t.numel())} / crown={len(crown_all)}",
        flush=True,
    )

  def contact_loss(verts_m: torch.Tensor) -> torch.Tensor:
    """Keep adjacent crowns near a slightly tighter-than-template spacing."""
    v = verts_m
    v_ref = apply_m(base0).detach()
    loss = v.new_zeros(())
    n = 0
    for chain in adj_chains:
      fdis = [f for f in chain if f in fdi_to_i]
      for a, b in zip(fdis, fdis[1:]):
        ia, ib = per_fdi_t[a], per_fdi_t[b]
        d = torch.linalg.norm(v[ia].mean(0) - v[ib].mean(0))
        d0 = torch.linalg.norm(v_ref[ia].mean(0) - v_ref[ib].mean(0)).detach()
        target = 0.70 * d0
        loss = loss + (d - target).pow(2)
        n += 1
    return loss / max(n, 1)

  def shape_lock_loss(verts_m: torch.Tensor) -> torch.Tensor:
    v_r = apply_m(base0).detach()
    return ((verts_m[crown_all_t] - v_r[crown_all_t]) ** 2).mean()

  def fdi_full_centroid(verts_m: torch.Tensor, i: int) -> torch.Tensor:
    v = verts_m.to(device)
    idx = per_fdi_t[channels[i]].to(device)
    xy = project_world_to_pixels(v[idx], rt, k, width, height, flip_y=True)
    return xy.mean(0)

  def arch_centroid_loss(verts_m: torch.Tensor, which: str) -> torch.Tensor:
    v = verts_m
    loss = v.new_zeros(())
    n = 0
    for i, f in enumerate(channels):
      if which == "up" and f >= 30:
        continue
      if which == "low" and f < 30:
        continue
      pc = fdi_pred_centroid(v, i)
      loss = loss + ((pc - gt_cent[i]) ** 2).sum()
      n += 1
    return loss / max(n, 1)

  def mean_centroid_errs(verts_m: torch.Tensor) -> tuple[float, float, float]:
    up_e, lo_e = [], []
    for i, f in enumerate(channels):
      pc = fdi_pred_centroid(verts_m, i)
      e = float(torch.linalg.norm(pc - gt_cent[i]).detach())
      (up_e if f < 30 else lo_e).append(e)
    return float(np.mean(up_e + lo_e)), float(np.mean(up_e)), float(np.mean(lo_e))

  def edge_losses_from_verts(verts_m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    v = verts_m.to(device)
    xy_u = project_world_to_pixels(v[crown_up_t.to(device)], rt, k, width, height, flip_y=True)
    pred_ub = soft_edge_y_from_points(xy_u[:, 1], bottom=True)
    gt_ub = soft_bottom_y_stable(gt_full[up_ch].sum(0).clamp(0, 1), ys_full)
    l_ub = (pred_ub - gt_ub).pow(2)
    xy_l = project_world_to_pixels(v[crown_low_t.to(device)], rt, k, width, height, flip_y=True)
    pred_lt = soft_edge_y_from_points(xy_l[:, 1], bottom=False)
    gt_lt = soft_top_y(gt_full[low_ch].sum(0).clamp(0, 1), ys_full)
    l_lt = (pred_lt - gt_lt).pow(2)
    return l_ub, l_lt

  def gum_lock_loss(verts_m: torch.Tensor) -> torch.Tensor:
    # Target gums = current rigid pose (aa/t), independent of id_teeth.
    return ((verts_m[gum_t] - apply_m(base0)[gum_t]) ** 2).mean()

  def sem_losses(verts_m: torch.Tensor, *, full_res: bool = False):
    """Paper-style losses: bidirectional L_sem + L_semXY + outside + one-sided bite.

    L_sem uses union support (pred|gt) so both underfill and overfill get gradients.
    Lip-covered pred is detached (occlusion-aware).
    """
    v = verts_m.to(device)
    s = 1.0 if full_res else float(np.clip(args.render_scale, 0.25, 1.0))
    if abs(s - 1.0) < 1e-6:
      h_r, w_r = height, width
      k_r = k
      y0r, y1r, x0r, x1r = y0, y1, x0, x1
      gt = gt_roi
      gt_c_all = gt_cent
      allowed = allowed_full[y0:y1, x0:x1]
    else:
      h_r = int(round(height * s)) // 2 * 2
      w_r = int(round(width * s)) // 2 * 2
      sx, sy = w_r / float(width), h_r / float(height)
      fx, fy, cx, cy = k.unbind(0)
      k_r = torch.stack([fx * sx, fy * sy, cx * sx, cy * sy])
      y0r, y1r = int(round(y0 * sy)), int(round(y1 * sy))
      x0r, x1r = int(round(x0 * sx)), int(round(x1 * sx))
      if (y1r - y0r) % 2:
        y1r = min(h_r, y1r + 1)
      if (x1r - x0r) % 2:
        x1r = min(w_r, x1r + 1)
      gt = F.interpolate(
          gt_full[None], size=(h_r, w_r), mode="bilinear", align_corners=False,
      )[0][:, y0r:y1r, x0r:x1r]
      allowed = F.interpolate(
          allowed_full[None, None], size=(h_r, w_r), mode="bilinear",
          align_corners=False,
      )[0, 0, y0r:y1r, x0r:x1r]
      gt_c_all = gt_cent * torch.tensor([sx, sy], device=device)

    pred_full = render_semantic_masks(
        renderer, v, faces_full, v_attr, rt, k_r, (h_r, w_r),
    )
    _, lip_full = render_semantic_and_lips(
        renderer, v, faces_lips, v_attr * 0, v_lip, rt, k_r, (h_r, w_r),
    )
    pred = pred_full[:, y0r:y1r, x0r:x1r]
    lip = lip_full[y0r:y1r, x0r:x1r]
    pred_sem = torch.where(lip[None] > 0.4, pred.detach(), pred)
    nonlip = (lip < 0.4).float()

    # Bidirectional semantic L1 on union(pred, gt) ∩ non-lip.
    union = (((gt > 0.15) | (pred_sem > 0.15)).float() * nonlip[None])
    if float(union.sum()) < 1e-4:
      l_sem = pred.new_zeros(())
    else:
      l_sem = ((pred_sem - gt).abs() * union).sum() / union.sum().clamp(min=1e-4)

    # Per-tooth FILL: GT mass not covered (underfill only) — "mask 要填满".
    # Lower channels get w_fill_low so mandibular arch fills first.
    l_fill = pred.new_zeros(())
    l_fill_low = pred.new_zeros(())
    n_fill = 0
    n_fill_low = 0
    for i, f in enumerate(channels):
      g = gt[i]
      if float(g.sum()) < 5.0:
        continue
      miss = (g - pred_sem[i]).clamp(min=0.0) * (g > 0.15).float() * nonlip
      term = miss.sum() / g.sum().clamp(min=1.0)
      if f >= 30:
        l_fill = l_fill + float(args.w_fill_low) * term
        l_fill_low = l_fill_low + term
        n_fill_low += 1
      else:
        l_fill = l_fill + term
      n_fill += 1
    l_fill = l_fill / max(n_fill, 1)
    l_fill_low = l_fill_low / max(n_fill_low, 1)

    # Outside dilated GT union (arch-level protrusion).
    pr_arch = pred_sem.sum(0).clamp(0, 1)
    l_out = (pr_arch * (1.0 - allowed) * nonlip).sum() / gt.sum().clamp(min=1.0)

    hh, ww = pred.shape[-2:]
    ys = torch.arange(hh, device=device, dtype=torch.float32)[:, None].expand(hh, ww)
    xs = torch.arange(ww, device=device, dtype=torch.float32)[None, :].expand(hh, ww)

    gt_up = gt[up_ch].sum(0).clamp(0, 1) if up_ch else pred.new_zeros(hh, ww)
    pr_up = pred_sem[up_ch].sum(0).clamp(0, 1) if up_ch else pred.new_zeros(hh, ww)
    gt_lo = gt[low_ch].sum(0).clamp(0, 1) if low_ch else pred.new_zeros(hh, ww)
    pr_lo = pred_sem[low_ch].sum(0).clamp(0, 1) if low_ch else pred.new_zeros(hh, ww)
    # Lower FP: pred mass outside lower GT (stops lower arch climbing out of masks).
    if float(gt_lo.sum()) > 1.0:
      l_fp_low = (pr_lo * (gt_lo < 0.12).float() * nonlip).sum() / gt_lo.sum().clamp(min=1.0)
    else:
      l_fp_low = pred.new_zeros(())
    # Upper FP: outside upper GT union + past per-column bottom.
    if float(gt_up.sum()) > 1.0:
      l_fp_up = (pr_up * (gt_up < 0.12).float() * nonlip).sum() / gt_up.sum().clamp(min=1.0)
    else:
      l_fp_up = pred.new_zeros(())

    # Per-tooth FP + FILL already: each FDI must cover GT and not spill outside it.
    l_fp_tooth = pred.new_zeros(())
    n_fp_t = 0
    for i, f in enumerate(channels):
      g = gt[i]
      if float(g.sum()) < 5.0:
        continue
      p = pred_sem[i]
      spill = (p.clamp(0.0, 1.0) * (g < 0.10).float() * nonlip).sum() / g.sum().clamp(min=1.0)
      # Upper: extra penalty for mass past this tooth's soft bottom (12/23 classic).
      if f < 30:
        gy, gok = soft_edge_y_per_col(g, ys, bottom=True)
        past = (ys > gy[None, :]) & gok[None, :]
        spill = spill + 2.0 * (
            p * past.float() * nonlip
        ).sum() / g.sum().clamp(min=1.0)
      l_fp_tooth = l_fp_tooth + spill
      n_fp_t += 1
    l_fp_tooth = l_fp_tooth / max(n_fp_t, 1)

    has_up = float(gt_up.sum()) > 1.0
    has_lo = float(gt_lo.sum()) > 1.0
    gt_ub = soft_bottom_y_stable(gt_up, ys) if has_up else pred.new_zeros(())
    gt_lt = soft_top_y(gt_lo, ys) if has_lo else pred.new_zeros(())
    pr_ub = soft_bottom_y_stable(pr_up, ys) if has_up else pred.new_zeros(())
    pr_lt = soft_top_y(pr_lo, ys) if has_lo else pred.new_zeros(())

    # Per-column one-sided edge: same-x 不能超出 (上牙下缘 / 下牙上缘).
    l_col_edge = pred.new_zeros(())
    l_edge_over = pred.new_zeros(())
    l_bite = pred.new_zeros(())
    w_up = float(args.w_col_up)
    if has_up:
      gy, gok = soft_edge_y_per_col(gt_up, ys, bottom=True)
      py, pok = soft_edge_y_per_col(pr_up, ys, bottom=True)
      both = gok & pok
      if bool(both.any()):
        l_col_edge = l_col_edge + w_up * F.relu(py[both] - gy[both]).pow(2).mean()
      l_edge_over = l_edge_over + w_up * F.relu(pr_ub - gt_ub).pow(2)
      # Pixel past GT per-column bottom (only where GT column exists).
      past = (ys > gy[None, :]) & gok[None, :]
      l_bite = l_bite + w_up * (
          pr_up * past.float() * nonlip
      ).sum() / gt_up.sum().clamp(min=1.0)
      l_fp_up = l_fp_up + (
          pr_up * past.float() * nonlip
      ).sum() / gt_up.sum().clamp(min=1.0)
    if has_lo:
      gy, gok = soft_edge_y_per_col(gt_lo, ys, bottom=False)
      py, pok = soft_edge_y_per_col(pr_lo, ys, bottom=False)
      both = gok & pok
      if bool(both.any()):
        l_col_edge = l_col_edge + float(args.w_col_low) * F.relu(
            gy[both] - py[both]
        ).pow(2).mean()
      l_edge_over = l_edge_over + float(args.w_col_low) * F.relu(gt_lt - pr_lt).pow(2)
      past = (ys < gy[None, :]) & gok[None, :]
      l_bite = l_bite + float(args.w_col_low) * (
          pr_lo * past.float() * nonlip
      ).sum() / gt_lo.sum().clamp(min=1.0)

    # Also per-FDI column edges (local tooth bite).
    l_tooth_col = pred.new_zeros(())
    n_tc = 0
    for i, f in enumerate(channels):
      g = gt[i]
      if float(g.sum()) < 8.0:
        continue
      p = pred_sem[i]
      bottom = f < 30
      gy, gok = soft_edge_y_per_col(g, ys, bottom=bottom)
      py, pok = soft_edge_y_per_col(p, ys, bottom=bottom)
      both = gok & pok
      if not bool(both.any()):
        continue
      w_tc = w_up if bottom else float(args.w_col_low)
      if bottom:
        l_tooth_col = l_tooth_col + w_tc * F.relu(py[both] - gy[both]).pow(2).mean()
      else:
        l_tooth_col = l_tooth_col + w_tc * F.relu(gy[both] - py[both]).pow(2).mean()
      n_tc += 1
    if n_tc > 0:
      l_col_edge = l_col_edge + l_tooth_col / n_tc

    # L_semXY: per-FDI soft centroid match (OT-lite).
    l_xy = v.new_zeros(())
    n_xy = 0
    for i, f in enumerate(channels):
      if float(gt[i].sum()) < 5.0:
        continue
      mass = pred_sem[i].sum().clamp(min=0.0)
      if mass < 1.0:
        idx = per_fdi_t[f].to(device)
        xy = project_world_to_pixels(v[idx], rt, k_r, w_r, h_r, flip_y=True)
        pc = xy.mean(0)
      else:
        pc = soft_centroid(pred_sem[i], ys, xs) + torch.tensor(
            [float(x0r), float(y0r)], device=device,
        )
      l_xy = l_xy + ((pc - gt_c_all[i]) ** 2).sum()
      n_xy += 1
    l_xy = l_xy / max(n_xy, 1)

    l_ub = (pr_ub - gt_ub).pow(2) if has_up else pred.new_zeros(())
    l_lt = (pr_lt - gt_lt).pow(2) if has_lo else pred.new_zeros(())
    return {
        "l_sem": l_sem,
        "l_fill": l_fill,
        "l_fill_low": l_fill_low,
        "l_xy": l_xy,
        "l_out": l_out,
        "l_fp_low": l_fp_low,
        "l_fp_up": l_fp_up,
        "l_fp_tooth": l_fp_tooth,
        "l_bite": l_bite,
        "l_edge_over": l_edge_over,
        "l_col_edge": l_col_edge,
        "l_ub": l_ub,
        "l_lt": l_lt,
        "pr_ub": pr_ub,
        "gt_ub": gt_ub,
        "pr_lt": pr_lt,
        "gt_lt": gt_lt,
        "has_up": has_up,
        "has_lo": has_lo,
        "pred_full": pred_full.detach(),
    }

  def snapshot_rigid():
    return {
        "aa_up": aa_up.detach().clone(), "t_up": t_up.detach().clone(),
        "aa_low": aa_low.detach().clone(), "t_low": t_low.detach().clone(),
    }

  def restore_rigid(snap):
    with torch.no_grad():
      aa_up.copy_(snap["aa_up"]); t_up.copy_(snap["t_up"])
      aa_low.copy_(snap["aa_low"]); t_low.copy_(snap["t_low"])

  def save_overlay(path: Path, verts_m: torch.Tensor, tag: str):
    with torch.no_grad():
      pred = sem_losses(verts_m, full_res=True)["pred_full"]
    ov = overlay_pred_gt(image, pred.cpu().numpy(), gt_full.cpu().numpy())
    cv2.putText(ov, tag, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.imwrite(str(path), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
    cv2.imwrite(
        str(path.with_name(path.stem + "_crop.jpg")),
        cv2.cvtColor(ov[y0:y1, x0:x1], cv2.COLOR_RGB2BGR),
    )

  # ---------- Stage1: full-arch 6-DoF (optional) ----------
  if args.skip_rigid:
    print("==> Stage1 SKIPPED (freeze init rigid)")
    for p_ in (aa_up, t_up, aa_low, t_low):
      p_.requires_grad_(False)
  else:
    print("==> Stage1 rigid 6-DoF on FULL arch (crown edge targets)")
    opt_r = torch.optim.Adam(
        [
            {"params": [aa_up, t_up], "lr": args.lr_rigid},
            {"params": [aa_low, t_low], "lr": args.lr_rigid},
        ]
    )
    best = {"score": float("inf"), "snap": snapshot_rigid()}
    for it in range(args.rigid_iters):
      opt_r.zero_grad()
      verts_m = apply_m(base0)
      l_ub_v, l_lt_v = edge_losses_from_verts(verts_m)
      l_cent_u = arch_centroid_loss(verts_m, "up")
      l_cent_l = arch_centroid_loss(verts_m, "low")
      use_rast = True
      sm = sem_losses(verts_m)
      l_ub = 0.5 * l_ub_v + 0.5 * sm["l_ub"]
      l_lt = 0.5 * l_lt_v + 0.5 * sm["l_lt"]
      l_sem_term = 0.35 * args.w_sem * sm["l_sem"]
      l_xy_term = args.w_semxy * sm["l_xy"]
      l_fill_term = 0.80 * args.w_fill * sm["l_fill"]
      l_fp_term = (
          1.0 * args.w_fp_up * sm["l_fp_up"]
          + 0.7 * args.w_fp_low * sm["l_fp_low"]
          + 0.5 * args.w_fp_tooth * sm["l_fp_tooth"]
      )
      l_out_term = 0.35 * args.w_out * sm["l_out"]
      l_sem = sm["l_sem"]
      l_reg = (
          aa_up.pow(2).sum() + aa_low.pow(2).sum()
          + t_up.pow(2).sum() + t_low.pow(2).sum()
      )
      loss = (
          1.5 * args.w_edge * l_ub
          + args.w_edge * l_lt
          + args.w_centroid * (l_cent_u + l_cent_l)
          + l_sem_term + l_xy_term + l_fill_term + l_fp_term + l_out_term
          + args.w_rigid_reg * l_reg
      )
      loss.backward()
      opt_r.step()
      with torch.no_grad():
        aa_up.clamp_(-0.5, 0.5); aa_low.clamp_(-0.5, 0.5)
        t_up.clamp_(-0.03, 0.03); t_low.clamp_(-0.03, 0.03)
        mu, uu, ll = mean_centroid_errs(apply_m(base0))
        e_ub = float(l_ub.detach().sqrt())
        e_lt = float(l_lt.detach().sqrt())
      score = (
          1.5 * e_ub + e_lt + 0.25 * uu + 0.25 * ll
          + 8.0 * float(sm["l_fp_up"].detach())
          + 4.0 * float(sm["l_fp_low"].detach())
          + 3.0 * float(sm["l_fill"].detach())
      )
      if np.isfinite(score) and score < best["score"]:
        best = {"score": score, "snap": snapshot_rigid()}
      if it % 25 == 0 or it == args.rigid_iters - 1:
        print(
            f"  [{it:04d}] loss={float(loss.detach()):.2f} ub={e_ub:.2f} lt={e_lt:.2f} "
            f"up|c|={uu:.1f} low|c|={ll:.1f} rast={int(use_rast)}"
        )
    restore_rigid(best["snap"])

  with torch.no_grad():
    verts_rigid = apply_m(base0).detach()
    gum_ref.copy_(verts_rigid[gum_t])
    refresh_occluded(verts_rigid)
    mu, uu, ll = mean_centroid_errs(verts_rigid)
  print(f"  Stage1 freeze up|c|={uu:.1f} low|c|={ll:.1f}  gum_ref locked")
  save_overlay(
      args.out_dir / "02_after_rigid_before_shape_overlay.jpg",
      verts_rigid,
      "frozen rigid / before shape; red=pred crown green=GT",
  )

  # ---------- Stage2A: mild id_teeth (no per-tooth) — fix over-shrink ----------
  print("==> Stage2A mild id_teeth only (rigid frozen)")
  for p_ in (aa_up, t_up, aa_low, t_low):
    p_.requires_grad_(False)
  id_teeth = torch.zeros(80, requires_grad=True)
  tooth_t = torch.zeros(n_c, 3, requires_grad=False)
  if args.init_teeth is not None:
    zt = np.load(args.init_teeth)
    with torch.no_grad():
      id_teeth.copy_(torch.as_tensor(zt["id_teeth"], dtype=torch.float32))
      # Stage2A optimizes id_teeth only; keep tooth_t=0 here (Stage2B handles deltas).
    print(f"warm-start id_teeth from {args.init_teeth} |teeth|={float(id_teeth.detach().norm()):.2f}")
  use_per_tooth = not bool(args.no_per_tooth)
  opt_a = torch.optim.Adam([id_teeth], lr=args.lr_shape * 0.55)
  best_a = {"score": float("inf"), "teeth": id_teeth.detach().clone()}
  print(
      f"  weights sem={args.w_sem} fill={args.w_fill} fill_low×={args.w_fill_low} "
      f"col={args.w_col_edge} col_up×={args.w_col_up} col_low×={args.w_col_low} "
      f"fp_tooth={args.w_fp_tooth} fp_up={args.w_fp_up} fp_low={args.w_fp_low} "
      f"semxy={args.w_semxy} out={args.w_out} bite={args.w_bite} prior={args.w_prior} "
      f"contact={args.w_contact} freeze_occ={int(args.freeze_occluded)}"
  )
  n_a = max(80, args.shape_iters * 2 // 3)
  for it in range(n_a):
    opt_a.zero_grad()
    verts_m = apply_shape(id_teeth, tooth_t)
    sm = sem_losses(verts_m)
    l_prior = id_teeth.pow(2).mean()
    l_lock = lock_fdi_loss(verts_m)
    l_contact = contact_loss(verts_m)
    loss = (
        args.w_sem * sm["l_sem"]
        + args.w_fill * sm["l_fill"]
        + args.w_col_edge * sm["l_col_edge"]
        + args.w_semxy * sm["l_xy"]
        + args.w_out * sm["l_out"]
        + args.w_fp_low * sm["l_fp_low"]
        + args.w_fp_up * sm["l_fp_up"]
        + args.w_fp_tooth * sm["l_fp_tooth"]
        + args.w_bite * sm["l_bite"]
        + args.w_edge_over * sm["l_edge_over"]
        + args.w_prior * l_prior
        + args.w_lock34 * l_lock
        + args.w_contact * l_contact
    )
    with torch.no_grad():
      mu, uu, ll = mean_centroid_errs(verts_m.detach())
      # Boundary-first: overflow kills the checkpoint; fill still matters.
      cur = (
          float(sm["l_sem"].detach())
          + 0.30 * float(sm["l_fill"].detach())
          + 0.15 * float(sm["l_fill_low"].detach())
          + 0.35 * float(sm["l_col_edge"].detach())
          + 0.50 * float(sm["l_out"].detach())
          + 0.70 * float(sm["l_fp_low"].detach())
          + 1.00 * float(sm["l_fp_up"].detach())
          + 1.20 * float(sm["l_fp_tooth"].detach())
          + 0.40 * float(sm["l_bite"].detach())
          + 0.15 * float(l_lock.detach())
          + 0.02 * float(id_teeth.detach().norm())
      )
      if cur < best_a["score"]:
        best_a = {"score": cur, "teeth": id_teeth.detach().clone()}
    loss.backward()
    opt_a.step()
    with torch.no_grad():
      id_teeth.clamp_(-float(args.id_teeth_clamp), float(args.id_teeth_clamp))
    if it % 25 == 0 or it == n_a - 1:
      print(
          f"  A[{it:04d}] loss={float(loss.detach()):.2f} sem={float(sm['l_sem'].detach()):.3f} "
          f"fill={float(sm['l_fill'].detach()):.3f} fillL={float(sm['l_fill_low'].detach()):.3f} "
          f"col={float(sm['l_col_edge'].detach()):.3f} "
          f"fpT={float(sm['l_fp_tooth'].detach()):.3f} fpU={float(sm['l_fp_up'].detach()):.3f} "
          f"out={float(sm['l_out'].detach()):.3f} bite={float(sm['l_bite'].detach()):.3f} "
          f"lock={float(l_lock.detach()):.5f} contact={float(l_contact.detach()):.5f} "
          f"|teeth|={float(id_teeth.detach().norm()):.2f} up|c|={uu:.1f} low|c|={ll:.1f}"
      )
  with torch.no_grad():
    id_teeth.copy_(best_a["teeth"])
    id_teeth.requires_grad_(False)
  print(f"  Stage2A done |teeth|={float(id_teeth.norm()):.2f}")

  # ---------- Stage2B: closed-form centroid snap + tiny Adam refine ----------
  if use_per_tooth:
    print("==> Stage2B per-tooth: centroid snap then tiny Adam")
    tooth_t = torch.zeros(n_c, 3, requires_grad=False)
    n_snap = max(40, int(args.pertooth_iters) // 4)
    n_adam = max(80, int(args.pertooth_iters) - n_snap)
    fx, fy, cx, cy = [float(x) for x in k.detach().cpu().tolist()]
    R = rt[:3, :3].detach().cpu()
    tvec = rt[:3, 3].detach().cpu()

    def tooth_image_centroid(verts_m: torch.Tensor, fdi_i: int) -> torch.Tensor:
      return fdi_pred_centroid(verts_m, fdi_i)

    for it in range(n_snap):
      with torch.no_grad():
        verts_m = apply_shape(id_teeth, tooth_t).detach().cpu()
        for i, f in enumerate(channels):
          if float(gt_full[i].sum()) < 8.0:
            continue
          idx = per_fdi_t[f]
          c_w = verts_m[idx].mean(0)
          c_cam = c_w @ R.T + tvec
          z = float(c_cam[2].clamp(min=1e-4))
          pc = tooth_image_centroid(verts_m, i).cpu()
          gc = gt_cent[i].detach().cpu()
          du = float(gc[0] - pc[0])
          dv = float(gc[1] - pc[1])
          d_cam = torch.tensor([du * z / fx, dv * z / fy, 0.0], dtype=torch.float32)
          d_w = d_cam @ R
          # Locked / float-prone: skip snap (esp. upper 12/23 overshoot).
          if f in float_fdi or f in set(lock_fdis):
            continue
          step = 0.18 if f >= 30 else 0.22
          tooth_t[i] = tooth_t[i] + step * d_w
        clamp_tooth_t_(tooth_t)
        for f in lock_fdis:
          if f in fdi_to_i:
            tooth_t[fdi_to_i[f]].zero_()
        if it % 10 == 0 or it == n_snap - 1:
          sm = sem_losses(apply_shape(id_teeth, tooth_t))
          mu, uu, ll = mean_centroid_errs(apply_shape(id_teeth, tooth_t))
          print(
              f"  snap[{it:03d}] sem={float(sm['l_sem']):.3f} fill={float(sm['l_fill']):.3f} "
              f"fillL={float(sm['l_fill_low']):.3f} col={float(sm['l_col_edge']):.3f} "
              f"fpT={float(sm['l_fp_tooth']):.3f} fpU={float(sm['l_fp_up']):.3f} "
              f"|dt|={float(tooth_t.norm()):.4f} up|c|={uu:.1f} low|c|={ll:.1f}"
          )

    tooth_t = tooth_t.detach().clone().requires_grad_(True)
    opt_b = torch.optim.Adam([tooth_t], lr=max(args.lr_shape * 0.03, 4e-4))
    with torch.no_grad():
      sm0 = sem_losses(apply_shape(id_teeth, tooth_t))
      score0 = (
          float(sm0["l_sem"]) + 0.35 * float(sm0["l_fill"])
          + 0.20 * float(sm0["l_fill_low"])
          + 0.25 * float(sm0["l_col_edge"])
          + 0.60 * float(sm0["l_out"])
          + 0.80 * float(sm0["l_fp_low"])
          + 1.10 * float(sm0["l_fp_up"])
          + 1.30 * float(sm0["l_fp_tooth"])
          + 0.35 * float(sm0["l_bite"]) + 0.002 * float(sm0["l_xy"])
          + 0.20 * float(tooth_t.pow(2).mean())
      )
    best_b = {"score": score0, "tooth_t": tooth_t.detach().clone()}
    print(f"  adam_iters={n_adam} init_score={score0:.4f} |dt|={float(tooth_t.detach().norm()):.4f}")
    for it in range(n_adam):
      opt_b.zero_grad()
      verts_m = apply_shape(id_teeth, tooth_t)
      sm = sem_losses(verts_m)
      l_tooth, l_adj, l_fl = tooth_reg_terms(tooth_t)
      l_lock = lock_fdi_loss(verts_m)
      l_contact = contact_loss(verts_m)
      loss = (
          args.w_sem * sm["l_sem"]
          + 1.10 * args.w_fill * sm["l_fill"]
          + 1.00 * args.w_col_edge * sm["l_col_edge"]
          + args.w_semxy * sm["l_xy"]
          + args.w_out * sm["l_out"]
          + args.w_fp_low * sm["l_fp_low"]
          + args.w_fp_up * sm["l_fp_up"]
          + args.w_fp_tooth * sm["l_fp_tooth"]
          + args.w_bite * sm["l_bite"]
          + 0.9 * args.w_edge_over * sm["l_edge_over"]
          + args.w_tooth_reg * l_tooth
          + args.w_adj * l_adj
          + args.w_float * l_fl
          + args.w_lock34 * l_lock
          + args.w_contact * l_contact
      )
      loss.backward()
      opt_b.step()
      with torch.no_grad():
        clamp_tooth_t_(tooth_t)
        for f in lock_fdis:
          if f in fdi_to_i:
            tooth_t[fdi_to_i[f]].zero_()
        sm_b = sem_losses(apply_shape(id_teeth, tooth_t))
        mu, uu, ll = mean_centroid_errs(apply_shape(id_teeth, tooth_t))
        cur = (
            float(sm_b["l_sem"]) + 0.35 * float(sm_b["l_fill"])
            + 0.20 * float(sm_b["l_fill_low"])
            + 0.25 * float(sm_b["l_col_edge"])
            + 0.60 * float(sm_b["l_out"])
            + 0.80 * float(sm_b["l_fp_low"])
            + 1.10 * float(sm_b["l_fp_up"])
            + 1.30 * float(sm_b["l_fp_tooth"])
            + 0.35 * float(sm_b["l_bite"]) + 0.002 * float(sm_b["l_xy"])
            + 0.20 * float(tooth_t.pow(2).mean())
        )
        if cur < best_b["score"] - 1e-6:
          best_b = {"score": cur, "tooth_t": tooth_t.detach().clone()}
      if it % 30 == 0 or it == n_adam - 1:
        dt12 = float(tooth_t[fdi_to_i[12]].detach().norm()) if 12 in fdi_to_i else -1.0
        dt23 = float(tooth_t[fdi_to_i[23]].detach().norm()) if 23 in fdi_to_i else -1.0
        print(
            f"  adam[{it:03d}] sem={float(sm_b['l_sem']):.3f} fill={float(sm_b['l_fill']):.3f} "
            f"fpT={float(sm_b['l_fp_tooth']):.3f} fpU={float(sm_b['l_fp_up']):.3f} "
            f"col={float(sm_b['l_col_edge']):.3f} "
            f"|dt|={float(tooth_t.detach().norm()):.4f} |12|={dt12:.4f} |23|={dt23:.4f} "
            f"best={best_b['score']:.4f} up|c|={uu:.1f} low|c|={ll:.1f}"
        )
    with torch.no_grad():
      tooth_t.copy_(best_b["tooth_t"])
      for f in lock_fdis:
        if f in fdi_to_i:
          tooth_t[fdi_to_i[f]].zero_()
      tooth_t.requires_grad_(False)
    print(f"  Stage2B done |dt|={float(tooth_t.norm()):.4f} best={best_b['score']:.4f}")
  else:
    print("==> Stage2B skipped (--no_per_tooth)")

  with torch.no_grad():
    verts_final = apply_shape(id_teeth.detach(), tooth_t.detach()).detach()
    mu, uu, ll = mean_centroid_errs(verts_final)
    gum_final = float(gum_lock_loss(verts_final).sqrt())
    sm_f = sem_losses(verts_final)
  print(
      f"  restored Stage2 |teeth|={float(id_teeth.detach().norm()):.2f} "
      f"|dt|={float(tooth_t.detach().norm()):.4f} "
      f"up={uu:.1f} low={ll:.1f} gum_rms={gum_final:.5f} "
      f"sem={float(sm_f['l_sem']):.3f} fill={float(sm_f['l_fill']):.3f} "
      f"fillL={float(sm_f['l_fill_low']):.3f} "
      f"col={float(sm_f['l_col_edge']):.3f} out={float(sm_f['l_out']):.3f} "
      f"fpT={float(sm_f['l_fp_tooth']):.3f} fpU={float(sm_f['l_fp_up']):.3f} "
      f"fpL={float(sm_f['l_fp_low']):.3f} bite={float(sm_f['l_bite']):.3f}"
  )
  save_overlay(
      args.out_dir / "03_after_shape_overlay.jpg",
      verts_final,
      "after shape (per-tooth fill + no-overflow + lock12/23)",
  )
  ov_r = cv2.imread(str(args.out_dir / "02_after_rigid_before_shape_overlay_crop.jpg"))
  ov_s = cv2.imread(str(args.out_dir / "03_after_shape_overlay_crop.jpg"))
  if ov_r is not None and ov_s is not None:
    h = min(ov_r.shape[0], ov_s.shape[0])
    compare = np.concatenate([ov_r[:h], ov_s[:h]], 1)
    cv2.putText(compare, "rigid", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(
        compare, "shape", (ov_r.shape[1] + 12, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
    )
    cv2.imwrite(str(args.out_dir / "04_rigid_vs_shape_crop.jpg"), compare)

  # ---------- renders ----------
  print("==> render full teeth+gums mesh")
  try:
    rend = NVDiffRenderer(
        use_opengl=False, lighting_type="front", lighting_space="camera",
        flip_y_raster=False,
    ).to(device)
    with torch.no_grad():
      out_r = rend.render_rgba_vis(
          verts_final.to(device)[None], faces_full, rt[None], k[None],
          (height, width), background_color=[0.0, 0.0, 0.0],
      )
      rgba = out_r["rgba"][0].detach().float().cpu().numpy()
    rgb = np.clip(rgba[..., :3] * 255, 0, 255).astype(np.uint8)
    alpha = rgba[..., 3]
    cv2.imwrite(
        str(args.out_dir / "01_optimized_teeth_gums_mesh.jpg"),
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
    )
    comp = np.clip(
        image.astype(np.float32) * (1 - 0.65 * alpha[..., None])
        + rgb.astype(np.float32) * (0.65 * alpha[..., None]),
        0, 255,
    ).astype(np.uint8)
    cv2.imwrite(
        str(args.out_dir / "01_optimized_teeth_gums_on_photo.jpg"),
        cv2.cvtColor(comp, cv2.COLOR_RGB2BGR),
    )
    crop = np.concatenate(
        [image[y0:y1, x0:x1], rgb[y0:y1, x0:x1], comp[y0:y1, x0:x1]], 1,
    )
    cv2.imwrite(
        str(args.out_dir / "01_optimized_teeth_gums_crop_collage.jpg"),
        cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
    )
  except Exception as e:  # noqa: BLE001
    print("mesh render failed:", e)

  try:
    import trimesh
    trimesh.Trimesh(
        vertices=verts_final.cpu().numpy(), faces=faces_full_np, process=False,
    ).export(args.out_dir / "optimized_teeth_and_gums.obj")
  except Exception as e:  # noqa: BLE001
    print("obj export skipped:", e)

  with torch.no_grad():
    pred_f = sem_losses(verts_final, full_res=True)["pred_full"]
    pred_r = sem_losses(verts_rigid, full_res=True)["pred_full"]
  cv2.imwrite(
      str(args.out_dir / "gt_masks_color.jpg"),
      cv2.cvtColor(colorize_masks(gt_full.cpu().numpy()), cv2.COLOR_RGB2BGR),
  )
  cv2.imwrite(
      str(args.out_dir / "pred_masks_after_rigid_color.jpg"),
      cv2.cvtColor(colorize_masks(pred_r.cpu().numpy()), cv2.COLOR_RGB2BGR),
  )
  cv2.imwrite(
      str(args.out_dir / "pred_masks_after_shape_color.jpg"),
      cv2.cvtColor(colorize_masks(pred_f.cpu().numpy()), cv2.COLOR_RGB2BGR),
  )
  # Mouth crops for quick compare.
  cv2.imwrite(
      str(args.out_dir / "diag_gt_crop.jpg"),
      cv2.cvtColor(colorize_masks(gt_full.cpu().numpy())[y0:y1, x0:x1], cv2.COLOR_RGB2BGR),
  )
  cv2.imwrite(
      str(args.out_dir / "diag_rigid_pred_crop.jpg"),
      cv2.cvtColor(colorize_masks(pred_r.cpu().numpy())[y0:y1, x0:x1], cv2.COLOR_RGB2BGR),
  )
  cv2.imwrite(
      str(args.out_dir / "diag_shape_pred_crop.jpg"),
      cv2.cvtColor(colorize_masks(pred_f.cpu().numpy())[y0:y1, x0:x1], cv2.COLOR_RGB2BGR),
  )

  np.savez(
      args.out_dir / "fullstruct_crownloss_params.npz",
      aa_up=aa_up.detach().cpu().numpy(),
      t_up=t_up.detach().cpu().numpy(),
      aa_low=aa_low.detach().cpu().numpy(),
      t_low=t_low.detach().cpu().numpy(),
      id_teeth=id_teeth.detach().cpu().numpy(),
      tooth_t=tooth_t.detach().cpu().numpy(),
      channels=np.asarray(channels),
      occ_crown=occ_crown_t.detach().cpu().numpy() if occ_crown_t.numel()
      else np.zeros((0,), dtype=np.int64),
      lock_verts=lock_verts_t.detach().cpu().numpy() if lock_verts_t.numel()
      else np.zeros((0,), dtype=np.int64),
  )
  summary = {
      "mode": "smooth_coco_v2: vis-centroid rigid + occ-freeze + contact + strong upper fp",
      "skip_rigid": bool(args.skip_rigid),
      "init_rigid": str(args.init_rigid) if args.init_rigid else None,
      "init_teeth": str(args.init_teeth) if args.init_teeth else None,
      "up_centroid_err_px": uu,
      "low_centroid_err_px": ll,
      "mean_centroid_err_px": mu,
      "gum_lock_rms": gum_final,
      "hard_gum_freeze": bool(args.hard_gum_freeze),
      "w_gum_lock": args.w_gum_lock,
      "w_prior": args.w_prior,
      "w_sem": args.w_sem,
      "w_semxy": args.w_semxy,
      "w_out": args.w_out,
      "w_bite": args.w_bite,
      "w_edge_over": args.w_edge_over,
      "w_fill": args.w_fill,
      "w_fill_low": args.w_fill_low,
      "w_col_edge": args.w_col_edge,
      "w_col_up": args.w_col_up,
      "w_col_low": args.w_col_low,
      "w_fp_low": args.w_fp_low,
      "w_fp_up": args.w_fp_up,
      "w_fp_tooth": args.w_fp_tooth,
      "w_tooth_reg": args.w_tooth_reg,
      "w_adj": args.w_adj,
      "w_float": args.w_float,
      "w_lock34": args.w_lock34,
      "lock_fdi": lock_fdis,
      "w_contact": args.w_contact,
      "w_centroid": args.w_centroid,
      "freeze_occluded": bool(args.freeze_occluded),
      "visible_centroid": bool(args.visible_centroid),
      "n_occ_crown": int(occ_crown_t.numel()),
      "l_sem": float(sm_f["l_sem"]),
      "l_fill": float(sm_f["l_fill"]),
      "l_fill_low": float(sm_f["l_fill_low"]),
      "l_col_edge": float(sm_f["l_col_edge"]),
      "l_out": float(sm_f["l_out"]),
      "l_fp_low": float(sm_f["l_fp_low"]),
      "l_fp_up": float(sm_f["l_fp_up"]),
      "l_fp_tooth": float(sm_f["l_fp_tooth"]),
      "l_bite": float(sm_f["l_bite"]),
      "pertooth_iters": int(args.pertooth_iters),
      "id_teeth_clamp": args.id_teeth_clamp,
      "tooth_t_clamp": args.tooth_t_clamp,
      "tooth_t_clamp_float": args.tooth_t_clamp_float,
      "tooth_t_norm": float(tooth_t.detach().norm()),
      "dt34": float(tooth_t[fdi_to_i[34]].norm()) if 34 in fdi_to_i else None,
      "per_tooth": bool(use_per_tooth),
      "aa_up": aa_up.detach().cpu().tolist(),
      "t_up": t_up.detach().cpu().tolist(),
      "aa_low": aa_low.detach().cpu().tolist(),
      "t_low": t_low.detach().cpu().tolist(),
      "id_teeth_norm": float(id_teeth.detach().norm()),
      "n_arch_up": int(len(arch_up)),
      "n_arch_low": int(len(arch_low)),
      "n_crown": int(len(crown_all)),
      "n_gum": int(len(gum)),
      "outputs": [
          "01_optimized_teeth_gums_mesh.jpg",
          "02_after_rigid_before_shape_overlay.jpg",
          "03_after_shape_overlay.jpg",
          "04_rigid_vs_shape_crop.jpg",
      ],
  }
  (args.out_dir / "summary.json").write_text(
      json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
  )
  print(f"done → {args.out_dir}")
  print(
      f"key figs:\n"
      f"  1) {args.out_dir / '01_optimized_teeth_gums_mesh.jpg'}\n"
      f"  2) {args.out_dir / '02_after_rigid_before_shape_overlay.jpg'}\n"
      f"  3) {args.out_dir / '03_after_shape_overlay.jpg'}"
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
