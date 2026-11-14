#!/usr/bin/env python3
"""Teeth-mesh-only: freeze vis_both_rigid6_nocollide pose, fill with id_teeth.

Rigid 6-DoF is copied from vis_both_rigid6_nocollide (upper clip6 frozen,
lower refined to avoid arch collision). Face stays frozen. No per-tooth t.
Shape fills the coco GT without overflowing and without opening contacts.
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
REPO = next(p for p in [ROOT, *ROOT.parents] if (p / "GNM-main").is_dir()) / "GNM-main"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT))

from gnm.shape import gnm_pytorch, gnm_numpy  # pylint: disable=wrong-import-position
from nvdiffrast_render import NVDiffRenderer  # pylint: disable=wrong-import-position
from nvdiffrast_render.camera import project_world_to_pixels  # pylint: disable=wrong-import-position

from fit_gnm_teeth_single import (  # pylint: disable=wrong-import-position
    apply_arch_rigid,
    load_fdi_vertex_map,
    load_gt_masks,
    mouth_roi,
    overlay_pred_gt,
    parse_k,
    render_semantic_masks,
    teeth_faces_from_groups,
)
from vis_lower_rigid6 import col_edge, draw_col_polyline  # pylint: disable=wrong-import-position
from vis_both_rigid6 import (  # pylint: disable=wrong-import-position
    arch_collision,
    fdi_crown_indices,
    vertex_normals,
)


ADJ_CHAINS = (
    (11, 12, 13),
    (21, 22, 23),
    (31, 32, 33, 34, 35, 36),
    (41, 42, 43, 44, 45, 46),
)


def build_argparser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument(
      "--params", type=Path,
      default=Path("gnm_fit_out/fit_frame_0027_mp68_official_eyes/frame_000000_params.npz"),
      help="Frozen face used when the vis rigid6 poses were solved.",
  )
  p.add_argument(
      "--init_rigid", type=Path,
      default=Path("gnm_fit_out/vis_both_rigid6_nocollide/both_rigid6_params.npz"),
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
      default=Path("gnm_fit_out/fit_teeth_frame_0027_nocollide_fill"),
  )
  p.add_argument("--shape_iters", type=int, default=300)
  p.add_argument("--lr_shape", type=float, default=0.012)
  p.add_argument("--w_over", type=float, default=180.0)
  p.add_argument("--w_over_col", type=float, default=8.0)
  p.add_argument("--w_gap", type=float, default=80.0)
  p.add_argument("--w_fp", type=float, default=240.0)
  p.add_argument("--w_fill", type=float, default=140.0)
  p.add_argument("--w_fill_edge", type=float, default=120.0)
  p.add_argument("--w_fill_tooth", type=float, default=90.0)
  p.add_argument("--w_seam", type=float, default=80.0)
  p.add_argument("--w_coll", type=float, default=30.0)
  p.add_argument("--w_contact", type=float, default=80.0)
  p.add_argument("--w_prior", type=float, default=1.2)
  p.add_argument("--w_shape_lock", type=float, default=12.0)
  p.add_argument("--id_teeth_clamp", type=float, default=2.0)
  p.add_argument("--top_band", type=int, default=14)
  p.add_argument("--margin_mm", type=float, default=0.25)
  p.add_argument("--gate_mm", type=float, default=3.0)
  p.add_argument("--tiny_mass", type=float, default=20.0)
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
  rigid = np.load(args.init_rigid)
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

  n_verts = int(np.asarray(gnm_np.skinning_weights).shape[-1])
  fdi_npz = np.load(args.fdi_verts)
  _, crown_up_np = fdi_crown_indices(fdi_npz, True)
  _, crown_lo_np = fdi_crown_indices(fdi_npz, False)
  crown_up = torch.as_tensor(crown_up_np, dtype=torch.long)
  crown_lo = torch.as_tensor(crown_lo_np, dtype=torch.long)
  crown_all = torch.as_tensor(
      np.unique(np.concatenate([crown_up_np, crown_lo_np])), dtype=torch.long,
  )

  masks_meta = np.load(args.masks)
  visible = [int(x) for x in masks_meta["fdi"]]
  per_fdi, _, channels = load_fdi_vertex_map(args.fdi_verts, n_verts, visible)
  per_t = {f: torch.as_tensor(idx, dtype=torch.long) for f, idx in per_fdi.items()}
  vis_up = [f for f in channels if f < 30]
  vis_lo = [f for f in channels if f >= 30]
  gt_all = load_gt_masks(args.masks, channels, height, width).to(device)
  ch_of = {int(f): i for i, f in enumerate(channels)}
  gt_up = gt_all[[ch_of[f] for f in vis_up]]
  gt_lo = gt_all[[ch_of[f] for f in vis_lo]]
  y0, y1, x0, x1 = mouth_roi(gt_all, pad=40)

  up_idx = torch.as_tensor(
      np.asarray(gnm_np.vertex_group_indices("upper_teeth_and_gums"), np.int64),
      dtype=torch.long,
  )
  lo_idx = torch.as_tensor(
      np.asarray(gnm_np.vertex_group_indices("lower_teeth_and_gums"), np.int64),
      dtype=torch.long,
  )
  gum = np.setdiff1d(
      np.unique(np.concatenate([up_idx.numpy(), lo_idx.numpy()])),
      crown_all.numpy(),
  )
  gum_t = torch.as_tensor(gum, dtype=torch.long)
  faces_up = torch.as_tensor(
      teeth_faces_from_groups(gnm_np, ["upper_teeth_and_gums"]),
      dtype=torch.int32, device=device,
  )
  faces_lo = torch.as_tensor(
      teeth_faces_from_groups(gnm_np, ["lower_teeth_and_gums"]),
      dtype=torch.int32, device=device,
  )
  faces_both_np = teeth_faces_from_groups(
      gnm_np, ["upper_teeth_and_gums", "lower_teeth_and_gums"],
  )
  faces_both = torch.as_tensor(faces_both_np, dtype=torch.int32, device=device)
  faces_up_n = faces_up.detach().cpu().long()
  faces_lo_n = faces_lo.detach().cpu().long()

  def posed_verts(id_teeth: torch.Tensor) -> torch.Tensor:
    identity = torch.cat([id_head, id_eyes, id_teeth], dim=0)
    return gnm(
        identity[None], expression[None], rotations[None], translation[None],
    )[0]

  with torch.no_grad():
    base0 = posed_verts(torch.zeros(80)).detach()
  pivot_up = base0[up_idx].mean(0).detach()
  pivot_lo = base0[lo_idx].mean(0).detach()

  aa_up = torch.tensor(np.asarray(rigid["aa_up"], np.float32), requires_grad=False)
  t_up = torch.tensor(np.asarray(rigid["t_up"], np.float32), requires_grad=False)
  aa_lo = torch.tensor(np.asarray(rigid["aa_low"], np.float32), requires_grad=False)
  t_lo = torch.tensor(np.asarray(rigid["t_low"], np.float32), requires_grad=False)

  def apply_m(verts: torch.Tensor) -> torch.Tensor:
    v = apply_arch_rigid(verts, up_idx, aa_up, t_up, pivot_up)
    return apply_arch_rigid(v, lo_idx, aa_lo, t_lo, pivot_lo)

  ys = torch.arange(height, device=device, dtype=torch.float32)[:, None].expand(height, width)
  gt_up_u = gt_up.sum(0).clamp(0, 1)
  gt_lo_u = gt_lo.sum(0).clamp(0, 1)

  v_up = torch.zeros(n_verts, len(vis_up), device=device)
  for c, f in enumerate(vis_up):
    v_up[per_t[f].to(device), c] = 1.0
  v_lo = torch.zeros(n_verts, len(vis_lo), device=device)
  for c, f in enumerate(vis_lo):
    v_lo[per_t[f].to(device), c] = 1.0
  v_edge_up = torch.zeros(n_verts, 1, device=device)
  v_edge_lo = torch.zeros(n_verts, 1, device=device)
  v_edge_up[crown_up.to(device), 0] = 1.0
  v_edge_lo[crown_lo.to(device), 0] = 1.0

  ker = np.ones((5, 5), np.uint8)
  seam_np = np.zeros((height, width), np.float32)
  for chain in ADJ_CHAINS:
    fdis = [f for f in chain if f in ch_of]
    for a, b in zip(fdis, fdis[1:]):
      da = cv2.dilate((gt_all[ch_of[a]].detach().cpu().numpy() > 0.15).astype(np.uint8), ker)
      db = cv2.dilate((gt_all[ch_of[b]].detach().cpu().numpy() > 0.15).astype(np.uint8), ker)
      seam_np = np.maximum(seam_np, (da & db).astype(np.float32))
  seam = torch.as_tensor(seam_np, device=device)

  renderer = NVDiffRenderer(
      use_opengl=False, lighting_type="constant", lighting_space="camera",
      flip_y_raster=False,
  ).to(device)
  rend_mesh = NVDiffRenderer(
      use_opengl=False, lighting_type="front", lighting_space="camera",
      flip_y_raster=False,
  ).to(device)

  def arch_losses(verts_m: torch.Tensor, *, is_upper: bool) -> dict:
    bite_bottom = is_upper
    vis = vis_up if is_upper else vis_lo
    gt = gt_up if is_upper else gt_lo
    gt_u = gt_up_u if is_upper else gt_lo_u
    faces = faces_up if is_upper else faces_lo
    v_attr = v_up if is_upper else v_lo
    v_edge = v_edge_up if is_upper else v_edge_lo
    pred = render_semantic_masks(
        renderer, verts_m.to(device), faces, v_attr, rt, k, (height, width),
    )
    pr = pred.sum(0).clamp(0, 1)
    edge_full = render_semantic_masks(
        renderer, verts_m.to(device), faces, v_edge, rt, k, (height, width),
    )[0].clamp(0, 1)
    gy, gok = col_edge(gt_u, ys, bottom=bite_bottom)
    py, pok = col_edge(edge_full, ys, bottom=bite_bottom)
    pok = pok & gok
    both = gok & pok
    l_gap = pr.new_zeros(())
    l_over_col = pr.new_zeros(())
    if bool(both.any()):
      if bite_bottom:
        l_gap = F.relu(gy[both] - py[both]).pow(2).mean()
        l_over_col = F.relu(py[both] - gy[both]).pow(2).mean()
      else:
        l_gap = F.relu(py[both] - gy[both]).pow(2).mean()
        l_over_col = F.relu(gy[both] - py[both]).pow(2).mean()
    if bite_bottom:
      past = (ys > gy[None, :]) & gok[None, :]
      band = gok[None, :] & (ys <= gy[None, :]) & (ys > gy[None, :] - float(args.top_band))
    else:
      past = (ys < gy[None, :]) & gok[None, :]
      band = gok[None, :] & (ys >= gy[None, :]) & (ys < gy[None, :] + float(args.top_band))
    l_over_px = (pr * past.float()).sum() / gt_u.sum().clamp(min=1.0)
    l_fp = (pr * (gt_u < 0.12).float()).sum() / gt_u.sum().clamp(min=1.0)
    miss = (gt_u - pr).clamp(min=0.0) * (gt_u > 0.15).float()
    l_fill = miss.sum() / gt_u.sum().clamp(min=1.0)
    l_fill_edge = (
        ((gt_u - pr).clamp(min=0.0) * band.float()).sum()
        / band.float().sum().clamp(min=1.0)
    )
    l_fill_tooth = pr.new_zeros(())
    n_tooth = 0
    for i, f in enumerate(vis):
      gi = gt[i]
      if float(gi.sum()) < args.tiny_mass:
        continue
      pi = pred[i].clamp(0, 1)
      l_fill_tooth = l_fill_tooth + (
          ((gi - pi).clamp(min=0.0) * (gi > 0.15).float()).sum()
          / gi.sum().clamp(min=1.0)
      )
      n_tooth += 1
    if n_tooth:
      l_fill_tooth = l_fill_tooth / n_tooth
    pred_e = py[gok].mean() if bool(gok.any()) else py.new_zeros(())
    gt_e = gy[gok].mean() if bool(gok.any()) else gy.new_zeros(())
    return {
        "pred": pred, "pr": pr, "gy": gy, "gok": gok, "py": py, "pok": pok,
        "l_gap": l_gap, "l_over_col": l_over_col, "l_over_px": l_over_px,
        "l_fp": l_fp, "l_fill": l_fill, "l_fill_edge": l_fill_edge,
        "l_fill_tooth": l_fill_tooth, "pred_e": pred_e, "gt_e": gt_e,
    }

  def pack(sm: dict) -> torch.Tensor:
    # Clip over_col: a few bad columns were ~100 and drowned fill/fp.
    return (
        args.w_over * sm["l_over_px"]
        + args.w_over_col * sm["l_over_col"].clamp(max=8.0)
        + args.w_gap * sm["l_gap"].clamp(max=12.0)
        + args.w_fp * sm["l_fp"]
        + args.w_fill * sm["l_fill"]
        + args.w_fill_edge * sm["l_fill_edge"]
        + args.w_fill_tooth * sm["l_fill_tooth"]
    )

  def collision_pack(verts_m: torch.Tensor) -> dict:
    n_u = vertex_normals(verts_m, faces_up_n)
    n_l = vertex_normals(verts_m, faces_lo_n)
    return arch_collision(
        verts_m, crown_up, crown_lo, n_u, n_l, margin=margin, gate=gate,
    )

  def contact_spread(verts_m: torch.Tensor, verts_ref: torch.Tensor) -> torch.Tensor:
    loss = verts_m.new_zeros(())
    n = 0
    for chain in ADJ_CHAINS:
      fdis = [f for f in chain if f in per_t]
      for a, b in zip(fdis, fdis[1:]):
        ia, ib = per_t[a], per_t[b]
        d = torch.linalg.norm(verts_m[ia].mean(0) - verts_m[ib].mean(0))
        d0 = torch.linalg.norm(verts_ref[ia].mean(0) - verts_ref[ib].mean(0)).detach()
        nn = torch.cdist(verts_m[ia], verts_m[ib]).min()
        nn0 = torch.cdist(verts_ref[ia], verts_ref[ib]).min().detach()
        loss = loss + F.relu(d - d0).pow(2) + F.relu(nn - nn0).pow(2)
        n += 1
    return loss / max(n, 1)

  def seam_loss(pr_up: torch.Tensor, pr_lo: torch.Tensor) -> torch.Tensor:
    pr = (pr_up + pr_lo).clamp(0, 1)
    miss = seam * (1.0 - pr)
    return miss.sum() / seam.sum().clamp(min=1.0)

  def dump_vis(stem: str, verts_m: torch.Tensor, tag: str) -> None:
    with torch.no_grad():
      sm_u = arch_losses(verts_m, is_upper=True)
      sm_l = arch_losses(verts_m, is_upper=False)
      pred = torch.cat([sm_u["pred"], sm_l["pred"]], 0)
      ov = overlay_pred_gt(
          image, pred.detach().cpu().numpy(),
          torch.cat([gt_up, gt_lo], 0).detach().cpu().numpy(),
      )
      crop = ov[y0:y1, x0:x1].copy()
      edge = crop.copy()
      draw_col_polyline(edge, sm_u["gy"].cpu().numpy()[x0:x1], sm_u["gok"].cpu().numpy()[x0:x1], 0, (0, 220, 0), y0=y0)
      draw_col_polyline(edge, sm_u["py"].cpu().numpy()[x0:x1], sm_u["pok"].cpu().numpy()[x0:x1], 0, (220, 40, 40), y0=y0)
      draw_col_polyline(edge, sm_l["gy"].cpu().numpy()[x0:x1], sm_l["gok"].cpu().numpy()[x0:x1], 0, (0, 200, 200), y0=y0)
      draw_col_polyline(edge, sm_l["py"].cpu().numpy()[x0:x1], sm_l["pok"].cpu().numpy()[x0:x1], 0, (220, 40, 200), y0=y0)
      cv2.putText(crop, tag, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
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
    cv2.imwrite(str(args.out_dir / f"{stem}_overlay.jpg"), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(args.out_dir / f"{stem}_overlay_crop.jpg"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(args.out_dir / f"{stem}_edge_crop.jpg"), cv2.cvtColor(edge, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(args.out_dir / f"{stem}_mesh.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(
        str(args.out_dir / f"{stem}_mesh_crop.jpg"),
        cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        str(args.out_dir / f"{stem}_mesh_on_photo_crop.jpg"),
        cv2.cvtColor(blend[y0:y1, x0:x1], cv2.COLOR_RGB2BGR),
    )

  def print_arch(tag: str, sm: dict) -> None:
    print(
        f"  {tag} over={float(sm['l_over_px']):.4f} overCol={float(sm['l_over_col']):.3f} "
        f"gap={float(sm['l_gap']):.3f} fp={float(sm['l_fp']):.4f} "
        f"fill={float(sm['l_fill']):.3f} fillT={float(sm['l_fill_tooth']):.3f} "
        f"edge {float(sm['pred_e']):.1f}->{float(sm['gt_e']):.1f}"
    )

  print(
      f"NOCOLLIDE FILL  face={args.params.name}  rigid={args.init_rigid}  "
      f"ROI=({x0},{y0})-({x1},{y1})  vis_up={vis_up} vis_lo={vis_lo}"
  )
  print(f"  aa_up={aa_up.tolist()}")
  print(f"  t_up={t_up.tolist()}")
  print(f"  aa_low={aa_lo.tolist()}")
  print(f"  t_low={t_lo.tolist()}")

  with torch.no_grad():
    verts_rigid = apply_m(base0).detach()
    gum_ref = verts_rigid[gum_t].clone()
    sm_u0 = arch_losses(verts_rigid, is_upper=True)
    sm_l0 = arch_losses(verts_rigid, is_upper=False)
  dump_vis("00_rigid", verts_rigid, "copied nocollide rigid")
  print_arch("rigid UPPER", sm_u0)
  print_arch("rigid LOWER", sm_l0)

  print("==> id_teeth fill (rigid frozen, no per-tooth t, gums frozen)")
  id_teeth = torch.zeros(80, requires_grad=True)

  def apply_shape(teeth: torch.Tensor) -> torch.Tensor:
    v = apply_m(posed_verts(teeth)).clone()
    v[gum_t] = gum_ref
    return v

  opt = torch.optim.Adam([id_teeth], lr=args.lr_shape)
  best = {"score": float("inf"), "teeth": id_teeth.detach().clone(), "it": -1}
  for it in range(args.shape_iters):
    opt.zero_grad()
    verts = apply_shape(id_teeth)
    sm_u = arch_losses(verts, is_upper=True)
    sm_l = arch_losses(verts, is_upper=False)
    coll = collision_pack(verts)
    l_prior = id_teeth.pow(2).mean()
    l_lock = ((verts[crown_all] - verts_rigid[crown_all]) ** 2).mean()
    l_spread = contact_spread(verts, verts_rigid)
    l_seam = seam_loss(sm_u["pr"], sm_l["pr"])
    loss = (
        pack(sm_u) + pack(sm_l)
        + args.w_coll * coll["loss"]
        + args.w_contact * l_spread
        + args.w_seam * l_seam
        + args.w_prior * l_prior
        + args.w_shape_lock * l_lock
    )
    # Keep a real shape step: ignore the first 25 iters (still ~rigid).
    score = float(loss.detach())
    if it >= 25 and np.isfinite(score) and score < best["score"]:
      best = {"score": score, "teeth": id_teeth.detach().clone(), "it": it}
    loss.backward()
    opt.step()
    with torch.no_grad():
      id_teeth.clamp_(-float(args.id_teeth_clamp), float(args.id_teeth_clamp))
    if it % 40 == 0 or it == args.shape_iters - 1:
      print(
          f"  [{it:03d}] loss={float(loss.detach()):.2f} "
          f"fpU={float(sm_u['l_fp'].detach()):.3f} fpL={float(sm_l['l_fp'].detach()):.3f} "
          f"fillU={float(sm_u['l_fill'].detach()):.3f} fillL={float(sm_l['l_fill'].detach()):.3f} "
          f"overU={float(sm_u['l_over_px'].detach()):.3f} overL={float(sm_l['l_over_px'].detach()):.3f} "
          f"seam={float(l_seam.detach()):.3f} spread={float(l_spread.detach()):.5f} "
          f"|teeth|={float(id_teeth.detach().norm()):.2f}"
      )
      if it in (80, 160, 240) or it == args.shape_iters - 1:
        dump_vis(f"it{it:03d}", apply_shape(id_teeth).detach(), f"shape it {it}")

  with torch.no_grad():
    if best["it"] < 0:
      best = {"score": float("nan"), "teeth": id_teeth.detach().clone(), "it": args.shape_iters - 1}
    id_teeth.copy_(best["teeth"])
    id_teeth.requires_grad_(False)
    print(f"  restore it={best['it']} loss={best['score']:.2f} |teeth|={float(id_teeth.norm()):.2f}")
    verts_final = apply_shape(id_teeth).detach()
    sm_u_f = arch_losses(verts_final, is_upper=True)
    sm_l_f = arch_losses(verts_final, is_upper=False)
    coll_f = collision_pack(verts_final)
    gum_rms = float((verts_final[gum_t] - gum_ref).pow(2).mean().sqrt())
  dump_vis("01_shape", verts_final, "id_teeth fill")
  print_arch("shape UPPER", sm_u_f)
  print_arch("shape LOWER", sm_l_f)
  print(
      f"  |id_teeth|={float(id_teeth.norm()):.2f} gum_rms={gum_rms:.5f} "
      f"nPen={coll_f['n_pen_l']+coll_f['n_pen_u']} minSDF={coll_f['min_sdf']*1000:.2f}mm"
  )

  r_ov = cv2.imread(str(args.out_dir / "00_rigid_overlay_crop.jpg"))
  s_ov = cv2.imread(str(args.out_dir / "01_shape_overlay_crop.jpg"))
  if r_ov is not None and s_ov is not None:
    pair = np.concatenate([r_ov, s_ov], 1)
    cv2.putText(pair, "nocollide rigid", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(pair, "shape", (r_ov.shape[1] + 8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imwrite(str(args.out_dir / "02_rigid_vs_shape_overlay.jpg"), pair)
  r_m = cv2.imread(str(args.out_dir / "00_rigid_mesh_on_photo_crop.jpg"))
  s_m = cv2.imread(str(args.out_dir / "01_shape_mesh_on_photo_crop.jpg"))
  if r_m is not None and s_m is not None:
    cv2.imwrite(
        str(args.out_dir / "03_rigid_vs_shape_mesh.jpg"),
        np.concatenate([r_m, s_m], 1),
    )
  r_e = cv2.imread(str(args.out_dir / "00_rigid_edge_crop.jpg"))
  s_e = cv2.imread(str(args.out_dir / "01_shape_edge_crop.jpg"))
  if r_e is not None and s_e is not None:
    cv2.imwrite(
        str(args.out_dir / "04_rigid_vs_shape_edge.jpg"),
        np.concatenate([r_e, s_e], 1),
    )

  try:
    import trimesh
    trimesh.Trimesh(
        vertices=verts_final.cpu().numpy(), faces=faces_both_np, process=False,
    ).export(args.out_dir / "optimized_teeth_and_gums.obj")
  except Exception as e:  # noqa: BLE001
    print("obj export skipped:", e)

  np.savez(
      args.out_dir / "nocollide_fill_params.npz",
      aa_up=aa_up.detach().cpu().numpy(),
      t_up=t_up.detach().cpu().numpy(),
      aa_low=aa_lo.detach().cpu().numpy(),
      t_low=t_lo.detach().cpu().numpy(),
      id_teeth=id_teeth.detach().cpu().numpy(),
      tooth_t=np.zeros((len(channels), 3), dtype=np.float32),
      channels=np.asarray(channels),
  )
  summary = {
      "mode": "copy vis_both_rigid6_nocollide + id_teeth fill, no per-tooth t",
      "face": str(args.params),
      "init_rigid": str(args.init_rigid),
      "masks": str(args.masks),
      "per_tooth": False,
      "aa_up": aa_up.detach().cpu().tolist(),
      "t_up": t_up.detach().cpu().tolist(),
      "aa_low": aa_lo.detach().cpu().tolist(),
      "t_low": t_lo.detach().cpu().tolist(),
      "id_teeth_norm": float(id_teeth.norm()),
      "id_teeth_clamp": args.id_teeth_clamp,
      "restore_iter": best["it"],
      "gum_lock_rms": gum_rms,
      "rigid_up_fp": float(sm_u0["l_fp"]),
      "rigid_up_fill": float(sm_u0["l_fill"]),
      "rigid_low_fp": float(sm_l0["l_fp"]),
      "rigid_low_fill": float(sm_l0["l_fill"]),
      "shape_up_fp": float(sm_u_f["l_fp"]),
      "shape_up_fill": float(sm_u_f["l_fill"]),
      "shape_up_over": float(sm_u_f["l_over_px"]),
      "shape_low_fp": float(sm_l_f["l_fp"]),
      "shape_low_fill": float(sm_l_f["l_fill"]),
      "shape_low_over": float(sm_l_f["l_over_px"]),
      "n_pen": coll_f["n_pen_l"] + coll_f["n_pen_u"],
      "min_sdf_mm": coll_f["min_sdf"] * 1000,
  }
  (args.out_dir / "summary.json").write_text(
      json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
  )
  print("done →", args.out_dir)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
