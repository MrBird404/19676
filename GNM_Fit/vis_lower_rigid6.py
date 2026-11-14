#!/usr/bin/env python3
"""One-arch 6-DoF rigid: visualize bite-edge overlap vs overflow.

--arch lower: freeze upper; align lower 上端 (image-space top) to GT top.
--arch upper: freeze lower; align upper 下端 (image-space bottom) to GT bottom.
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
    soft_bottom_y_stable,
    soft_top_y,
    teeth_faces_from_groups,
)


def col_edge(
    mask: torch.Tensor,
    ys: torch.Tensor,
    *,
    bottom: bool,
    tau: float = 0.12,
    eps: float = 1e-6,
):
  """Per-column soft top (bottom=False) or bottom (bottom=True) y."""
  m = mask.clamp(min=0)
  active = m > 1e-4
  neg = torch.finfo(ys.dtype).min / 4
  signed = tau * ys if bottom else -tau * ys
  logits = torch.where(active, signed, torch.full_like(ys, neg))
  logits = logits - logits.amax(dim=0, keepdim=True).detach()
  w = torch.where(active, torch.exp(logits) * m, torch.zeros_like(m))
  y = (w * ys).sum(dim=0) / w.sum(dim=0).clamp(min=eps)
  ok = m.sum(dim=0) > 1e-3
  return y, ok


def draw_col_polyline(
    img: np.ndarray,
    y_col: np.ndarray,
    ok: np.ndarray,
    x0: int,
    color: tuple[int, int, int],
    y0: int = 0,
) -> None:
  pts = []
  for x, (yy, good) in enumerate(zip(y_col, ok)):
    if not good:
      if len(pts) >= 2:
        cv2.polylines(img, [np.asarray(pts, np.int32)], False, color, 2, cv2.LINE_AA)
      pts = []
      continue
    pts.append((x0 + x, int(round(float(yy) - y0))))
  if len(pts) >= 2:
    cv2.polylines(img, [np.asarray(pts, np.int32)], False, color, 2, cv2.LINE_AA)


def build_argparser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument(
      "--params", type=Path,
      default=Path("gnm_fit_out/fit_frame_0027_mp68_official_eyes/frame_000000_params.npz"),
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
  p.add_argument("--arch", choices=("lower", "upper"), default="lower")
  p.add_argument(
      "--out_dir", type=Path, default=None,
      help="Default: gnm_fit_out/vis_{arch}_rigid6_coco",
  )
  p.add_argument("--iters", type=int, default=350)
  p.add_argument("--lr", type=float, default=0.01)
  p.add_argument("--w_over", type=float, default=250.0, help="Pixels above GT top.")
  p.add_argument("--w_over_col", type=float, default=180.0, help="Per-column top overshoot.")
  p.add_argument("--w_gap", type=float, default=90.0, help="Per-column top undershoot (hole).")
  p.add_argument("--w_fp", type=float, default=160.0, help="Pred outside lower GT.")
  p.add_argument("--w_fill", type=float, default=35.0)
  p.add_argument("--w_fill_top", type=float, default=80.0, help="Underfill in top band.")
  p.add_argument("--top_band", type=int, default=14)
  p.add_argument("--w_cent", type=float, default=15.0)
  p.add_argument("--w_reg", type=float, default=1e-3)
  p.add_argument("--device", default="cuda")
  return p


def main() -> int:
  args = build_argparser().parse_args()
  device = torch.device(args.device if torch.cuda.is_available() else "cpu")
  if device.type != "cuda":
    raise SystemExit("CUDA required")
  if args.out_dir is None:
    args.out_dir = Path(f"gnm_fit_out/vis_{args.arch}_rigid6_coco")
  args.out_dir.mkdir(parents=True, exist_ok=True)
  is_upper = args.arch == "upper"
  bite_bottom = is_upper  # upper: align 下端; lower: align 上端
  group = "upper_teeth_and_gums" if is_upper else "lower_teeth_and_gums"
  edge_name = "bottom" if bite_bottom else "top"

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

  masks_meta = np.load(args.masks)
  visible = [int(x) for x in masks_meta["fdi"]]
  n_verts = int(np.asarray(gnm_np.skinning_weights).shape[-1])
  per_fdi, _, channels = load_fdi_vertex_map(args.fdi_verts, n_verts, visible)
  gt_all = load_gt_masks(args.masks, channels, height, width).to(device)
  arch_ch = [i for i, f in enumerate(channels) if (f < 30) == is_upper]
  arch_fdi = [channels[i] for i in arch_ch]
  gt_arch = gt_all[arch_ch]
  n_c = len(arch_ch)
  per_fdi_t = {f: torch.as_tensor(idx, dtype=torch.long) for f, idx in per_fdi.items()}

  arch_idx = np.asarray(gnm_np.vertex_group_indices(group), dtype=np.int64)
  arch_t = torch.as_tensor(arch_idx, dtype=torch.long)
  faces = torch.as_tensor(
      teeth_faces_from_groups(gnm_np, [group]),
      dtype=torch.int32, device=device,
  )

  identity0 = torch.cat([id_head, id_eyes, torch.zeros(80)], dim=0)
  with torch.no_grad():
    base0 = gnm(
        identity0[None], expression[None], rotations[None], translation[None],
    )[0].detach()
  pivot = base0[arch_t].mean(0).detach()

  aa = torch.zeros(3, requires_grad=True)
  tvec = torch.zeros(3, requires_grad=True)

  def apply_m(verts: torch.Tensor) -> torch.Tensor:
    return apply_arch_rigid(verts, arch_t, aa, tvec, pivot)

  ys = torch.arange(height, device=device, dtype=torch.float32)[:, None].expand(height, width)
  xs = torch.arange(width, device=device, dtype=torch.float32)[None, :].expand(height, width)
  gt_union = gt_arch.sum(0).clamp(0, 1)
  y0, y1, x0, x1 = mouth_roi(gt_arch, pad=40)

  v_attr = torch.zeros(n_verts, n_c, device=device)
  for c, f in enumerate(arch_fdi):
    v_attr[per_fdi_t[f].to(device), c] = 1.0

  # Full-arch crown occupancy for the bite edge (all 16 teeth), then clip
  # the polyline to the front/annotated GT columns so gingival spikes drop.
  fdi_npz = np.load(args.fdi_verts)
  all_fdi = sorted(int(k.split("_")[1]) for k in fdi_npz.files if k.startswith("fdi_"))
  arch_all_fdi = [f for f in all_fdi if (f < 30) == is_upper]
  v_edge = torch.zeros(n_verts, 1, device=device)
  for f in arch_all_fdi:
    idx = torch.as_tensor(np.asarray(fdi_npz[f"fdi_{f}"], dtype=np.int64), dtype=torch.long)
    v_edge[idx.to(device), 0] = 1.0
  print(f"  edge from all {args.arch} FDI {arch_all_fdi}, clip to GT cols of {arch_fdi}")

  renderer = NVDiffRenderer(
      use_opengl=False, lighting_type="constant", lighting_space="camera",
      flip_y_raster=False,
  ).to(device)
  rend_mesh = NVDiffRenderer(
      use_opengl=False, lighting_type="front", lighting_space="camera",
      flip_y_raster=False,
  ).to(device)

  gt_cent = []
  for i in range(n_c):
    m = gt_arch[i]
    mass = m.sum().clamp(min=1e-6)
    gt_cent.append(torch.stack([(m * xs).sum() / mass, (m * ys).sum() / mass]))
  gt_cent = torch.stack(gt_cent, 0)

  def losses(verts_m: torch.Tensor):
    pred = render_semantic_masks(
        renderer, verts_m.to(device), faces, v_attr, rt, k, (height, width),
    )
    pr = pred.sum(0).clamp(0, 1)
    edge_full = render_semantic_masks(
        renderer, verts_m.to(device), faces, v_edge, rt, k, (height, width),
    )[0].clamp(0, 1)
    gy, gok = col_edge(gt_union, ys, bottom=bite_bottom)
    py, pok = col_edge(edge_full, ys, bottom=bite_bottom)
    # Truncate full-arch incisal line to front/annotated teeth columns.
    pok = pok & gok
    both = gok & pok
    l_gap = pr.new_zeros(())
    l_over_col = pr.new_zeros(())
    if bool(both.any()):
      if bite_bottom:
        # Upper 下端: overflow if pred bottom > GT bottom; gap if pred too high.
        l_gap = F.relu(gy[both] - py[both]).pow(2).mean()
        l_over_col = F.relu(py[both] - gy[both]).pow(2).mean()
      else:
        l_gap = F.relu(py[both] - gy[both]).pow(2).mean()
        l_over_col = F.relu(gy[both] - py[both]).pow(2).mean()
    if bite_bottom:
      past = (ys > gy[None, :]) & gok[None, :]
      band = (
          gok[None, :]
          & (ys <= gy[None, :])
          & (ys > gy[None, :] - float(args.top_band))
      )
    else:
      past = (ys < gy[None, :]) & gok[None, :]
      band = (
          gok[None, :]
          & (ys >= gy[None, :])
          & (ys < gy[None, :] + float(args.top_band))
      )
    l_over_px = (pr * past.float()).sum() / gt_union.sum().clamp(min=1.0)
    l_fp = (pr * (gt_union < 0.12).float()).sum() / gt_union.sum().clamp(min=1.0)
    miss = (gt_union - pr).clamp(min=0.0) * (gt_union > 0.15).float()
    l_fill = miss.sum() / gt_union.sum().clamp(min=1.0)
    l_fill_top = (
        ((gt_union - pr).clamp(min=0.0) * band.float()).sum()
        / band.float().sum().clamp(min=1.0)
    )
    l_cent = pr.new_zeros(())
    for i, f in enumerate(arch_fdi):
      idx = per_fdi_t[f].to(device)
      xy = project_world_to_pixels(verts_m.to(device)[idx], rt, k, width, height, flip_y=True)
      l_cent = l_cent + ((xy.mean(0) - gt_cent[i]) ** 2).sum()
    l_cent = l_cent / max(n_c, 1)
    pred_e = py[gok].mean() if bool(gok.any()) else py.new_zeros(())
    gt_e = gy[gok].mean() if bool(gok.any()) else gy.new_zeros(())
    return {
        "pred": pred,
        "pr": pr,
        "gy": gy, "gok": gok, "py": py, "pok": pok,
        "l_gap": l_gap, "l_over_col": l_over_col, "l_over_px": l_over_px,
        "l_fp": l_fp, "l_fill": l_fill, "l_fill_top": l_fill_top, "l_cent": l_cent,
        "pred_e": pred_e, "gt_e": gt_e,
    }

  def dump_vis(stem: str, verts_m: torch.Tensor, tag: str) -> dict:
    with torch.no_grad():
      sm = losses(verts_m)
      pred_np = sm["pred"].detach().cpu().numpy()
      gt_np = gt_arch.detach().cpu().numpy()
      ov = overlay_pred_gt(image, pred_np, gt_np)
      crop = ov[y0:y1, x0:x1].copy()
      gy = sm["gy"].detach().cpu().numpy()[x0:x1]
      gok = sm["gok"].detach().cpu().numpy()[x0:x1]
      py = sm["py"].detach().cpu().numpy()[x0:x1]
      pok = sm["pok"].detach().cpu().numpy()[x0:x1]
      edge = crop.copy()
      draw_col_polyline(edge, gy, gok, 0, (0, 220, 0), y0=y0)
      draw_col_polyline(edge, py, pok, 0, (220, 40, 40), y0=y0)
      cv2.putText(crop, tag, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
      cv2.putText(
          edge, f"GT {edge_name}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 2,
      )
      cv2.putText(
          edge, f"pred {edge_name}", (8, 44),
          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 40, 40), 2,
      )
      rgba = rend_mesh.render_rgba_vis(
          verts_m.to(device)[None], faces, rt[None], k[None], (height, width),
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
    cv2.imwrite(str(args.out_dir / f"{stem}_top_edge_crop.jpg"), cv2.cvtColor(edge, cv2.COLOR_RGB2BGR))
    cv2.imwrite(
        str(args.out_dir / f"{stem}_mesh_crop.jpg"),
        cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        str(args.out_dir / f"{stem}_mesh_on_photo_crop.jpg"),
        cv2.cvtColor(blend[y0:y1, x0:x1], cv2.COLOR_RGB2BGR),
    )
    return sm

  print(
      f"{args.arch.upper()} 6-DoF only  bite={edge_name}  "
      f"ROI=({x0},{y0})-({x1},{y1})  fdi={arch_fdi}  faces={int(faces.shape[0])}"
  )
  dump_vis("00_init", base0, f"INIT {args.arch} (aa=0)")

  opt = torch.optim.Adam([aa, tvec], lr=args.lr)
  best = {
      "score": float("inf"),
      "aa": aa.detach().clone(),
      "t": tvec.detach().clone(),
  }
  for it in range(args.iters):
    opt.zero_grad()
    sm = losses(apply_m(base0))
    l_reg = aa.pow(2).sum() + tvec.pow(2).sum()
    loss = (
        args.w_over * sm["l_over_px"]
        + args.w_over_col * sm["l_over_col"]
        + args.w_gap * sm["l_gap"]
        + args.w_fp * sm["l_fp"]
        + args.w_fill * sm["l_fill"]
        + args.w_fill_top * sm["l_fill_top"]
        + args.w_cent * sm["l_cent"]
        + args.w_reg * l_reg
    )
    loss.backward()
    opt.step()
    with torch.no_grad():
      aa.clamp_(-0.55, 0.55)
      tvec.clamp_(-0.035, 0.035)
    score = (
        8.0 * float(sm["l_over_px"].detach())
        + 4.0 * float(sm["l_over_col"].detach())
        + 3.0 * float(sm["l_fp"].detach())
        + 1.2 * float(sm["l_gap"].detach())
        + 0.8 * float(sm["l_fill_top"].detach())
        + 0.4 * float(sm["l_fill"].detach())
    )
    if np.isfinite(score) and score < best["score"]:
      best = {
          "score": score,
          "aa": aa.detach().clone(),
          "t": tvec.detach().clone(),
      }
    if it % 25 == 0 or it == args.iters - 1:
      print(
          f"  [{it:03d}] loss={float(loss.detach()):.2f} "
          f"over={float(sm['l_over_px'].detach()):.4f} "
          f"overCol={float(sm['l_over_col'].detach()):.3f} "
          f"gap={float(sm['l_gap'].detach()):.3f} "
          f"fp={float(sm['l_fp'].detach()):.4f} "
          f"fill={float(sm['l_fill'].detach()):.3f} "
          f"fillEdge={float(sm['l_fill_top'].detach()):.3f} "
          f"{edge_name} {float(sm['pred_e'].detach()):.1f}->{float(sm['gt_e'].detach()):.1f}"
      )

  with torch.no_grad():
    aa.copy_(best["aa"])
    tvec.copy_(best["t"])
  verts = apply_m(base0).detach()
  sm_f = dump_vis("01_aligned", verts, f"{args.arch.upper()} 6-DoF")
  print(
      f"FINAL {edge_name} {float(sm_f['pred_e']):.1f}/{float(sm_f['gt_e']):.1f}  "
      f"over={float(sm_f['l_over_px']):.4f} gap={float(sm_f['l_gap']):.3f} "
      f"fp={float(sm_f['l_fp']):.4f} fill={float(sm_f['l_fill']):.3f} "
      f"fillEdge={float(sm_f['l_fill_top']):.3f}"
  )
  print(f"  aa={aa.detach().tolist()}")
  print(f"  t={tvec.detach().tolist()}")

  init_ov = cv2.imread(str(args.out_dir / "00_init_overlay_crop.jpg"))
  aln_ov = cv2.imread(str(args.out_dir / "01_aligned_overlay_crop.jpg"))
  init_ed = cv2.imread(str(args.out_dir / "00_init_top_edge_crop.jpg"))
  aln_ed = cv2.imread(str(args.out_dir / "01_aligned_top_edge_crop.jpg"))
  init_m = cv2.imread(str(args.out_dir / "00_init_mesh_on_photo_crop.jpg"))
  aln_m = cv2.imread(str(args.out_dir / "01_aligned_mesh_on_photo_crop.jpg"))
  if init_ov is not None and aln_ov is not None:
    pair = np.concatenate([init_ov, aln_ov], 1)
    cv2.putText(pair, "init", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(
        pair, "6-DoF", (init_ov.shape[1] + 10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
    )
    cv2.imwrite(str(args.out_dir / "02_init_vs_aligned_overlay.jpg"), pair)
  if init_ed is not None and aln_ed is not None:
    cv2.imwrite(
        str(args.out_dir / "03_init_vs_aligned_top_edge.jpg"),
        np.concatenate([init_ed, aln_ed], 1),
    )
  if init_m is not None and aln_m is not None:
    cv2.imwrite(
        str(args.out_dir / "04_init_vs_aligned_mesh.jpg"),
        np.concatenate([init_m, aln_m], 1),
    )

  aa_np = aa.detach().cpu().numpy()
  t_np = tvec.detach().cpu().numpy()
  zeros3 = np.zeros(3, dtype=np.float32)
  np.savez(
      args.out_dir / f"{args.arch}_rigid6_params.npz",
      aa_up=aa_np if is_upper else zeros3,
      t_up=t_np if is_upper else zeros3,
      aa_low=zeros3 if is_upper else aa_np,
      t_low=zeros3 if is_upper else t_np,
      id_teeth=np.zeros(80, dtype=np.float32),
  )
  summary = {
      "mode": f"{args.arch}_arch_6dof_{edge_name}_edge",
      "arch": args.arch,
      "aa": aa.detach().cpu().tolist(),
      "t": tvec.detach().cpu().tolist(),
      "pred_edge_y": float(sm_f["pred_e"]),
      "gt_edge_y": float(sm_f["gt_e"]),
      "l_over_px": float(sm_f["l_over_px"]),
      "l_over_col": float(sm_f["l_over_col"]),
      "l_gap": float(sm_f["l_gap"]),
      "l_fp": float(sm_f["l_fp"]),
      "l_fill": float(sm_f["l_fill"]),
      "l_fill_edge": float(sm_f["l_fill_top"]),
      "fdi": arch_fdi,
  }
  (args.out_dir / "summary.json").write_text(
      json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
  )
  print("done →", args.out_dir)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())

