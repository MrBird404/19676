#!/usr/bin/env python3
"""Compose frozen GNM face params with optimized teeth params and render.

Face stays frozen (id_head / id_eyes / expression / RT / K).
Teeth replace identity[173:253] (= id_teeth), then apply arch rigid + per-tooth t,
with gum verts hard-locked to the post-rigid pose (same as fit script).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent / "GNM-main"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT))

from gnm.shape import gnm_pytorch, gnm_numpy  # pylint: disable=wrong-import-position
from nvdiffrast_render import NVDiffRenderer  # pylint: disable=wrong-import-position
from fit_gnm_teeth_single import (  # pylint: disable=wrong-import-position
  apply_arch_rigid,
  faces_excluding_groups,
  load_fdi_vertex_map,
  parse_k,
  teeth_faces_from_groups,
)


def build_argparser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument(
      "--face_params", type=Path,
      default=Path("gnm_fit_out/fit_frame_0027_hybrid468/frame_000000_params.npz"),
  )
  p.add_argument(
      "--teeth_params", type=Path,
      default=Path("gnm_fit_out/fit_teeth_frame_0027_shape_refill_lock/fullstruct_crownloss_params.npz"),
  )
  p.add_argument(
      "--fdi_verts", type=Path,
      default=Path("sam3_teeth_out/gnm_bind/gnm_fdi_vertex_indices.npz"),
  )
  p.add_argument("--image", type=Path, default=Path("sam3_teeth_out/frame_0027.png"))
  p.add_argument(
      "--out_dir", type=Path,
      default=Path("gnm_fit_out/fit_teeth_frame_0027_shape_refill_lock/compose_face"),
  )
  p.add_argument("--device", default="cuda")
  return p


def main() -> int:
  args = build_argparser().parse_args()
  device = torch.device(args.device if torch.cuda.is_available() else "cpu")
  if device.type != "cuda":
    raise SystemExit("CUDA required")
  args.out_dir.mkdir(parents=True, exist_ok=True)

  face = np.load(args.face_params)
  teeth = np.load(args.teeth_params)
  image = cv2.cvtColor(cv2.imread(str(args.image), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
  width, height = map(int, face["image_size"])

  gnm = gnm_pytorch.GNM.from_local(
      version=gnm_pytorch.GNMMajorVersion.V3, variant=gnm_pytorch.GNMVariant.HEAD,
  )
  gnm_np = gnm_numpy.GNM.from_local(
      version=gnm_numpy.GNMMajorVersion.V3, variant=gnm_numpy.GNMVariant.HEAD,
  )

  id_head = torch.as_tensor(face["id_head"], dtype=torch.float32)
  id_eyes = torch.as_tensor(face["id_eyes"], dtype=torch.float32)
  id_teeth = torch.as_tensor(teeth["id_teeth"], dtype=torch.float32)
  expression = torch.as_tensor(face["expression"], dtype=torch.float32)
  rotations = torch.as_tensor(face["rotations"], dtype=torch.float32)
  translation = torch.as_tensor(face["translation"], dtype=torch.float32)
  rt = torch.as_tensor(face["RT"], dtype=torch.float32, device=device)
  k = parse_k(face["K"]).to(device)

  channels = [int(x) for x in teeth["channels"]]
  n_verts = int(np.asarray(gnm_np.skinning_weights).shape[-1])
  per_fdi, _, _ = load_fdi_vertex_map(args.fdi_verts, n_verts, channels)
  per_fdi_t = {f: torch.as_tensor(idx, dtype=torch.long) for f, idx in per_fdi.items()}

  arch_up = np.asarray(gnm_np.vertex_group_indices("upper_teeth_and_gums"), dtype=np.int64)
  arch_low = np.asarray(gnm_np.vertex_group_indices("lower_teeth_and_gums"), dtype=np.int64)
  arch_up_t = torch.as_tensor(arch_up, dtype=torch.long)
  arch_low_t = torch.as_tensor(arch_low, dtype=torch.long)
  crown_all = np.unique(np.concatenate([per_fdi[f] for f in channels]))
  gum = np.setdiff1d(np.unique(np.concatenate([arch_up, arch_low])), crown_all)
  gum_t = torch.as_tensor(gum, dtype=torch.long)

  aa_up = torch.as_tensor(teeth["aa_up"], dtype=torch.float32)
  t_up = torch.as_tensor(teeth["t_up"], dtype=torch.float32)
  aa_low = torch.as_tensor(teeth["aa_low"], dtype=torch.float32)
  t_low = torch.as_tensor(teeth["t_low"], dtype=torch.float32)
  tooth_t = torch.as_tensor(teeth["tooth_t"], dtype=torch.float32)

  def posed(id_t: torch.Tensor) -> torch.Tensor:
    identity = torch.cat([id_head, id_eyes, id_t], dim=0)
    return gnm(
        identity[None], expression[None], rotations[None], translation[None],
    )[0]

  def apply_m(verts: torch.Tensor) -> torch.Tensor:
    v = apply_arch_rigid(verts, arch_up_t, aa_up, t_up, pivot_up)
    return apply_arch_rigid(v, arch_low_t, aa_low, t_low, pivot_low)

  with torch.no_grad():
    base0 = posed(torch.zeros(80)).detach()
  pivot_up = base0[arch_up_t].mean(0)
  pivot_low = base0[arch_low_t].mean(0)

  with torch.no_grad():
    # Face mesh (default teeth=0, no arch rigid) — for reference.
    verts_face = posed(torch.zeros(80)).detach()
    # Combined: optimized teeth shape + arch rigid + per-tooth + gum lock.
    verts = apply_m(posed(id_teeth))
    for i, f in enumerate(channels):
      verts = verts.clone()
      verts[per_fdi_t[f]] = verts[per_fdi_t[f]] + tooth_t[i]
    v_rigid = apply_m(base0)
    verts = verts.clone()
    verts[gum_t] = v_rigid[gum_t]
    for name in ("occ_crown", "lock_verts"):
      if name in teeth.files:
        idx = torch.as_tensor(teeth[name], dtype=torch.long)
        if idx.numel() > 0:
          verts[idx] = v_rigid[idx]

  # Merged identity for saving / downstream.
  identity_merged = np.concatenate(
      [face["id_head"], face["id_eyes"], teeth["id_teeth"].astype(np.float32)],
      axis=0,
  )
  np.savez_compressed(
      args.out_dir / "face_teeth_merged_params.npz",
      # face (frozen)
      id_head=face["id_head"],
      id_eyes=face["id_eyes"],
      expression=face["expression"],
      rotations=face["rotations"],
      translation=face["translation"],
      K=face["K"],
      RT=face["RT"],
      image_size=face["image_size"],
      # teeth (optimized)
      id_teeth=teeth["id_teeth"],
      aa_up=teeth["aa_up"],
      t_up=teeth["t_up"],
      aa_low=teeth["aa_low"],
      t_low=teeth["t_low"],
      tooth_t=teeth["tooth_t"],
      channels=np.asarray(channels),
      # convenience
      identity=identity_merged,
      note=np.asarray(
          "identity uses optimized id_teeth; still apply aa/t + tooth_t + gum lock at runtime",
      ),
  )

  # Drop tongue + mouth_sock so oral cavity is empty and teeth are clearly visible.
  faces_head_np = faces_excluding_groups(gnm_np, {"tongue", "mouth_sock"})
  faces_head = torch.as_tensor(faces_head_np, dtype=torch.int32, device=device)
  faces_teeth = torch.as_tensor(
      teeth_faces_from_groups(gnm_np, ["upper_teeth_and_gums", "lower_teeth_and_gums"]),
      dtype=torch.int32, device=device,
  )

  rend = NVDiffRenderer(
      use_opengl=False, lighting_type="front", lighting_space="camera",
      flip_y_raster=False,
  ).to(device)

  def render_rgba(verts_m: torch.Tensor, faces: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
      out = rend.render_rgba_vis(
          verts_m.to(device)[None], faces, rt[None], k[None], (height, width),
          background_color=[0.0, 0.0, 0.0],
      )
      rgba = out["rgba"][0].detach().float().cpu().numpy()
    rgb = np.clip(rgba[..., :3] * 255, 0, 255).astype(np.uint8)
    return rgb, rgba[..., 3]

  # Full head with personalized teeth (no tongue / mouth_sock).
  head_rgb, head_a = render_rgba(verts, faces_head)
  teeth_rgb, teeth_a = render_rgba(verts, faces_teeth)
  face_only_rgb, face_only_a = render_rgba(verts_face, faces_head)

  def composite(bg: np.ndarray, rgb: np.ndarray, a: np.ndarray, w: float = 0.7) -> np.ndarray:
    aa = a[..., None]
    return np.clip(
        bg.astype(np.float32) * (1.0 - w * aa) + rgb.astype(np.float32) * (w * aa),
        0, 255,
    ).astype(np.uint8)

  def photo_bg_black_oral(a_head: np.ndarray, a_teeth: np.ndarray) -> np.ndarray:
    """Black-fill oral cavity holes so photo tongue does not show through."""
    bg = image.copy()
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    oral = cv2.dilate((a_teeth > 0.12).astype(np.uint8), ker) > 0
    hole = oral & (a_head < 0.28)
    bg[hole] = (12, 12, 12)
    return bg

  bg_head = photo_bg_black_oral(head_a, teeth_a)
  bg_face = photo_bg_black_oral(face_only_a, teeth_a)
  on_photo_head = composite(bg_head, head_rgb, head_a, 0.85)
  on_photo_teeth = composite(bg_head, teeth_rgb, teeth_a, 0.8)
  side = np.concatenate([
      composite(bg_face, face_only_rgb, face_only_a, 0.85),
      on_photo_head,
  ], 1)
  cv2.putText(side, "face only (id_teeth=0), no tongue", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
  cv2.putText(
      side, "face + optimized teeth, no tongue", (image.shape[1] + 20, 40),
      cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
  )

  # ROI crop around mouth from teeth alpha
  ys, xs = np.where(teeth_a > 0.2)
  pad = 50
  y0, y1 = max(0, int(ys.min()) - pad), min(height, int(ys.max()) + pad)
  x0, x1 = max(0, int(xs.min()) - pad), min(width, int(xs.max()) + pad)

  def save_rgb(path: Path, rgb: np.ndarray) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print("wrote", path)

  # Mesh-only (black bg) — clearest no-tongue view.
  save_rgb(args.out_dir / "compose_head_mesh.jpg", head_rgb)
  save_rgb(args.out_dir / "compose_head_mesh_no_tongue.jpg", head_rgb)
  save_rgb(args.out_dir / "compose_head_mesh_crop_no_tongue.jpg", head_rgb[y0:y1, x0:x1])
  save_rgb(args.out_dir / "compose_head_on_photo.jpg", on_photo_head)
  save_rgb(args.out_dir / "compose_head_on_photo_no_tongue.jpg", on_photo_head)
  save_rgb(args.out_dir / "compose_teeth_on_photo.jpg", on_photo_teeth)
  save_rgb(args.out_dir / "compose_face_vs_teeth_on_photo.jpg", side)
  save_rgb(args.out_dir / "compose_face_vs_teeth_on_photo_no_tongue.jpg", side)
  save_rgb(args.out_dir / "compose_head_on_photo_crop.jpg", on_photo_head[y0:y1, x0:x1])
  save_rgb(args.out_dir / "compose_head_on_photo_crop_no_tongue.jpg", on_photo_head[y0:y1, x0:x1])
  save_rgb(args.out_dir / "compose_teeth_on_photo_crop.jpg", on_photo_teeth[y0:y1, x0:x1])
  save_rgb(
      args.out_dir / "compose_face_vs_teeth_crop.jpg",
      np.concatenate([
          composite(bg_face, face_only_rgb, face_only_a, 0.85)[y0:y1, x0:x1],
          on_photo_head[y0:y1, x0:x1],
      ], 1),
  )
  save_rgb(
      args.out_dir / "compose_face_vs_teeth_crop_no_tongue.jpg",
      np.concatenate([
          composite(bg_face, face_only_rgb, face_only_a, 0.85)[y0:y1, x0:x1],
          on_photo_head[y0:y1, x0:x1],
      ], 1),
  )
  # Mesh-only side-by-side on black.
  mesh_side = np.concatenate([
      face_only_rgb[y0:y1, x0:x1],
      head_rgb[y0:y1, x0:x1],
  ], 1)
  save_rgb(args.out_dir / "compose_face_vs_teeth_mesh_crop_no_tongue.jpg", mesh_side)

  try:
    import trimesh
    trimesh.Trimesh(
        vertices=verts.cpu().numpy(), faces=faces_head_np, process=False,
    ).export(args.out_dir / "compose_full_head_with_teeth_no_tongue.obj")
    trimesh.Trimesh(
        vertices=verts.cpu().numpy(),
        faces=teeth_faces_from_groups(
            gnm_np, ["upper_teeth_and_gums", "lower_teeth_and_gums"],
        ),
        process=False,
    ).export(args.out_dir / "compose_teeth_and_gums_only.obj")
    print("wrote objs")
  except Exception as e:  # noqa: BLE001
    print("obj export skipped:", e)

  print("done →", args.out_dir)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
