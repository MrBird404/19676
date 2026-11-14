#!/usr/bin/env python3
"""VHAP-main photometric stages on a finished GNM landmark fit.

Color path matches VHAP: painted/authored skin base + learnable residual
(``tex_extra``), SH lights, L1 on covered pixels, white bg, no silhouette.
Albedo is clamped to ``[1e-4, 1-1e-4]``. Jaw landmarks stay off. Eyeball
joints stay at the pre-photo SAM aim.

Deferred (not learned here): eyes / teeth / gums / tongue keep authored UV
sheets; eye color and pupil size are later stages.

Single frame init stages:
  rgb_init_texture  cam, shape, skin residual, lights; jaw ray-contour (two-sided)
  rgb_init_all      + pose, neck/jaw, face expr PCA150; tex/lights frozen
  rgb_init_offset   + static offset (teeth, eyes, scalp, ears pinned)

``align_texture_except`` includes hair / boundary / neck like VHAP (UV
detach; RGB still updates the skin residual). Ears and the shoulder cut
stay out of boundary alignment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh

ROOT = Path(__file__).resolve().parent
REPO = next(p for p in [ROOT, *ROOT.parents] if (p / "GNM-main").is_dir()) / "GNM-main"
UV_MASKS = ROOT.parent / "notes" / "gnm_uv" / "uv_masks.npz"
UV_DIR = ROOT.parent / "notes" / "gnm_uv" / "UV"
# Each part has its own [0, 1] sheet, so they are packed into tiles.
# Only the skin tile is optimized.
_ATLAS_COLS = 4
_ATLAS_ROWS = 2
_TILES = {
    "skin": (0, 0),
    "eye_left": (1, 0),
    "eye_right": (2, 0),
    "teeth": (3, 0),
    "gum_upper": (0, 1),
    "gum_lower": (1, 1),
    "tongue": (2, 1),
}
_TILE_NAMES = list(_TILES)
# Same order as VHAP-main tex_clusters. Renderer id 1 is "no cluster".
_CLUSTER = {
    "skin": 2,
    "hair": 3,
    "boundary": 4,
    "lips_tight": 5,
    "teeth": 6,
    "sclerae": 7,
    "irises": 8,
    "neck": 9,
    "ears": 10,
    "occluded": 11,
}
if str(REPO) not in sys.path:
  sys.path.insert(0, str(REPO))
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))


def scale_for_photo(height: int, width: int, max_side: int) -> tuple[int, int, float]:
  long_side = max(height, width)
  if long_side <= max_side:
    return height, width, 1.0
  s = max_side / float(long_side)
  return max(int(round(height * s)), 1), max(int(round(width * s)), 1), s


def uv_from_triangles(gnm_np) -> tuple[np.ndarray, np.ndarray]:
  tuv = np.asarray(gnm_np.triangle_uvs, dtype=np.float32)
  faces_uv = np.arange(tuv.shape[0] * 3, dtype=np.int32).reshape(-1, 3)
  verts_uv = tuv.reshape(-1, 2)
  return verts_uv, faces_uv


def _sheet_rgb(name: str, size: int) -> np.ndarray:
  """PNG row 0 is v=1. Returned array is nvdiffrast order, row 0 = v=0."""
  path = UV_DIR / f"{name}.png"
  bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
  if bgr is None:
    raise FileNotFoundError(path)
  if bgr.shape[0] != size or bgr.shape[1] != size:
    bgr = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
  rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
  return np.flipud(rgb)


def material_ids(gnm_np, faces: np.ndarray) -> np.ndarray:
  """0 = skin. Teeth win over gums, gums and tongue win over the eyeballs."""
  n_verts = int(np.asarray(gnm_np(None, None, None, None)).shape[0])
  mid = np.zeros(faces.shape[0], dtype=np.int64)
  name_of = {name: i for i, name in enumerate(_TILE_NAMES)}

  def paint(group: str, tile: str, need: int = 2) -> None:
    fid = _faces_with(faces, _group_ids(gnm_np, group), n_verts, need)
    mid[fid] = name_of[tile]

  paint("left_eye", "eye_left")
  paint("right_eye", "eye_right")
  paint("upper_teeth_and_gums", "gum_upper")
  paint("lower_teeth_and_gums", "gum_lower")
  paint("tongue", "tongue")
  paint("teeth", "teeth", need=3)
  counts = {name: int((mid == i).sum()) for i, name in enumerate(_TILE_NAMES)}
  print(
      "[photo] uv tiles  " + "  ".join(f"{name}={counts[name]}" for name in _TILE_NAMES),
      flush=True,
  )
  return mid


def pack_material_uvs(verts_uv: np.ndarray, faces_uv: np.ndarray, material: np.ndarray) -> np.ndarray:
  packed = np.array(verts_uv, dtype=np.float32, copy=True)
  for tile, (col, row) in _TILES.items():
    fid = np.nonzero(material == _TILE_NAMES.index(tile))[0]
    if fid.size == 0:
      continue
    idx = faces_uv[fid].reshape(-1)
    packed[idx, 0] = (packed[idx, 0] + col) / _ATLAS_COLS
    packed[idx, 1] = (packed[idx, 1] + row) / _ATLAS_ROWS
  return packed


# Gutter outside the skin UV island. The crown pole sits on the last painted
# texel of skin.png, and linear-mipmap sampling otherwise reads the black empty
# texels beside the seam.
_UV_PAD = 8


def _shift(img: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
  out = img.new_zeros(img.shape)
  h, w = img.shape[-2:]
  ys, yd = max(dy, 0), max(-dy, 0)
  xs, xd = max(dx, 0), max(-dx, 0)
  hh, ww = h - abs(dy), w - abs(dx)
  out[..., yd:yd + hh, xd:xd + ww] = img[..., ys:ys + hh, xs:xs + ww]
  return out


def _skin_cover_mask(face_uv: np.ndarray, size: int) -> np.ndarray:
  """Texels covered by skin triangles. Row 0 is v=0, same layout as _sheet_rgb."""
  mask = np.zeros((size, size), np.uint8)
  x = np.clip(np.rint(face_uv[..., 0] * (size - 1)), 0, size - 1).astype(np.int32)
  y = np.clip(np.rint((1.0 - face_uv[..., 1]) * (size - 1)), 0, size - 1).astype(np.int32)
  polys = np.stack([x, y], axis=-1)
  cv2.fillPoly(mask, [p for p in polys], 1)
  return np.ascontiguousarray(np.flipud(mask)) > 0


def _dilate_albedo(albedo: torch.Tensor, valid: torch.Tensor, steps: int) -> torch.Tensor:
  """Copy island colors outward so the filter footprint stays off the black gutter."""
  if steps <= 0:
    return albedo
  out = albedo
  filled = valid
  for _ in range(steps):
    acc = albedo.new_zeros(albedo.shape)
    cnt = albedo.new_zeros(valid.shape)
    for dy in (-1, 0, 1):
      for dx in (-1, 0, 1):
        if dy == 0 and dx == 0:
          continue
        acc = acc + _shift(out, dy, dx)
        cnt = cnt + _shift(filled.to(albedo.dtype), dy, dx)
    take = (cnt > 0) & ~filled
    out = torch.where(take, acc / cnt.clamp(min=1.0), out)
    filled = filled | take
  return out


def _tv(tex: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
  y = (tex[:, :, :-1, :] - tex[:, :, 1:, :]).pow(2)
  x = (tex[:, :, :, :-1] - tex[:, :, :, 1:]).pow(2)
  if valid is None:
    return y.mean() + x.mean()
  my = (valid[:-1, :] & valid[1:, :]).to(tex.dtype)
  mx = (valid[:, :-1] & valid[:, 1:]).to(tex.dtype)
  nch = tex.shape[1]
  return (y * my).sum() / (my.sum() * nch).clamp(min=1.0) + (x * mx).sum() / (mx.sum() * nch).clamp(min=1.0)


def _group_ids(gnm_np, name: str) -> np.ndarray:
  names = set(gnm_np.vertex_group_names)
  if name not in names:
    return np.zeros(0, dtype=np.int64)
  return np.asarray(gnm_np.vertex_group_indices(name), dtype=np.int64)


def _faces_with(faces: np.ndarray, verts: np.ndarray, n_verts: int, need: int = 2) -> np.ndarray:
  if verts.size == 0:
    return np.zeros(0, dtype=np.int64)
  flag = np.zeros(n_verts, dtype=np.uint8)
  flag[verts] = 1
  return np.nonzero(flag[faces].sum(axis=1) >= need)[0].astype(np.int64)


def _boundary_vertices(faces: np.ndarray, n_verts: int) -> np.ndarray:
  edges: dict[tuple[int, int], int] = {}
  for tri in faces:
    a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
    for i, j in ((a, b), (b, c), (c, a)):
      e = (i, j) if i < j else (j, i)
      edges[e] = edges.get(e, 0) + 1
  mark = np.zeros(n_verts, dtype=bool)
  for (i, j), count in edges.items():
    if count == 1:
      mark[i] = True
      mark[j] = True
  return np.nonzero(mark)[0].astype(np.int64)


def _edge_index(faces: np.ndarray) -> np.ndarray:
  pairs = []
  seen = set()
  for tri in faces:
    a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
    for i, j in ((a, b), (b, c), (c, a)):
      e = (i, j) if i < j else (j, i)
      if e not in seen:
        seen.add(e)
        pairs.append(e)
  return np.asarray(pairs, dtype=np.int64)


def _centroid_uv(gnm_np) -> np.ndarray:
  return np.asarray(gnm_np.triangle_uvs, dtype=np.float32).mean(axis=1)


def _hits_mask(mask: np.ndarray, uv: np.ndarray) -> np.ndarray:
  """PNG row 0 is v=1. uv is per-face centroid."""
  height, width = mask.shape
  x = np.clip(np.rint(uv[:, 0] * (width - 1)).astype(np.int32), 0, width - 1)
  y = np.clip(np.rint((1.0 - uv[:, 1]) * (height - 1)).astype(np.int32), 0, height - 1)
  return np.asarray(mask, dtype=bool)[y, x]


def _tex_mask(mask: np.ndarray, size: int, device: torch.device) -> torch.Tensor:
  """nvdiffrast row 0 is v=0, so the atlas PNG is flipped before resize."""
  flipped = np.flipud(np.asarray(mask, dtype=bool))
  small = cv2.resize(
      flipped.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST,
  ) > 0
  return torch.as_tensor(np.ascontiguousarray(small), device=device)


def build_photo_regions(gnm_np, faces: np.ndarray) -> dict:
  """Scalp above the forehead is hair and stays in the photometric fit. Ears stay out."""
  rest = np.asarray(gnm_np(None, None, None, None), dtype=np.float32)
  n_verts = int(rest.shape[0])
  skin = _group_ids(gnm_np, "skin")
  hockey = set(_group_ids(gnm_np, "hockey_mask").tolist())
  ears = _group_ids(gnm_np, "ears")
  forehead = _group_ids(gnm_np, "forehead_region")
  y_cap = float(np.percentile(rest[forehead, 1], 80)) if forehead.size else 0.35
  neck_y = 0.18
  scalp = np.zeros(n_verts, dtype=bool)
  for i in skin.tolist():
    y = float(rest[i, 1])
    z = float(rest[i, 2])
    if y < neck_y:
      continue
    if i in hockey and y < y_cap:
      continue
    if y >= y_cap or z < 0.04:
      scalp[i] = True
  occluded = np.zeros(n_verts, dtype=bool)
  occluded[ears] = True
  boundary = _boundary_vertices(faces, n_verts)
  bottom = boundary[rest[boundary, 1] < 0.12]
  neck = skin[rest[skin, 1] < neck_y]
  eyes = _group_ids(gnm_np, "eyes")
  teeth_pin = np.unique(np.concatenate([
      _group_ids(gnm_np, "teeth"),
      _group_ids(gnm_np, "gums"),
      _group_ids(gnm_np, "tongue"),
  ]))
  pin = np.unique(np.concatenate([
      np.nonzero(occluded)[0],
      bottom,
      eyes,
      teeth_pin,
  ])).astype(np.int64)
  occ_f = _faces_with(faces, np.nonzero(occluded)[0], n_verts)
  neck_f = _faces_with(faces, neck, n_verts)
  bottom_f = _faces_with(faces, bottom, n_verts)
  color_except = np.unique(np.concatenate([occ_f, bottom_f])).astype(np.int64)

  if not UV_MASKS.is_file():
    raise FileNotFoundError(f"UV masks missing: {UV_MASKS}")
  masks = np.load(UV_MASKS)
  uv = _centroid_uv(gnm_np)
  skin_f = _faces_with(faces, skin, n_verts, need=2)
  on_skin = np.zeros(faces.shape[0], dtype=bool)
  on_skin[skin_f] = True

  def masked(name: str) -> np.ndarray:
    hit = _hits_mask(masks[name], uv) & on_skin
    return np.nonzero(hit)[0].astype(np.int64)

  hair_f = np.unique(np.concatenate([
      masked("hair"),
      _faces_with(faces, np.nonzero(scalp)[0], n_verts),
  ])).astype(np.int64)
  boundary_f = masked("boundary")
  boundary_f = np.setdiff1d(boundary_f, hair_f, assume_unique=False)
  tight_f = masked("lips_tight")
  neck_uv_f = masked("neck")
  neck_f = np.unique(np.concatenate([neck_f, neck_uv_f])).astype(np.int64)
  neck_f = np.setdiff1d(neck_f, hair_f, assume_unique=False)
  # VHAP: hair/boundary/neck in align_texture_except (texc detach).
  texture_except = np.unique(np.concatenate([
      occ_f, neck_f, bottom_f, boundary_f, hair_f,
  ])).astype(np.int64)

  teeth_f = _faces_with(faces, _group_ids(gnm_np, "teeth"), n_verts, need=3)
  sclera_f = _faces_with(faces, _group_ids(gnm_np, "scleras"), n_verts, need=3)
  iris_ids = np.unique(np.concatenate([
      _group_ids(gnm_np, "irises"),
      _group_ids(gnm_np, "pupils"),
  ]))
  iris_f = _faces_with(faces, iris_ids, n_verts, need=3)
  ear_f = _faces_with(faces, ears, n_verts, need=2)

  cid = np.ones(faces.shape[0], dtype=np.int64)
  def paint(fids: np.ndarray, cluster: int) -> None:
    if fids.size:
      cid[fids] = cluster
  paint(skin_f, _CLUSTER["skin"])
  paint(neck_f, _CLUSTER["neck"])
  paint(hair_f, _CLUSTER["hair"])
  paint(boundary_f, _CLUSTER["boundary"])
  paint(tight_f, _CLUSTER["lips_tight"])
  paint(teeth_f, _CLUSTER["teeth"])
  paint(sclera_f, _CLUSTER["sclerae"])
  paint(iris_f, _CLUSTER["irises"])
  paint(ear_f, _CLUSTER["ears"])
  paint(occ_f, _CLUSTER["occluded"])

  boundary_v = np.unique(faces[boundary_f]).astype(np.int64) if boundary_f.size else np.zeros(0, dtype=np.int64)
  occ_v = np.nonzero(occluded)[0].astype(np.int64)
  # Bottom stays excluded on every stage so the neck cut is not pulled to the frame.
  # VHAP rgb_init_texture: align_boundary_except = hair, boundary (+ occluded).
  # Later stages keep face boundary in AA (silhouette can move); jaw ray loss holds outline.
  hair_v = np.unique(faces[hair_f]).astype(np.int64) if hair_f.size else np.zeros(0, dtype=np.int64)
  bound_texture = np.unique(np.concatenate([occ_v, bottom, boundary_v, hair_v])).astype(np.int64)
  bound_later = np.unique(np.concatenate([occ_v, bottom])).astype(np.int64)
  counts = {name: int((cid == cid_i).sum()) for name, cid_i in _CLUSTER.items()}
  print(
      f"[photo] regions  verts={n_verts}  hair_faces={hair_f.size}  scalp_verts={int(scalp.sum())}  "
      f"ears={int(occluded.sum())}  bottom={bottom.size}  neck_faces={neck_f.size}  pin={pin.size}  "
      f"texture_except={texture_except.size}  y_cap={y_cap:.3f}",
      flush=True,
  )
  print(
      "[photo] clusters  "
      + "  ".join(f"{name}={counts[name]}" for name in (
          "skin", "hair", "boundary", "lips_tight", "teeth", "sclerae", "irises",
      )),
      flush=True,
  )
  return {
      "n_verts": n_verts,
      "pin": pin,
      "color_except": color_except,
      "texture_except": texture_except,
      "bound_texture": bound_texture,
      "bound_later": bound_later,
      "fid2cid": cid,
      "edges": _edge_index(faces),
      "mask_teeth": np.asarray(masks["teeth"], dtype=bool),
      "mask_sclera": np.asarray(masks["sclera"], dtype=bool),
  }


def dump_photo_preview(out_dir: Path, tag: str, gt_rgb: np.ndarray, pred_rgba: np.ndarray) -> None:
  pred = np.clip(pred_rgba[..., :3], 0, 1)
  alpha = np.clip(pred_rgba[..., 3], 0, 1)
  overlay = gt_rgb * (1.0 - alpha[..., None]) + pred * alpha[..., None]
  err = np.abs(gt_rgb - pred).mean(-1)
  err_u8 = cv2.applyColorMap(np.clip(err * 4.0 * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
  row = np.concatenate([
      np.clip(gt_rgb * 255, 0, 255).astype(np.uint8),
      np.clip(overlay * 255, 0, 255).astype(np.uint8),
      np.clip(pred * 255, 0, 255).astype(np.uint8),
      cv2.cvtColor(err_u8, cv2.COLOR_BGR2RGB),
  ], axis=1)
  path = out_dir / f"stage_{tag}_photo.jpg"
  cv2.imwrite(str(path), cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
  render_path = out_dir / f"stage_{tag}_render.jpg"
  cv2.imwrite(str(render_path), cv2.cvtColor(np.clip(pred * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
  print(f"  [dump] {path.name}  {render_path.name}", flush=True)


def run_photometric(
    *,
    args,
    state,
    gnm,
    gnm_np,
    posed_verts_world,
    sample_lmk3d,
    project_xy,
    dump_stage,
    stage_bgr: np.ndarray,
    gt_xy: np.ndarray,
    gt_t: torch.Tensor,
    width: int,
    height: int,
    mouth_locked: bool,
    metric_reg,
    weighted_landmark_loss,
    w_lmk: np.ndarray,
    faces: np.ndarray,
) -> dict:
  del gnm, mouth_locked
  if stage_bgr is None:
    print("[photo] skip: no frame image", flush=True)
    return {}
  if not torch.cuda.is_available():
    print("[photo] skip: CUDA required for nvdiffrast photometric", flush=True)
    return {}

  from nvdiffrast_render import NVDiffRenderer

  device = torch.device("cuda")
  ph, pw, scale = scale_for_photo(height, width, int(args.photo_max_side))
  bgr = stage_bgr
  if scale != 1.0:
    bgr = cv2.resize(stage_bgr, (pw, ph), interpolation=cv2.INTER_AREA)
  rgb_np = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
  gt_rgb = torch.as_tensor(rgb_np, dtype=torch.float32, device=device)[None]
  if scale == 1.0:
    k4 = state.camera_k(width, height).detach()
  else:
    k4 = state.camera_k(width, height).detach() * scale
    # cx, cy are the last two; focal scales, and so do the principals.
  k_t = k4.to(device)[None]
  faces_t = torch.as_tensor(faces, dtype=torch.int32, device=device)
  verts_uv, faces_uv = uv_from_triangles(gnm_np)
  material = material_ids(gnm_np, faces)
  skin_face_uv = verts_uv[faces_uv[material == 0]]
  verts_uv = pack_material_uvs(verts_uv, faces_uv, material)
  verts_uv_t = torch.as_tensor(verts_uv, dtype=torch.float32, device=device)
  faces_uv_t = torch.as_tensor(faces_uv, dtype=torch.int32, device=device)

  regions = build_photo_regions(gnm_np, faces)
  pin = torch.as_tensor(regions["pin"], dtype=torch.long)
  edges = torch.as_tensor(regions["edges"], dtype=torch.long)
  tex_except = torch.as_tensor(regions["texture_except"], dtype=torch.long, device=device)
  bound_texture = torch.as_tensor(regions["bound_texture"], dtype=torch.long, device=device)
  bound_later = torch.as_tensor(regions["bound_later"], dtype=torch.long, device=device)
  # Lower-face verts for one-sided outer-contour vs detected jaw points 0..16.
  jaw_vid_np = np.unique(np.concatenate([
      _group_ids(gnm_np, "chin_region"),
      _group_ids(gnm_np, "left_cheek_region"),
      _group_ids(gnm_np, "right_cheek_region"),
      _group_ids(gnm_np, "left_parotid_region"),
      _group_ids(gnm_np, "right_parotid_region"),
  ])).astype(np.int64)
  jaw_vids = torch.as_tensor(jaw_vid_np, dtype=torch.long)
  tile = int(args.photo_tex_res)
  skin_cover = torch.as_tensor(_skin_cover_mask(skin_face_uv, tile), device=device)
  print(
      f"[photo] skin uv cover {int(skin_cover.sum())} texels, gutter {_UV_PAD}px",
      flush=True,
  )
  # VHAP-style: authored skin base (fixed) + residual tex_extra (learned).
  skin_rgb = np.clip(_sheet_rgb("skin", tile), 1e-4, 1.0 - 1e-4)
  skin_base = torch.as_tensor(
      skin_rgb.transpose(2, 0, 1)[None], dtype=torch.float32, device=device,
  )
  tex_res = torch.nn.Parameter(torch.zeros(1, 3, tile, tile, dtype=torch.float32, device=device))
  atlas_h = _ATLAS_ROWS * tile
  atlas_w = _ATLAS_COLS * tile
  fixed = np.zeros((1, 3, atlas_h, atlas_w), dtype=np.float32)
  for name, (col, row) in _TILES.items():
    if name == "skin":
      continue
    # Deferred parts: authored sheets only, never optimized here.
    sheet = _sheet_rgb(name, tile).transpose(2, 0, 1)
    y0, x0 = row * tile, col * tile
    fixed[0, :, y0:y0 + tile, x0:x0 + tile] = sheet
  fixed_atlas = torch.as_tensor(fixed, device=device)
  # VHAP lights_uniform: band0 = sqrt(4*pi), rest 0.
  lights_uniform = torch.zeros(1, 9, 3, dtype=torch.float32, device=device)
  lights_uniform[:, 0, :] = float(np.sqrt(4.0 * np.pi))
  lights = torch.nn.Parameter(lights_uniform.clone())
  n_verts = int(regions["n_verts"])
  offset = torch.nn.Parameter(torch.zeros(n_verts, 3, dtype=torch.float32))

  w_photo = np.array(w_lmk, dtype=np.float32, copy=True)
  if w_photo.shape[0] >= 17:
    w_photo[:17] = 0.0
  conf = torch.as_tensor(w_photo, dtype=torch.float32)
  print(
      f"[photo] VHAP color  skin=base+residual  deferred eyes/teeth/gums/tongue  "
      f"white bg  {pw}x{ph}  jaw weight 0  "
      f"nose mean w={float(w_photo[27:36].mean()):.1f}  no silhouette loss",
      flush=True,
  )

  renderer = NVDiffRenderer(
      use_opengl=False,
      lighting_type="SH",
      lighting_space="world",
      disturb_rate_fg=0.5,
      disturb_rate_bg=0.5,
      fid2cid=torch.as_tensor(regions["fid2cid"], dtype=torch.long),
      shade_smooth=True,
      flip_y_raster=False,
  ).to(device)
  with torch.no_grad():
    init_appearance = getattr(args, "init_appearance", None)
    if init_appearance is not None:
      saved = np.load(init_appearance)
      if "residual" in saved.files:
        res = np.asarray(saved["residual"], dtype=np.float32)
      else:
        albedo = np.clip(saved["albedo"].astype(np.float32), 1e-4, 1.0 - 1e-4)
        if albedo.ndim == 3 and albedo.shape[0] != 3:
          albedo = albedo.transpose(2, 0, 1)
        if albedo.ndim == 3:
          albedo = albedo[None]
        base_np = skin_base.detach().cpu().numpy()
        res = albedo - base_np
      if res.ndim == 3:
        res = res[None]
      if res.shape[-2:] != (tile, tile):
        raise ValueError(f"saved residual/albedo {res.shape} does not match skin tile {tile}")
      tex_res.copy_(torch.as_tensor(res, device=tex_res.device))
      lights.copy_(torch.as_tensor(saved["lights"].astype(np.float32), device=lights.device))
      print(f"[photo] loaded residual+SH from {init_appearance}", flush=True)

  stats: dict = {"photo_size": [pw, ph], "photo_scale": scale}

  def world_verts() -> torch.Tensor:
    return posed_verts_world() + offset

  def skin_albedo() -> torch.Tensor:
    # VHAP: painted/authored base + residual, then clamp.
    alb = (skin_base + tex_res).clamp(1e-4, 1.0 - 1e-4)
    return _dilate_albedo(alb, skin_cover, _UV_PAD)

  def skin_atlas() -> torch.Tensor:
    atlas = fixed_atlas.clone()
    y0, x0 = _TILES["skin"][1] * tile, _TILES["skin"][0] * tile
    atlas[:, :, y0:y0 + tile, x0:x0 + tile] = skin_albedo()
    return atlas

  def render_pack(disturb: bool, bound_ids: torch.Tensor):
    verts = world_verts()
    verts_g = verts.to(device)[None].contiguous()
    rt = state.camera_rt().to(device)[None]
    rast = renderer.rasterize(
        verts_g, faces_t, rt, k_t, (ph, pw), use_cache=False, require_grad=True
    )
    out = renderer.render_rgba(
        rast,
        verts_g[0],
        faces_t,
        verts_uv_t,
        faces_uv_t,
        skin_atlas(),
        lights.to(device),
        background_color=[1.0, 1.0, 1.0],
        align_texture_except_fid=tex_except,
        align_boundary_except_vid=bound_ids,
        enable_disturbance=disturb,
    )
    return verts, rast, out

  def jaw_outer_loss(verts_world: torch.Tensor) -> torch.Tensor:
    """Ray-radius lower-outline vs detected jaw points (two-sided).

    Mid-jaw 3..13 only (skip near-ear 0..2 / 14..16). Detected 2D points are
    visible-face outline, not anatomical iBUG. Along each ray from a face
    center, mesh extent should match that point (±slack deadzone). Not
    point-to-point vertex↔landmark.
    """
    w = float(getattr(args, "w_jaw_outer", 20.0))
    # Mid contour only: 3..13 inclusive (11 pts). Need at least index 13.
    jaw_ids = list(range(3, 14))
    if w <= 0.0 or jaw_vids.numel() == 0 or gt_t.shape[0] <= jaw_ids[-1]:
      return verts_world.new_zeros(())
    slack = float(getattr(args, "jaw_outer_slack", 2.0))
    # Face center: nose tip if present, else mean of non-jaw landmarks.
    if gt_t.shape[0] > 30:
      center = gt_t[30]
    else:
      center = gt_t[17:].mean(dim=0)
    mesh_xy = project_xy(verts_world[jaw_vids])
    rel = mesh_xy - center
    loss = verts_world.new_zeros(())
    n_used = 0
    for i in jaw_ids:
      d = gt_t[i] - center
      gt_r = torch.linalg.norm(d)
      if float(gt_r) < 1.0:
        continue
      u = d / gt_r
      mesh_r = (rel * u.unsqueeze(0)).sum(dim=-1).max()
      # Outside and inside: stick outline to detected radius (±slack).
      err = mesh_r - gt_r
      loss = loss + torch.relu(err - slack).pow(2) + torch.relu(-err - slack).pow(2)
      n_used += 1
    if n_used == 0:
      return verts_world.new_zeros(())
    return w * loss / float(n_used)

  def photo_loss(rast, out, w_rgb: float, w_lmk_term: float, use_offset_reg: bool):
    rgba = out["rgba"]
    pred = rgba[..., :3]
    # Every covered pixel paints the texture, including cap, neck, and clothes.
    vis = (rgba[..., 3:] > 0).detach()
    rgb_loss = ((pred - gt_rgb).abs() * vis).sum() / (vis.sum() * 3).clamp(min=1.0)
    verts_w = world_verts()
    pred_xy = project_xy(sample_lmk3d(verts_w))
    lmk_loss = weighted_landmark_loss(pred_xy, conf)
    jaw_out = jaw_outer_loss(verts_w)
    extra = metric_reg(
        state.id_head.requires_grad,
        (
            (state.expr_w is not None and state.expr_w.requires_grad)
            or state.ex_lower.requires_grad
            or state.ex_left.requires_grad
            or state.ex_right.requires_grad
        ),
        state.rot_head.requires_grad,
    )
    albedo = skin_albedo()
    tv = _tv(albedo, skin_cover)
    diffuse = out["diffuse_detach_normal"]
    reg_diffuse = torch.relu(diffuse.max() - 1.0) + diffuse.var()
    loss = w_rgb * rgb_loss
    loss = loss + w_lmk_term * lmk_loss.to(loss.device)
    loss = loss + jaw_out.to(loss.device)
    loss = loss + extra.to(loss.device)
    # VHAP default reg_tex_tv=1e4; we use a slightly lower weight for more high-freq.
    w_tv = float(getattr(args, "reg_tex_tv", 5.0e3))
    loss = loss + w_tv * tv.to(loss.device)
    loss = loss + 100.0 * reg_diffuse
    lap = tex_res.new_zeros(())
    off = tex_res.new_zeros(())
    if use_offset_reg:
      delta = offset[edges[:, 0]] - offset[edges[:, 1]]
      lap = delta.pow(2).mean()
      free = torch.ones(offset.shape[0], dtype=torch.bool)
      free[pin] = False
      off = offset[free].abs().mean() if bool(free.any()) else offset.new_zeros(())
      loss = loss + 1.0e5 * lap.to(loss.device) + 3.0e2 * off.to(loss.device)
    with torch.no_grad():
      err = torch.linalg.norm(pred_xy.detach() - gt_t, dim=-1)
    return loss, {
        "loss": float(loss.detach()),
        "rgb": float(rgb_loss.detach()),
        "lmk": float(err.mean()),
        "nose": float(err[27:36].mean()) if err.numel() >= 36 else float(err.mean()),
        "jaw_out": float(jaw_out.detach()),
        "tv": float(tv.detach()),
        "lap": float(lap.detach()),
        "off": float(off.detach()) if torch.is_tensor(off) else float(off),
    }

  def run_loop(name: str, iters: int, groups: list[dict], w_rgb: float, w_lmk_term: float,
               disturb: bool, use_offset_reg: bool, bound_ids: torch.Tensor) -> dict:
    groups = [g for g in groups if g["params"]]
    if iters <= 0 or not groups:
      return {}
    opt = torch.optim.Adam(groups)
    last_log: dict = {}
    print(f"\n==> {name}  iters={iters}  {ph}x{pw}  disturb={disturb}", flush=True)
    for i in range(iters):
      opt.zero_grad()
      _, rast, out = render_pack(disturb, bound_ids)
      loss, last_log = photo_loss(rast, out, w_rgb, w_lmk_term, use_offset_reg)
      loss.backward()
      if offset.grad is not None:
        offset.grad[pin] = 0
      opt.step()
      with torch.no_grad():
        offset[pin] = 0
        offset.clamp_(-0.008, 0.008)
        state.focal_norm.clamp_(0.3, 5.0)
        if state.id_head.requires_grad:
          state.id_head.clamp_(-5.0, 5.0)
        if state.id_eyes.requires_grad:
          state.id_eyes.clamp_(-5.0, 5.0)
        if state.expr_w is not None and state.expr_w.requires_grad:
          state.expr_w.clamp_(-1.5, 2.0)
        if state.ex_left.requires_grad:
          state.ex_left.clamp_(-4.0, 4.0)
        if state.ex_right.requires_grad:
          state.ex_right.clamp_(-4.0, 4.0)
        if state.ex_lower.requires_grad:
          state.ex_lower.clamp_(-4.0, 4.0)
      if i == 0 or (i + 1) % max(iters // 8, 1) == 0 or i + 1 == iters:
        print(
            f"  [{name}] {i+1:4d}/{iters}  rgb={last_log['rgb']:.4f}  "
            f"lmk={last_log['lmk']:.2f}  nose={last_log['nose']:.2f}  "
            f"jaw_out={last_log.get('jaw_out', 0.0):.4f}  "
            f"tv={last_log['tv']:.6f}  off={last_log['off']:.5f}",
            flush=True,
        )
    with torch.no_grad():
      _, _, out = render_pack(False, bound_ids)
      dump_photo_preview(
          args.out_dir,
          name.replace("gnm_", ""),
          rgb_np,
          out["rgba"][0].detach().float().cpu().numpy(),
      )
    return last_log

  for t in (
      state.id_head, state.id_eyes, state.ex_lower, state.rot_neck, state.rot_head,
      state.rot_leye, state.rot_reye, state.translation, state.cam_aa, state.cam_t,
      state.focal_norm, offset,
  ):
    t.requires_grad_(False)

  tex_iters = int(args.photo_tex_iters)
  rgb_iters = int(args.photo_iters)
  off_iters = int(getattr(args, "photo_offset_iters", 500))
  stop_after = getattr(args, "stop_after", "tex")
  lmk_w = float(getattr(args, "w_landmark", 10.0))
  # Eyeball joints stay at the pre-photo rotation.
  state.rot_leye.requires_grad_(False)
  state.rot_reye.requires_grad_(False)

  shape_params = [state.id_head, state.id_eyes]
  cam_params = [state.cam_aa, state.cam_t, state.focal_norm]
  for t in shape_params + cam_params:
    t.requires_grad_(True)
  appearance = [
      {"params": [tex_res], "lr": 5e-3},
      {"params": [lights], "lr": 5e-3},
      {"params": shape_params, "lr": 5e-3},
      {"params": cam_params, "lr": 5e-3},
  ]
  print("[photo] rgb_init_texture  cam, shape, texture, lights", flush=True)
  log_tex = run_loop(
      "rgb_init_texture",
      tex_iters,
      appearance,
      w_rgb=30.0,
      w_lmk_term=lmk_w,
      disturb=True,
      use_offset_reg=False,
      bound_ids=bound_texture,
  )
  if dump_stage is not None:
    dump_stage("4_photo_tex", "rgb_init_texture", stage_bgr)
  stats["photo_tex"] = log_tex
  stats["stop_after"] = stop_after
  if stop_after == "tex":
    print("[photo] stop after rgb_init_texture", flush=True)

  pose_rot = [state.rot_neck, state.rot_head]
  # Face expr: PCA150 (expr_w) when set; else raw 350. Eye joints stay frozen.
  if state.expr_w is not None:
    expr_params = [state.expr_w]
    expr_label = f"expr PCA{int(state.expr_w.numel())}"
  else:
    expr_params = [state.ex_left, state.ex_right, state.ex_lower]
    expr_label = "expr 350 (leye/reye/lower)"
  if stop_after in ("rgb", "offset"):
    # Freeze tex/lights so RGB pulls geometry/expr instead of repainting albedo.
    # Shape stays free for fine adjust; jaw outer-contour loss blocks cheek blow-up.
    tex_res.requires_grad_(False)
    lights.requires_grad_(False)
    print(
        f"[photo] rgb_init_all  + pose, neck/jaw, shape, {expr_label}; "
        "freeze tex+lights; eye joints fixed",
        flush=True,
    )
    for t in shape_params + pose_rot + [state.translation] + expr_params:
      t.requires_grad_(True)
    if state.expr_w is not None:
      state.ex_left.requires_grad_(False)
      state.ex_right.requires_grad_(False)
      state.ex_lower.requires_grad_(False)
    log_rgb = run_loop(
        "rgb_init_all",
        rgb_iters,
        [
            {"params": shape_params, "lr": 5e-3},
            {"params": cam_params, "lr": 5e-3},
            {"params": expr_params, "lr": 5e-2},
            {"params": pose_rot, "lr": 5e-3},
            {"params": [state.translation], "lr": 1e-3},
        ],
        w_rgb=30.0,
        w_lmk_term=lmk_w,
        disturb=True,
        use_offset_reg=False,
        bound_ids=bound_later,
    )
    if dump_stage is not None:
      dump_stage("5_photo", "rgb_init_all, jaw landmarks off", stage_bgr)
    stats["photo_all"] = log_rgb

  if stop_after == "offset":
    print("[photo] rgb_init_offset  static offset, teeth/eyes/scalp/ears pinned", flush=True)
    tex_res.requires_grad_(True)
    lights.requires_grad_(True)
    for t in shape_params:
      t.requires_grad_(True)
    offset.requires_grad_(True)
    for t in expr_params:
      t.requires_grad_(True)
    log_off = run_loop(
        "rgb_init_offset",
        off_iters,
        appearance + [
            {"params": expr_params, "lr": 5e-2},
            {"params": pose_rot, "lr": 5e-3},
            {"params": [state.translation], "lr": 1e-3},
            {"params": [offset], "lr": 5e-4},
        ],
        w_rgb=30.0,
        w_lmk_term=lmk_w,
        disturb=True,
        use_offset_reg=True,
        bound_ids=bound_later,
    )
    if dump_stage is not None:
      dump_stage("6_offset", "rgb_init_offset, scalp/ears/teeth/eyes pinned", stage_bgr)
    stats["photo_offset"] = log_off

  with torch.no_grad():
    verts = world_verts().detach().cpu().numpy()
    pred = project_xy(sample_lmk3d(world_verts())).detach().cpu().numpy()
  err = np.linalg.norm(pred - gt_xy, axis=1)
  stats["rmse_all"] = float(err.mean())
  stats["rmse_nose"] = float(err[27:36].mean())
  stats["rmse_jaw"] = float(err[:17].mean())
  stats["offset_mm_p95"] = float(np.percentile(np.linalg.norm(offset.detach().cpu().numpy(), axis=1), 95) * 1000.0)
  with torch.no_grad():
    alb_out = skin_albedo().detach().cpu().numpy()
    res_out = tex_res.detach().cpu().numpy()
  np.savez(
      args.out_dir / "photo_appearance.npz",
      albedo=alb_out,
      residual=res_out,
      skin_base=skin_base.detach().cpu().numpy(),
      lights=lights.detach().cpu().numpy(),
      static_offset=offset.detach().cpu().numpy(),
      pin=regions["pin"],
  )
  trimesh.Trimesh(vertices=verts, faces=faces, process=False).export(args.out_dir / "frame_photo.obj")
  vis = stage_bgr.copy()
  for i in range(pred.shape[0]):
    gx, gy = int(round(gt_xy[i, 0])), int(round(gt_xy[i, 1]))
    px, py = int(round(pred[i, 0])), int(round(pred[i, 1]))
    cv2.circle(vis, (gx, gy), 3, (0, 255, 0), -1)
    cv2.line(vis, (gx, gy), (px, py), (0, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(vis, (px, py), 3, (0, 0, 255), -1)
  cv2.imwrite(str(args.out_dir / "frame_photo_overlay.jpg"), vis)
  print(
      f"[photo] done  lmk={stats['rmse_all']:.2f}px  nose={stats['rmse_nose']:.2f}  "
      f"jaw={stats['rmse_jaw']:.2f}  offset_p95={stats['offset_mm_p95']:.2f}mm",
      flush=True,
  )
  for t in (
      state.id_head, state.id_eyes, state.ex_lower, state.rot_neck, state.rot_head,
      state.rot_leye, state.rot_reye, state.translation, state.cam_aa, state.cam_t,
      state.focal_norm, offset,
  ):
    t.requires_grad_(False)
  return stats


def _build_and_run(args) -> int:
  from gnm.shape import gnm_numpy, gnm_pytorch, gnm_utils
  import gnm_mediapipe_landmarks as mp_lmk
  from nvdiffrast_render.camera import project_world_to_pixels
  from fit_gnm import FitState, ibug68_vhap_weights

  params = np.load(args.params)
  lmk_npz = np.load(args.landmarks, allow_pickle=True)
  gt_xy = np.asarray(lmk_npz["landmarks"][args.frame_index], dtype=np.float32)
  width, height = [int(x) for x in lmk_npz["image_size"]]
  if args.image is None:
    image = Path(str(lmk_npz["image_paths"][args.frame_index]))
  else:
    image = args.image
  stage_bgr = cv2.imread(str(image))
  if stage_bgr is None:
    raise FileNotFoundError(image)
  if stage_bgr.shape[1] != width or stage_bgr.shape[0] != height:
    stage_bgr = cv2.resize(stage_bgr, (width, height), interpolation=cv2.INTER_AREA)

  gnm = gnm_pytorch.GNM.from_local(
      version=gnm_pytorch.GNMMajorVersion.V3,
      variant=gnm_pytorch.GNMVariant.HEAD,
  )
  gnm_np = gnm_numpy.GNM.from_local(
      version=gnm_numpy.GNMMajorVersion.V3,
      variant=gnm_numpy.GNMVariant.HEAD,
  )
  faces = np.asarray(gnm.triangles.detach().cpu().numpy())
  del gnm_utils
  lmk_idx, lmk_w = mp_lmk.load_landmarks(args.landmark_file)

  def p(v):
    return torch.tensor(np.asarray(v), dtype=torch.float32).requires_grad_(True)

  rot = np.asarray(params["rotations"], dtype=np.float32)
  if "expression_left_eye" in params.files:
    ex_left = np.asarray(params["expression_left_eye"], dtype=np.float32)
  else:
    ex_left = np.zeros(100, dtype=np.float32)
  if "expression_right_eye" in params.files:
    ex_right = np.asarray(params["expression_right_eye"], dtype=np.float32)
  else:
    ex_right = np.zeros(100, dtype=np.float32)
  state = FitState(
      id_head=p(params["id_head"]),
      id_eyes=p(params["id_eyes"]),
      ex_left=p(ex_left),
      ex_right=p(ex_right),
      ex_lower=p(params["expression_lower_face"]),
      rot_neck=p(rot[0]),
      rot_head=p(rot[1]),
      rot_leye=p(rot[2]),
      rot_reye=p(rot[3]),
      translation=p(params["translation"]),
      cam_aa=p(params["cam_aa"]),
      cam_t=p(params["cam_t"]),
      focal_norm=p(params["focal_norm"]),
  )
  # Same face PCA150 as vhap_monocular landmark fit (350 -> K).
  k_pca = int(getattr(args, "expr_pca_dim", 150))
  if k_pca > 0:
    pca_path = Path(getattr(args, "expr_pca", "") or "")
    if not pca_path.is_file():
      pca_path = Path(args.params).resolve().parent / f"gnm_face_expr_pca{k_pca}.npz"
    if not pca_path.is_file():
      raise FileNotFoundError(
          f"expr PCA dims={k_pca} requested but missing {pca_path}; "
          "pass --expr_pca or set --expr_pca_dim 0"
      )
    pca = np.load(pca_path)
    dirs = np.asarray(pca["dirs"], dtype=np.float32)
    if dirs.ndim != 2 or dirs.shape[0] != 350:
      raise ValueError(f"PCA dirs should be (350, K), got {dirs.shape} from {pca_path}")
    k_use = min(k_pca, dirs.shape[1])
    dirs = dirs[:, :k_use]
    if "expression_pca_w" in params.files:
      w0 = np.asarray(params["expression_pca_w"], dtype=np.float32).reshape(-1)[:k_use]
      if w0.shape[0] < k_use:
        w0 = np.pad(w0, (0, k_use - w0.shape[0]))
    else:
      w0 = np.zeros(k_use, dtype=np.float32)
    state.expr_dirs = torch.as_tensor(dirs, dtype=torch.float32)
    state.expr_w = torch.tensor(w0, dtype=torch.float32, requires_grad=True)
    args._expr_w0 = torch.tensor(w0, dtype=torch.float32)  # landmark anchor for all-stage reg
    state.ex_left.requires_grad_(False)
    state.ex_right.requires_grad_(False)
    state.ex_lower.requires_grad_(False)
    print(
        f"[photo] face expr PCA{k_use} from {pca_path.name}  "
        f"w0_L2={float(np.linalg.norm(w0)):.3f}",
        flush=True,
    )
  idx_t = torch.as_tensor(lmk_idx, dtype=torch.long)
  bary_t = torch.as_tensor(lmk_w, dtype=torch.float32)
  gt_t = torch.as_tensor(gt_xy)

  def posed_verts_world() -> torch.Tensor:
    return gnm(
        state.identity()[None],
        state.expression()[None],
        state.rotations()[None],
        state.translation[None],
    )[0]

  def sample_lmk3d(verts_world: torch.Tensor) -> torch.Tensor:
    return (verts_world[idx_t] * bary_t.unsqueeze(-1)).sum(dim=1)

  def project_xy(lmk3d: torch.Tensor) -> torch.Tensor:
    return project_world_to_pixels(
        lmk3d, state.camera_rt(), state.camera_k(width, height), width, height, flip_y=True,
    )

  def metric_reg(reg_id: bool, reg_ex: bool, reg_joints: bool) -> torch.Tensor:
    # Match fit_gnm vhap_monocular: coefficient mean-square, not id_scale@identity.
    reg = torch.tensor(0.0)
    if reg_id:
      id_coef = torch.cat([state.id_head.reshape(-1), state.id_eyes.reshape(-1)], dim=0)
      reg = reg + args.reg_id * (id_coef ** 2).mean()
    if reg_ex:
      if state.expr_w is not None and state.expr_w.requires_grad:
        # Anchor to landmark PCA init so all-stage photometry cannot inflate cheeks.
        w0 = getattr(args, "_expr_w0", None)
        if w0 is not None:
          # Pull toward landmark PCA init (stops cheek/jaw inflation under photo loss).
          reg = reg + float(getattr(args, "reg_ex_anchor", 1.0)) * ((state.expr_w - w0) ** 2).mean()
        reg = reg + args.reg_ex * (state.expr_w ** 2).mean()
      else:
        ex_parts = []
        if state.ex_left.requires_grad:
          ex_parts.append(state.ex_left.reshape(-1))
        if state.ex_right.requires_grad:
          ex_parts.append(state.ex_right.reshape(-1))
        if state.ex_lower.requires_grad:
          ex_parts.append(state.ex_lower.reshape(-1))
        if ex_parts:
          reg = reg + args.reg_ex * torch.cat(ex_parts, dim=0).pow(2).mean()
    if reg_joints:
      # Root rot_neck is free (like VHAP rotation); do not pull it to 0.
      for t in (state.rot_head, state.rot_leye, state.rot_reye):
        reg = reg + args.reg_joint * (t ** 2).mean()
      reg = reg + args.reg_jaw_yz * (state.rot_head[1:] ** 2).mean()
    return reg

  def weighted_landmark_loss(pred_xy: torch.Tensor, conf: torch.Tensor) -> torch.Tensor:
    err = torch.linalg.norm(gt_t - pred_xy, dim=-1)
    return (err * conf).sum() / conf.sum().clamp(min=1.0)

  w = ibug68_vhap_weights(nose_boost=args.nose_weight, mouth_boost=1.0, eye_boost=1.0, brow_boost=1.0)
  args.out_dir.mkdir(parents=True, exist_ok=True)
  photo = run_photometric(
      args=args,
      state=state,
      gnm=gnm,
      gnm_np=gnm_np,
      posed_verts_world=posed_verts_world,
      sample_lmk3d=sample_lmk3d,
      project_xy=project_xy,
      dump_stage=lambda *_a, **_k: None,
      stage_bgr=stage_bgr,
      gt_xy=gt_xy,
      gt_t=gt_t,
      width=width,
      height=height,
      mouth_locked=False,
      metric_reg=metric_reg,
      weighted_landmark_loss=weighted_landmark_loss,
      w_lmk=w,
      faces=faces,
  )
  with torch.no_grad():
    ex_full = state.expression().detach().cpu().numpy()
  save_kw = dict(
      id_head=state.id_head.detach().numpy(),
      id_eyes=state.id_eyes.detach().numpy(),
      expression_left_eye=ex_full[:100],
      expression_right_eye=ex_full[100:200],
      expression_lower_face=ex_full[200:350],
      expression=ex_full,
      rotations=state.rotations().detach().numpy(),
      translation=state.translation.detach().numpy(),
      cam_aa=state.cam_aa.detach().numpy(),
      cam_t=state.cam_t.detach().numpy(),
      focal_norm=float(state.focal_norm.detach()),
      image_size=np.array([width, height], dtype=np.int32),
      rmse_all=photo.get("rmse_all", 0.0),
      rmse_nose=photo.get("rmse_nose", 0.0),
  )
  if state.expr_w is not None:
    save_kw["expression_pca_w"] = state.expr_w.detach().cpu().numpy()
    save_kw["expression_pca_dim"] = np.int32(state.expr_w.numel())
  np.savez(args.out_dir / "frame_photo_params.npz", **save_kw)
  summary = {k: v for k, v in photo.items() if k not in ("photo_tex", "photo_all", "photo_offset")}
  summary["photo_tex"] = photo.get("photo_tex", {})
  summary["photo_all"] = photo.get("photo_all", {})
  summary["photo_offset"] = photo.get("photo_offset", {})
  summary["image"] = str(image)
  summary["params_in"] = str(args.params)
  (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
  print(json.dumps(summary, indent=2), flush=True)
  return 0


def main() -> int:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--params", type=Path, required=True)
  p.add_argument("--landmarks", type=Path, required=True)
  p.add_argument("--landmark_file", type=Path, required=True)
  p.add_argument("--out_dir", type=Path, required=True)
  p.add_argument("--image", type=Path, default=None)
  p.add_argument("--frame_index", type=int, default=0)
  p.add_argument("--photo_max_side", type=int, default=1024)
  p.add_argument("--photo_tex_res", type=int, default=2048,
                 help="Skin tile resolution. VHAP tex_resolution default is 2048.")
  p.add_argument("--reg_tex_tv", type=float, default=5.0e3,
                 help="TV on skin albedo. VHAP uses 1e4; default is slightly lower.")
  p.add_argument("--photo_tex_iters", type=int, default=500)
  p.add_argument("--photo_iters", type=int, default=500)
  p.add_argument("--photo_offset_iters", type=int, default=500)
  p.add_argument("--stop_after", choices=("tex", "rgb", "offset"), default="tex")
  p.add_argument("--init_appearance", type=Path, default=None,
                 help="Load albedo and SH from a previous texture stage, then continue.")
  p.add_argument("--w_landmark", type=float, default=10.0)
  p.add_argument("--nose_weight", type=float, default=10.0)
  p.add_argument("--reg_id", type=float, default=0.3)
  p.add_argument("--reg_ex", type=float, default=0.03)
  p.add_argument(
      "--reg_ex_anchor",
      type=float,
      default=1.0,
      help="When PCA expr is on, weight of ||expr_w - w_landmark||^2 in photo stages.",
  )
  p.add_argument(
      "--w_jaw_outer",
      type=float,
      default=20.0,
      help="Two-sided lower-outline vs detected jaw points 3..13 (ray radius; not point-to-point). 0 disables.",
  )
  p.add_argument(
      "--jaw_outer_slack",
      type=float,
      default=2.0,
      help="Pixels of deadzone around detected radius before jaw_outer penalty (both in and out).",
  )
  p.add_argument("--reg_joint", type=float, default=0.3)
  p.add_argument("--reg_jaw_yz", type=float, default=3.0)
  p.add_argument(
      "--expr_pca_dim",
      type=int,
      default=150,
      help="Face expression PCA dim (350->K), same as fit_gnm vhap_monocular. 0 = raw 350.",
  )
  p.add_argument(
      "--expr_pca",
      type=Path,
      default=None,
      help="Optional gnm_face_expr_pcaK.npz; default is next to --params.",
  )
  args = p.parse_args()
  return _build_and_run(args)


if __name__ == "__main__":
  raise SystemExit(main())
