#!/usr/bin/env python3
"""Export GNM template + each 68-point set as MeshLab PLY (mesh + colored balls)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent / "GNM-main"
OUT = ROOT.parent / "gnm_fit_out" / "lmk68_audit"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT))

from gnm.shape import gnm_numpy  # pylint: disable=wrong-import-position
import gnm_mediapipe_landmarks as mp_lmk  # pylint: disable=wrong-import-position
import gnm_landmark_sets as lmk_sets  # pylint: disable=wrong-import-position

GROUPS = [
    (range(0, 17), (255, 200, 0)),
    (range(17, 22), (255, 165, 0)),
    (range(22, 27), (255, 255, 0)),
    (range(27, 36), (0, 220, 0)),
    (range(36, 42), (255, 0, 255)),
    (range(42, 48), (0, 128, 255)),
    (range(48, 60), (220, 0, 0)),
    (range(60, 68), (0, 220, 220)),
]


def color_of(i: int) -> tuple[int, int, int]:
  for rng, rgb in GROUPS:
    if i in rng:
      return rgb
  return (255, 255, 255)


def icosphere(radius: float, color: tuple[int, int, int]) -> trimesh.Trimesh:
  ball = trimesh.creation.icosphere(subdivisions=2, radius=float(radius))
  rgba = np.array([*color, 255], dtype=np.uint8)
  ball.visual.face_colors = np.tile(rgba, (len(ball.faces), 1))
  ball.visual.vertex_colors = np.tile(rgba, (len(ball.vertices), 1))
  return ball


def export_ply(verts, faces, pts, path: Path, radius: float) -> None:
  head = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
  head.visual.face_colors = np.tile(np.array([200, 200, 205, 255], np.uint8), (len(head.faces), 1))
  parts = [head]
  for i, p in enumerate(pts):
    ball = icosphere(radius, color_of(i))
    ball.apply_translation(np.asarray(p, dtype=np.float64))
    parts.append(ball)
  trimesh.util.concatenate(parts).export(path)
  print(f"wrote {path}  n={len(pts)}")


def sample_txt(verts, path: Path) -> np.ndarray:
  idx, wt = mp_lmk.load_landmarks(path)
  return mp_lmk.sample(verts, idx, wt)


def main() -> int:
  gnm = gnm_numpy.GNM.from_local(
      version=gnm_numpy.GNMMajorVersion.V3, variant=gnm_numpy.GNMVariant.HEAD)
  verts = np.asarray(gnm(None, None, None, None), dtype=np.float64)
  faces = np.asarray(gnm.triangles, dtype=np.int32)
  radius = 1.8e-3

  official = REPO / "gnm/shape/data/landmarks/head_sparse_68.txt"
  ibug = ROOT / "head_sparse_68_ibug_order.txt"
  dense468 = ROOT / "head_dense_468.txt"
  hybrid = ROOT / "head_sparse_68_mp_body_official_eyes.txt"

  pts_official = sample_txt(verts, official)
  pts_ibug = sample_txt(verts, ibug)
  pts_468 = sample_txt(verts, dense468)
  pts_mp468 = pts_468[lmk_sets.MP68_IBUG]
  pts_mp2dlib = lmk_sets.mp468_to_dlib68(pts_468)
  pts_hybrid = sample_txt(verts, hybrid)

  same_official_ibug = bool(np.allclose(pts_official, pts_ibug, atol=1e-8))

  OUT.mkdir(parents=True, exist_ok=True)
  jobs = [
      ("01_official_gnm68_mesh_lmk68.ply", pts_official,
       "官方 GNM head_sparse_68.txt，未改顺序"),
      ("02_ibug_order_mesh_lmk68.ply", pts_ibug,
       "head_sparse_68_ibug_order.txt（对官方做 jaw 2-6 反序）。当前与官方数值相同。"),
      ("03_mp2dlib_mesh_lmk68.ply", pts_mp2dlib,
       "head_dense_468.txt 的 468 个 3D 点经 Mediapipe_2_Dlib 平均成 68"),
      ("04_mp468_subset_mesh_lmk68.ply", pts_mp468,
       "head_dense_468.txt 按教科书 MP68_IBUG 下标直接抽出 68 个"),
      ("05_hybrid_official_eyes_mesh_lmk68.ply", pts_hybrid,
       "head_sparse_68_mp_body_official_eyes.txt（身体 MP、眼睛官方）"),
  ]
  manifest = {"official_equals_ibug": same_official_ibug, "files": []}
  for name, pts, note in jobs:
    path = OUT / name
    export_ply(verts, faces, pts, path, radius)
    manifest["files"].append({"file": str(path), "note": note})

  (OUT / "meshlab_lmk68_sets.json").write_text(
      json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
  print("official == ibug:", same_official_ibug)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
