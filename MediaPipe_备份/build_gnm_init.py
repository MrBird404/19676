#!/usr/bin/env python3
"""Build ``frame_*_gnm_init.npz``: neutral GNM + true-perspective (K, RT) init.

No weak-perspective / no vertex [1,-1,-1] bake. World = GNM native.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent / "GNM-main"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT))

from gnm.shape import gnm_numpy  # pylint: disable=wrong-import-position
import gnm_mediapipe_landmarks as mp_lmk  # pylint: disable=wrong-import-position
import gnm_landmark_sets as lmk_sets  # pylint: disable=wrong-import-position
from nvdiffrast_render.camera import (  # pylint: disable=wrong-import-position
    k_from_focal,
    perspective_init_from_iod,
    rt_from_aa_t,
)

GNM_68_FILE = REPO / "gnm" / "shape" / "data" / "landmarks" / "head_sparse_68.txt"
GNM_68_IBUG_FILE = ROOT / "head_sparse_68_ibug_order.txt"
V3_68_FILE = ROOT.parent / "MediaPipe" / "head_sparse_68_v3_改.txt"


def main() -> int:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--landmarks", type=Path, required=True)
  p.add_argument("--lmk_set", choices=("dense468", "sparse68"), default="sparse68")
  p.add_argument(
      "--gt68_map",
      choices=("star", "identity", "mp68", "mp2dlib", "mp68_ibug", "mp68_v1"),
      default="star",
      help="2D 68 GT. star/identity = already iBUG 68, paired with "
           "MediaPipe/head_sparse_68_v3_改.txt. "
           "mp* = take 68 from MediaPipe 468 (legacy).",
  )
  p.add_argument("--landmark_file", type=Path, default=None)
  p.add_argument("--out", type=Path, required=True)
  p.add_argument("--frame_index", type=int, default=0)
  p.add_argument("--focal_norm", type=float, default=1.5,
                 help="fx=fy=focal_norm*max(H,W), VHAP-style.")
  args = p.parse_args()
  if args.landmark_file is None:
    if args.lmk_set == "sparse68":
      if args.gt68_map == "mp2dlib":
        if not GNM_68_IBUG_FILE.is_file():
          lmk_sets.write_gnm68_ibug_order(GNM_68_FILE, GNM_68_IBUG_FILE)
        args.landmark_file = GNM_68_IBUG_FILE
      else:
        args.landmark_file = V3_68_FILE
    else:
      args.landmark_file = ROOT / "head_dense_468.txt"

  data = np.load(args.landmarks, allow_pickle=True)
  gt_all = data["landmarks"][args.frame_index].astype(np.float32)
  if args.lmk_set == "sparse68":
    gt_xy = lmk_sets.sparse68_gt_xy(gt_all, args.gt68_map)
    iod_idx = lmk_sets.IBUG_IOD
  else:
    gt_xy = gt_all
    iod_idx = (33, 263)
  width, height = [int(x) for x in data["image_size"]]
  video_frame = int(data["frame_indices"][args.frame_index])
  video = str(data["video"]) if "video" in data.files else ""

  gnm = gnm_numpy.GNM.from_local(
      version=gnm_numpy.GNMMajorVersion.V3,
      variant=gnm_numpy.GNMVariant.HEAD,
  )
  verts_native = np.asarray(gnm(None, None, None, None), dtype=np.float32)
  faces = np.asarray(gnm.triangles)

  idx, w = mp_lmk.load_landmarks(args.landmark_file)
  lmk3d = mp_lmk.sample(verts_native, idx, w).astype(np.float32)
  cam_aa, cam_t, focal_norm, pred_xy = perspective_init_from_iod(
      gt_xy, lmk3d, width, height, iod_idx=iod_idx, focal_norm=args.focal_norm,
  )
  k = k_from_focal(focal_norm, width, height)
  rt = rt_from_aa_t(cam_aa, cam_t)
  err_px = np.linalg.norm(pred_xy - gt_xy, axis=1)

  args.out.parent.mkdir(parents=True, exist_ok=True)
  np.savez(
      args.out,
      video_frame=video_frame,
      video=video,
      image_size=np.array([width, height], dtype=np.int32),
      description=(
          f"Neutral GNM + true-perspective K/RT init ({args.lmk_set}"
          f"{'' if args.lmk_set != 'sparse68' else ', gt68_map=' + args.gt68_map})."
      ),
      gt68_map=args.gt68_map if args.lmk_set == "sparse68" else "",
      landmark_file=str(args.landmark_file),
      camera_model="perspective_k_rt",
      vertices_native=verts_native,
      faces=faces.astype(np.int32),
      landmarks_3d_world=lmk3d,
      gt_landmarks_xy=gt_xy,
      gnm_init_landmarks_xy=pred_xy,
      init_cam_aa=cam_aa.astype(np.float32),
      init_cam_t=cam_t.astype(np.float32),
      init_focal_norm=np.float32(focal_norm),
      init_K=k,
      init_RT=rt,
      identity=np.zeros(gnm.identity_dim, dtype=np.float32),
      expression=np.zeros(gnm.expression_dim, dtype=np.float32),
      rotations=np.zeros((4, 3), dtype=np.float32),
      translation=np.zeros(3, dtype=np.float32),
      err_px=err_px.astype(np.float32),
  )
  print(
      f"wrote {args.out}  {args.lmk_set} n={len(gt_xy)}  "
      f"gt68_map={args.gt68_map if args.lmk_set == 'sparse68' else '-'}  "
      f"lmk_file={args.landmark_file}  "
      f"perspective  focal_norm={focal_norm:.3f}  "
      f"init_err mean={err_px.mean():.1f}px  p90={np.percentile(err_px, 90):.1f}px",
      flush=True,
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
