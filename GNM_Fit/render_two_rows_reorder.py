#!/usr/bin/env python3
"""Produce the fixed 2x6 stage figure from ONE pipeline run directory.

Layout (6 cols x 2 rows, each cell 512px):
  Row1: GT | rigid | lmk+eye | neck-shoulder | GT-vs-neck-line | ear
  Row2: L1+blue mask | L1 render | L1 mesh | L2+blue mask | L2 render | L2 mesh

Usage:
  python GNM_Fit/render_two_rows_reorder.py [RUN_DIR]
    RUN_DIR  : one run tree produced by pipeline_trap_reorder.sh
               (default: ../gnm_fit_out/expression_v3/pipeline_run_reorder)
    RENDER_OUT (env, optional): output jpg path.

Expected stage sub-dirs inside RUN_DIR:
  01_rigid, 02_lmk_eye, 03_neck_trapPCA, 04_ear_semantic, 05_ear_coef, 06_l1, 07_l2
(all panel data is read only from RUN_DIR; requires GNM-main + nvdiffrast_render).
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import render_grid as rg  # noqa: E402

EV3 = Path("/root/autodl-tmp/gnm_fit_out/expression_v3")
if any(a in ("-h", "--help") for a in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)
_args = [a for a in sys.argv[1:] if not a.startswith("-")]
RUN = Path(_args[0]) if _args else EV3 / "pipeline_run_reorder"
if not RUN.is_dir():
    sys.exit(f"[render_two_rows] RUN dir not found: {RUN}  (refusing to write a broken figure)")
OUT = Path(os.environ.get("RENDER_OUT", "/root/autodl-tmp/notes/gnm_uv/测试用图片/GNM v4 定版2行 reorder GT_rigid_lmkeye_neck_线_ear + L1_L2.jpg"))

S = {"rigid": "01_rigid", "lmkeye": "02_lmk_eye", "neck": "03_neck_trapPCA",
     "ear": "05_ear_coef", "l1": "06_l1", "l2": "07_l2"}


def fname(stage):
    return "frame_photo_params.npz" if stage in ("l1", "l2") else "frame_000000_params.npz"


cfg = json.loads(json.dumps(rg.DEFAULTS))
cfg["gt"] = str(EV3 / "images/000001.jpg")
cfg["stages"] = {"l1": str(RUN / S["l1"]), "l2": str(RUN / S["l2"])}
cfg["models"] = {k: str(RUN / S[k] / fname(k)) for k in S}
R = rg.Renderer(cfg)


def shoulder_panel():
    im = cv2.imread(str(RUN / S["neck"] / "shoulder_before_after_clay.jpg"))
    if im is None:
        return R._placeholder("no shoulder img")
    h, w = im.shape[:2]; w3 = w // 3
    return R._fit(im[:, w3:2 * w3])


def lab(im, t):
    cv2.putText(im, t, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(im, t, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return im


row1 = [R.panel_gt(), R.panel_mesh("rigid"), R.panel_mesh("lmkeye"), R.panel_mesh("neck"), shoulder_panel(), R.panel_mesh("ear")]
row2 = [R.panel_mesh_overlay("l1"), R.panel_render("l1"), R.panel_mesh("l1"),
        R.panel_mesh_overlay("l2"), R.panel_render("l2"), R.panel_mesh("l2")]
l1 = ["GT", "rigid mesh", "lmk+eye mesh", "neck-shoulder mesh", "GT vs neck line", "ear mesh"]
l2 = ["L1 +blue mask", "L1 render", "L1 mesh", "L2 +blue mask", "L2 render", "L2 mesh"]

r1 = np.concatenate([lab(im, t) for im, t in zip(row1, l1)], axis=1)
r2 = np.concatenate([lab(im, t) for im, t in zip(row2, l2)], axis=1)
out = np.concatenate([r1, r2], axis=0)
OUT.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(OUT), out)
print("wrote", OUT, out.shape)
