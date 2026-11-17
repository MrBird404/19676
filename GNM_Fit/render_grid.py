#!/usr/bin/env python3
"""Unified panel renderer: a 2D grid of tokens -> one image.

One file covers every panel type used across the pipeline:
  GT / GT+mask / mesh clay / photometric L1,L2 render / overlay ...

Token grammar (case-insensitive, ':' separated):
  gt                         input photo
  mask                       GT + ear-mask overlay
  overlay:<stage>            overlay jpg saved by a stage dir (e.g. l1, l2)
  render:<stage>             textured render jpg saved by a stage dir
  mesh:<model>[@<view>]      clay mesh from that model's params (view: front/side/back/top)
  clay:<model>[@<view>]      alias of mesh

Grid string: rows separated by ';', cells by ','. Example:
  --grid "GT,mask,overlay:l1,render:l1;GT,mesh:gnm,mesh:gnm@side,mesh:vhap"

Registry defaults live in DEFAULTS below; override/extend with --config panels.json:
  { "gt": "...jpg", "mask": "...npz",
    "stages": {"l1": "/abs/dir", "l2": "/abs/dir"},
    "models": {"gnm": "/abs/frame_000000_params.npz", "vhap": "/abs/....npz"} }
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
REPO = next(p for p in [ROOT, *ROOT.parents] if (p / "GNM-main").is_dir()) / "GNM-main"
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(ROOT))
from fit_gnm import FitState  # noqa: E402
from gnm.shape import gnm_pytorch, gnm_numpy  # noqa: E402
from nvdiffrast_render import NVDiffRenderer  # noqa: E402

EXPR = "/root/autodl-tmp/gnm_fit_out/expression_v3"
DEFAULTS = {
    "gt": f"{EXPR}/images/000001.jpg",
    "mask": "/root/autodl-tmp/notes/gnm_uv/sam_ears/ear_masks.npz",
    "stages": {
        "l1": f"{EXPR}/photo_l1_coef_v8",
        "l2": f"{EXPR}/photo_l2b_coef_v8",
        "base": f"{EXPR}/ear_coef_v8",
        "vhap": f"{EXPR}/photo_l1_sam_eyes_vhap_residual",
        "vhap_r2048": f"{EXPR}/photo_l1_sam_eyes_vhap_r2048",
        "l1_old": f"{EXPR}/photo_l1_after_neck_len_more_jaw",
    },
    "models": {
        "gnm":    f"{EXPR}/ear_coef_v8/frame_000000_params.npz",
        "base":   f"{EXPR}/ear_coef_v8/frame_000000_params.npz",
        "gnm_l1": f"{EXPR}/photo_l1_coef_v8/frame_photo_params.npz",
        "gnm_l2": f"{EXPR}/photo_l2b_coef_v8/frame_photo_params.npz",
        "vhap":       f"{EXPR}/photo_l1_sam_eyes_vhap_residual/frame_photo_params.npz",
        "vhap_r2048": f"{EXPR}/photo_l1_sam_eyes_vhap_r2048/frame_photo_params.npz",
        "vhap_base":  f"{EXPR}/lmk_align_vhap_frame0_v4_sam_eyes/frame_000000_params.npz",
    },
}
SIZE = 512
VIEWS = {"front": 0.0, "side": 90.0, "back": 180.0}


def _tp(v):
    return torch.tensor(np.asarray(v), dtype=torch.float32)


def state_of(p):
    rot = np.asarray(p["rotations"], np.float32)
    return FitState(
        id_head=_tp(p["id_head"]), id_eyes=_tp(p["id_eyes"]),
        ex_left=_tp(p["expression_left_eye"] if "expression_left_eye" in p.files else np.zeros(100, np.float32)),
        ex_right=_tp(p["expression_right_eye"] if "expression_right_eye" in p.files else np.zeros(100, np.float32)),
        ex_lower=_tp(p["expression_lower_face"]),
        rot_neck=_tp(rot[0]), rot_head=_tp(rot[1]), rot_leye=_tp(rot[2]), rot_reye=_tp(rot[3]),
        translation=_tp(p["translation"]), cam_aa=_tp(p["cam_aa"]), cam_t=_tp(p["cam_t"]),
        focal_norm=_tp(p["focal_norm"]),
    )


def mesh_verts(npz_path, gnm):
    p = np.load(npz_path)
    st = state_of(p)
    with torch.no_grad():
        posed = gnm(st.identity()[None], st.expression()[None], st.rotations()[None], st.translation[None])[0]
    off = torch.as_tensor(np.asarray(p["static_offset"], np.float32)) if "static_offset" in p.files else torch.zeros_like(posed)
    if "semantic_dirs" in p.files and "semantic_coef" in p.files:
        dirs = torch.as_tensor(np.asarray(p["semantic_dirs"], np.float32))
        coef = torch.as_tensor(np.asarray(p["semantic_coef"], np.float32))
        off = off + torch.einsum("k,kvi->vi", coef, dirs)
    return posed + off, st


class Renderer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gnm = gnm_pytorch.GNM.from_local(version=gnm_pytorch.GNMMajorVersion.V3, variant=gnm_pytorch.GNMVariant.HEAD)
        self.faces = torch.as_tensor(np.asarray(self.gnm.triangles.cpu().numpy()), dtype=torch.int32, device=self.dev)
        self.rend = NVDiffRenderer(use_opengl=False, lighting_type="front", lighting_space="camera",
                                   flip_y_raster=False).to(self.dev)
        self._mesh_cache = {}
        self.render_wh = (1024, 576)

    def _rot(self, view):
        a = np.deg2rad(VIEWS.get(view, 0.0))
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def _rgba(self, verts, st, view="front"):
        rw, rh = self.render_wh
        if view == "top":
            a = np.deg2rad(90.0); c, s = np.cos(a), np.sin(a)
            R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        else:
            R = self._rot(view)
        V = verts.float().cpu().numpy()
        C = V.mean(0); V = (V - C) @ R.T + C
        with torch.no_grad():
            rgba = self.rend.render_rgba_vis(
                torch.as_tensor(V, dtype=torch.float32).to(self.dev)[None], self.faces,
                st.camera_rt().detach()[None].to(self.dev), st.camera_k(rw, rh).detach()[None].to(self.dev),
                (rh, rw), background_color=[0.12, 0.12, 0.14])["rgba"][0].float().cpu().numpy()
        return rgba

    def clay(self, verts, st, view="front"):
        rgba = self._rgba(verts, st, view)
        return cv2.cvtColor(np.clip(rgba[..., :3] * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    def panel_mesh_overlay(self, model, k=0.45):
        path = self.cfg["models"].get(model, "")
        if not path or not Path(path).is_file():
            return self._placeholder(f"no data '{model}'")
        rw, rh = self.render_wh
        verts, st = mesh_verts(path, self.gnm)
        a = self._rgba(verts, st)[..., 3]                    # mesh coverage (rh, rw)
        gt = cv2.imread(self.cfg["gt"])
        if gt is None:
            return self._placeholder("GT missing")
        gt = cv2.resize(gt, (rw, rh)).astype(np.float32)
        blue = np.array([255.0, 0.0, 0.0], np.float32)       # BGR blue
        a3 = a[..., None]
        out = gt * (1 - k * a3) + blue * (k * a3)
        return self._fit(np.clip(out, 0, 255).astype(np.uint8))

    def _fit(self, im):
        h, w = im.shape[:2]; s = min(h, w)
        y0 = (h - s) // 2; x0 = (w - s) // 2
        return cv2.resize(im[y0:y0 + s, x0:x0 + s], (SIZE, SIZE), interpolation=cv2.INTER_AREA)

    def _img(self, path):
        im = cv2.imread(str(path))
        return None if im is None else self._fit(im)

    def panel_gt(self):
        im = self._img(self.cfg["gt"])
        return im if im is not None else self._placeholder("GT missing")

    def panel_mask(self):
        gt = self._img(self.cfg["gt"])
        if gt is None:
            return self._placeholder("GT missing")
        mp = Path(self.cfg.get("mask", ""))
        if not mp.is_file():
            return gt.copy()
        m = np.load(mp)
        L = cv2.resize(np.asarray(m["left"], np.float32), (SIZE, SIZE), cv2.INTER_NEAREST) > 0.5
        R = cv2.resize(np.asarray(m["right"], np.float32), (SIZE, SIZE), cv2.INTER_NEAREST) > 0.5
        ov = gt.copy(); ov[L] = (0, 255, 0); ov[R] = (0, 0, 255)
        out = cv2.addWeighted(gt, 0.6, ov, 0.4, 0)
        cnts, _ = cv2.findContours((L | R).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (0, 255, 255), 2)
        return out

    def _stage_dir(self, stage):
        d = Path(self.cfg["stages"].get(stage, ""))
        return d if d.is_dir() else None

    def panel_render(self, stage):
        d = self._stage_dir(stage)
        if d is None:
            return self._placeholder(f"no stage '{stage}'")
        for c in ("stage_rgb_init_texture_render.jpg", "stage_rgb_init_all_render.jpg",
                  "stage_rgb_init_offset_render.jpg", "frame_photo_overlay.jpg"):
            if (d / c).is_file():
                return self._img(d / c)
        return self._placeholder(f"no render in '{stage}'")

    def panel_overlay(self, stage):
        d = self._stage_dir(stage)
        if d is None:
            return self._placeholder(f"no stage '{stage}'")
        for c in ("frame_photo_overlay.jpg", "stage_rgb_init_all_overlay.jpg",
                  "stage_rgb_init_texture_overlay.jpg", "stage_rgb_init_all_render.jpg"):
            if (d / c).is_file():
                return self._img(d / c)
        return self._placeholder(f"no overlay in '{stage}'")

    def panel_mesh(self, model, view="front"):
        path = self.cfg["models"].get(model, "")
        if not path or not Path(path).is_file():
            return self._placeholder(f"no data '{model}'")
        key = (model, view)
        if key not in self._mesh_cache:
            verts, st = mesh_verts(path, self.gnm)
            self._mesh_cache[key] = self._fit(self.clay(verts, st, view))
        return self._mesh_cache[key].copy()

    def _placeholder(self, msg):
        im = np.full((SIZE, SIZE, 3), 40, np.uint8)
        cv2.putText(im, msg[:34], (16, SIZE // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        return im

    def panel_img(self, path):
        im = self._img(path) if path and Path(path).is_file() else None
        return im if im is not None else self._placeholder("img missing")

    def panel_grid(self, path, idx):
        p = Path(path)
        if not p.is_file():
            return self._placeholder("grid missing")
        im = cv2.imread(str(p))
        H, W = im.shape[:2]
        black = [bool(np.all(np.abs(im[:, k].astype(np.int16)) < 8)) for k in range(W)]
        runs = []; i = 0
        while i < W:
            if not black[i]:
                j = i
                while j < W and not black[j]:
                    j += 1
                if j - i > 50:
                    runs.append((i, j))
                i = j
            else:
                i += 1
        if not runs:
            return self._placeholder("grid no panels")
        a, b = runs[idx % len(runs)]
        return self._fit(im[:, a:b])

    def resolve(self, token):
        parts = [t.strip() for t in token.split(":")]
        k = parts[0].lower()
        if k in ("gt",):
            return self.panel_gt()
        if k in ("mask", "gtmask"):
            return self.panel_mask()
        if k == "render":
            return self.panel_render(parts[1] if len(parts) > 1 else "l1")
        if k == "overlay":
            return self.panel_overlay(parts[1] if len(parts) > 1 else "l1")
        if k in ("meshoverlay", "overlaym"):
            return self.panel_mesh_overlay(parts[1] if len(parts) > 1 else "gnm_l1")
        if k == "img":
            return self.panel_img(":".join(parts[1:]))
        if k == "grid":
            return self.panel_grid(":".join(parts[1:-1]), int(parts[-1]))
        if k in ("mesh", "clay"):
            model = parts[1] if len(parts) > 1 else "gnm"
            view = "front"
            if "@" in model:
                model, view = model.split("@", 1)
            elif len(parts) > 2:
                view = parts[2]
            return self.panel_mesh(model, view)
        return self._placeholder(f"unknown '{token}'")


def label(img, text):
    cv2.putText(img, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True, help="rows by ';', cells by ',' e.g. 'GT,mask,render:l1;GT,mesh:gnm,mesh:vhap'")
    ap.add_argument("--out", default="/root/autodl-tmp/notes/gnm_uv/测试用图片/grid.jpg")
    ap.add_argument("--config", default="", help="JSON overriding gt/mask/stages/models")
    ap.add_argument("--gt", default="", help="override GT image path")
    ap.add_argument("--mask", default="", help="override ear-mask npz path")
    args = ap.parse_args()

    cfg = json.loads(json.dumps(DEFAULTS))  # deep copy
    if args.config:
        over = json.loads(Path(args.config).read_text())
        for kk in ("gt", "mask"):
            if kk in over:
                cfg[kk] = over[kk]
        for kk in ("stages", "models"):
            if kk in over:
                cfg[kk].update(over[kk])
    if args.gt:
        cfg["gt"] = args.gt
    if args.mask:
        cfg["mask"] = args.mask

    R = Renderer(cfg)
    rows = [r for r in args.grid.split(";") if r.strip()]
    grid = []
    for ri, row in enumerate(rows):
        cells = [c for c in row.split(",")]
        panels = []
        for ci, tok in enumerate(cells):
            img = R.resolve(tok)
            panels.append(label(img, f"{ri+1}.{ci+1} {tok}"))
        grid.append(np.concatenate(panels, axis=1))
    out = np.concatenate(grid, axis=0)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, out)
    print("wrote", args.out, f"{len(rows)}x{len(rows[0].split(','))}")


if __name__ == "__main__":
    main()
