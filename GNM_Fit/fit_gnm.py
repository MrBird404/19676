#!/usr/bin/env python3
"""GNM v3 HEAD landmark fit — VHAP-style 3-stage + true-perspective camera.

``--lmk_set sparse68`` (default): 2D GT is iBUG 68, 3D is
``GNM_Fit/head_sparse_68_v3_改.txt``. ``dense468`` is the legacy MediaPipe path.

Camera: OpenGL-like K + RT (no weak-perspective, no vertex [1,-1,-1] bake).
World = GNM native. Projection matches ``nvdiffrast_render.camera``.

Stages:
  1. rigid      — camera + head rigid body (neck root + translation). No identity,
     expression, jaw, or eyes. ``vhap_monocular`` matches VHAP ``lmk_init_rigid``.
  2. shape      — identity, joints, expression, and pose together (landmark only).
  3. expression — optional later refine. ``--stop_after shape`` skips it.
     Photometric fitting stays off unless ``--photometric``. It compares
     rendered color to the white-background frame. There is no mask-contour loss.

Each stage writes a photo GT/pred overlay and a true-perspective mesh head
with the current 68 (or 468) landmarks:
``stage_{tag}_render.jpg``, ``stage_{tag}_render_lmk.jpg``,
``stage_{tag}_render_lmk_crop.jpg``. Crops are concatenated into
``stages_mesh_lmk_strip.jpg``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh

ROOT = Path(__file__).resolve().parent
REPO = next(p for p in [ROOT, *ROOT.parents] if (p / "GNM-main").is_dir()) / "GNM-main"
GNM_68_FILE = REPO / "gnm" / "shape" / "data" / "landmarks" / "head_sparse_68.txt"
GNM_68_IBUG_FILE = ROOT / "head_sparse_68_ibug_order.txt"
V3_68_FILE = ROOT / "head_sparse_68_v3_改.txt"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT))

from gnm.shape import gnm_numpy, gnm_pytorch, gnm_utils  # pylint: disable=wrong-import-position
import expr_dirs  # pylint: disable=wrong-import-position
import gnm_landmark_sets as lmk_sets  # pylint: disable=wrong-import-position
import gnm_mediapipe_landmarks as mp_lmk  # pylint: disable=wrong-import-position
from nvdiffrast_render.camera import (  # pylint: disable=wrong-import-position
    k_from_focal,
    project_world_to_pixels,
)


def axis_angle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
  angle = torch.linalg.norm(aa, dim=-1, keepdim=True).clamp(min=1e-8)
  axis = aa / angle
  x, y, z = axis.unbind(-1)
  zeros = torch.zeros_like(x)
  k = torch.stack([zeros, -z, y, z, zeros, -x, -y, x, zeros], dim=-1)
  k = k.reshape(aa.shape[:-1] + (3, 3))
  eye = torch.eye(3, device=aa.device, dtype=aa.dtype).expand(k.shape)
  return eye + torch.sin(angle)[..., None] * k + (1.0 - torch.cos(angle)[..., None]) * (k @ k)


LANDMARK_COUNT = 468


def uniform_weights() -> np.ndarray:
  """All dense468 points active with weight 1 (stage1 rigid)."""
  return np.ones(LANDMARK_COUNT, dtype=np.float32)


def vhap_stage_weights(
    stage: str,
    nose_w: float,
    mouth_w: float,
    mouth_sock_w: float,
    oval_w: float,
    eye_w: float,
    brow_w: float,
    base_w: float = 1.0,
) -> np.ndarray:
  """Per-point confidence; base is ``base_w`` on all 468 indices."""
  w = np.full(LANDMARK_COUNT, base_w, dtype=np.float32)
  if stage == "rigid":
    return np.ones(LANDMARK_COUNT, dtype=np.float32)

  if stage == "mouth_open":
    # Outer vermilion + jaw only. Inner 13/14 are depth-stacked and veto opening.
    w[:] = 0.0
    for i in lmk_sets.MP_UPPER_LIP_OUTER:
      w[i] = mouth_w
    for i in lmk_sets.MP_LOWER_LIP_OUTER:
      w[i] = mouth_w * 2.5
    for i in lmk_sets.MP_JAW68:
      w[i] = oval_w
    return w

  if stage == "shape":
    # Identity: contour / nose / brows / eyes; lips+sock if mouth_w>0 (incl. inner).
    w[:] = 0.0
    for i in lmk_sets.MP_NOSE_DENSE:
      w[i] = nose_w
    for i in lmk_sets.MP_CONTOUR_DUAL | lmk_sets.MP_JAW68:
      w[i] = max(w[i], oval_w)
    for i in lmk_sets.MP_EYES:
      w[i] = eye_w
    for i in lmk_sets.MP_BROWS:
      w[i] = brow_w
    if mouth_w > 0:
      for i in lmk_sets.MP_LIPS:
        w[i] = max(w[i], mouth_w)
      for i in lmk_sets.MP_LOWER_LIP_OUTER:
        w[i] = max(w[i], mouth_w * 2.0)
      for i in lmk_sets.MP_UPPER_LIP_INNER | lmk_sets.MP_LOWER_LIP_INNER:
        w[i] = max(w[i], mouth_w)
    if mouth_sock_w > 0:
      for i in lmk_sets.MP_MOUTH_SOCK:
        w[i] = max(w[i], mouth_sock_w)
      for i in (
          lmk_sets.MP_PERIORAL_ON_TEETH_MPMESH
          | lmk_sets.MP_MOUTH_NEIGHBOR_LEFTOVER
          | lmk_sets.MP_MENTOLABIAL
      ):
        w[i] = max(w[i], mouth_sock_w)
    return w

  # expression: mouth-only refine; rest of face stays at stage2 via frozen identity
  w[:] = 0.0
  if mouth_w > 0:
    for i in lmk_sets.MP_LIPS:
      w[i] = mouth_w
    for i in lmk_sets.MP_LOWER_LIP_OUTER:
      w[i] = mouth_w * 2.5
    for i in lmk_sets.MP_UPPER_LIP_INNER | lmk_sets.MP_LOWER_LIP_INNER:
      w[i] = mouth_w * 2.0
  if mouth_sock_w > 0:
    for i in lmk_sets.MP_MOUTH_SOCK:
      w[i] = max(w[i], mouth_sock_w)
    for i in (
        lmk_sets.MP_PERIORAL_ON_TEETH_MPMESH
        | lmk_sets.MP_MOUTH_NEIGHBOR_LEFTOVER
        | lmk_sets.MP_MENTOLABIAL
    ):
      w[i] = max(w[i], mouth_sock_w)
  if oval_w > 0:
    for i in lmk_sets.MP_JAW68:
      w[i] = oval_w
  if brow_w > 0:
    for i in lmk_sets.MP_BROWS:
      w[i] = brow_w
  if eye_w > 0:
    for i in lmk_sets.MP_EYES:
      w[i] = eye_w
  if nose_w > 0:
    for i in lmk_sets.MP_NOSE_DENSE:
      w[i] = nose_w
  return w


def ibug68_vhap_weights(
    nose_boost: float = 10.0,
    mouth_boost: float = 1.0,
    eye_boost: float = 1.0,
    brow_boost: float = 1.0,
) -> np.ndarray:
  """VHAP monocular landmark confidence (base=1, region boosts on iBUG ranges).

  Matches ``vhap.config.base.LossWeightConfig``:
  nose 27-36 ×10, mouth 48-68 ×1, eyes 36-48 ×1, brows 17-26 ×1, jaw 0-16 ×1.
  """
  w = np.ones(68, dtype=np.float32)
  for i in lmk_sets.IBUG_NOSE:
    w[i] *= nose_boost
  for i in lmk_sets.IBUG_MOUTH:
    w[i] *= mouth_boost
  for i in lmk_sets.IBUG_EYES:
    w[i] *= eye_boost
  for i in lmk_sets.IBUG_BROWS:
    w[i] *= brow_boost
  return w


def ibug68_stage_weights(
    stage: str,
    nose_w: float,
    mouth_w: float,
    oval_w: float,
    eye_w: float,
    brow_w: float,
    base_w: float = 1.0,
    inner_mouth_w: float | None = None,
) -> np.ndarray:
  """Per-point confidence on official GNM/iBUG 68."""
  n = 68
  w = np.full(n, base_w, dtype=np.float32)
  if stage == "rigid":
    return np.ones(n, dtype=np.float32)

  inner_w = mouth_w if inner_mouth_w is None else inner_mouth_w

  if stage == "mouth_open":
    w[:] = 0.0
    for i in lmk_sets.IBUG_MOUTH_OUTER:
      w[i] = mouth_w
    for i in lmk_sets.IBUG_MOUTH_LOWER_OUTER:
      w[i] = mouth_w * 2.0
    for i in lmk_sets.IBUG_MOUTH_INNER:
      w[i] = inner_w * 2.0
    for i in lmk_sets.IBUG_JAW:
      w[i] = oval_w
    return w

  if stage == "shape":
    for i in lmk_sets.IBUG_NOSE:
      w[i] = max(w[i], nose_w)
    if mouth_w > 0:
      for i in lmk_sets.IBUG_MOUTH:
        w[i] = max(w[i], mouth_w)
    for i in lmk_sets.IBUG_JAW:
      w[i] = max(w[i], oval_w)
    for i in lmk_sets.IBUG_EYES:
      w[i] = max(w[i], eye_w)
    for i in lmk_sets.IBUG_BROWS:
      w[i] = max(w[i], brow_w)
    return w

  for i in lmk_sets.IBUG_MOUTH_OUTER:
    w[i] = max(w[i], mouth_w)
  for i in lmk_sets.IBUG_MOUTH_LOWER_OUTER:
    w[i] = max(w[i], mouth_w * 2.0)
  for i in lmk_sets.IBUG_MOUTH_INNER:
    w[i] = max(w[i], inner_w * 2.0)
  for i in lmk_sets.IBUG_BROWS:
    w[i] = max(w[i], brow_w)
  for i in lmk_sets.IBUG_EYES:
    w[i] = max(w[i], eye_w)
  for i in lmk_sets.IBUG_NOSE:
    w[i] = max(w[i], nose_w * 0.5)
  for i in lmk_sets.IBUG_JAW:
    w[i] = max(w[i], oval_w)
  return w


@dataclass
class FitState:
  id_head: torch.Tensor
  id_eyes: torch.Tensor
  ex_left: torch.Tensor
  ex_right: torch.Tensor
  ex_lower: torch.Tensor
  rot_neck: torch.Tensor
  rot_head: torch.Tensor
  rot_leye: torch.Tensor
  rot_reye: torch.Tensor
  translation: torch.Tensor
  cam_aa: torch.Tensor
  cam_t: torch.Tensor
  focal_norm: torch.Tensor

  @classmethod
  def from_init(cls, init: np.lib.npyio.NpzFile) -> FitState:
    def p(v=None, shape=None):
      if shape is not None:
        t = torch.zeros(shape, dtype=torch.float32)
      else:
        t = torch.as_tensor(v, dtype=torch.float32)
      return t.requires_grad_(True)

    if "init_cam_aa" not in init.files:
      raise ValueError(
          "init npz has no init_cam_aa — rebuild with build_gnm_init.py "
          "(perspective). Weak-perspective inits are no longer supported."
      )
    return cls(
        id_head=p(None, 170),
        id_eyes=p(None, 3),
        ex_left=p(None, 100),
        ex_right=p(None, 100),
        ex_lower=p(None, 150),
        rot_neck=p(None, 3),
        rot_head=p(None, 3),
        rot_leye=p(None, 3),
        rot_reye=p(None, 3),
        translation=p(None, 3),
        cam_aa=p(init["init_cam_aa"]),
        cam_t=p(init["init_cam_t"]),
        focal_norm=p(float(init["init_focal_norm"])),
    )

  def identity(self) -> torch.Tensor:
    teeth = torch.zeros(80)
    return torch.cat([self.id_head, self.id_eyes, teeth], dim=0)

  # 设上之后，表情只走这几条方向，ex_lower 本身不优化。
  expr_dirs: torch.Tensor | None = None
  expr_w: torch.Tensor | None = None

  def expression(self) -> torch.Tensor:
    # 383 = left_eye 100 + right_eye 100 + lower_face 150 + tongue 32 + pupils 1.
    # expr_dirs (350, K): face PCA, c_face = dirs @ w.
    # expr_dirs (150, K): safe_expr mouth directions in lower-face only.
    if self.expr_dirs is not None:
      mapped = self.expr_dirs @ self.expr_w
      if mapped.shape[0] == 350:
        left, right, ex = mapped[:100], mapped[100:200], mapped[200:350]
      else:
        ex = mapped
        left = torch.zeros(100, dtype=ex.dtype, device=ex.device)
        right = torch.zeros(100, dtype=ex.dtype, device=ex.device)
    else:
      ex = self.ex_lower
      left = self.ex_left
      right = self.ex_right
    return torch.cat([
        left,
        right,
        ex,
        torch.zeros(32, dtype=ex.dtype, device=ex.device),
        torch.zeros(1, dtype=ex.dtype, device=ex.device),
    ], dim=0)

  def rotations(self) -> torch.Tensor:
    return torch.stack(
        [self.rot_neck, self.rot_head, self.rot_leye, self.rot_reye], dim=0
    )

  def camera_rt(self) -> torch.Tensor:
    r = axis_angle_to_matrix(self.cam_aa)
    rt = torch.eye(4, dtype=self.cam_aa.dtype, device=self.cam_aa.device)
    rt[:3, :3] = r
    rt[:3, 3] = self.cam_t
    return rt

  def camera_k(self, width: int, height: int) -> torch.Tensor:
    f = self.focal_norm * float(max(width, height))
    half_w = self.focal_norm.new_tensor(width * 0.5)
    half_h = self.focal_norm.new_tensor(height * 0.5)
    return torch.stack([f, f, half_w, half_h])

  def snapshot(self) -> dict[str, torch.Tensor]:
    fields = (
        "id_head", "id_eyes", "ex_left", "ex_right", "ex_lower",
        "rot_neck", "rot_head", "rot_leye", "rot_reye", "translation",
        "cam_aa", "cam_t", "focal_norm",
    )
    snap = {k: getattr(self, k).detach().clone() for k in fields}
    if self.expr_w is not None:
      snap["expr_w"] = self.expr_w.detach().clone()
    return snap

  def restore(self, snap: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
      for k, v in snap.items():
        getattr(self, k).copy_(v)


IBUG68_CHAINS = (
    (list(range(0, 17)), (0, 180, 255), 2),
    (list(range(17, 22)), (0, 165, 255), 2),
    (list(range(22, 27)), (0, 255, 255), 2),
    (list(range(27, 31)), (0, 220, 0), 2),
    (list(range(31, 36)), (0, 220, 0), 2),
    (list(range(36, 42)) + [36], (255, 0, 255), 2),
    (list(range(42, 48)) + [42], (255, 128, 0), 2),
    (list(range(48, 60)) + [48], (0, 0, 220), 2),
    ([60, 61, 62, 63, 64, 65, 66, 67, 60], (0, 220, 220), 2),
)


def _ibug68_color(i: int) -> tuple[int, int, int]:
  for chain, col, _ in IBUG68_CHAINS:
    if i in chain:
      return col
  return (0, 0, 255)


def _crop_around_xy(img: np.ndarray, xy: np.ndarray, pad: int = 90) -> np.ndarray:
  h, w = img.shape[:2]
  xs, ys = xy[:, 0], xy[:, 1]
  x0 = max(int(np.floor(xs.min())) - pad, 0)
  y0 = max(int(np.floor(ys.min())) - pad, 0)
  x1 = min(int(np.ceil(xs.max())) + pad, w)
  y1 = min(int(np.ceil(ys.max())) + pad, h)
  if x1 <= x0 or y1 <= y0:
    return img
  return img[y0:y1, x0:x1]


def _draw_landmarks_on_mesh(bgr: np.ndarray, xy: np.ndarray) -> np.ndarray:
  vis = bgr.copy()
  n = int(xy.shape[0])
  if n == 68:
    for chain, col, thick in IBUG68_CHAINS:
      for a, b in zip(chain[:-1], chain[1:]):
        pa = (int(round(xy[a, 0])), int(round(xy[a, 1])))
        pb = (int(round(xy[b, 0])), int(round(xy[b, 1])))
        cv2.line(vis, pa, pb, (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.line(vis, pa, pb, col, thick, cv2.LINE_AA)
    for i, (x, y) in enumerate(xy):
      p = (int(round(float(x))), int(round(float(y))))
      c = _ibug68_color(i)
      cv2.circle(vis, p, 6, (0, 0, 0), -1, cv2.LINE_AA)
      cv2.circle(vis, p, 4, c, -1, cv2.LINE_AA)
      cv2.putText(vis, str(i), (p[0] + 5, p[1] - 4),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 2, cv2.LINE_AA)
      cv2.putText(vis, str(i), (p[0] + 5, p[1] - 4),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.32, c, 1, cv2.LINE_AA)
  else:
    r = 3 if n <= 68 else 1
    for x, y in xy:
      p = (int(round(float(x))), int(round(float(y))))
      cv2.circle(vis, p, r + 1, (0, 0, 0), -1, cv2.LINE_AA)
      cv2.circle(vis, p, r, (0, 0, 255), -1, cv2.LINE_AA)
  return vis


def render_fitted_frontal(
    verts: np.ndarray,
    faces: np.ndarray,
    lmk3d: np.ndarray,
    rt: np.ndarray,
    k4: np.ndarray,
    height: int,
    width: int,
    out_dir: Path,
    frame_tag: str,
    title: str = "",
    renderer=None,
) -> dict[str, str]:
  """True-perspective shaded frontal mesh at video resolution + landmarks.

  Uses the fitted K+RT (same camera as landmark projection). Returns paths written.
  """
  if not torch.cuda.is_available():
    print("[render] skip: CUDA required for nvdiffrast", flush=True)
    return {}

  from nvdiffrast_render import NVDiffRenderer  # pylint: disable=import-outside-toplevel

  device = torch.device("cuda")
  verts_t = torch.as_tensor(verts, dtype=torch.float32, device=device)[None]
  faces_t = torch.as_tensor(faces, dtype=torch.int32, device=device)
  rt_t = torch.as_tensor(rt, dtype=torch.float32, device=device)[None]
  k_t = torch.as_tensor(k4, dtype=torch.float32, device=device)[None]
  lmk_t = torch.as_tensor(lmk3d, dtype=torch.float32, device=device)

  if renderer is None:
    renderer = NVDiffRenderer(
        use_opengl=False,
        lighting_type="front",
        lighting_space="camera",
        flip_y_raster=False,
    ).to(device)

  with torch.no_grad():
    out = renderer.render_rgba_vis(
        verts_t, faces_t, rt_t, k_t, (height, width),
        background_color=[0.12, 0.12, 0.14],
    )
    rgba = out["rgba"][0].detach().float().cpu().numpy()
    xy = project_world_to_pixels(
        lmk_t, rt_t[0], k_t[0], width, height, flip_y=True,
    ).detach().cpu().numpy()

  rgb_u8 = np.clip(rgba[..., :3] * 255.0, 0, 255).astype(np.uint8)
  bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
  render_path = out_dir / f"{frame_tag}_render.jpg"
  cv2.imwrite(str(render_path), bgr)

  vis = _draw_landmarks_on_mesh(bgr, xy)
  if title:
    cv2.putText(
        vis, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3,
    )
    cv2.putText(
        vis, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
    )
  lmk_path = out_dir / f"{frame_tag}_render_lmk.jpg"
  cv2.imwrite(str(lmk_path), vis)
  crop = _crop_around_xy(vis, xy, pad=110)
  crop_path = out_dir / f"{frame_tag}_render_lmk_crop.jpg"
  cv2.imwrite(str(crop_path), crop)
  print(
      f"[render] wrote {render_path.name} + {lmk_path.name} + {crop_path.name}  "
      f"{width}x{height}  n_lmk={lmk3d.shape[0]}",
      flush=True,
  )
  return {
      "render": str(render_path),
      "render_lmk": str(lmk_path),
      "render_lmk_crop": str(crop_path),
  }


def write_stage_mesh_strip(out_dir: Path, tags: list[str]) -> str | None:
  """Horizontal strip of per-stage cropped mesh+68 visualizations."""
  imgs = []
  for tag in tags:
    path = out_dir / f"stage_{tag}_render_lmk_crop.jpg"
    im = cv2.imread(str(path)) if path.is_file() else None
    if im is None:
      continue
    imgs.append((tag, im))
  if not imgs:
    return None
  h = min(im.shape[0] for _, im in imgs)
  row = []
  for tag, im in imgs:
    scale = h / im.shape[0]
    w = max(int(round(im.shape[1] * scale)), 1)
    resized = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
    cv2.putText(resized, tag, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
    cv2.putText(resized, tag, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    row.append(resized)
  strip = np.concatenate(row, axis=1)
  out = out_dir / "stages_mesh_lmk_strip.jpg"
  cv2.imwrite(str(out), strip)
  print(f"[render] wrote {out.name}  {strip.shape[1]}x{strip.shape[0]}", flush=True)
  return str(out)


def main() -> int:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--landmarks", type=Path, required=True)
  p.add_argument("--init", type=Path, required=True)
  p.add_argument("--out_dir", type=Path, required=True)
  p.add_argument("--lmk_set", choices=("dense468", "sparse68"), default="sparse68")
  p.add_argument(
      "--gt68_map",
      choices=("star", "identity", "mp68", "mp2dlib", "mp68_ibug", "mp68_v1"),
      default="star",
      help="2D 68 GT. star/identity = already iBUG 68, paired with "
           "GNM_Fit/head_sparse_68_v3_改.txt. "
           "mp* = subset from MediaPipe 468 (legacy).",
  )
  p.add_argument("--landmark_file", type=Path, default=None)
  p.add_argument("--frame_index", type=int, default=0)
  p.add_argument(
      "--focal_norm",
      type=float,
      default=None,
      help="Override init focal. fx=fy=focal_norm*max(H,W).",
  )
  p.add_argument(
      "--freeze_focal",
      action="store_true",
      help="Keep focal_norm fixed (use with --focal_norm).",
  )
  p.add_argument("--version_name", type=str, default="vhap_3stage")
  p.add_argument(
      "--preset",
      choices=("default", "vhap_monocular", "safe_expr"),
      default="default",
      help="vhap_monocular: VHAP base LossWeightConfig + lmk_init_* style stages "
           "(balanced mouth boost=1, nose×10, reg_expr/shape/jaw_yz).",
  )
  p.add_argument(
      "--stop_after",
      choices=("rigid", "shape", "all"),
      default="all",
      help="rigid: camera + head rigid only. shape: then identity/joints/expr/pose. "
           "all: also the later expression refine. Never runs a mask-contour loss.",
  )
  p.add_argument("--rigid_iters", type=int, default=500)
  p.add_argument("--shape_iters", type=int, default=1000)
  p.add_argument("--expr_iters", type=int, default=1000)
  p.add_argument("--w_landmark", type=float, default=10.0)
  p.add_argument("--nose_weight", type=float, default=10.0)
  p.add_argument("--mouth_weight", type=float, default=3.0,
                 help="Stage2 lip weight if shape_mouth_weight unset; else expr default.")
  p.add_argument("--shape_mouth_weight", type=float, default=16.0,
                 help="Stage2 lip+mouth weight (0=skip mouth).")
  p.add_argument("--mouth_sock_weight", type=float, default=8.0,
                 help="Peri-mouth fill in stage2 when shape_mouth_weight>0.")
  p.add_argument("--oval_weight", type=float, default=12.0,
                 help="Contour belt weight in stage2 shape.")
  p.add_argument("--eye_weight", type=float, default=3.0)
  p.add_argument("--brow_weight", type=float, default=5.0)
  p.add_argument("--expr_mouth_weight", type=float, default=48.0,
                 help="Stage3 lip weight (expression only; rest of face frozen in loss).")
  p.add_argument("--expr_mouth_sock_weight", type=float, default=24.0,
                 help="Stage3 peri-mouth fill.")
  p.add_argument("--expr_oval_weight", type=float, default=1.0,
                 help="Stage3 lower-jaw only (keep tiny so mouth can move).")
  p.add_argument("--expr_brow_weight", type=float, default=0.0,
                 help="Stage3 brow weight (0=leave stage2 brows).")
  p.add_argument("--expr_eye_weight", type=float, default=0.0,
                 help="Stage3 eye weight (0=leave stage2 eyes).")
  p.add_argument("--expr_base_weight", type=float, default=0.0,
                 help="Stage3 weight on non-mouth landmarks (0=mouth-only).")
  p.add_argument("--mouth_open_iters", type=int, default=0,
                 help="After rigid: optimize expression to open the mouth (0=skip).")
  p.add_argument("--w_mouth_open", type=float, default=0.0,
                 help="Aux loss: match outer-lip vertical span (468: 0/17; 68: 51/57).")
  p.add_argument("--mouth_open_jaw_weight", type=float, default=2.0,
                 help="Jaw68 weight during mouth-open stage.")
  p.add_argument(
      "--freeze_identity_in_expr",
      action=argparse.BooleanOptionalAction,
      default=True,
      help="Stage3: do not optimize id_head (mouth via expression only).",
  )
  p.add_argument(
      "--identity_from",
      type=Path,
      default=None,
      help="Load id_head from a fit params npz and keep it frozen.",
  )
  p.add_argument(
      "--mouth_basis",
      action="store_true",
      help="Fit the mouth with the 150 lower-face expression coefficients.",
  )
  p.add_argument(
      "--photometric",
      action="store_true",
      help="After landmarks: VHAP rgb_init_texture, rgb_init_all, rgb_init_offset. No silhouette.",
  )
  p.add_argument("--mask", type=Path, default=None, help="Unused. Masks are for compositing, not a loss.")
  p.add_argument("--photo_max_side", type=int, default=1024)
  p.add_argument("--photo_tex_res", type=int, default=1024)
  p.add_argument("--photo_tex_iters", type=int, default=500)
  p.add_argument("--photo_iters", type=int, default=500)
  p.add_argument("--photo_offset_iters", type=int, default=500)
  p.add_argument("--photo_mask_iters", type=int, default=0, help="Ignored. Silhouette stage is not used.")
  p.add_argument("--reg_id", type=float, default=0.05)
  p.add_argument("--reg_ex", type=float, default=0.03)
  p.add_argument("--reg_joint", type=float, default=0.05)
  p.add_argument(
      "--reg_jaw_yz",
      type=float,
      default=0.0,
      help="Extra L2 on rot_head y/z (VHAP reg_jaw_yz=3; reduces mouth twist).",
  )
  p.add_argument("--lr_pose", type=float, default=5e-3)
  p.add_argument("--lr_cam", type=float, default=1e-2)
  p.add_argument("--lr_id", type=float, default=2e-2)
  p.add_argument("--lr_ex", type=float, default=5e-2)
  p.add_argument("--grad_clip", type=float, default=5.0)
  p.add_argument(
      "--landmark_loss",
      choices=("l1", "huber"),
      default="l1",
      help="Per-landmark loss (huber = robust to bad correspondence).",
  )
  p.add_argument("--huber_delta", type=float, default=5.0,
                 help="Huber transition in pixels (landmark_loss=huber).")
  p.add_argument(
      "--render_mesh",
      action=argparse.BooleanOptionalAction,
      default=True,
      help="After fit: nvdiffrast true-perspective mesh at video size + red 3D lmks.",
  )
  p.add_argument(
      "--expr_pca_dim",
      type=int,
      default=0,
      help="If >0 with vhap_monocular: PCA the 350 face expression bases down to this "
           "many coeffs (FLAME-like). 0 = optimize all left/right/lower dims.",
  )
  args = p.parse_args()
  if args.preset == "vhap_monocular":
    # Align with VHAP-main output/.../config.yml LossWeightConfig + LearningRateConfig
    # for landmark-only stages (lmk_init_rigid / lmk_init_all style).
    # Expression: face PCA (--expr_pca_dim, default 150) or full left/right/lower.
    args.version_name = "vhap_monocular_lmk"
    if args.expr_pca_dim <= 0:
      args.expr_pca_dim = 150
    args.w_landmark = 10.0
    args.nose_weight = 10.0
    args.shape_mouth_weight = 1.0
    args.mouth_sock_weight = 0.0
    args.oval_weight = 1.0
    args.eye_weight = 1.0
    args.brow_weight = 1.0
    args.expr_mouth_weight = 1.0
    args.expr_mouth_sock_weight = 0.0
    args.expr_oval_weight = 1.0
    args.expr_brow_weight = 1.0
    args.expr_eye_weight = 1.0
    args.expr_base_weight = 1.0
    args.mouth_open_iters = 0
    args.w_mouth_open = 0.0
    args.freeze_identity_in_expr = False
    args.reg_id = 0.3
    args.reg_ex = 0.03
    args.reg_joint = 0.3
    args.reg_jaw_yz = 3.0
    args.lr_pose = 5e-3
    args.lr_cam = 5e-3
    args.lr_id = 5e-3
    args.lr_ex = 5e-2
    args.rigid_iters = 500
    args.shape_iters = 500
    # VHAP stops after lmk_init_all. The extra expression stage is not part of that.
    args.expr_iters = 0
    args.grad_clip = 0.0
    print(
        "[preset=vhap_monocular] balanced lmk (mouth×1 nose×10)  "
        f"reg_id={args.reg_id} reg_ex={args.reg_ex} reg_jaw_yz={args.reg_jaw_yz}  "
        f"lr_ex={args.lr_ex} freeze_id={args.freeze_identity_in_expr}  "
        f"expr_pca_dim={args.expr_pca_dim}",
        flush=True,
    )
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
  if args.lmk_set == "sparse68" and args.gt68_map == "mp2dlib":
    if Path(args.landmark_file).resolve() == GNM_68_FILE.resolve():
      print(
          "[warn] gt68_map=mp2dlib with official GNM 68 (jaw #75). "
          "Prefer GNM_Fit/head_sparse_68_ibug_order.txt",
          flush=True,
      )
  if args.expr_brow_weight is None:
    args.expr_brow_weight = 0.0
  shape_mouth = (
      args.shape_mouth_weight
      if args.shape_mouth_weight is not None
      else 0.0
  )
  shape_sock = args.mouth_sock_weight if shape_mouth > 0 else 0.0

  torch.set_num_threads(8)
  args.out_dir.mkdir(parents=True, exist_ok=True)

  lmk_npz = np.load(args.landmarks, allow_pickle=True)
  init = np.load(args.init, allow_pickle=True)
  gt_all = lmk_npz["landmarks"][args.frame_index].astype(np.float32)
  if args.lmk_set == "sparse68":
    gt_xy = lmk_sets.sparse68_gt_xy(gt_all, args.gt68_map)
  else:
    gt_xy = gt_all
  n_lmk = int(gt_xy.shape[0])
  width, height = [int(x) for x in lmk_npz["image_size"]]
  video_frame = int(lmk_npz["frame_indices"][args.frame_index])
  video_path = str(lmk_npz["video"]) if "video" in lmk_npz.files else None

  if args.lmk_set == "sparse68":
    if args.preset == "safe_expr":
      # 鼻子、眼、嘴保持原权重。下颌轮廓压低。眉毛提到和眼睛一样，让额头跟着眉走一点。
      w_safe = np.ones(68, dtype=np.float32)
      w_safe[0:17] = 0.15
      w_safe[17:27] = 1.0
      w_safe[27:36] = 10.0
      w_rigid = w_safe.copy()
      w_shape = w_safe.copy()
      w_expr = w_safe.copy()
      w_open = w_safe.copy()
      args.reg_ex = 0.15
      args.reg_jaw_yz = 3.0
      args.rigid_iters = 500
      args.expr_iters = 400
      args.version_name = "safe_expr"
      print(
          "[preset=safe_expr] identity after rigid (jaw weight 1, clamp ±2.5), "
          "then 4 expression directions; brow weight 1; focal free then frozen",
          flush=True,
      )
    elif args.preset == "vhap_monocular":
      w_vhap = ibug68_vhap_weights(
          nose_boost=args.nose_weight,
          mouth_boost=args.expr_mouth_weight,
          eye_boost=args.eye_weight,
          brow_boost=args.brow_weight,
      )
      w_rigid = np.ones(68, dtype=np.float32)
      w_shape = w_vhap.copy()
      w_expr = w_vhap.copy()
      w_open = ibug68_stage_weights(
          "mouth_open", args.nose_weight, max(args.expr_mouth_weight, 12.0),
          args.mouth_open_jaw_weight, 0.0, 0.0, base_w=0.0,
          inner_mouth_w=max(args.expr_mouth_weight, 12.0),
      )
    else:
      w_rigid = np.ones(68, dtype=np.float32)
      w_shape = ibug68_stage_weights(
          "shape", args.nose_weight, shape_mouth, args.oval_weight,
          args.eye_weight, args.brow_weight,
      )
      w_expr = ibug68_stage_weights(
          "expr", args.nose_weight, args.expr_mouth_weight, args.expr_oval_weight,
          args.eye_weight, args.expr_brow_weight, base_w=args.expr_base_weight,
          inner_mouth_w=args.expr_mouth_weight,
      )
      w_open = ibug68_stage_weights(
          "mouth_open", args.nose_weight, max(args.expr_mouth_weight, 12.0),
          args.mouth_open_jaw_weight, 0.0, 0.0, base_w=0.0,
          inner_mouth_w=max(args.expr_mouth_weight, 12.0),
      )
  else:
    w_rigid = uniform_weights()
    w_shape = vhap_stage_weights(
        "shape", args.nose_weight, shape_mouth, shape_sock,
        args.oval_weight, args.eye_weight, args.brow_weight,
    )
    w_expr = vhap_stage_weights(
        "expr", 0.4 * args.nose_weight, args.expr_mouth_weight,
        args.expr_mouth_sock_weight,
        args.expr_oval_weight, args.expr_eye_weight, args.expr_brow_weight,
        base_w=args.expr_base_weight,
    )
    w_open = vhap_stage_weights(
        "mouth_open", args.nose_weight, max(args.expr_mouth_weight, 12.0),
        3.0, args.mouth_open_jaw_weight, 0.0, 0.0, base_w=0.0,
    )
  print(
      f"==> {args.version_name}  {args.lmk_set} n={n_lmk}  "
      f"gt68_map={args.gt68_map if args.lmk_set == 'sparse68' else '-'}  "
      f"lmk_file={args.landmark_file}  "
      f"shape: oval={args.oval_weight} mouth={shape_mouth}  "
      f"expr: mouth={args.expr_mouth_weight} oval={args.expr_oval_weight} "
      f"base={args.expr_base_weight} open_iters={args.mouth_open_iters} "
      f"freeze_id={args.freeze_identity_in_expr}",
      flush=True,
  )

  gnm = gnm_pytorch.GNM.from_local(
      version=gnm_pytorch.GNMMajorVersion.V3,
      variant=gnm_pytorch.GNMVariant.HEAD,
  )
  gnm_np = gnm_numpy.GNM.from_local(
      version=gnm_numpy.GNMMajorVersion.V3,
      variant=gnm_numpy.GNMVariant.HEAD,
  )
  id_scale = torch.as_tensor(gnm_utils.scaling_for_identity_loss(gnm_np), dtype=torch.float32)
  ex_scale_full = gnm_utils.scaling_for_expression_loss(gnm_np)
  ex_scale_lower = torch.as_tensor(
      ex_scale_full[200:350, 200:350], dtype=torch.float32
  )
  # left_eye + right_eye + lower_face (tongue/pupils stay 0).
  ex_scale_face = torch.as_tensor(ex_scale_full[:350, :350], dtype=torch.float32)
  faces = np.asarray(gnm.triangles.detach().cpu().numpy())

  lmk_idx, lmk_w = mp_lmk.load_landmarks(args.landmark_file)
  if len(lmk_idx) != n_lmk:
    raise ValueError(
        f"landmark_file has {len(lmk_idx)} points but GT {args.lmk_set} has {n_lmk}"
    )
  state = FitState.from_init(init)
  if args.preset == "vhap_monocular" and args.expr_pca_dim > 0:
    # SVD of the 350 face bases in vertex space. Coeffs live in the top-K
    # right-singular subspace: c_face = Vt[:K].T @ w, same span truncation as FLAME.
    face_basis = np.asarray(gnm_np.expression_basis[:350], dtype=np.float64)
    m_mat = face_basis.reshape(350, -1).T  # (V*3, 350)
    _u, s_vals, vt = np.linalg.svd(m_mat, full_matrices=False)
    k_pca = min(int(args.expr_pca_dim), 350)
    energy = (s_vals[:k_pca] ** 2).sum() / (s_vals ** 2).sum()
    pca_dirs = vt[:k_pca].T.astype(np.float32)  # (350, K)
    state.expr_dirs = torch.as_tensor(pca_dirs)
    state.expr_w = torch.zeros(k_pca, dtype=torch.float32, requires_grad=True)
    state.ex_left.requires_grad_(False)
    state.ex_right.requires_grad_(False)
    state.ex_lower.requires_grad_(False)
    pca_path = args.out_dir / f"gnm_face_expr_pca{k_pca}.npz"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        pca_path,
        dirs=pca_dirs,
        singular_values=s_vals.astype(np.float32),
        energy_covered=np.float32(energy),
        k=np.int32(k_pca),
    )
    print(
        f"[expr_pca] face 350 -> {k_pca}  energy={100.0 * energy:.1f}%  "
        f"reg on w like FLAME  wrote {pca_path.name}",
        flush=True,
    )
  if args.focal_norm is not None:
    with torch.no_grad():
      state.focal_norm.fill_(float(args.focal_norm))
  if args.freeze_focal:
    state.focal_norm.requires_grad_(False)
    fx = float(state.focal_norm) * float(max(width, height))
    print(f"[freeze_focal] fx=fy={fx:.1f}px  focal_norm={float(state.focal_norm):.6f}", flush=True)
  mouth_locked = False
  mouth_gap_px = 0.0
  mouth_width_px = 0.0
  if (
      args.preset == "safe_expr"
      and args.lmk_set == "sparse68"
      and (args.mouth_basis or args.identity_from is not None)
  ):
    # 61/67、62/66、63/65 是上下内唇成对的点。闭口时它们叠成一条线。
    pair_gaps = [
        float(np.linalg.norm(gt_xy[a] - gt_xy[b]))
        for a, b in ((61, 67), (62, 66), (63, 65))
    ]
    mouth_gap_px = float(np.mean(pair_gaps))
    mouth_width_px = float(np.linalg.norm(gt_xy[48] - gt_xy[54]))
    mouth_locked = mouth_width_px > 1.0 and mouth_gap_px <= 0.04 * mouth_width_px
    print(
        f"[safe_expr] inner-lip gap={mouth_gap_px:.2f}px  "
        f"width={mouth_width_px:.1f}px  "
        f"ratio={mouth_gap_px / max(mouth_width_px, 1.0):.3f}  "
        f"{'seal opening, fit closed expression' if mouth_locked else 'fit mouth'}",
        flush=True,
    )
  if args.preset == "safe_expr":
    state.id_head.requires_grad_(False)
    state.id_eyes.requires_grad_(False)
    if args.mouth_basis or args.identity_from is not None:
      state.ex_lower.requires_grad_(True)
      state.expr_dirs = None
      state.expr_w = None
      w_rigid[48:68] = 0.15
      if args.identity_from is not None:
        saved = np.load(args.identity_from)
        with torch.no_grad():
          state.id_head.copy_(torch.as_tensor(saved["id_head"], dtype=torch.float32))
        print(
            f"[safe_expr] identity frozen L2={float(torch.linalg.norm(state.id_head)):.3f}  "
            f"mouth uses lower-face 150  from {args.identity_from}",
            flush=True,
        )
      else:
        print("[safe_expr] mouth uses lower-face 150, identity solved on this frame", flush=True)
    else:
      neutral = np.asarray(gnm_np(None, None, None, None), dtype=np.float64)
      dirs_np, dir_names = expr_dirs.build_directions(
          neutral,
          np.asarray(lmk_idx),
          np.asarray(lmk_w),
          np.asarray(gnm_np.expression_basis),
      )
      state.expr_dirs = torch.as_tensor(dirs_np)
      state.expr_w = torch.zeros(dirs_np.shape[1], dtype=torch.float32, requires_grad=True)
      state.ex_lower.requires_grad_(False)
      print(f"[safe_expr] directions {dir_names}", flush=True)
  gt_t = torch.as_tensor(gt_xy)
  idx_t = torch.as_tensor(lmk_idx, dtype=torch.long)
  bary_t = torch.as_tensor(lmk_w, dtype=torch.float32)

  if args.lmk_set == "sparse68":
    idx_contour = torch.as_tensor(sorted(lmk_sets.IBUG_JAW), dtype=torch.long)
    idx_nose = torch.as_tensor(sorted(lmk_sets.IBUG_NOSE), dtype=torch.long)
    idx_lips = torch.as_tensor(sorted(lmk_sets.IBUG_MOUTH), dtype=torch.long)
    idx_sock = torch.as_tensor(sorted(lmk_sets.IBUG_MOUTH_INNER), dtype=torch.long)
    idx_eyes = torch.as_tensor(sorted(lmk_sets.IBUG_EYES), dtype=torch.long)
    idx_brows = torch.as_tensor(sorted(lmk_sets.IBUG_BROWS), dtype=torch.long)
    idx_upper_outer = torch.as_tensor(sorted(lmk_sets.IBUG_MOUTH_UPPER_OUTER), dtype=torch.long)
    idx_lower_outer = torch.as_tensor(sorted(lmk_sets.IBUG_MOUTH_LOWER_OUTER), dtype=torch.long)
    inner_up, inner_lo = lmk_sets.IBUG_INNER_UP, lmk_sets.IBUG_INNER_LO
    outer_up, outer_lo = lmk_sets.IBUG_OUTER_UP, lmk_sets.IBUG_OUTER_LO
  else:
    idx_contour = torch.as_tensor(sorted(lmk_sets.MP_CONTOUR_DUAL | lmk_sets.MP_JAW68), dtype=torch.long)
    idx_nose = torch.as_tensor(sorted(lmk_sets.MP_NOSE_DENSE), dtype=torch.long)
    idx_lips = torch.as_tensor(sorted(lmk_sets.MP_LIPS), dtype=torch.long)
    idx_sock = torch.as_tensor(sorted(lmk_sets.MP_MOUTH_SOCK), dtype=torch.long)
    idx_eyes = torch.as_tensor(sorted(lmk_sets.MP_EYES), dtype=torch.long)
    idx_brows = torch.as_tensor(sorted(lmk_sets.MP_BROWS), dtype=torch.long)
    idx_upper_outer = torch.as_tensor(sorted(lmk_sets.MP_UPPER_LIP_OUTER), dtype=torch.long)
    idx_lower_outer = torch.as_tensor(sorted(lmk_sets.MP_LOWER_LIP_OUTER), dtype=torch.long)
    inner_up, inner_lo = 13, 14
    outer_up, outer_lo = 0, 17

  def posed_verts_world() -> torch.Tensor:
    return gnm(
        state.identity()[None],
        state.expression()[None],
        state.rotations()[None],
        state.translation[None],
    )[0]

  def sample_lmk3d(verts_world: torch.Tensor) -> torch.Tensor:
    v = verts_world[idx_t]
    return (v * bary_t.unsqueeze(-1)).sum(dim=1)

  def project_xy(lmk3d: torch.Tensor) -> torch.Tensor:
    return project_world_to_pixels(
        lmk3d,
        state.camera_rt(),
        state.camera_k(width, height),
        width,
        height,
        flip_y=True,
    )

  def metrics(pred_xy: torch.Tensor) -> dict[str, float]:
    err = torch.sqrt(((pred_xy.detach() - gt_t) ** 2).sum(-1))
    pred_np = pred_xy.detach()
    inner_gap = float(torch.linalg.norm(pred_np[inner_up] - pred_np[inner_lo]))
    gt_inner_gap = float(torch.linalg.norm(gt_t[inner_up] - gt_t[inner_lo]))
    outer_span = float(pred_np[outer_lo, 1] - pred_np[outer_up, 1])
    gt_outer_span = float(gt_t[outer_lo, 1] - gt_t[outer_up, 1])
    return {
        "rmse_all": float(err.mean()),
        "rmse_active": float(err.mean()),
        "rmse_contour": float(err[idx_contour].mean()),
        "rmse_nose": float(err[idx_nose].mean()),
        "rmse_eyes": float(err[idx_eyes].mean()),
        "rmse_brows": float(err[idx_brows].mean()),
        "rmse_lips": float(err[idx_lips].mean()),
        "rmse_mouth_sock": float(err[idx_sock].mean()) if idx_sock.numel() else 0.0,
        "rmse_lower_outer": float(err[idx_lower_outer].mean()),
        "inner_lip_gap_px": inner_gap,
        "gt_inner_lip_gap_px": gt_inner_gap,
        "outer_lip_span_px": outer_span,
        "gt_outer_lip_span_px": gt_outer_span,
    }

  def dump_stage(tag: str, title: str, bgr: np.ndarray | None) -> dict:
    with torch.no_grad():
      verts_np = posed_verts_world().numpy()
      lmk3d_np = sample_lmk3d(torch.as_tensor(verts_np)).detach().numpy()
      pred = project_xy(torch.as_tensor(lmk3d_np)).cpu().numpy()
      m = metrics(torch.as_tensor(pred))
      rt_np = state.camera_rt().detach().numpy()
      k4_np = state.camera_k(width, height).detach().numpy()
    label = (
        f"{title}  all={m['rmse_all']:.1f} ctr={m['rmse_contour']:.1f} "
        f"nose={m['rmse_nose']:.1f} brow={m['rmse_brows']:.1f} "
        f"lips={m['rmse_lips']:.1f} lo={m['rmse_lower_outer']:.1f} "
        f"span={m['outer_lip_span_px']:.0f}/{m['gt_outer_lip_span_px']:.0f} "
        f"inner={m['inner_lip_gap_px']:.0f}/{m['gt_inner_lip_gap_px']:.0f}"
    )
    if bgr is not None:
      vis = bgr.copy()
      r = 3 if n_lmk == 68 else 2
      for i in range(n_lmk):
        gx, gy = int(gt_xy[i, 0]), int(gt_xy[i, 1])
        px, py = int(pred[i, 0]), int(pred[i, 1])
        cv2.circle(vis, (gx, gy), r, (0, 255, 0), -1)
        cv2.line(vis, (gx, gy), (px, py), (0, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(vis, (px, py), r, (0, 0, 255), -1)
      cv2.putText(vis, label, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
      cv2.putText(vis, label, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
      path = args.out_dir / f"stage_{tag}_gt_pred_lines.jpg"
      cv2.imwrite(str(path), vis)
      print(f"  [dump] {path.name}  {label}", flush=True)
    if args.render_mesh:
      try:
        render_fitted_frontal(
            verts_np,
            faces,
            lmk3d_np,
            rt_np,
            k4_np,
            height,
            width,
            args.out_dir,
            f"stage_{tag}",
            title=f"{tag} mesh+{n_lmk}  {label}",
            renderer=mesh_renderer,
        )
      except Exception as exc:  # noqa: BLE001
        print(f"[render] stage {tag} mesh failed: {exc}", flush=True)
    stage_mesh_tags.append(tag)
    return m

  stage_bgr = None
  if "image_paths" in lmk_npz.files:
    img_path = str(lmk_npz["image_paths"][args.frame_index])
    stage_bgr = cv2.imread(img_path)
    if stage_bgr is None:
      print(f"[warn] cannot read image_paths[{args.frame_index}] {img_path}", flush=True)
  if stage_bgr is None and video_path:
    src = Path(video_path)
    if src.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
      stage_bgr = cv2.imread(str(src))
    else:
      cap = cv2.VideoCapture(video_path)
      cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame)
      ok, stage_bgr = cap.read()
      cap.release()
      if not ok:
        stage_bgr = None

  mesh_renderer = None
  stage_mesh_tags: list[str] = []
  if args.render_mesh and torch.cuda.is_available():
    from nvdiffrast_render import NVDiffRenderer  # pylint: disable=import-outside-toplevel
    mesh_renderer = NVDiffRenderer(
        use_opengl=False,
        lighting_type="front",
        lighting_space="camera",
        flip_y_raster=False,
    ).to(torch.device("cuda"))

  t0 = time.time()
  dump_stage("0_init", "init", stage_bgr)

  def weighted_landmark_loss(pred_xy: torch.Tensor, conf: torch.Tensor) -> torch.Tensor:
    if args.preset == "vhap_monocular":
      # VHAP normalize_image_points: pixels -> [-1, 1], then L1, then mean.
      # Nose boost stays inside the mean, so it is not diluted by sum(conf).
      scale = pred_xy.new_tensor([float(width), float(height)])
      pred_n = 2.0 * (pred_xy - scale * 0.5) / scale
      gt_n = 2.0 * (gt_t - scale * 0.5) / scale
      per = (pred_n - gt_n).abs().sum(dim=-1)
      return (per * conf).mean()
    err = torch.linalg.norm(gt_t - pred_xy, dim=-1)
    if args.landmark_loss == "huber":
      d = args.huber_delta
      per = torch.where(err < d, 0.5 * err * err, d * (err - 0.5 * d))
    else:
      per = err
    return (per * conf).sum() / conf.sum().clamp(min=1.0)

  def mouth_open_aux(pred_xy: torch.Tensor) -> torch.Tensor:
    if args.w_mouth_open <= 0:
      return pred_xy.new_zeros(())
    pred_span = pred_xy[outer_lo, 1] - pred_xy[outer_up, 1]
    gt_span = gt_t[outer_lo, 1] - gt_t[outer_up, 1]
    lower = pred_xy[idx_lower_outer].mean(dim=0)
    gt_lower = gt_t[idx_lower_outer].mean(dim=0)
    loss = (pred_span - gt_span).abs() + (lower[1] - gt_lower[1]).abs()
    if args.lmk_set == "sparse68":
      pred_inner = pred_xy[inner_lo, 1] - pred_xy[inner_up, 1]
      gt_inner = gt_t[inner_lo, 1] - gt_t[inner_up, 1]
      loss = loss + (pred_inner - gt_inner).abs()
    return args.w_mouth_open * loss

  def metric_reg(reg_id: bool, reg_ex: bool, reg_joints: bool) -> torch.Tensor:
    reg = torch.tensor(0.0)
    if reg_id:
      scaled = id_scale @ state.identity()
      reg = reg + args.reg_id * (scaled ** 2).mean()
    if reg_ex:
      if state.expr_w is not None:
        reg = reg + args.reg_ex * (state.expr_w ** 2).mean()
      elif args.preset == "vhap_monocular":
        ex_face = torch.cat([state.ex_left, state.ex_right, state.ex_lower], dim=0)
        scaled_ex = ex_scale_face @ ex_face
        reg = reg + args.reg_ex * (scaled_ex ** 2).mean()
      else:
        scaled_ex = ex_scale_lower @ state.ex_lower
        reg = reg + args.reg_ex * (scaled_ex ** 2).mean()
    if reg_joints and args.preset == "vhap_monocular":
      # VHAP compute_joint_L2_energy: reg_neck/reg_jaw 0.3, reg_eyes 0.03.
      # Jaw y/z is multiplied by reg_jaw again, and negative jaw-x is penalized.
      reg = reg + args.reg_joint * (state.rot_neck ** 2).mean()
      jaw = state.rot_head
      jaw_term = (jaw ** 2).mean() + torch.relu(-jaw[0]) * 10.0
      jaw_term = jaw_term + (jaw[1:] ** 2).mean() * args.reg_jaw_yz
      reg = reg + args.reg_joint * jaw_term
      reg = reg + 0.03 * (state.rot_leye ** 2).mean()
      reg = reg + 0.03 * (state.rot_reye ** 2).mean()
      reg = reg + 0.03 * ((state.rot_leye - state.rot_reye) ** 2).mean()
    elif reg_joints:
      for t in (state.rot_neck, state.rot_head, state.rot_leye, state.rot_reye, state.cam_aa):
        reg = reg + args.reg_joint * (t ** 2).mean()
      reg = reg + args.reg_joint * 0.1 * (state.cam_t ** 2).mean()
      reg = reg + args.reg_joint * 0.01 * ((state.focal_norm - 1.5) ** 2)
    if args.reg_jaw_yz > 0 and args.preset != "vhap_monocular":
      # VHAP: penalize jaw/head rotation about y/z to reduce mouth twist.
      reg = reg + args.reg_jaw_yz * (state.rot_head[1:] ** 2).mean()
    return reg

  lip_fwd_mm = [0.0]
  lip_open_mm = [0.0]

  def run_stage(
      name: str,
      iters: int,
      groups: list[dict],
      conf_np: np.ndarray,
      reg_id: bool,
      reg_ex: bool,
      reg_joints: bool,
      score_mode: str = "default",
      after_backward=None,
      after_step=None,
      extra_loss=None,
  ) -> dict:
    conf = torch.as_tensor(conf_np, dtype=torch.float32)
    opt = torch.optim.Adam(groups)
    base_lrs = [g["lr"] for g in opt.param_groups]
    best_score = float("inf")
    best = state.snapshot()
    best_m: dict = {}
    # VHAP keeps a constant Adam step and the last iterate.
    vhap_solver = args.preset == "vhap_monocular"
    print(f"\n==> {name}  iters={iters}  cosine={not vhap_solver}", flush=True)
    for i in range(iters):
      t = i / max(iters - 1, 1)
      lr_scale = 1.0 if vhap_solver else 0.1 + 0.9 * (0.5 * (1.0 + np.cos(np.pi * t)))
      for g, base in zip(opt.param_groups, base_lrs):
        g["lr"] = base * lr_scale
      opt.zero_grad()
      pred = project_xy(sample_lmk3d(posed_verts_world()))
      open_aux = (
          mouth_open_aux(pred)
          if score_mode in ("mouth", "mouth_open")
          else pred.new_zeros(())
      )
      extra = extra_loss() if extra_loss is not None else pred.new_zeros(())
      loss = (
          args.w_landmark * weighted_landmark_loss(pred, conf)
          + open_aux
          + extra
          + metric_reg(reg_id, reg_ex, reg_joints)
      )
      loss.backward()
      if after_backward is not None:
        after_backward()
      if args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(
            [par for g in groups for par in g["params"]], args.grad_clip
        )
      cur = metrics(pred)
      err = torch.sqrt(((pred.detach() - gt_t) ** 2).sum(-1))
      active = conf > 0.05
      if score_mode == "shape":
        if shape_mouth > 0 and args.preset == "safe_expr":
          score = (
              0.25 * float(err[active].mean())
              + 0.20 * cur["rmse_contour"]
              + 0.35 * cur["rmse_lips"]
              + 0.20 * cur["rmse_brows"]
          )
        elif shape_mouth > 0:
          score = (
              0.30 * float(err[active].mean())
              + 0.25 * cur["rmse_contour"]
              + 0.45 * cur["rmse_lips"]
          )
        else:
          score = 0.6 * float(err[active].mean()) + 0.4 * cur["rmse_contour"]
      elif score_mode == "lips":
        score = cur["rmse_lips"]
      elif score_mode == "lips_flat":
        score = (
            cur["rmse_lips"]
            + 1.5 * float(lip_fwd_mm[0])
            + 2.0 * float(lip_open_mm[0])
        )
      elif score_mode == "mouth":
        span_err = abs(cur["outer_lip_span_px"] - cur["gt_outer_lip_span_px"])
        inner_err = abs(cur["inner_lip_gap_px"] - cur["gt_inner_lip_gap_px"])
        score = (
            0.50 * cur["rmse_lips"]
            + 0.30 * cur["rmse_lower_outer"]
            + 0.10 * span_err
            + 0.10 * inner_err
        )
      elif score_mode == "mouth_open":
        span_err = abs(cur["outer_lip_span_px"] - cur["gt_outer_lip_span_px"])
        inner_err = abs(cur["inner_lip_gap_px"] - cur["gt_inner_lip_gap_px"])
        if args.lmk_set == "sparse68":
          score = cur["rmse_lower_outer"] + 0.25 * span_err + 0.5 * inner_err
        else:
          score = cur["rmse_lower_outer"] + 0.25 * span_err
      else:
        score = float(err[active].mean()) if active.any() else cur["rmse_all"]
      if score < best_score:
        best_score = score
        best = state.snapshot()
        best_m = dict(cur)
        best_m["best_iter"] = i + 1
        best_m["score"] = score
      opt.step()
      with torch.no_grad():
        state.focal_norm.clamp_(min=0.3, max=5.0)
        if state.expr_w is not None:
          state.expr_w.clamp_(-1.5, 2.0)
        if args.mouth_basis and state.ex_lower.requires_grad:
          state.ex_lower.clamp_(-4.0, 4.0)
        if args.preset == "safe_expr" and state.id_head.requires_grad:
          state.id_head.clamp_(-2.5, 2.5)
        if after_step is not None:
          after_step()
      if i == 0 or (i + 1) % max(iters // 10, 1) == 0:
        print(
            f"  [{name}] {i+1:4d}/{iters}  "
            f"all={cur['rmse_all']:.2f} ctr={cur['rmse_contour']:.2f} "
            f"brow={cur['rmse_brows']:.2f} lips={cur['rmse_lips']:.2f}  "
            f"lo={cur['rmse_lower_outer']:.1f} "
            f"span={cur['outer_lip_span_px']:.0f}/{cur['gt_outer_lip_span_px']:.0f} "
            f"inner={cur['inner_lip_gap_px']:.0f}/{cur['gt_inner_lip_gap_px']:.0f}  "
            f"fwd={lip_fwd_mm[0]:.1f}mm  "
            f"best={best_score:.2f}@{best_m.get('best_iter', 0)}",
            flush=True,
        )
    if not vhap_solver:
      state.restore(best)
    else:
      with torch.no_grad():
        pred_last = project_xy(sample_lmk3d(posed_verts_world()))
      best_m = dict(metrics(pred_last))
      best_m["best_iter"] = iters
      best_m["score"] = best_m.get("rmse_all", 0.0)
    print(
        f"  [{name}] best @{best_m.get('best_iter')}  "
        f"all={best_m.get('rmse_all', 0):.2f} brow={best_m.get('rmse_brows', 0):.2f} "
        f"lips={best_m.get('rmse_lips', 0):.2f}",
        flush=True,
    )
    return best_m

  def trainable(groups):
    kept = []
    for group in groups:
      params = [t for t in group["params"] if t.requires_grad]
      if params:
        kept.append({**group, "params": params})
    return kept

  # Stage1: camera + head rigid body. Neck is the GNM root, so it is the
  # rigid rotation. Jaw (rot_head) and eyes stay at zero — they change the face.
  if args.preset == "vhap_monocular":
    # VHAP fixes the camera RT and only solves focal length.
    # The head rigid body is root rotation (GNM neck) plus translation.
    state.cam_aa.requires_grad_(False)
    state.cam_t.requires_grad_(False)
    rigid_groups = [
        {"params": [state.rot_neck], "lr": args.lr_pose},
        {"params": [state.focal_norm], "lr": args.lr_cam},
        {"params": [state.translation], "lr": 1e-3},
    ]
  else:
    rigid_groups = [
        {"params": [state.rot_neck, state.rot_head], "lr": args.lr_pose},
        {"params": [state.cam_aa], "lr": args.lr_pose},
        {"params": [state.cam_t], "lr": args.lr_cam},
        {"params": [state.focal_norm], "lr": args.lr_cam * 0.2},
        {"params": [state.translation], "lr": args.lr_pose * 0.2},
    ]
  last = run_stage(
      "gnm_lmk_init_rigid",
      args.rigid_iters,
      trainable(rigid_groups),
      w_rigid,
      reg_id=False,
      reg_ex=False,
      reg_joints=True,
  )
  dump_stage("1_rigid", f"stage1 rigid only ({args.lmk_set})", stage_bgr)
  if args.stop_after == "rigid":
    print("[stop_after=rigid] landmark rigid done; shape and later stages skipped", flush=True)
  elif args.preset == "safe_expr":
    state.focal_norm.requires_grad_(False)
    fx = float(state.focal_norm) * float(max(width, height))
    print(
        f"[safe_expr] focal frozen after rigid  fx=fy={fx:.1f}px  "
        f"focal_norm={float(state.focal_norm):.4f}",
        flush=True,
    )
    if args.identity_from is None:
      # 身份补脸宽、外轮廓，并按眉毛把额头带一点。嘴留给后面的表情方向。
      state.id_head.requires_grad_(True)
      w_id = w_expr.copy()
      w_id[0:17] = 1.0
      if args.mouth_basis:
        w_id[48:68] = 0.15
      args.reg_id = 0.3
      run_stage(
          "gnm_lmk_init_identity",
          400,
          [{"params": [state.id_head], "lr": 1e-2}],
          w_id,
          reg_id=True,
          reg_ex=False,
          reg_joints=False,
          score_mode="shape",
      )
      dump_stage("2_identity", "stage2 identity head ±2.5", stage_bgr)
      with torch.no_grad():
        id_now = state.id_head.detach()
        print(
            f"[safe_expr] identity after shape  L2={float(torch.linalg.norm(id_now)):.3f}  "
            f"max={float(id_now.abs().max()):.3f}  "
            f"n>|0.5|={int((id_now.abs() > 0.5).sum())}",
            flush=True,
        )
    else:
      print("[safe_expr] identity stage skipped, mouth uses expression directions", flush=True)

  # Stage2: identity, joints, expression, and pose together. Landmark only.
  # safe_expr 的身份已经在上面单独跑过。
  if args.stop_after != "rigid" and args.preset != "safe_expr":
    if args.preset == "vhap_monocular":
      # VHAP-like: one low-dim face expression (PCA) or full regional dims.
      if state.expr_w is not None and state.expr_dirs is not None and state.expr_dirs.shape[0] == 350:
        ex_params = [state.expr_w]
      else:
        ex_params = [state.ex_left, state.ex_right, state.ex_lower]
      shape_groups = [
          {"params": [state.id_head, state.id_eyes], "lr": args.lr_id},
          {"params": ex_params, "lr": args.lr_ex},
          {"params": [state.rot_leye, state.rot_reye], "lr": args.lr_pose},
          {"params": [state.rot_neck], "lr": args.lr_pose},
          {"params": [state.rot_head], "lr": args.lr_pose},
          {"params": [state.focal_norm], "lr": args.lr_cam},
          {"params": [state.translation], "lr": 1e-3},
      ]
    else:
      shape_groups = [
          {"params": [state.id_head, state.id_eyes], "lr": args.lr_id},
          {"params": [state.rot_leye, state.rot_reye], "lr": args.lr_pose},
          {"params": [state.rot_neck, state.rot_head], "lr": args.lr_pose * 0.5},
          {"params": [state.cam_aa], "lr": args.lr_pose * 0.3},
          {"params": [state.cam_t, state.focal_norm], "lr": args.lr_cam * 0.2},
          {"params": [state.translation], "lr": args.lr_pose * 0.15},
      ]
    if args.preset != "vhap_monocular" and (shape_mouth > 0):
      shape_groups.insert(1, {"params": [state.ex_lower], "lr": args.lr_ex})
    last = run_stage(
        "gnm_lmk_init_shape",
        args.shape_iters,
        trainable(shape_groups),
        w_shape,
        reg_id=True,
        reg_ex=shape_mouth > 0 or args.preset == "vhap_monocular",
        reg_joints=True,
        score_mode="shape",
    )
    dump_stage(
        "2_shape",
        f"stage2 shape (mouth={shape_mouth:g} sock={shape_sock:g} preset={args.preset})",
        stage_bgr,
    )

  run_later = args.stop_after == "all"
  if run_later and args.preset != "safe_expr" and args.mouth_open_iters > 0:
    print(
        f"==> mouth-open AFTER shape  iters={args.mouth_open_iters}  "
        f"w_open={args.w_mouth_open}  jaw_w={args.mouth_open_jaw_weight}  "
        f"(expression loss, no reanchor)",
        flush=True,
    )
    run_stage(
        "gnm_lmk_mouth_open",
        args.mouth_open_iters,
        [
            {"params": [state.ex_lower], "lr": args.lr_ex * 1.5},
            {"params": [state.rot_neck, state.rot_head], "lr": args.lr_pose * 0.15},
        ],
        w_open,
        reg_id=False,
        reg_ex=True,
        reg_joints=True,
        score_mode="mouth_open",
    )
    dump_stage("2b_mouth_open", "stage2b mouth-open (expr loss, no reanchor)", stage_bgr)

  # 嘴只用前 24 个分量，并惩罚唇点朝相机（+Z）探出。下颌铰链不动。
  # 检测到闭口时，上下内唇保持贴住，唇线的表情仍然拟合。
  if (
      run_later
      and args.preset == "safe_expr"
      and (args.mouth_basis or args.identity_from is not None)
  ):
    w_lip = np.full(68, 0.15, dtype=np.float32)
    w_lip[27:36] = 10.0
    w_lip[48:68] = 1.0
    lip_sl = slice(48, 68)
    seal_pairs = ((61, 67), (62, 66), (63, 65))
    with torch.no_grad():
      rest_v = gnm(
          state.identity()[None],
          torch.zeros(1, 383),
          torch.zeros(1, 4, 3),
          torch.zeros(1, 3),
      )[0]
      rest_lmk = sample_lmk3d(rest_v).detach()
      rest_lip_z = rest_lmk[lip_sl, 2]
      rest_sep = torch.stack([
          torch.linalg.norm(rest_lmk[a] - rest_lmk[b]) for a, b in seal_pairs
      ])
    if mouth_locked:
      print(
          f"[safe_expr] closed mouth keeps its expression  gap={mouth_gap_px:.2f}px  "
          f"inner lips stay together",
          flush=True,
      )

    def zero_tail_grad() -> None:
      if state.ex_lower.grad is not None:
        state.ex_lower.grad[24:] = 0

    def zero_tail() -> None:
      state.ex_lower[24:].zero_()

    def lip_extra() -> torch.Tensor:
      cur_v = gnm(
          state.identity().detach()[None],
          state.expression()[None],
          torch.zeros(1, 4, 3),
          torch.zeros(1, 3),
      )[0]
      lmk = sample_lmk3d(cur_v)
      dz = lmk[lip_sl, 2] - rest_lip_z
      fwd = torch.relu(dz)
      lip_fwd_mm[0] = float(fwd.sum().detach()) * 1000.0
      loss = fwd.sum() * 4000.0
      if mouth_locked:
        sep = torch.stack([
            torch.linalg.norm(lmk[a] - lmk[b]) for a, b in seal_pairs
        ])
        opened = torch.relu(sep - rest_sep)
        lip_open_mm[0] = float(opened.sum().detach()) * 1000.0
        loss = loss + opened.sum() * 4000.0
      return loss

    last = run_stage(
        "gnm_lmk_mouth_closed" if mouth_locked else "gnm_lmk_mouth24",
        400,
        [{"params": [state.ex_lower], "lr": args.lr_ex}],
        w_lip,
        reg_id=False,
        reg_ex=True,
        reg_joints=False,
        score_mode="lips_flat",
        after_backward=zero_tail_grad,
        after_step=zero_tail,
        extra_loss=lip_extra,
    )
    dump_stage(
        "3_mouth_closed" if mouth_locked else "3_mouth24",
        (
            "stage3 closed expression, opening sealed"
            if mouth_locked
            else "stage3 first 24, lip +Z penalized"
        ),
        stage_bgr,
    )
    vhap_stage3 = False
    expr_groups = []
  # Stage3: expression refine. VHAP-like: full-face landmarks + joints (not mouth-only).
  # safe_expr：只动四条表情方向、少量下颌和眼球，身份和焦距不再动。
  elif args.preset == "safe_expr":
    vhap_stage3 = False
    expr_groups = [
        {"params": [state.expr_w], "lr": 5e-2},
        {"params": [state.rot_head], "lr": args.lr_pose * 0.15},
        {"params": [state.rot_leye, state.rot_reye], "lr": args.lr_pose * 0.3},
    ]
  elif run_later:
    vhap_stage3 = args.preset == "vhap_monocular" or args.expr_base_weight > 0
    expr_groups = [
        {"params": [state.ex_lower], "lr": args.lr_ex},
    ]
    if not args.freeze_identity_in_expr:
      expr_groups.insert(1, {"params": [state.id_head], "lr": args.lr_id * 0.15})
    if vhap_stage3:
      expr_groups.extend([
          {"params": [state.rot_neck, state.rot_head], "lr": args.lr_pose * 0.5},
          {"params": [state.rot_leye, state.rot_reye], "lr": args.lr_pose},
          {"params": [state.cam_aa], "lr": args.lr_pose * 0.3},
          {"params": [state.cam_t, state.focal_norm], "lr": args.lr_cam * 0.2},
          {"params": [state.translation], "lr": args.lr_pose * 0.15},
      ])
  if not run_later:
    expr_groups = []
    print(
        f"[stop_after={args.stop_after}] later expression refine skipped; no mask loss",
        flush=True,
    )
  if expr_groups and args.expr_iters > 0:
    expr_groups = trainable(expr_groups)
    last = run_stage(
        "gnm_lmk_init_expression",
        args.expr_iters,
        expr_groups,
        w_expr,
        reg_id=not args.freeze_identity_in_expr,
        reg_ex=True,
        reg_joints=vhap_stage3,
        score_mode="default" if vhap_stage3 else "mouth",
    )
    dump_stage(
        "3_expression",
        (
            "stage3 safe_expr directions"
            if args.preset == "safe_expr"
            else (
                f"stage3 vhap-balanced (mouth×{args.expr_mouth_weight:g} nose×{args.nose_weight:g})"
                if vhap_stage3
                else f"stage3 mouth-only (w={args.expr_mouth_weight:g} sock={args.expr_mouth_sock_weight:g})"
            )
        ),
        stage_bgr,
    )

  photo_stats: dict = {}
  if args.photometric and args.stop_after != "all":
    print("[stop_after] photometric / mask stage skipped", flush=True)
  elif args.photometric:
    import photo_fit  # pylint: disable=import-outside-toplevel

    photo_stats = photo_fit.run_photometric(
        args=args,
        state=state,
        gnm=gnm,
        gnm_np=gnm_np,
        posed_verts_world=posed_verts_world,
        sample_lmk3d=sample_lmk3d,
        project_xy=project_xy,
        dump_stage=dump_stage,
        stage_bgr=stage_bgr,
        gt_xy=gt_xy,
        gt_t=gt_t,
        width=width,
        height=height,
        mouth_locked=mouth_locked,
        metric_reg=metric_reg,
        weighted_landmark_loss=weighted_landmark_loss,
        w_lmk=w_expr,
        faces=faces,
    )
    if photo_stats:
      last = {**last, **{k: v for k, v in photo_stats.items() if k not in ("photo_tex", "photo_all", "photo_offset")}}
      last["photo_tex"] = photo_stats.get("photo_tex", {})
      last["photo_all"] = photo_stats.get("photo_all", {})
      last["photo_offset"] = photo_stats.get("photo_offset", {})

  elapsed = time.time() - t0
  with torch.no_grad():
    verts = posed_verts_world().numpy()
    pred = project_xy(sample_lmk3d(posed_verts_world())).cpu().numpy()
    final_m = metrics(torch.as_tensor(pred))
    id_h = float(torch.linalg.norm(state.id_head))
    if state.expr_w is not None:
      print(
          "[safe_expr] weights "
          + " ".join(
              f"{n}={float(v):+.3f}"
              for n, v in zip(expr_dirs.NAMES, state.expr_w)
          ),
          flush=True,
      )
    elif args.mouth_basis or args.identity_from is not None:
      ex = state.ex_lower.detach()
      print(
          f"[safe_expr] ex_lower L2={float(torch.linalg.norm(ex)):.3f}  "
          f"max={float(ex.abs().max()):.3f}  "
          f"n>|0.5|={int((ex.abs() > 0.5).sum())}  "
          f"tail24={int((ex[24:].abs() > 1e-4).sum())}  "
          f"mouth_locked={mouth_locked}  open={lip_open_mm[0]:.2f}mm",
          flush=True,
      )
    rt_np = state.camera_rt().detach().numpy()
    k_np = state.camera_k(width, height).detach().numpy()

  mesh_path = args.out_dir / f"frame_{video_frame:06d}.obj"
  trimesh.Trimesh(vertices=verts, faces=faces, process=False).export(mesh_path)

  overlay_path = args.out_dir / f"frame_{video_frame:06d}_overlay.jpg"
  if stage_bgr is not None:
    vis = stage_bgr.copy()
    r = 3 if n_lmk == 68 else 1
    for i in range(n_lmk):
      gx, gy = int(gt_xy[i, 0]), int(gt_xy[i, 1])
      px, py = int(pred[i, 0]), int(pred[i, 1])
      cv2.circle(vis, (gx, gy), r, (0, 255, 0), -1)
      cv2.line(vis, (gx, gy), (px, py), (0, 255, 255), 1, cv2.LINE_AA)
      cv2.circle(vis, (px, py), r, (0, 0, 255), -1)
    cv2.putText(
        vis,
        f"{args.version_name} {args.lmk_set} persp all={final_m['rmse_all']:.1f} "
        f"lips={final_m['rmse_lips']:.1f} "
        f"span={final_m['outer_lip_span_px']:.0f}/{final_m['gt_outer_lip_span_px']:.0f} "
        f"inner={final_m['inner_lip_gap_px']:.0f}/{final_m['gt_inner_lip_gap_px']:.0f}",
        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
    )
    cv2.imwrite(str(overlay_path), vis)

  overlay_lmk_path = args.out_dir / f"frame_{video_frame:06d}_overlay_lmk.jpg"
  star_gt_path = args.out_dir / f"frame_{video_frame:06d}_star_gt.jpg"
  if stage_bgr is not None:
    vis_gt = _draw_landmarks_on_mesh(stage_bgr, gt_xy)
    cv2.putText(
        vis_gt, f"STAR/iBUG 68 GT  {args.gt68_map}",
        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3,
    )
    cv2.putText(
        vis_gt, f"STAR/iBUG 68 GT  {args.gt68_map}",
        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
    )
    cv2.imwrite(str(star_gt_path), vis_gt)
    vis_lmk = _draw_landmarks_on_mesh(stage_bgr, pred)
    cv2.putText(
        vis_lmk,
        f"{args.version_name} GNM official 68 pred  all={final_m['rmse_all']:.1f} "
        f"lips={final_m['rmse_lips']:.1f}",
        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3,
    )
    cv2.putText(
        vis_lmk,
        f"{args.version_name} GNM official 68 pred  all={final_m['rmse_all']:.1f} "
        f"lips={final_m['rmse_lips']:.1f}",
        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
    )
    cv2.imwrite(str(overlay_lmk_path), vis_lmk)

  render_paths: dict[str, str] = {}
  if args.render_mesh:
    lmk3d_np = sample_lmk3d(torch.as_tensor(verts)).detach().numpy()
    k4_np = state.camera_k(width, height).detach().numpy()
    try:
      render_paths = render_fitted_frontal(
          verts,
          faces,
          lmk3d_np,
          rt_np,
          k4_np,
          height,
          width,
          args.out_dir,
          f"frame_{video_frame:06d}",
          title=(
              f"{args.version_name} render+lmk3d  all={final_m['rmse_all']:.1f} "
              f"lips={final_m['rmse_lips']:.1f}"
          ),
          renderer=mesh_renderer,
      )
    except Exception as exc:  # noqa: BLE001 — keep fit outputs even if render fails
      print(f"[render] failed: {exc}", flush=True)

  np.savez(
      args.out_dir / f"frame_{video_frame:06d}_params.npz",
      version=args.version_name,
      method="vhap_3stage_perspective",
      camera_model="perspective_k_rt",
      identity=state.identity().detach().numpy(),
      expression=state.expression().detach().numpy(),
      id_head=state.id_head.detach().numpy(),
      id_eyes=state.id_eyes.detach().numpy(),
      expression_left_eye=(
          state.expression().detach().numpy()[:100]
          if state.expr_dirs is not None and state.expr_dirs.shape[0] == 350
          else state.ex_left.detach().numpy()
      ),
      expression_right_eye=(
          state.expression().detach().numpy()[100:200]
          if state.expr_dirs is not None and state.expr_dirs.shape[0] == 350
          else state.ex_right.detach().numpy()
      ),
      expression_lower_face=(
          state.expression().detach().numpy()[200:350]
          if state.expr_dirs is not None and state.expr_dirs.shape[0] == 350
          else state.ex_lower.detach().numpy()
      ),
      **(
          {
              "expression_pca_w": state.expr_w.detach().numpy(),
              "expression_pca_dim": np.int32(state.expr_w.numel()),
          }
          if state.expr_w is not None and state.expr_dirs is not None
          and state.expr_dirs.shape[0] == 350
          else {}
      ),
      rotations=state.rotations().detach().numpy(),
      translation=state.translation.detach().numpy(),
      cam_aa=state.cam_aa.detach().numpy(),
      cam_t=state.cam_t.detach().numpy(),
      focal_norm=float(state.focal_norm.detach()),
      K=k_np,
      RT=rt_np,
      image_size=np.array([width, height], dtype=np.int32),
      **final_m,
      **{
          k: v for k, v in last.items()
          if k not in final_m and np.isscalar(v)
      },
  )

  summary = {
      "version": args.version_name,
      "preset": args.preset,
      "method": "vhap_3stage_perspective",
      "camera_model": "perspective_k_rt",
      "lmk_set": args.lmk_set,
      "gt68_map": args.gt68_map if args.lmk_set == "sparse68" else None,
      "landmark_file": str(args.landmark_file),
      "n_lmk": n_lmk,
      "rigid_iters": args.rigid_iters,
      "shape_iters": args.shape_iters,
      "expr_iters": args.expr_iters,
      "w_landmark": args.w_landmark,
      "nose_weight": args.nose_weight,
      "mouth_weight": args.mouth_weight,
      "oval_weight": args.oval_weight,
      "shape_mouth_weight": shape_mouth,
      "expr_mouth_weight": args.expr_mouth_weight,
      "expr_mouth_sock_weight": args.expr_mouth_sock_weight,
      "expr_oval_weight": args.expr_oval_weight,
      "expr_base_weight": args.expr_base_weight,
      "mouth_open_iters": args.mouth_open_iters,
      "w_mouth_open": args.w_mouth_open,
      "freeze_identity_in_expr": args.freeze_identity_in_expr,
      "eye_weight": args.eye_weight,
      "brow_weight": args.brow_weight,
      "reg_id": args.reg_id,
      "reg_ex": args.reg_ex,
      "reg_joint": args.reg_joint,
      "reg_jaw_yz": args.reg_jaw_yz,
      "landmark_loss": args.landmark_loss,
      "huber_delta": args.huber_delta,
      "elapsed_sec": elapsed,
      "mesh": str(mesh_path),
      "overlay": str(overlay_path),
      "overlay_lmk": str(overlay_lmk_path) if stage_bgr is not None else None,
      "stage_mesh_tags": stage_mesh_tags,
      "stages_mesh_lmk_strip": write_stage_mesh_strip(args.out_dir, stage_mesh_tags),
      **render_paths,
      "id_head_L2": id_h,
      "mouth_locked": mouth_locked,
      "mouth_inner_gap_px": mouth_gap_px,
      "mouth_width_px": mouth_width_px,
      "photometric": bool(args.photometric),
      **final_m,
      **last,
  }
  (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
  print(json.dumps(summary, indent=2), flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
