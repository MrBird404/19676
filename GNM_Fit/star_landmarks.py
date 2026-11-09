"""STAR 300W → iBUG/Dlib 68（与官方 GNM 68 配对）。

不经过 MediaPipe 468 映射。裁切优先 dlib（与 VHAP 相同）；没有 dlib 时
用子进程跑 MediaPipe 只取脸框，避免本进程 mediapipe+torch 冲突。
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
# 代码在 VHAP 自带的 STAR；权重在 detectors/STAR/checkpoints（vendor 里是软链）。
STAR_ROOT = REPO_ROOT / "VHAP-main" / "vendor" / "STAR-master"
CKPT = STAR_ROOT / "checkpoints" / "300W_STARLoss_NME_2_87.pkl"
DLIB_DAT = STAR_ROOT / "checkpoints" / "shape_predictor_68_face_landmarks.dat"

_NET = None
_DEVICE: torch.device | None = None


def _mod(name: str, **attrs):
  m = types.ModuleType(name)
  for k, v in attrs.items():
    setattr(m, k, v)
  sys.modules[name] = m
  return m


def _load_file(name: str, path: Path):
  spec = importlib.util.spec_from_file_location(name, path)
  mod = importlib.util.module_from_spec(spec)
  sys.modules[name] = mod
  spec.loader.exec_module(mod)
  return mod


def _ensure_star_imports() -> None:
  if "star.lib.backbone.stackedHGNetV1" in sys.modules:
    return
  if "torch.utils.tensorboard" not in sys.modules:
    tb = _mod("torch.utils.tensorboard")
    class _SW:
      def __init__(self, *a, **k):
        pass
      def close(self):
        pass
    tb.SummaryWriter = _SW
  if str(STAR_ROOT) not in sys.path:
    sys.path.insert(0, str(STAR_ROOT))
  dec = _load_file(
      "star_decoder_default",
      STAR_ROOT / "star/lib/dataset/decoder/decoder_default.py",
  )
  ds = _mod("star.lib.dataset", get_decoder=lambda decoder_type="default": dec.decoder_default())
  ds.get_encoder = lambda *a, **k: None
  import star  # noqa: F401
  lib = types.ModuleType("star.lib")
  lib.__path__ = [str(STAR_ROOT / "star/lib")]
  sys.modules["star.lib"] = lib
  bb = types.ModuleType("star.lib.backbone")
  bb.__path__ = [str(STAR_ROOT / "star/lib/backbone")]
  sys.modules["star.lib.backbone"] = bb


class GetCropMatrix:
  def __init__(self, image_size: int, target_face_scale: float, align_corners: bool = True):
    self.image_size = image_size
    self.target_face_scale = target_face_scale
    self.align_corners = align_corners

  def process(self, scale: float, center_w: float, center_h: float) -> np.ndarray:
    to_w = to_h = (self.image_size - 1) if self.align_corners else self.image_size
    scale_mu = self.image_size / (scale * self.target_face_scale * 200.0)
    acos, asin = scale_mu, 0.0
    fx, fy, tx, ty = center_w, center_h, to_w / 2.0, to_h / 2.0
    return np.array([
        [acos, -asin, tx - acos * fx + asin * fy],
        [asin, acos, ty - asin * fx - acos * fy],
        [0.0, 0.0, 1.0],
    ], np.float32)


def _denorm(points: torch.Tensor, input_size: int = 256) -> torch.Tensor:
  return ((points + 1) * torch.tensor(
      [input_size, input_size], device=points.device).view(1, 1, 2) - 1) / 2


def _unwarp(src: np.ndarray, coeff: np.ndarray) -> np.ndarray:
  dst = np.zeros_like(src, dtype=np.float32)
  for i in range(src.shape[0]):
    dst[i, 0] = coeff[0, 0] * src[i, 0] + coeff[0, 1] * src[i, 1] + coeff[0, 2]
    dst[i, 1] = coeff[1, 0] * src[i, 0] + coeff[1, 1] * src[i, 1] + coeff[1, 2]
  return dst


def _box_to_scale_center(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float]:
  scale = min(x2 - x1, y2 - y1) / 200.0 * 1.05
  return scale, (x1 + x2) / 2.0, (y1 + y2) / 2.0


def crop_from_dlib(bgr: np.ndarray) -> tuple[float, float, float] | None:
  try:
    import dlib
  except ImportError:
    return None
  if not DLIB_DAT.is_file():
    return None
  rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
  detector = dlib.get_frontal_face_detector()
  boxes = detector(rgb, 1)
  if not boxes:
    return None
  pred = dlib.shape_predictor(str(DLIB_DAT))
  face = pred(rgb, boxes[0])
  pts = np.array([[face.part(i).x, face.part(i).y] for i in range(68)], dtype=np.float64)
  return _box_to_scale_center(pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max())


def crop_from_mp_subprocess(image_path: Path) -> tuple[float, float, float]:
  """子进程只跑 MediaPipe，取 468 外接框。"""
  code = (
      "import sys,cv2,numpy as np\n"
      f"sys.path.insert(0,{str(ROOT)!r})\n"
      "from tools.detect import detect_mp468, read_bgr\n"
      f"p=read_bgr(__import__('pathlib').Path({str(image_path)!r}))\n"
      "pts=detect_mp468(p)\n"
      "print(pts[:,0].min(),pts[:,1].min(),pts[:,0].max(),pts[:,1].max())\n"
  )
  proc = subprocess.run(
      [sys.executable, "-c", code],
      check=True, capture_output=True, text=True,
  )
  lines = [ln for ln in proc.stdout.splitlines() if ln.strip() and not ln.startswith("WARNING")]
  # last numeric line
  nums = None
  for ln in reversed(proc.stdout.splitlines()):
    parts = ln.split()
    if len(parts) == 4:
      try:
        nums = [float(x) for x in parts]
        break
      except ValueError:
        continue
  if nums is None:
    raise RuntimeError(f"MediaPipe bbox subprocess failed:\n{proc.stderr}\n{proc.stdout}")
  return _box_to_scale_center(*nums)


def resolve_crop(bgr: np.ndarray, image_path: Path | None = None) -> tuple[float, float, float, str]:
  crop = crop_from_dlib(bgr)
  if crop is not None:
    return (*crop, "dlib")
  if image_path is not None:
    return (*crop_from_mp_subprocess(image_path), "mediapipe_bbox")
  raise RuntimeError("no dlib and no image_path for MediaPipe bbox")


def load_star_net(device: torch.device | None = None):
  global _NET, _DEVICE
  if _NET is not None:
    return _NET, _DEVICE
  if not CKPT.is_file():
    raise FileNotFoundError(f"STAR 300W weights missing: {CKPT}")
  _ensure_star_imports()
  from star.conf.alignment import Alignment as StarCfg
  from star.lib.backbone.stackedHGNetV1 import StackedHGNetV1

  args = argparse.Namespace()
  args.config_name = "alignment"
  cfg = StarCfg(args)
  device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
  net = StackedHGNetV1(
      config=cfg,
      classes_num=cfg.classes_num,
      edge_info=cfg.edge_info,
      nstack=cfg.nstack,
      add_coord=cfg.add_coord,
      decoder_type=cfg.decoder_type,
  )
  ckpt = torch.load(CKPT, map_location=device)
  net.load_state_dict(ckpt["net"])
  net.to(device).eval()
  _NET, _DEVICE = net, device
  return net, device


def _star68_at_crop(
    bgr: np.ndarray, scale: float, cx: float, cy: float,
) -> np.ndarray:
  net, device = load_star_net()
  rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
  matrix = GetCropMatrix(256, 1.0, True).process(float(scale), float(cx), float(cy))
  inp = cv2.warpPerspective(rgb, matrix, dsize=(256, 256), flags=cv2.INTER_LINEAR, borderValue=0)
  tensor = torch.from_numpy(inp).float().permute(2, 0, 1)[None] / 255.0 * 2.0 - 1.0
  tensor = tensor.to(device)
  with torch.no_grad():
    lmk = net(tensor)[-1][0]
  pts = _denorm(lmk).detach().cpu().numpy()[0]
  return _unwarp(pts, np.linalg.inv(matrix)).astype(np.float32)


def detect_star68(
    bgr: np.ndarray,
    image_path: Path | None = None,
    scale_center: tuple[float, float, float] | None = None,
    refine_crop: bool = True,
) -> np.ndarray:
  """返回 (68, 2) 像素坐标，iBUG 顺序。

  VHAP：dlib 68 的外接框定 scale。没有 dlib 时先用粗框跑一遍 STAR，
  再按这 68 点的外接框重裁（与 dlib 68 extent 同一公式），避免 MediaPipe
  468 框偏紧把下巴/耳朵挤歪。
  """
  if scale_center is None:
    scale, cx, cy, src = resolve_crop(bgr, image_path)
  else:
    scale, cx, cy = scale_center
    src = "given"
  pts = _star68_at_crop(bgr, scale, cx, cy)
  if refine_crop and src != "dlib":
    scale2, cx2, cy2 = _box_to_scale_center(
        float(pts[:, 0].min()), float(pts[:, 1].min()),
        float(pts[:, 0].max()), float(pts[:, 1].max()),
    )
    pts = _star68_at_crop(bgr, scale2, cx2, cy2)
  return pts
