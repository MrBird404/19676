"""命令行：vis2d / vis3d / ply / all。

在 MediaPipe 目录下：
  python -m tools vis2d --image ../gnm_fit_out/neutral_frames/frame_0001.png
  python -m tools vis3d --views front,left
  python -m tools ply
  python -m tools all --image ../gnm_fit_out/neutral_frames/frame_0001.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from .detect import detect_mp468, mp468_to_68, read_bgr
from .gnm_io import load_gnm_template, sample_official68
from .paths import OFFICIAL_68, REPO_ROOT
from .ply68 import export_lmk68_ply
from .render3d import VIEWS, render_official68_view
from .style import crop_around_xy, draw_ibug68, draw_mp468_points

DEFAULT_OUT = REPO_ROOT / "gnm_fit_out" / "lmk68_audit" / "tools"


def _add_out(p: argparse.ArgumentParser) -> None:
  p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)


def cmd_vis2d(args: argparse.Namespace) -> int:
  bgr = read_bgr(args.image)
  pts468 = detect_mp468(bgr, model=args.model)
  pts68 = mp468_to_68(pts468, args.gt68_map)
  args.out_dir.mkdir(parents=True, exist_ok=True)
  stem = args.image.stem

  vis468 = draw_mp468_points(
      bgr, pts468, title=f"MP468  n=468  {args.image.name}", label=args.label_468)
  p468 = args.out_dir / f"{stem}_mp468.jpg"
  p468c = args.out_dir / f"{stem}_mp468_crop.jpg"
  cv2.imwrite(str(p468), vis468)
  cv2.imwrite(str(p468c), crop_around_xy(vis468, pts468, pad=80))

  vis68 = draw_ibug68(
      bgr, pts68,
      title=f"MP468→68 ({args.gt68_map})  {args.image.name}",
      radius=4, label=not args.no_index,
  )
  p68 = args.out_dir / f"{stem}_mp68_{args.gt68_map}.jpg"
  p68c = args.out_dir / f"{stem}_mp68_{args.gt68_map}_crop.jpg"
  cv2.imwrite(str(p68), vis68)
  cv2.imwrite(str(p68c), crop_around_xy(vis68, pts68, pad=80))

  print(f"wrote {p468}")
  print(f"wrote {p68}")
  return 0


def cmd_vis3d(args: argparse.Namespace) -> int:
  views = [v.strip() for v in args.views.split(",") if v.strip()]
  unknown = [v for v in views if v not in VIEWS]
  if unknown:
    raise SystemExit(f"unknown views {unknown}, expected {sorted(VIEWS)}")
  args.out_dir.mkdir(parents=True, exist_ok=True)
  verts, faces, gnm = load_gnm_template()
  pts68 = sample_official68(verts, args.landmark_file)
  renderer = None
  for view in views:
    out = args.out_dir / f"official68_{view}.jpg"
    result = render_official68_view(
        view=view,
        out_path=out,
        size=args.size,
        landmark_file=args.landmark_file,
        verts=verts, faces=faces, gnm=gnm, pts68=pts68,
        renderer=renderer,
        label=not args.no_index,
    )
    renderer = result["renderer"]
    print(f"wrote {result['path']}")
  return 0


def cmd_ply(args: argparse.Namespace) -> int:
  verts, faces, _ = load_gnm_template()
  pts68 = sample_official68(verts, args.landmark_file)
  path = args.out if args.out is not None else args.out_dir / "official68_mesh_lmk68.ply"
  path = Path(path)
  export_lmk68_ply(verts, faces, pts68, path, radius=args.radius)
  print(f"wrote {path}")
  return 0


def cmd_all(args: argparse.Namespace) -> int:
  if args.image is not None:
    cmd_vis2d(args)
  cmd_vis3d(args)
  if args.out is None:
    args.out = args.out_dir / "official68_mesh_lmk68.ply"
  cmd_ply(args)
  return 0


def build_parser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(description=__doc__)
  sub = p.add_subparsers(dest="cmd", required=True)

  p2 = sub.add_parser("vis2d", help="照片：468 只点 + 68 连线（468→68）")
  p2.add_argument("--image", type=Path, required=True)
  p2.add_argument("--model", type=Path, default=None)
  p2.add_argument(
      "--gt68_map", choices=("mp2dlib", "mp68_ibug", "mp68", "mp68_v1"), default="mp2dlib",
      help="468→68：默认用 Mediapipe_2_Dlib 对应表",
  )
  p2.add_argument("--label_468", action="store_true")
  p2.add_argument("--no_index", action="store_true")
  _add_out(p2)
  p2.set_defaults(func=cmd_vis2d)

  p3 = sub.add_parser("vis3d", help="中性头官方 68：正脸 / 左脸")
  p3.add_argument("--views", type=str, default="front,left")
  p3.add_argument("--size", type=int, default=1280)
  p3.add_argument("--landmark_file", type=Path, default=OFFICIAL_68)
  p3.add_argument("--no_index", action="store_true")
  _add_out(p3)
  p3.set_defaults(func=cmd_vis3d)

  pp = sub.add_parser("ply", help="官方 68 + 头网 → MeshLab PLY")
  pp.add_argument("--out", type=Path, default=None)
  pp.add_argument("--landmark_file", type=Path, default=OFFICIAL_68)
  pp.add_argument("--radius", type=float, default=1.8e-3)
  _add_out(pp)
  pp.set_defaults(func=cmd_ply)

  pa = sub.add_parser("all", help="vis2d（若给图）+ vis3d + ply")
  pa.add_argument("--image", type=Path, default=None)
  pa.add_argument("--model", type=Path, default=None)
  pa.add_argument("--gt68_map", choices=("mp2dlib", "mp68_ibug", "mp68", "mp68_v1"), default="mp2dlib")
  pa.add_argument("--label_468", action="store_true")
  pa.add_argument("--no_index", action="store_true")
  pa.add_argument("--views", type=str, default="front,left")
  pa.add_argument("--size", type=int, default=1280)
  pa.add_argument("--landmark_file", type=Path, default=OFFICIAL_68)
  pa.add_argument("--out", type=Path, default=None)
  pa.add_argument("--radius", type=float, default=1.8e-3)
  _add_out(pa)
  pa.set_defaults(func=cmd_all)
  return p


def main(argv: list[str] | None = None) -> int:
  args = build_parser().parse_args(argv)
  return int(args.func(args))
