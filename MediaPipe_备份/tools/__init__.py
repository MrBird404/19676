"""MediaPipe 可视化工具包：2D 468/68、3D 官方 68、MeshLab PLY。"""

from __future__ import annotations

__all__ = [
    "detect_mp468",
    "mp468_to_68",
    "draw_mp468_points",
    "draw_ibug68",
    "load_gnm_template",
    "sample_official68",
    "render_official68_view",
    "export_lmk68_ply",
]


def __getattr__(name: str):
  if name == "detect_mp468":
    from .detect import detect_mp468
    return detect_mp468
  if name == "mp468_to_68":
    from .detect import mp468_to_68
    return mp468_to_68
  if name in ("draw_mp468_points", "draw_ibug68"):
    from . import style
    return getattr(style, name)
  if name in ("load_gnm_template", "sample_official68"):
    from . import gnm_io
    return getattr(gnm_io, name)
  if name == "render_official68_view":
    from .render3d import render_official68_view
    return render_official68_view
  if name == "export_lmk68_ply":
    from .ply68 import export_lmk68_ply
    return export_lmk68_ply
  raise AttributeError(name)
