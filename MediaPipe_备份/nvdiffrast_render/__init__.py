"""True-perspective nvdiffrast helpers extracted from VHAP for GNM.

Vendors ``nvdiffrast`` from ``MediaPipe/vendor/nvdiffrast-backface-culling``
(same layout as VHAP-main/vendor/, full copy, not a symlink).
"""

from __future__ import annotations

import sys
from pathlib import Path

_VENDOR = Path(__file__).resolve().parent.parent / "vendor" / "nvdiffrast-backface-culling"
if _VENDOR.is_dir() and str(_VENDOR) not in sys.path:
  sys.path.insert(0, str(_VENDOR))

from .render_nvdiffrast import NVDiffRenderer, get_SH_shading
from . import vector_ops

__all__ = ["NVDiffRenderer", "get_SH_shading", "vector_ops"]
