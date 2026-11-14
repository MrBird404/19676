"""仓库内固定路径。"""

from __future__ import annotations

from pathlib import Path

TOOLS = Path(__file__).resolve().parent
MEDIAPIPE = TOOLS.parent
REPO_ROOT = next(p for p in [MEDIAPIPE, *MEDIAPIPE.parents] if (p / "GNM-main").is_dir())
GNM_MAIN = REPO_ROOT / "GNM-main"
OFFICIAL_68 = GNM_MAIN / "gnm" / "shape" / "data" / "landmarks" / "head_sparse_68.txt"
SPARSE68_V2_MOUTH_OUT2 = MEDIAPIPE / "head_sparse_68_v2_mouth_out2.txt"
SPARSE68_V3 = MEDIAPIPE / "head_sparse_68_v3.txt"
DETECTORS = REPO_ROOT / "detectors"
MP2DLIB_ROOT = DETECTORS / "Mediapipe_2_Dlib_Landmarks-main"
MP_MODEL = DETECTORS / "mediapipe" / "face_landmarker.task"
