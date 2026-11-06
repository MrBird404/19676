"""仓库内固定路径。"""

from __future__ import annotations

from pathlib import Path

TOOLS = Path(__file__).resolve().parent
MEDIAPIPE = TOOLS.parent
REPO_ROOT = MEDIAPIPE.parent
GNM_MAIN = REPO_ROOT / "GNM-main"
OFFICIAL_68 = GNM_MAIN / "gnm" / "shape" / "data" / "landmarks" / "head_sparse_68.txt"
SPARSE68_V2_MOUTH_OUT2 = MEDIAPIPE / "head_sparse_68_v2_mouth_out2.txt"
SPARSE68_V3 = MEDIAPIPE / "head_sparse_68_v3.txt"
MP2DLIB_ROOT = REPO_ROOT / "Mediapipe_2_Dlib_Landmarks-main"
MP_MODEL = MEDIAPIPE / "models" / "face_landmarker.task"
