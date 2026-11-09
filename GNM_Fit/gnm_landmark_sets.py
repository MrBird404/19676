"""Shared MediaPipe 468 index sets for GNM fitting / correspondence tools."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_MP2DLIB_ROOT = Path(__file__).resolve().parent.parent / "detectors" / "Mediapipe_2_Dlib_Landmarks-main"
if _MP2DLIB_ROOT.is_dir() and str(_MP2DLIB_ROOT) not in sys.path:
  sys.path.insert(0, str(_MP2DLIB_ROOT))
from mediapipe2dlib.mp2dlib import mp2dlib_correspondence as MP2DLIB_PAIRS  # noqa: E402

# Textbook MediaPipe-468 -> iBUG-68 (oval order: ear -> chin -> ear).
MP68_IBUG = np.array([
    162, 234, 93, 58, 172, 136, 149, 148, 152, 377, 378, 365, 397, 288, 323, 454, 389,
    70, 63, 105, 66, 107,
    336, 296, 334, 293, 300,
    168, 197, 5, 4,
    75, 97, 2, 326, 305,
    33, 160, 158, 133, 153, 144,
    362, 385, 387, 263, 373, 380,
    61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181,
    78, 82, 13, 312, 308, 317, 14, 87,
], dtype=np.int64)

# Original correspondence pack (MediaPipe/correspondence_v1): one MediaPipe
# index per iBUG point, no midpoints. Same numbers as MP68_IBUG. The 3D file
# head_sparse_68_v1.txt is these rows of that pack's own head_dense_468, not
# of the current v2 head_dense_468.txt. v2's table is
# build_mediapipe_correspondence.MP68.
MP68_V1 = MP68_IBUG.copy()

# GNM official 68 jaw is NOT iBUG oval order on the left: along the silhouette
# it runs 0,1,6,5,4,3,2,7,8,... (indices 2-6 reversed). Pairing MP68_IBUG with
# official GNM 68 made jaw2↔jaw6 / jaw3↔jaw5 100–280px crossed lines.
# MP68 jaw = Hungarian match of FACE_OVAL to official GNM 68 on the template XY.
MP68 = MP68_IBUG.copy()
MP68[0:17] = np.array(
    [127, 93, 149, 150, 172, 58, 132, 148, 152, 400, 378, 379, 397, 288, 361, 323, 356],
    dtype=np.int64,
)

# iBUG / GNM official 68 (same order as MP68).
IBUG_JAW = set(range(0, 17))
IBUG_BROW_R = set(range(17, 22))
IBUG_BROW_L = set(range(22, 27))
IBUG_BROWS = IBUG_BROW_L | IBUG_BROW_R
IBUG_NOSE = set(range(27, 36))
IBUG_EYE_R = set(range(36, 42))
IBUG_EYE_L = set(range(42, 48))
IBUG_EYES = IBUG_EYE_L | IBUG_EYE_R
IBUG_MOUTH_OUTER = set(range(48, 60))
IBUG_MOUTH_INNER = set(range(60, 68))
IBUG_MOUTH = IBUG_MOUTH_OUTER | IBUG_MOUTH_INNER
IBUG_MOUTH_UPPER_OUTER = {48, 49, 50, 51, 52, 53, 54}
IBUG_MOUTH_LOWER_OUTER = {48, 54, 55, 56, 57, 58, 59}  # corners + lower arc
IBUG_IOD = (36, 45)  # outer canthi
IBUG_INNER_UP, IBUG_INNER_LO = 62, 66
IBUG_OUTER_UP, IBUG_OUTER_LO = 51, 57

# GNM ``head_sparse_68.txt`` jaw #75: stored rows 2-6 are reversed vs iBUG oval.
# dest[i] = src[perm[i]] puts official GNM 68 into textbook iBUG order.
# Pair this with mp2dlib 2D GT. Do NOT pair mp2dlib with the uncorrected file.
GNM68_OFFICIAL_TO_IBUG = np.arange(68, dtype=np.int64)
GNM68_OFFICIAL_TO_IBUG[2:7] = np.array([6, 5, 4, 3, 2], dtype=np.int64)


def mp468_to_dlib68(lmks_mp: np.ndarray) -> np.ndarray:
  """MediaPipe 468/478 xy -> Dlib/iBUG 68. Uses local mp2dlib (6 midpoints)."""
  pts = np.asarray(lmks_mp, dtype=np.float64)
  pairs = np.asarray(MP2DLIB_PAIRS, dtype=np.int64)
  return pts[pairs].mean(axis=1).astype(np.float32)


def sparse68_gt_xy(lmks_mp: np.ndarray, gt68_map: str = "star") -> np.ndarray:
  """2D GT for ``--lmk_set sparse68``.

  ``star``: already-iBUG 68 from STAR 300W (or any (68,2) detector). Pair with
  official GNM ``head_sparse_68.txt``.
  ``mp68`` / ``mp68_ibug`` / ``mp68_v1`` / ``mp2dlib``: take 68 from MediaPipe 468 (legacy).
  ``mp68_v1`` is the original-pack index table (same indices as ``mp68_ibug``).
  """
  pts = np.asarray(lmks_mp, dtype=np.float32)
  if pts.ndim != 2:
    raise ValueError(f"expected (N,2+), got {pts.shape}")
  xy = pts[:, :2]
  if gt68_map in ("star", "identity"):
    if xy.shape[0] != 68:
      raise ValueError(
          f"gt68_map={gt68_map} expects (68,2), got {xy.shape}. "
          "Run extract_star_landmarks.py or pass a 68-point npz."
      )
    return xy.astype(np.float32)
  if xy.shape[0] == 68 and gt68_map in ("mp68", "mp68_ibug", "mp68_v1", "mp2dlib"):
    raise ValueError(
        f"gt68_map={gt68_map} expects MediaPipe 468, got 68. Use --gt68_map star."
    )
  if gt68_map == "mp2dlib":
    return mp468_to_dlib68(xy)
  if gt68_map == "mp68":
    return xy[MP68].astype(np.float32)
  if gt68_map in ("mp68_ibug", "mp68_v1"):
    table = MP68_V1 if gt68_map == "mp68_v1" else MP68_IBUG
    return xy[table].astype(np.float32)
  raise ValueError(f"unknown gt68_map={gt68_map!r}")


def write_gnm68_ibug_order(src: Path, dst: Path) -> Path:
  """Rewrite official GNM 68 with jaw 2-6 reversed -> iBUG oval order."""
  data = np.loadtxt(src)
  if data.shape != (68, 6):
    raise ValueError(f"expected official GNM 68 (68,6), got {data.shape} from {src}")
  dst.parent.mkdir(parents=True, exist_ok=True)
  header = (
      "GNM head_sparse_68.txt with jaw rows 2-6 reversed to iBUG oval order "
      "(GNM #75). Pair with mp2dlib 2D GT, not with MP68."
  )
  np.savetxt(
      dst,
      data[GNM68_OFFICIAL_TO_IBUG],
      fmt="%d %.6f %d %.6f %d %.6f",
      header=header,
  )
  return dst

MP_FACE_OVAL = {
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
    378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109,
}

MP_CONTOUR_OUTER = MP_FACE_OVAL | {
    32, 68, 69, 71, 104, 108, 139, 151, 199, 208, 262, 298, 299, 301, 333, 337,
    369, 396, 428,
    34, 127, 162, 227, 234, 264, 356, 368, 389, 447, 454,
    58, 93, 132, 137, 147, 172, 177, 213, 215, 288, 323, 361, 366, 376, 397,
    401, 433, 435,
    50, 135, 136, 138, 150, 169, 170, 187, 192, 207, 210, 211, 212, 214, 216,
    280, 364, 365, 367, 379, 394, 395, 411, 416, 427, 430, 431, 432, 434, 436,
    140, 148, 149, 152, 171, 175, 176, 377, 378, 400,
}

MP_CONTOUR_INNER = {
    34, 68, 69, 71, 104, 108, 135, 137, 138, 139, 140, 151, 169, 170, 171,
    175, 177, 215, 227, 264, 298, 299, 301, 333, 337, 364, 366, 367, 368,
    369, 394, 395, 396, 401, 435, 447,
}

MP_CONTOUR_DUAL = MP_CONTOUR_OUTER | MP_CONTOUR_INNER
MP_JAW68 = set(MP68_IBUG[0:17].tolist())
MP_NOSE68 = set(MP68_IBUG[27:36].tolist())

MP_NOSE_DENSE = {
    1, 2, 4, 5, 6, 19, 44, 45, 48, 49, 51, 64, 75, 79, 94, 97, 98, 99, 102,
    114, 115, 122, 125, 126, 129, 131, 134, 141, 168, 174, 188, 193, 195,
    196, 197, 198, 209, 217, 220, 235, 236, 237, 240, 241, 242, 245, 248,
    274, 275, 278, 279, 281, 294, 305, 309, 326, 327, 328, 331, 343, 344,
    351, 354, 358, 360, 363, 370, 399, 412, 417, 419, 420, 429, 437, 439,
    440, 455, 456, 457, 460, 461, 462, 465,
}

MP_EYES = {
    7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173,
    246, 249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388,
    390, 398, 466,
}

MP_LIPS = {
    0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146,
    178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314, 317,
    318, 321, 324, 375, 402, 405, 409, 415,
}

MP_UPPER_LIP_OUTER = {61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291}
MP_LOWER_LIP_OUTER = {61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291}
MP_UPPER_LIP_INNER = {78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308}
MP_LOWER_LIP_INNER = {78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308}

MP_MOUTH_SOCK = {
    11, 12, 15, 16, 38, 41, 42, 72, 73, 74, 85, 86, 89, 90, 179, 180, 183,
    184, 185, 268, 271, 272, 287, 302, 303, 304, 315, 316, 319, 320, 403,
    404, 407, 408, 409,
}

# mp_mesh barycentric anchors that land on GNM teeth/gums (not in LIPS/sock).
# Use original head_dense_468.txt rows for these.
MP_PERIORAL_ON_TEETH_MPMESH = {
    43, 57, 77, 106, 182, 322, 335, 393, 410, 422,
}

# Near-mouth indices that were NOT in the first orig patch. mp_mesh still owns
# them: 76/186/306 sit centimetres inside the oral cavity / back of the head,
# 62 is on the lower lip while MediaPipe 62 is the left inner-corner (upper).
# Mirrors: 92↔322, 186↔410, 273↔43, 306↔76, 292↔62, 307↔77, 325↔96.
MP_MOUTH_NEIGHBOR_LEFTOVER = {
    62, 76, 92, 96, 186, 273, 292, 306, 307, 325,
}

# Mentolabial / chin-of-mouth belt: orig is on lower_lip_region, but these
# indices are not in LIPS/sock/periT/leftover so stage3 used to ignore them
# and they collapse into the mouth opening.
MP_MENTOLABIAL = {
    18, 83, 199, 200, 201, 204, 205, 208, 421, 428,
}

# mp_mesh anchors whose barycentric mass sits entirely on the eyeball
# (eyes / scleras / irises). Re-project onto sockets/orbital; do not copy
# the whole original 468 file.
MP_ON_EYEBALL_MPMESH = {
    22, 23, 28, 56, 153, 155, 249, 253, 254, 373, 374, 381, 382, 386, 388,
    390, 414, 441, 442,
}

MP_MOUTH_FROM_ORIG = (
    MP_LIPS | MP_UPPER_LIP_OUTER | MP_LOWER_LIP_OUTER
    | MP_UPPER_LIP_INNER | MP_LOWER_LIP_INNER | MP_MOUTH_SOCK
    | MP_PERIORAL_ON_TEETH_MPMESH
    | MP_MOUTH_NEIGHBOR_LEFTOVER
    | MP_MENTOLABIAL
)

# Brow polylines (MediaPipe canonical dense indices).
MP_BROW_R = {70, 63, 105, 66, 107, 55, 65, 52, 53, 46}
MP_BROW_L = {336, 296, 334, 293, 300, 276, 283, 282, 295, 285}
MP_BROWS = MP_BROW_L | MP_BROW_R

# GNM region labels where mp_mesh correspondence is systematically bad.
GNM_UNTRUSTED_GROUPS = frozenset({
    "left_parotid_region",
    "right_parotid_region",
    "left_temple_region",
    "right_temple_region",
    "left_cheek_region",
    "right_cheek_region",
})

GNM_SOFT_CHIN = frozenset({"chin_region"})

# Interior groups used for identity PCA lift.
GNM_IDENTITY_GROUPS = frozenset({
    "nose_region",
    "forehead_region",
    "left_orbital_region",
    "right_orbital_region",
    "left_infraorbital_region",
    "right_infraorbital_region",
    "left_brow_region",
    "right_brow_region",
    "middle_brow_region",
    "eye_sockets",
    "skin",
    "left_zygomatic_region",
    "right_zygomatic_region",
})

GNM_EXPRESSION_GROUPS = frozenset({
    "mouth_sock",
    "upper_lip",
    "lower_lip",
    "upper_lip_region",
    "lower_lip_region",
})
