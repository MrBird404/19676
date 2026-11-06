"""四条下半脸表情方向：张嘴、嘴角、嘴宽、唇部分开。

只使用 lower_face 最平滑的前 24 个分量。每条方向在达到自己的毫米目标时，
按住另外三项测量，系数长度封在 4σ。
"""

from __future__ import annotations

import numpy as np

N_SMOOTH = 24
MAX_SIGMA = 4.0
GOAL_W = 400.0
HOLD_W = 100.0
# +1 权重对应的毫米目标，与 face-to-voice 的嘴部细调同一量级。
GOALS = {
    "jaw_open": 14.0,
    "corner_raise": 4.0,
    "mouth_stretch": 8.0,
    "lip_part": 6.0,
}
NAMES = ("jaw_open", "corner_raise", "mouth_stretch", "lip_part")
# iBUG：8 颏、33 鼻底、48/54 嘴角、62/66 内唇中点。
CHIN, NOSE = 8, 33
CORNER_L, CORNER_R = 48, 54
LIP_UP, LIP_LO = 62, 66


def build_directions(
    verts: np.ndarray,
    lmk_idx: np.ndarray,
    lmk_w: np.ndarray,
    expression_basis: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
  """返回 (150, 4) 的方向，对应 ex_lower 的前 24 维。"""
  basis = np.asarray(expression_basis, dtype=np.float64)
  idx = np.asarray(lmk_idx)
  wt = np.asarray(lmk_w, dtype=np.float64)

  def sample(disp: np.ndarray | None = None) -> np.ndarray:
    v = verts if disp is None else verts + disp
    return (v[idx] * wt[..., None]).sum(1)

  rest = sample()
  disps = []
  for j in range(N_SMOOTH):
    disps.append(sample(basis[200 + j]) - rest)
  disps = np.stack(disps, 0)

  def row_axis(ia: int, ib: int) -> np.ndarray:
    return (disps[:, ia, 1] - disps[:, ib, 1]) * 1000.0

  def row_dist(ia: int, ib: int) -> np.ndarray:
    p = rest[ia] - rest[ib]
    u = p / np.linalg.norm(p)
    return ((disps[:, ia] - disps[:, ib]) * u).sum(-1) * 1000.0

  rows = {
      "jaw_open": row_axis(NOSE, CHIN),
      "corner_raise": 0.5 * (row_axis(CORNER_L, NOSE) + row_axis(CORNER_R, NOSE)),
      "mouth_stretch": row_dist(CORNER_L, CORNER_R),
      "lip_part": row_axis(LIP_UP, LIP_LO),
  }
  dirs = []
  for name in NAMES:
    a_rows, target, weight = [], [], []
    for measure, jac in rows.items():
      a_rows.append(jac)
      if measure == name:
        target.append(GOALS[name])
        weight.append(GOAL_W)
      else:
        target.append(0.0)
        weight.append(HOLD_W)
    A = np.stack(a_rows, 0)
    tw = np.asarray(target, dtype=np.float64)
    W = np.diag(weight)
    x = np.linalg.solve(A.T @ W @ A + np.eye(N_SMOOTH), A.T @ W @ tw)
    length = float(np.linalg.norm(x))
    if length > MAX_SIGMA:
      x *= MAX_SIGMA / length
    full = np.zeros(150, dtype=np.float32)
    full[:N_SMOOTH] = x.astype(np.float32)
    reached = A @ x
    print(
        f"[expr-dir] {name:16}  sigma {np.linalg.norm(x):.2f}  "
        f"mm { {m: round(float(v), 2) for m, v in zip(rows, reached)} }",
        flush=True,
    )
    dirs.append(full)
  return np.stack(dirs, axis=1), list(NAMES)
