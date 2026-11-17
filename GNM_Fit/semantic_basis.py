#!/usr/bin/env python3
"""Semantic localized basis (neck_thickness / neck_length / ear_size /
ear_protrusion) built from the GNM identity basis.

Math mirrors face-to-voice `semantic.py`: a localized, symmetric direction is
obtained by a ridge-regularized least-squares fit of a desired per-vertex target
field, penalising displacement OUTSIDE the region (Q_out, cKDTree falloff) and
left/right asymmetry (Q_sym):

    N = I + lam_out*Q_out + lam_sym*Q_sym ;  u = solve(N, Jc) ; delta = u . B

We use direct target displacement fields (radial / vertical stretch / outward)
rather than anthropometric measures — same DOF, self-contained, no external cfg.

Run `python semantic_basis.py --self_test` to build neck bases and measure how
much of an existing static_offset over the neck/shoulder region they can rebuild.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = next(p for p in [ROOT, *ROOT.parents] if (p / "GNM-main").is_dir()) / "GNM-main"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT))


def _gnm():
  from gnm.shape import gnm_numpy
  return gnm_numpy.GNM.from_local(version=gnm_numpy.GNMMajorVersion.V3,
                                  variant=gnm_numpy.GNMVariant.HEAD)


def outside_weight(rest, region_idx, falloff_m, y_band=None):
  from scipy.spatial import cKDTree
  inside = np.zeros(rest.shape[0], bool)
  inside[region_idx] = True
  if y_band is not None:
    inside &= (rest[:, 1] >= y_band[0]) & (rest[:, 1] <= y_band[1])
  if inside.sum() == 0:
    return np.ones(rest.shape[0])
  d, _ = cKDTree(rest[inside]).query(rest)
  return np.clip(d / falloff_m, 0.0, 1.0).astype(np.float64)


def solve_measure(B, rest, grad, region_idx, y_band=None,
                  lam_out=30.0, lam_sym=40.0, falloff_mm=15.0, ridge=1e-3,
                  target_mm=1.0):
  """Localized, symmetric direction that changes a scalar measurement by
  `target_mm`. `grad` is d(measure)/d(vertex) per vertex, shape (V,3).
  Mirrors face-to-voice semantic.py: Jc = grad . B ; u = N^-1 Jc ; normalise+scale.
  Returns delta (V,3) mm and coef u (C,)."""
  C, V, _ = B.shape
  Jc = np.einsum("cvi,vi->c", B, grad)  # mm per coefficient
 
  wo = outside_weight(rest, region_idx, falloff_mm / 1000.0, y_band)
  Bo = B.reshape(C, -1)
  w = np.repeat(wo ** 2, 3)
  Q_out = (Bo * w) @ Bo.T / max(wo.sum(), 1.0)
  mirror = getattr(solve_measure, "_mirror", None)
  if mirror is not None:
    Bx = B
    flip = np.array([-1.0, 1.0, 1.0])
    asym = (Bx - Bx[:, mirror, :] * flip).reshape(C, -1)
    Q_sym = asym @ asym.T / V
  else:
    Q_sym = 0.0
  N = np.eye(C) + lam_out * Q_out + lam_sym * Q_sym + ridge * np.eye(C)
  u0 = np.linalg.solve(N, Jc)
  n = np.linalg.norm(u0)
  if n < 1e-12:
    return np.zeros((V, 3)), np.zeros(C)
  u0 = u0 / n
  per_unit = float(Jc @ u0)  # measure change per unit scale (mm)
  scale = target_mm / per_unit if abs(per_unit) > 1e-9 else 0.0
  u = u0 * scale
  delta = np.tensordot(u, B, axes=1)
  return delta, u


def build(basis_mm=None, scale=True):
  g = _gnm()
  B = np.asarray(g.vertex_identity_basis, np.float64)
  if scale:
    try:
      from gnm.shape import gnm_utils
      sc = np.asarray(gnm_utils.scaling_for_identity_loss(g), np.float64)
      if sc.ndim == 1 and sc.shape[0] == B.shape[0]:
        B = B * sc[:, None, None]
    except Exception:
      pass
  B = B * 1000.0
  rest = np.asarray(g(None, None, None, None), np.float64)
  grp = {n: np.unique(np.asarray(g.vertex_group_indices(n), np.int64)) for n in g.vertex_group_names}
  solve_measure._mirror = np.asarray(g.mirror_indices, np.int64)
  V = rest.shape[0]
  skin = grp["skin"]; ears = grp["ears"]
  ys = rest[:, 1]
  neck = np.intersect1d(skin, np.where((ys >= 0.10) & (ys <= 0.22))[0])
  neck = np.setdiff1d(neck, ears)

  dirs = {}
  # neck_thickness: measure = mean radial (x,z) displacement over neck band.
  c = rest[neck].mean(0)
  rad = rest[neck] - c; rad[:, 1] = 0.0
  nrm = np.linalg.norm(rad, axis=1, keepdims=True); nrm[nrm < 1e-9] = 1.0
  grad = np.zeros((V, 3)); grad[neck] = (rad / nrm) / neck.size
  dirs["neck_thickness"] = solve_measure(B, rest, grad, neck, y_band=[0.12, 0.20], target_mm=3.0)[0]

  # neck_length: measure = vertical distance upper vs lower neck band (+y upper, -y lower)
  grad = np.zeros((V, 3))
  upper = neck[ys[neck] >= 0.15]; lower = neck[ys[neck] < 0.15]
  grad[upper, 1] = 1.0 / max(upper.size, 1)
  grad[lower, 1] = -1.0 / max(lower.size, 1)
  dirs["neck_length"] = solve_measure(B, rest, grad, neck, y_band=[0.10, 0.20], target_mm=4.0)[0]

  # ear_size: vertical top-bottom stretch (upper ear +y, lower ear -y), per side jointly
  es = np.zeros((V, 3))
  ec_y = rest[ears, 1].mean()
  up = ears[rest[ears, 1] >= ec_y]; lo = ears[rest[ears, 1] < ec_y]
  es[up, 1] = 1.0 / max(up.size, 1); es[lo, 1] = -1.0 / max(lo.size, 1)
  dirs["ear_size"] = solve_measure(B, rest, es, ears, target_mm=4.0)[0]
  # ear_protrusion: outward x (sign)
  ep = np.zeros((V, 3)); ep[ears, 0] = np.sign(rest[ears, 0]) / ears.size
  dirs["ear_protrusion"] = solve_measure(B, rest, ep, ears, target_mm=3.0)[0]

  return g, rest, B, grp, dirs, neck


def _self_test(args):
  g, rest, B, grp, dirs, neck = build()
  ears = grp["ears"]
  print("[basis] built:", {k: f"{np.linalg.norm(v,axis=1).mean():.2f}mm/build" for k, v in dirs.items()})
  # localization of each dir: region vs face magnitude
  face = np.setdiff1d(grp["skin"], ears)
  for k, v in dirs.items():
    reg = ears if k.startswith("ear") else neck
    mreg = np.linalg.norm(v[reg], axis=1).mean()
    mface = np.linalg.norm(v[face], axis=1).mean()
    print(f"  {k:16s} region={mreg:.2f}mm face={mface:.3f}mm ratio={mreg/(mface+1e-9):.1f}")

  # projection test: rebuild an existing static_offset over the neck region
  p = np.load(args.params)
  off = np.asarray(p["static_offset"], np.float64) * 1000.0  # mm
  target = off[neck]
  basis = np.stack([dirs["neck_thickness"][neck].reshape(-1),
                    dirs["neck_length"][neck].reshape(-1)], axis=1)  # (3N,2)
  coef, *_ = np.linalg.lstsq(basis, target.reshape(-1), rcond=None)
  recon = (basis @ coef).reshape(-1, 3)
  tgt_mag = np.linalg.norm(target, axis=1).mean()
  res = np.linalg.norm(target - recon, axis=1)
  print(f"[proj] neck region {neck.size} verts  target|.|={tgt_mag:.2f}mm  "
        f"residual|.|={res.mean():.2f}mm  explained={100*(1-res.mean()/(tgt_mag+1e-9)):.0f}%  coef={coef.round(3)}")
  # summarize where the original offset actually lives (top regions by mean |off|)
  from collections import Counter
  for name in ["ears", "skin"]:
    idx = grp[name]
    print(f"  |off| mean on {name}: {np.linalg.norm(off[idx],axis=1).mean():.2f}mm")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--self_test", action="store_true")
  ap.add_argument("--params", type=Path,
                  default=Path("/root/autodl-tmp/gnm_fit_out/expression_v3/ear_align_v2_vertmore/frame_000000_params.npz"))
  ap.add_argument("--out", type=Path, default=Path("/root/autodl-tmp/gnm_fit_out/semantic_neck_basis.npz"))
  args = ap.parse_args()
  if args.self_test:
    _self_test(args)
  else:
    g, rest, B, grp, dirs, neck = build()
    np.savez(args.out, **{k: v.astype(np.float32) for k, v in dirs.items()})
    print("wrote", args.out)


if __name__ == "__main__":
  main()
