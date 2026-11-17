#!/usr/bin/env python3
"""Store-coefficient representation for ear/neck (replaces baked ear offset).

Ear: rigid translation (lat/vert/front) stays BAKED into static_offset (fixed);
     prot/size become coefficients (lightly optimisable later).
Neck: neck_thickness/neck_length kept as identity-combo directions, coefficients
      optimisable (init 0, i.e. fine-tune on top of the baked neck).

Output params carry:
  static_offset  = lmk(neck) + ear rigid translation   (FIXED)
  semantic_dirs  = (K,V,3) metres/unit  [ear_prot, ear_size, neck_thick, neck_len]
  semantic_coef  = (K,) init [ear_prot, ear_size, 0, 0]
  semantic_names = names
Round-trip: posed + static_offset + Σ coef·dir == posed + original ear_align offset.
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

from gnm.shape import gnm_numpy  # noqa: E402
from fit_ear_mask import build_blend_weights  # noqa: E402
import semantic_basis as sb  # noqa: E402

SIG = dict(sig_x=0.06, sig_y_up=0.12, sig_y_dn=0.035, sig_z=0.07)


def ear_dirs(rest, ear_vids, trans_scale, g_sb):
  """Rebuild fit_ear_semantic's normalised basis (metres per unit, 1mm/unit on ears)."""
  w, Cr = build_blend_weights(rest.astype(np.float32), ear_vids, **SIG)
  w = w.astype(np.float64)
  xs = np.sign(rest[:, 0]); xs[xs == 0] = 1.0
  rigid = [
      np.stack([xs, 0 * xs, 0 * xs], 1),
      np.stack([0 * xs, np.ones_like(xs), 0 * xs], 1),
      np.stack([0 * xs, 0 * xs, np.ones_like(xs)], 1),
  ]
  _, _, _, _, sdirs, _ = sb.build()
  mirror = np.asarray(g_sb.mirror_indices, np.int64); flip = np.array([-1.0, 1.0, 1.0])
  sem = {}
  for key, nm in (("ear_protrusion", "prot"), ("ear_size", "size")):
    d = sdirs[key] / 1000.0
    sem[nm] = 0.5 * (d + d[mirror] * flip)
  out = {}
  names_r = ["lat", "vert", "front"]
  for i, vec in enumerate(rigid):
    b = vec * w[:, None] * trans_scale
    out[names_r[i]] = b
  for nm in ("prot", "size"):
    out[nm] = sem[nm] * w[:, None]
  # normalise each so mean |.| on ears = 1 mm
  for k in out:
    m = np.linalg.norm(out[k][ear_vids], axis=1).mean()
    if m > 1e-12:
      out[k] = out[k] / m * 1e-3
  return out


def _smooth(t):
  t = np.clip(t, 0.0, 1.0)
  return t * t * (3.0 - 2.0 * t)


def vertex_normals(rest, tri):
  tri = np.asarray(tri.cpu().numpy() if hasattr(tri, "cpu") else tri, np.int64)
  fn = np.cross(rest[tri[:, 1]] - rest[tri[:, 0]], rest[tri[:, 2]] - rest[tri[:, 0]])
  n = np.zeros_like(rest)
  for i in range(3):
    np.add.at(n, tri[:, i], fn)
  ln = np.linalg.norm(n, axis=1, keepdims=True); ln[ln < 1e-12] = 1.0
  n = n / ln
  C = rest.mean(0)
  sg = np.sign(np.einsum("vi,vi->v", n, rest - C)); sg[sg == 0] = 1.0
  return n * sg[:, None]


def head_round_dir(rest, normals, skin, ears):
  """ONE unified bulge: outward along surface normal over a broad side+upper-cranium
  patch (side/temple -> parietal -> crown). No separate lateral widening. The forehead
  centre is gated out and the crown is tapered, so it bulges the roundness, not the height.
  Returns a unit field (normal * weight); bake as head_w(mm)/1000 * this."""
  X = rest[:, 0]; Y = rest[:, 1]
  ear_top = rest[ears, 1].max()
  crown = rest[skin, 1].max()
  fy = min(crown - 0.030, 0.370)                 # approx forehead top
  gl = _smooth((np.abs(X) - 0.045) / 0.030)      # lateral gate (sides; skip forehead centre)
  side = _smooth((Y - (ear_top - 0.020)) / 0.050) * gl
  top = _smooth((Y - (fy - 0.020)) / 0.030)      # above forehead -> crown/parietal
  top *= (1.0 - 0.5 * _smooth((Y - (crown - 0.040)) / 0.040))  # taper crown (avoid过长)
  w = np.maximum(side, top)
  w[ears] = 0.0
  return normals * w[:, None]


def head_lower_dir(rest, skin, ears):
  """Press the crown down (-Y); weight high near the very top, fading by parietal.
  Sides and forehead untouched. Returns unit field (down); bake as head_lower(mm)/1000 * this."""
  Y = rest[:, 1]
  crown = rest[skin, 1].max()
  wt = _smooth((Y - (crown - 0.075)) / 0.075)
  wt[ears] = 0.0
  d = np.zeros((rest.shape[0], 3), np.float64)
  d[:, 1] = -wt
  return d


def neck_dirs(rest, neck_idx):
  _, _, _, _, sdirs, _ = sb.build()
  out = {}
  for key, nm in (("neck_thickness", "neck_thick"), ("neck_length", "neck_len")):
    d = sdirs[key] / 1000.0
    m = np.linalg.norm(d[neck_idx], axis=1).mean()
    if m > 1e-12:
      d = d / m * 1e-3
    out[nm] = d
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--base_params", type=Path,
                  default=Path("/root/autodl-tmp/gnm_fit_out/expression_v3/ear_semantic_v2/frame_000000_params.npz"),
                  help="fit_ear_semantic output (static_offset = lmk + full ear deform; has ear_coef/ear_basis_names)")
  ap.add_argument("--out_dir", type=Path,
                  default=Path("/root/autodl-tmp/gnm_fit_out/expression_v3/ear_coef_v1"))
  ap.add_argument("--trans_scale", type=float, default=1.3)
  ap.add_argument("--head_w", type=float, default=0.0,
                  help="bake head normal-bulge (mm) into static_offset (side+upper cranium roundness).")
  ap.add_argument("--head_lower", type=float, default=0.0,
                  help="bake top-lowering (mm) into static_offset: press the crown down, sides/forehead untouched.")
  args = ap.parse_args()
  args.out_dir.mkdir(parents=True, exist_ok=True)

  g = gnm_numpy.GNM.from_local(version=gnm_numpy.GNMMajorVersion.V3, variant=gnm_numpy.GNMVariant.HEAD)
  rest = np.asarray(g(None, None, None, None), np.float64)
  ear_vids = np.unique(np.asarray(g.vertex_group_indices("ears"), np.int64))
  skin = np.unique(np.asarray(g.vertex_group_indices("skin"), np.int64))
  ys = rest[:, 1]
  neck_idx = np.setdiff1d(np.intersect1d(skin, np.where((ys >= 0.10) & (ys <= 0.22))[0]), ear_vids)

  p = np.load(args.base_params)
  names = [str(x) for x in p["ear_basis_names"]]
  coef = np.asarray(p["ear_coef"], np.float64)
  cd = dict(zip(names, coef))
  off_full = np.asarray(p["static_offset"], np.float64)

  ed = ear_dirs(rest, ear_vids, args.trans_scale, g)
  nd = neck_dirs(rest, neck_idx)
  sem_part = cd["prot"] * ed["prot"] + cd["size"] * ed["size"]

  base_off_decomp = off_full - sem_part
  dirs = np.stack([ed["prot"], ed["size"], nd["neck_thick"], nd["neck_len"]], 0).astype(np.float32)
  sem_names = np.array(["ear_prot", "ear_size", "neck_thick", "neck_len"])
  sem_coef = np.array([cd["prot"], cd["size"], 0.0, 0.0], np.float32)

  # round-trip verify (metres)
  recon = base_off_decomp + sem_coef[0] * dirs[0] + sem_coef[1] * dirs[1] + sem_coef[2] * dirs[2] + sem_coef[3] * dirs[3]
  err = np.linalg.norm(recon - off_full, axis=1)
  print(f"[coef] round-trip max|err|={err.max()*1000:.5f}mm mean={err.mean()*1000:.6f}mm "
        f"(ear region mean={err[ear_vids].mean()*1000:.6f}mm)")
  print(f"[coef] base_off p95={np.percentile(np.linalg.norm(base_off_decomp,axis=1),95)*1000:.2f}mm  "
        f"coef={dict(zip(['ear_prot','ear_size','neck_thick','neck_len'], sem_coef.round(3)))}")

  hd = head_round_dir(rest, vertex_normals(rest, g.triangles), skin, ear_vids)
  hl = head_lower_dir(rest, skin, ear_vids)
  base_off = base_off_decomp + (args.head_w / 1000.0) * hd + (args.head_lower / 1000.0) * hl
  if args.head_w:
    print(f"[coef] head_round {args.head_w}mm normal-bulge baked -> "
          f"{np.linalg.norm((args.head_w/1000.0*hd)[skin],axis=1).mean()*1000:.2f}mm mean on skin")
  if args.head_lower:
    print(f"[coef] head_lower {args.head_lower}mm top-press baked -> "
          f"crown drop {np.linalg.norm((args.head_lower/1000.0*hl)[skin],axis=1).max()*1000:.2f}mm max")

  save = {k: np.asarray(p[k]) for k in p.files if k not in ("ear_coef", "ear_basis_names")}
  save["static_offset"] = base_off.astype(np.float32)
  save["semantic_dirs"] = dirs
  save["semantic_names"] = sem_names
  save["semantic_coef"] = sem_coef
  np.savez(args.out_dir / "frame_000000_params.npz", **save)
  print("wrote", args.out_dir / "frame_000000_params.npz")


if __name__ == "__main__":
  main()
