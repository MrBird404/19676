"""Eyelid <-> protruding iris/pupil (bulge) clearance, rest-referenced.

The eyelid opening sits right at the iris, so the lid margin is flush with the
front-facing iris disk at rest -- a raw "keep the lid outside the bulge" rule
would wrongly lift the margin.  Instead:

  * rotate the globe about its true sclera-sphere centre (sphere-invariant, so
    only the iris/pupil bulge moves on the surface, the sclera does not shift);
  * among FRONT lid vertices that are OUTSIDE the bulge at rest (rest sdf>0,
    i.e. the lid skin actually covering the globe, not the margin rim), find the
    ones the rotated bulge pushes inward (sdf < margin) and push them back out
    to `margin` along the bulge normal.

So the margin rim is untouched; only the bulge poking under the covering lid is
corrected, conforming to the bulge.  The eyeball is fixed; only lid skin moves.

Groups: eyeball = scleras|irises|pupils ; bulge = irises|pupils ; lid = eye_exteriors
"""

from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree

EYEBALL_GROUPS = ("scleras", "irises", "pupils")
BULGE_GROUPS = ("irises", "pupils")
LID_GROUP = "eye_exteriors"


def _ids(g, n):
    return np.asarray(g(n), dtype=np.int64).reshape(-1)


def groups(g):
    eb = np.unique(np.concatenate([_ids(g, x) for x in EYEBALL_GROUPS]))
    bg = np.unique(np.concatenate([_ids(g, x) for x in BULGE_GROUPS]))
    return eb, bg, _ids(g, LID_GROUP)


def _vn(V, faces, ids):
    m = np.zeros(V.shape[0], bool); m[ids] = True
    f = faces[m[faces].all(1)]
    vn = np.zeros_like(V)
    if f.size == 0:
        return vn
    tri = V[f]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
    for k in range(3):
        np.add.at(vn, f[:, k], fn)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12
    return vn


def _nearest_sdf(src, dst, dst_n):
    dist, nn = cKDTree(dst).query(src)
    rel = src - dst[nn]
    return (rel * dst_n[nn]).sum(-1), dist, nn


def _sphere_center(P):
    A = np.concatenate([-2.0 * P, np.ones((len(P), 1))], axis=1)
    b = (P ** 2).sum(1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    return sol[:3]


def _sides(V, g):
    eb, bg, lid = groups(g)
    scl = _ids(g, "scleras")
    out = {}
    for nm, m in (("L", V[:, 0] < 0), ("R", V[:, 0] >= 0)):
        E = eb[m[eb]]; B = bg[m[bg]]
        if E.size == 0 or B.size == 0:
            continue
        O = _sphere_center(V[np.intersect1d(scl, E)])
        f = V[B].mean(0) - O
        f = f / (np.linalg.norm(f) + 1e-12)
        out[nm] = dict(O=O, f=f, B=B, L=lid[m[lid]])
    return out


def _front(V, side, lid_ids, n_bg, nn):
    lid_front = ((V[lid_ids] - side["O"]) @ side["f"]) > 0.0
    nn_front = (n_bg[side["B"]][nn] @ side["f"]) > 0.0
    return lid_front & nn_front


def eligibility(rest_V, faces, g, gate=2e-3, floor=2e-5):
    """Per side: lid vertex ids that are OUTSIDE the bulge at rest (rest sdf>floor)."""
    n_bg = _vn(rest_V, faces, groups(g)[1])
    el = {}
    for nm, side in _sides(rest_V, g).items():
        L = side["L"]; B = side["B"]
        if L.size == 0:
            el[nm] = np.zeros(0, np.int64)
            continue
        s0, d0, nn0 = _nearest_sdf(rest_V[L], rest_V[B], n_bg[B])
        fr = _front(rest_V, side, L, n_bg, nn0) & (d0 < gate)
        el[nm] = L[fr & (s0 > floor)]
    return el


def _stats(V, faces, g, el, margin, gate):
    n_bg = _vn(V, faces, groups(g)[1])
    worst = 1e9; n_below = 0; n_el = int(sum(len(v) for v in el.values()))
    for nm, side in _sides(V, g).items():
        E_ = set(el.get(nm, np.zeros(0, np.int64)).tolist())
        L = side["L"]; B = side["B"]
        if L.size == 0 or not E_:
            continue
        sdf, dist, nn = _nearest_sdf(V[L], V[B], n_bg[B])
        fr = _front(V, side, L, n_bg, nn) & (dist < gate)
        allow = np.fromiter((i in E_ for i in L.tolist()), bool, L.size)
        m = fr & allow
        if m.any():
            worst = min(worst, float(sdf[m].min()))
            n_below += int((sdf[m] < margin).sum())
    return {"n_eligible": n_el, "n_below_margin": n_below,
            "min_sdf_mm": (worst * 1000.0 if worst < 1e8 else float("nan"))}


def enforce_lid_clearance(V, rest_V, faces, g, margin=1e-4, gate=2e-3, iters=12):
    """Rotated bulge vs covering lid: push only rest-outside lid verts to margin."""
    V = np.asarray(V, dtype=np.float64).copy()
    el = eligibility(rest_V, faces, g, gate)
    n_bg = _vn(V, faces, groups(g)[1])          # current (rotated) bulge
    pre = _stats(V, faces, g, el, margin, gate)
    for _ in range(max(1, iters)):
        moved = 0.0
        for nm, side in _sides(V, g).items():
            E_ = set(el.get(nm, np.zeros(0, np.int64)).tolist())
            L = side["L"]; B = side["B"]
            if L.size == 0 or not E_:
                continue
            sdf, dist, nn = _nearest_sdf(V[L], V[B], n_bg[B])
            fr = _front(V, side, L, n_bg, nn) & (dist < gate)
            allow = np.fromiter((i in E_ for i in L.tolist()), bool, L.size)
            pen = np.where(fr & allow, np.maximum(0.0, margin - sdf), 0.0)
            if pen.max() < 1e-9:
                continue
            V[L] += pen[:, None] * n_bg[B][nn]
            moved = max(moved, float(pen.max()))
        if moved < 1e-7:
            break
    post = _stats(V, faces, g, el, margin, gate)
    return V, {"pre": pre, "post": post, "margin_mm": margin * 1000.0}


def rotate_eyeball(V, g, side_deg):
    """Rotate each globe about its true sclera-sphere centre (sphere-invariant)."""
    V = np.asarray(V, dtype=np.float64).copy()
    eb, _, _ = groups(g)
    scl = _ids(g, "scleras")
    for nm, (pitch, yaw) in side_deg.items():
        m = (V[:, 0] < 0) if nm == "L" else (V[:, 0] >= 0)
        E = eb[m[eb]]
        O = _sphere_center(V[np.intersect1d(scl, E)])
        a, b = np.deg2rad(pitch), np.deg2rad(yaw)
        Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
        Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
        V[E] = (V[E] - O) @ (Ry @ Rx).T + O
    return V
