"""Regenerate the MediaPipe-468 <-> GNM correspondence from scratch.

Use this instead of the shipped `head_dense_468.txt` if GNM's topology changes in
a future release, or if you want to adjust which anatomy each landmark may snap to.

    pip install numpy scipy
    curl -LO https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/mediapipe/modules/face_geometry/data/canonical_face_model.obj
    python build_mediapipe_correspondence.py --canonical canonical_face_model.obj --out head_dense_468.txt

Method
------
1. Evaluate GNM's shipped 68 landmarks in 3D, correcting their order first
   (google/GNM#75, see GNM68_REVERSED_SLICE).
2. Fit MediaPipe's canonical mesh to them with a Umeyama similarity transform,
   using the 468 -> 68 table MP68. The mirrored ordering is also tried and the
   lower-residual fit wins.
3. Warp the result with a thin-plate spline pinned at the 68 points, so that
   local shape differences between the two faces (eye size, mouth opening) are
   removed rather than averaged away.
4. Project every canonical vertex onto the GNM surface, constrained to triangles
   of an anatomically appropriate vertex group, and store the barycentric anchor.

Unofficial and community-built. Not affiliated with Google.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree

from gnm.shape import gnm_landmarks, gnm_numpy

# MediaPipe canonical 468 -> iBUG 68. A row naming two vertices means "the
# midpoint of these two": six iBUG points fall between MediaPipe vertices.
#
# Based on PeizhiYan's mp2dlib table, with the symmetry fix from its issue #3
# (row 14 is mid(288, 361), mirroring row 4's mid(132, 58)).
#
# Inner lips 61/63/65/67 differ from mp2dlib and from the widely copied
# StackOverflow table. Both send them to 82/312/317/87, which are the inner-lip
# points next to the CENTRE, while iBUG places them about halfway to the corners.
# Measured as a fraction of arc length along each inner-lip contour:
#
#   iBUG   target (GNM's own)   was          now
#    61         0.27            82  0.39     81  0.29
#    63         0.73            312 0.61     311 0.71
#    65         0.27            317 0.39     402 0.29
#    67         0.74            87  0.61     178 0.71
#
# Every other eye and lip row was checked the same way and is within 0.08.
MP68 = [
    # face contour
    [127], [234], [93], [132, 58], [58, 172], [136], [150], [176], [152],
    [400], [379], [365], [397, 288], [288, 361], [323], [454], [356],
    # right brow
    [70], [63], [105], [66], [107],
    # left brow
    [336], [296], [334], [293], [300],
    # nose bridge and base
    [168, 6], [197, 195], [5], [4],
    [75], [97], [2], [326], [305],
    # right eye
    [33], [160], [158], [133], [153], [144],
    # left eye
    [362], [385], [387], [263], [373], [380],
    # outer lips
    [61], [39], [37], [0], [267], [269], [291],
    [321], [314], [17], [84], [91],
    # inner lips
    [78], [81], [13], [311], [308], [402], [14], [178],
]

# GNM v3.0's head_sparse_68.txt stores iBUG jaw anchors 2-6 in reverse order
# (google/GNM#75). Checked geometrically: as shipped, the jaw chain has two
# ~101 mm jumps between otherwise ~22 mm steps, and reversing this slice is the
# only contiguous reversal that removes them (path 506.2 mm -> 347.0 mm). The fit
# below pairs points BY INDEX, so a scrambled order matches the wrong ones.
GNM68_REVERSED_SLICE = slice(2, 7)

# Landmark groups that constrain where each MediaPipe point may project.
MP_UPPER_LIP = {61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
                78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308}
MP_LOWER_LIP = {61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
                78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308}
MP_CORNERS = {61, 291, 78, 308}

# Eyelid margins, split upper and lower. This split is load-bearing: projected
# onto one undivided region, the upper and lower lid land on the same triangle
# and the palpebral aperture collapses from about 6 mm to about 0.5 mm, silently.
MP_LID_UPPER = {157, 158, 159, 160, 161, 246, 384, 385, 386, 387, 388, 466}
MP_LID_LOWER = {144, 145, 153, 154, 163, 7, 373, 374, 380, 381, 390, 249}

MP_EYE_R = {33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246}
MP_EYE_L = {362, 249, 390, 373, 374, 380, 381, 382, 398, 384, 385, 386, 387, 388, 466, 263}
MP_BROW_R = {70, 63, 105, 66, 107, 55, 65, 52, 53, 46}
MP_BROW_L = {336, 296, 334, 293, 300, 276, 283, 282, 295, 285}

# Regions allowed to project onto triangles that touch interior geometry. The
# eyelid margin is a genuine skin/cavity boundary, so it needs them. Everywhere
# else, a triangle touching the mouth cavity lets a lip landmark slide onto the
# oral mucosa, so those triangles are excluded.
CAVITY_OK_KEYS = {"lid_upper", "lid_lower", "eye_l", "eye_r"}


def gnm68_fix_order(pts):
    """Undo google/GNM#75. Returns a corrected copy."""
    out = pts.copy()
    out[GNM68_REVERSED_SLICE] = pts[GNM68_REVERSED_SLICE][::-1]
    return out


def mp68_points(canon):
    """The iBUG-68 points of a 468 set, averaging rows that name two vertices."""
    return np.array([canon[idx].mean(axis=0) for idx in MP68])


def umeyama(src, dst):
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    cov = b.T @ a / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / (a ** 2).sum() * len(src)
    t = md - s * R @ ms
    rms = float(np.sqrt(((s * src @ R.T + t - dst) ** 2).sum(1).mean()))
    return s, R, t, rms


def mirror_68(idx_list):
    """Reorder an iBUG-68 list so each entry swaps with its left/right mirror."""
    m = list(range(68))
    m[0:17] = list(range(16, -1, -1))
    m[17:22], m[22:27] = list(range(26, 21, -1)), list(range(21, 16, -1))
    m[31:36] = list(range(35, 30, -1))
    m[36:42] = [45, 44, 43, 42, 47, 46]
    m[42:48] = [39, 38, 37, 36, 41, 40]
    m[48:55] = [54, 53, 52, 51, 50, 49, 48]
    m[55:60] = [59, 58, 57, 56, 55]
    m[60:65] = [64, 63, 62, 61, 60]
    m[65:68] = [67, 66, 65]
    return [idx_list[i] for i in m]


def closest_point_barycentric(p, tri):
    """Closest point on a triangle, as barycentric coordinates (clamped)."""
    a, b, c = tri
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = ab @ ap, ac @ ap
    if d1 <= 0 and d2 <= 0:
        return np.array([1.0, 0.0, 0.0])
    bp = p - b
    d3, d4 = ab @ bp, ac @ bp
    if d3 >= 0 and d4 <= d3:
        return np.array([0.0, 1.0, 0.0])
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        v = d1 / (d1 - d3)
        return np.array([1 - v, v, 0.0])
    cp = p - c
    d5, d6 = ab @ cp, ac @ cp
    if d6 >= 0 and d5 <= d6:
        return np.array([0.0, 0.0, 1.0])
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        w = d2 / (d2 - d6)
        return np.array([1 - w, 0.0, w])
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return np.array([0.0, 1 - w, w])
    denom = 1.0 / (va + vb + vc)
    v, w = vb * denom, vc * denom
    return np.array([1 - v - w, v, w])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--canonical", type=Path, required=True,
                    help="MediaPipe canonical_face_model.obj")
    ap.add_argument("--out", type=Path, default=Path("head_dense_468.txt"))
    ap.add_argument("--no-warp", action="store_true",
                    help="similarity transform only, skip the thin-plate warp (for comparison)")
    ap.add_argument("--no-gnm75-fix", action="store_true",
                    help="keep GNM's shipped anchor order (for comparison)")
    args = ap.parse_args()

    gnm = gnm_numpy.GNM.from_local(
        version=gnm_numpy.GNMMajorVersion.V3, variant=gnm_numpy.GNMVariant.HEAD)
    neutral = np.asarray(gnm(None, None, None, None))

    canon = np.array([[float(x) for x in ln.split()[1:4]]
                      for ln in args.canonical.read_text().splitlines()
                      if ln.startswith("v ")])
    assert canon.shape == (468, 3), f"expected 468 canonical verts, got {canon.shape}"

    lm = gnm_landmarks.load_landmarks(gnm_landmarks.GNMLandmarksType.HEAD_SPARSE_68)
    gnm68 = (neutral[lm.indices] * lm.weights[..., None]).sum(axis=1)
    if not args.no_gnm75_fix:
        gnm68 = gnm68_fix_order(gnm68)
        print("applied google/GNM#75 anchor-order fix (iBUG 2-6)")

    # step 2: global similarity fit
    ref = mp68_points(canon)
    ref_m = np.array([canon[idx].mean(axis=0) for idx in mirror_68(MP68)])
    fits = {tag: umeyama(p, gnm68) for tag, p in (("normal", ref), ("mirrored", ref_m))}
    best = min(fits, key=lambda k: fits[k][3])
    s, R, t, _ = fits[best]
    print(f"umeyama: normal {fits['normal'][3]*1000:.2f} mm, "
          f"mirrored {fits['mirrored'][3]*1000:.2f} mm, using {best}")
    canon_g = s * canon @ R.T + t

    # step 3: thin-plate warp
    # One similarity transform cannot reconcile two faces of different SHAPE.
    # MediaPipe's canonical face has relatively smaller eyes and a narrower,
    # more closed mouth than GNM's, so after a global fit its eyelid and inner-lip points sit
    # inside GNM's openings and projection flattens them. Pinning a thin-plate
    # spline at the 68 points removes that local mismatch. It is only correct
    # because MP68 is semantically right: a wrong pin drags the face with it.
    #
    # The pins are symmetrised first. GNM's head is left/right symmetric but its
    # shipped 68 anchors are not quite (the chin centre, iBUG 8, sits 1.35 mm off
    # the midline). Averaging each anchor with its mirror image keeps that
    # asymmetry out of all 468 landmarks.
    if not args.no_warp:
        src = mp68_points(canon_g)
        perm = mirror_68(list(range(68)))
        target = 0.5 * (gnm68 + gnm68[perm] * np.array([-1.0, 1.0, 1.0]))
        warp = RBFInterpolator(src, target - src, kernel="thin_plate_spline", smoothing=0.0)
        before = np.sqrt(((src - gnm68) ** 2).sum(1).mean()) * 1000
        canon_g = canon_g + warp(canon_g)
        after = np.sqrt(((mp68_points(canon_g) - target) ** 2).sum(1).mean()) * 1000
        print(f"thin-plate warp: 68-point residual {before:.2f} mm -> {after:.2f} mm")

    # step 4: constrained projection
    def group(name):
        return set(np.asarray(gnm.vertex_group_indices(name)).tolist()) \
            if name in gnm.vertex_group_names else set()

    skin = group("skin_exterior") or group("skin")
    upper_lip = group("upper_lip") | group("upper_lip_region")
    lower_lip = group("lower_lip") | group("lower_lip_region")
    orbital_l, orbital_r = group("left_orbital_region"), group("right_orbital_region")
    brow_l, brow_r = group("left_brow_region"), group("right_brow_region")
    exclude = (group("eyes") | group("teeth") | group("gums")
               | group("tongue") | group("mouth_sock"))
    generic = skin - exclude

    # Upper and lower lid bands, per eye. The socket group is intersected with
    # skin_exterior first: `eye_sockets` also contains the conjunctival surface
    # behind the lid (448 of its 608 vertices), and a lid landmark projected onto
    # that sits inside the eye.
    socket = np.asarray(sorted(group("eye_sockets") & skin), dtype=int)
    lid_upper, lid_lower = set(), set()
    for side in (socket[neutral[socket, 0] >= 0], socket[neutral[socket, 0] < 0]):
        if not side.size:
            continue
        cy = neutral[side, 1].mean()
        lid_upper |= set(side[neutral[side, 1] >= cy].tolist())
        lid_lower |= set(side[neutral[side, 1] < cy].tolist())

    regions = {
        "corners": upper_lip | lower_lip, "upper_lip": upper_lip, "lower_lip": lower_lip,
        "lid_upper": lid_upper or generic, "lid_lower": lid_lower or generic,
        "eye_l": orbital_l or generic, "eye_r": orbital_r or generic,
        "brow_l": brow_l | orbital_l, "brow_r": brow_r | orbital_r, "generic": generic,
    }

    def region_for(i):
        if i in MP_CORNERS: return "corners"
        if i in MP_LID_UPPER: return "lid_upper"
        if i in MP_LID_LOWER: return "lid_lower"
        if i in MP_UPPER_LIP: return "upper_lip"
        if i in MP_LOWER_LIP: return "lower_lip"
        if i in MP_EYE_L: return "eye_l"
        if i in MP_EYE_R: return "eye_r"
        if i in MP_BROW_L: return "brow_l"
        if i in MP_BROW_R: return "brow_r"
        return "generic"

    tris = np.asarray(gnm.triangles_group("~eye_exteriors")).astype(np.int64)
    cache, rows, errs = {}, [], []
    for i in range(468):
        key = region_for(i)
        if key not in cache:
            sub = tris[np.isin(tris, list(regions[key])).any(axis=1)]
            if key not in CAVITY_OK_KEYS:
                sub = sub[~np.isin(sub, list(exclude)).any(axis=1)]
            cache[key] = (sub, cKDTree(neutral[sub].mean(axis=1)))
        sub, tree = cache[key]
        _, cand = tree.query(canon_g[i], k=min(24, len(sub)))
        best_d, best_row = np.inf, None
        for ti in np.atleast_1d(cand):
            tri = sub[ti]
            bary = closest_point_barycentric(canon_g[i], neutral[tri])
            d = np.linalg.norm(canon_g[i] - (neutral[tri] * bary[:, None]).sum(0))
            if d < best_d:
                best_d, best_row = d, (tri, bary)
        rows.append(best_row)
        errs.append(best_d)

    with open(args.out, "w") as f:
        for tri, bary in rows:
            f.write(f"{tri[0]} {bary[0]:.6f} {tri[1]} {bary[1]:.6f} "
                    f"{tri[2]} {bary[2]:.6f}\n")

    e = np.asarray(errs) * 1000
    print(f"wrote {args.out}: projection residual mean {e.mean():.2f} mm, "
          f"p95 {np.percentile(e, 95):.2f} mm, max {e.max():.2f} mm")

    # Openings are measured vertically. A straight 3D distance cannot tell an
    # open mouth from upper and lower lip points pushed apart in depth, which
    # looks closed from the front.
    p = np.array([(neutral[tri] * bary[:, None]).sum(0) for tri, bary in rows]) * 1000
    ap_r, ap_l = p[159, 1] - p[145, 1], p[386, 1] - p[374, 1]
    lip = p[13, 1] - p[14, 1]
    ok = min(ap_r, ap_l) > 3 and lip > 2
    print(f"openings: eyes R {ap_r:.2f} mm, L {ap_l:.2f} mm, inner lip {lip:.2f} mm "
          f"({'ok' if ok else 'COLLAPSED, check the lid split and the warp'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())