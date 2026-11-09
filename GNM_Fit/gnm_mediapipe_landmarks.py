"""MediaPipe 468-landmark correspondence for the GNM head model: loader and self-check.

`head_dense_468.txt` uses the same format as GNM's shipped `head_sparse_68.txt`
(three `vertex_index weight` pairs per row, barycentric weights summing to 1). GNM's
`gnm_landmarks.load_landmarks()` dispatches off the `GNMLandmarksType` enum and only
knows HEAD_SPARSE_68, so read the file directly, as below.

Because the anchors are barycentric on the GNM topology, they hold for any identity,
expression and pose without re-registration.

    python gnm_mediapipe_landmarks.py          # runs the built-in self-check

Unofficial and community-built. Not affiliated with Google.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

LANDMARK_FILE = Path(__file__).with_name("head_dense_468.txt")
LANDMARK_COUNT = 468


def load_landmarks(path=LANDMARK_FILE):
    """Returns (indices, weights): int (N, 3) and float (N, 3).

    Default file is 468. Same barycentric format as official 68, so this also
    reads ``head_sparse_68.txt`` (the v2 pack rejected anything but 468).
    """
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] != 6:
        raise ValueError(f"expected (N, 6), got {data.shape}")
    return data[:, ::2].astype(np.int32), data[:, 1::2].astype(np.float64)


def sample(vertices: np.ndarray, indices: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Evaluate the 468 landmarks on a GNM mesh, returning (468, 3).

    `vertices` is any (17821, 3) GNM head mesh: the template, a sampled identity,
    an expression or a posed frame. The barycentric anchors do not care which.
    """
    return (vertices[indices] * weights[..., None]).sum(axis=1)


# MediaPipe canonical indices worth knowing. Sides are the SUBJECT's, verified
# against GNM's own left/right vertex groups; some MediaPipe docs use the viewer's.
MOUTH_CORNER_RIGHT = 61
MOUTH_CORNER_LEFT = 291
LIP_INNER_UPPER = 13
LIP_INNER_LOWER = 14
EYE_OUTER_RIGHT = 33
EYE_INNER_RIGHT = 133
EYE_OUTER_LEFT = 263
EYE_INNER_LEFT = 362
LID_UPPER_RIGHT, LID_LOWER_RIGHT = 159, 145
LID_UPPER_LEFT, LID_LOWER_LEFT = 386, 374

# Mirror-image measurements: (label, right-side pair, left-side pair). Each entry
# measures the same anatomy on both sides, so the two should agree on GNM's
# neutral head, which is left/right symmetric by construction.
MIRROR_PAIRS = [
    ("palpebral aperture", (159, 145), (386, 374)),
    ("lid pair 158/153",   (158, 153), (385, 380)),
    ("lid pair 160/144",   (160, 144), (387, 373)),
    ("lid pair 161/163",   (161, 163), (388, 390)),
    ("eye width",          (33, 133),  (263, 362)),
    ("brow to lid",        (105, 159), (334, 386)),
    ("commissure to eye",  (61, 33),   (291, 263)),
]
SYMMETRY_TOLERANCE_MM = 1.0

HIDDEN_GROUPS = ("mouth_sock", "teeth", "gums", "tongue", "eyes")


def self_check(vertices: np.ndarray) -> bool:
    """Assert the correspondence resolves the structures most people need.

    A landmark set can be numerically valid, with every anchor on the mesh and
    small residuals, and still be useless if landmarks that should be distinct
    collapse together. These checks are aimed at exactly that.
    """
    idx, w = load_landmarks()
    p = sample(vertices, idx, w)

    def mm(a, b):
        return float(np.linalg.norm(p[a] - p[b]) * 1000.0)

    def up(a, b):
        """Height of a above b. Openings are measured this way because a straight
        3D distance cannot tell an open mouth from upper and lower lip points
        pushed apart in depth, which looks closed from the front."""
        return float((p[a, 1] - p[b, 1]) * 1000.0)

    checks = [
        ("inter-ocular distance", mm(EYE_OUTER_RIGHT, EYE_OUTER_LEFT), 50.0, 120.0),
        ("mouth width", mm(MOUTH_CORNER_RIGHT, MOUTH_CORNER_LEFT), 25.0, 80.0),
        ("palpebral aperture R", up(LID_UPPER_RIGHT, LID_LOWER_RIGHT), 3.0, 20.0),
        ("palpebral aperture L", up(LID_UPPER_LEFT, LID_LOWER_LEFT), 3.0, 20.0),
        ("inner-lip opening", up(LIP_INNER_UPPER, LIP_INNER_LOWER), 2.0, 15.0),
        ("eye width R", mm(EYE_OUTER_RIGHT, EYE_INNER_RIGHT), 15.0, 45.0),
    ]
    ok = True
    print("self-check (values are for GNM's neutral template):")
    for name, val, lo, hi in checks:
        good = lo <= val <= hi
        ok &= good
        print(f"  [{'ok' if good else 'XX'}] {name:22s} {val:7.2f} mm   (expect {lo} to {hi})")

    # Range checks alone are weak: a landmark can leave its anatomy while the
    # distance it feeds stays plausible. On a symmetric head any left/right
    # disagreement is embedding error, so mirror pairs catch what ranges miss.
    print("\n  mirror symmetry (GNM's neutral head is symmetric by construction):")
    worst_name, worst = "", 0.0
    for name, r, l in MIRROR_PAIRS:
        d = float(abs(np.linalg.norm(p[r[0]] - p[r[1]])
                      - np.linalg.norm(p[l[0]] - p[l[1]])) * 1000.0)
        if d > worst:
            worst_name, worst = name, d
        good = d < SYMMETRY_TOLERANCE_MM
        ok &= good
        print(f"  [{'ok' if good else 'XX'}] {name:22s} L/R differ by {d:5.2f} mm "
              f"(tolerance {SYMMETRY_TOLERANCE_MM})")
    print(f"       worst: {worst_name} at {worst:.2f} mm")
    return ok


def check_visible_surface(gnm) -> bool:
    """Assert no landmark sits on geometry hidden inside the head.

    GNM's mesh has interior sheets: the oral mucosa behind the lips and the
    conjunctiva behind the eyelids. A landmark projected onto one is still
    exactly on the mesh and passes every distance check, but it renders as a
    point sunk into the mouth or the eye opening.
    """
    idx, w = load_landmarks()
    groups = {n: set(np.asarray(gnm.vertex_group_indices(n)).tolist())
              for n in gnm.vertex_group_names}
    hidden = set().union(*(groups.get(n, set()) for n in HIDDEN_GROUPS))

    touching = [i for i in range(LANDMARK_COUNT) if any(int(v) in hidden for v in idx[i])]
    inside = [i for i in touching
              if sum(w[i][k] for k in range(3) if int(idx[i][k]) in hidden) > 0.95]
    print("\n  visible surface:")
    print(f"  [{'ok' if not inside else 'XX'}] landmarks sitting inside the mouth or eye: "
          f"{len(inside)}")
    if inside:
        print(f"       {inside[:12]}")
    print(f"       landmarks on a triangle that touches interior geometry: {len(touching)}")
    return not inside


if __name__ == "__main__":
    from gnm.shape import gnm_numpy

    gnm = gnm_numpy.GNM.from_local(
        version=gnm_numpy.GNMMajorVersion.V3, variant=gnm_numpy.GNMVariant.HEAD)
    verts = np.asarray(gnm(None, None, None, None))
    print(f"GNM v3 HEAD: {verts.shape[0]} vertices")

    idx, w = load_landmarks()
    print(f"loaded {idx.shape[0]} landmarks; weights sum to 1: "
          f"{np.allclose(w.sum(axis=1), 1.0, atol=1e-3)}")

    lm = sample(verts, idx, w)
    print(f"sampled landmarks: {lm.shape}\n")

    passed = self_check(verts)
    passed &= check_visible_surface(gnm)

    # The anchors are identity- and expression-invariant: show it on a smile.
    e = np.zeros(gnm.expression_dim)
    lower_face = [i for i, n in enumerate(gnm.expression_names)
                  if n.startswith("lower_face_region_")]
    e[lower_face[1]] = -3.0                     # a smile in GNM's basis
    smiling = np.asarray(gnm(None, e, None, None))
    lm_smile = sample(smiling, idx, w)
    lift = (lm_smile[MOUTH_CORNER_RIGHT, 1] - lm[MOUTH_CORNER_RIGHT, 1]) * 1000
    print(f"\nsame anchors on a smiling mesh: right mouth corner rises {lift:+.2f} mm")
    print("\nRESULT:", "PASS" if passed else "FAIL")