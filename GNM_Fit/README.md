# MediaPipe 468-landmark correspondence for GNM Head

An unofficial mapping from MediaPipe FaceMesh's 468 canonical landmarks onto
Google's [GNM](https://github.com/google/GNM) head model, in the same file format
as GNM's shipped `head_sparse_68.txt`.

GNM ships a 68-point correspondence. MediaPipe, which most people actually run,
reports 468. This fills that gap.

**Not affiliated with Google. Community-built, provided as-is.**

| File | What it is |
|---|---|
| `head_dense_468.txt` | The correspondence: 468 rows, `i0 w0 i1 w1 i2 w2` (barycentric) |
| `gnm_mediapipe_landmarks.py` | Loader and self-check (run it to verify) |
| `build_mediapipe_correspondence.py` | Regenerates the file from scratch, and holds the 468 to 68 table |

Built against **GNM v3.0, `HEAD` variant** (17,821 vertices). If GNM's topology
changes, regenerate rather than reusing the file.

> **This is v2.** If you took the first version, replace it. It placed 41 of the
> 468 landmarks on the inside of the mouth, its eyelid landmarks sat behind the
> lid, and its eye and mouth contours did not match GNM's own anchors. See
> [What changed in v2](#what-changed-in-v2).

## Use

GNM's `gnm_landmarks.load_landmarks()` dispatches off the `GNMLandmarksType` enum
and only knows `HEAD_SPARSE_68`, so read this file directly:

```python
import numpy as np
from gnm.shape import gnm_numpy

emb = np.loadtxt("head_dense_468.txt")
idx, w = emb[:, ::2].astype(int), emb[:, 1::2]        # (468, 3) each

gnm = gnm_numpy.GNM.from_local(version=gnm_numpy.GNMMajorVersion.V3,
                               variant=gnm_numpy.GNMVariant.HEAD)
verts = np.asarray(gnm(None, None, None, None))       # any identity, expression, pose
landmarks = (verts[idx] * w[..., None]).sum(axis=1)   # (468, 3)
```

The anchors are barycentric on the GNM topology, so the same file works for every
identity, expression and pose without re-registration.

### Getting 68 points from the 468

If you need iBUG-68 points, use the `MP68` table in
`build_mediapipe_correspondence.py`. Where a row names two vertices, take their
midpoint:

```python
from build_mediapipe_correspondence import MP68
pts68 = np.array([landmarks[ids].mean(axis=0) for ids in MP68])
```

Use this table rather than the commonly copied ones. It differs from them at the
inner lips (see fix 2 below), and the commonly copied choice bunches the six
inner-lip points into the middle third of the mouth.

## Accuracy

All numbers are measured on GNM's neutral head.

**Against GNM's own 68 anchors.** These are GNM's iBUG landmarks with the order
fix from google/GNM#75 applied.

| Region | Mean | Max |
|---|---|---|
| Eyes | 0.21 mm | 0.60 mm |
| Lips | 0.07 mm | 0.22 mm |
| Brows | 0.04 mm | 0.12 mm |
| Nose | 0.27 mm | 0.80 mm |
| Jaw | 0.52 mm | 2.02 mm |
| All 68 | 0.23 mm | 2.02 mm |

Treat this table with care. The build pins a warp to these 68 points (fix 3), so
close agreement here is partly by construction. The jaw is loosest because
iBUG's jawline and MediaPipe's face oval are not quite the same curve.

**The checks that are not by construction** cover the other 400 landmarks:

| Check | v1 | v2 |
|---|---|---|
| Landmarks inside the mouth cavity | 41 | **0** |
| Mirror asymmetry, mean | 0.28 mm | **0.05 mm** |
| Mirror asymmetry, max | 5.81 mm | **0.37 mm** |
| Eye opening, 159 to 145 (R / L) | 3.43 / 4.56 mm | **8.25 / 8.23 mm** |

Mirror asymmetry is the most useful single number here. GNM's neutral head is
left/right symmetric by construction, so any disagreement between a landmark
and its left/right counterpart is embedding error, and it sets the noise floor under any left/right
measurement you build on these points. Note how lopsided v1's eyes were.

## What changed in v2

**1. GNM's shipped anchors are stored out of order** (google/GNM#75).
`head_sparse_68.txt` stores iBUG jaw anchors 2 to 6 in reverse. Checked
geometrically rather than taken on trust: as shipped, the jaw chain has two jumps
of about 101 mm between otherwise 22 mm steps, and reversing that slice is the
only contiguous reversal that removes them (total path 506.2 mm becomes
347.0 mm). The alignment pairs MediaPipe to GNM point by point, so the
scrambled order matched the wrong points to each other.

**2. A better 468 to 68 table.** Based on Peizhi Yan's
[mp2dlib](https://github.com/PeizhiYan/Mediapipe_2_Dlib_Landmarks), with the
symmetry fix from its
[issue #3](https://github.com/PeizhiYan/Mediapipe_2_Dlib_Landmarks/issues/3).
Six iBUG points fall between two MediaPipe vertices, and mp2dlib uses their
midpoint.

One further correction, which mp2dlib and the widely copied StackOverflow table
share. Both send iBUG inner-lip points 61/63/65/67 to MediaPipe 82/312/317/87,
the points next to the centre of the lip. iBUG places them about halfway to the
corners. As a fraction of arc length along each inner-lip contour:

| iBUG | GNM's own anchor | Was | Now |
|---|---|---|---|
| 61 | 0.27 | 82 at 0.39 | **81** at 0.29 |
| 63 | 0.73 | 312 at 0.61 | **311** at 0.71 |
| 65 | 0.27 | 317 at 0.39 | **402** at 0.29 |
| 67 | 0.74 | 87 at 0.61 | **178** at 0.71 |

Every other eye and lip row was checked the same way and is within 0.08.

**3. A thin-plate warp for shape differences.** A single rotate-and-scale cannot
reconcile two faces of different shape. MediaPipe's canonical face has
relatively smaller eyes and a narrower mouth than GNM's, so after a global fit
its eye and lip points sit inside GNM's openings and get flattened when
projected. v2 adds a thin-plate spline pinned at the 68 points, which removes
the local mismatch and interpolates smoothly everywhere else. This is only safe
because the table in fix 2 is right: a wrong pin drags the face with it.

**4. Symmetrised pins.** GNM's own 68 anchors are slightly asymmetric: mean
mirror error 0.13 mm, and the chin centre (iBUG 8) sits 1.35 mm off the midline
of a symmetric head. Warping onto them verbatim doubled our asymmetry, so each
pin is averaged with its left/right counterpart first.

**5. Landmarks kept on the visible surface.** GNM's mesh has interior sheets: the
oral mucosa behind the lips, and the inner surface of the eye socket behind the
eyelids. A landmark projected onto one is still exactly on the mesh and passes
every distance check, but it shows up as a point sunk into the mouth or eye.

In v1, 41 landmarks sat on the oral mucosa, including 15 of the 20 on the
inner-lip contour. Its eyelid landmarks sat on the socket's inner surface,
behind the lid. v2 restricts the eye socket group to exterior skin before the
lid bands are formed, and bars every landmark outside the eyelids from
triangles that touch the mouth or eye interior.

## Two things that will bite you if you build your own

**Split the eyelids.** The MediaPipe eyelid landmarks (159/145, 386/374 and
neighbours) must project onto GNM's `eye_sockets` group split into upper and lower
bands per eye. Use one undivided region and the upper and lower lid snap to the
same triangle. The palpebral aperture then reads about 0.5 mm instead of about
6 mm, nothing raises an error, and anything eye-related silently stops working.

**Measure openings vertically.** A straight 3D distance cannot tell an open mouth
from upper and lower lip points that have been pushed apart in depth. Those look
closed from the front. The self-check measures the eye and mouth openings as
heights, not distances, for this reason.

Run the self-check whenever the model version or the projection regions change:

```
python gnm_mediapipe_landmarks.py
```

## Side naming

MediaPipe's own docs are inconsistent about whether "left" means the subject's
left or the viewer's left. Verified against GNM's own `left`/`right` vertex
groups, these indices are **anatomical (subject) sides**:

- `61` is the subject's **right** mouth corner, `291` the subject's **left**
- `33` is the subject's **right** eye outer corner, `263` the subject's **left**

If you report per-side results to anyone clinical, pin this down explicitly.

## Regenerating

```bash
pip install numpy scipy
curl -LO https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/mediapipe/modules/face_geometry/data/canonical_face_model.obj
python build_mediapipe_correspondence.py --canonical canonical_face_model.obj --out head_dense_468.txt
```

This reproduces `head_dense_468.txt` byte for byte from GNM v3.0 (tested with
Python 3.13, NumPy 2.5 and SciPy 1.18). `--no-warp` and `--no-gnm75-fix` turn off
fixes 3 and 1 if you want to see what they change.

## Credits

Two of the v2 fixes came from a reader's review on the GNM issue tracker: the
pointer to mp2dlib and its issue #3, and the pointer to google/GNM#75. Chasing
those led to the rest.

## Licence

Derived from two Apache-2.0 works, Google's GNM head model and MediaPipe's
canonical face model, and released on the same terms. Please keep the
attribution to both.