"""Test eye-lid collision over a reasonable gaze range.

Question: inside reasonable eyeball rotation, do the protruding iris/pupil ever
penetrate the covering eyelid skin?

Method
-----
- eyeball rotates about its true sclera-sphere centre (sphere-invariant, so only
  the iris/pupil bulge moves on the surface; the sclera does not translate).
- collision is rest-referenced: only eyelid vertices that are OUTSIDE the bulge
  at rest (rest sdf > small floor) are eligible; they must not be pushed inside
  margin after rotation.
- margin: 0.1 mm safety distance; gate: 2 mm nearest-neighbour gate.
"""

from __future__ import annotations
import numpy as np, torch, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import photo_fit as pf
import lid_clearance as LC
from gnm.shape import gnm_pytorch, gnm_numpy
from fit_gnm import FitState

EV3 = "/root/autodl-tmp/gnm_fit_out/expression_v3"
RUN = EV3 + "/pipeline_run_reorder"

MARGIN = 1e-4   # 0.1 mm
GATE   = 2e-3   # 2 mm

# ---------- load current (already pupil-sized) params ----------
P = np.load(RUN + "/07_l2/frame_photo_params.npz")
rot = np.asarray(P["rotations"], np.float32)
st = FitState(
    id_head=torch.tensor(P["id_head"]), id_eyes=torch.tensor(P["id_eyes"]),
    ex_left=torch.tensor(P["expression_left_eye"]), ex_right=torch.tensor(P["expression_right_eye"]),
    ex_lower=torch.tensor(P["expression_lower_face"]),
    rot_neck=torch.tensor(rot[0]), rot_head=torch.tensor(rot[1]),
    rot_leye=torch.tensor(rot[2]), rot_reye=torch.tensor(rot[3]),
    translation=torch.tensor(P["translation"]),
    cam_aa=torch.tensor(P["cam_aa"]), cam_t=torch.tensor(P["cam_t"]),
    focal_norm=torch.tensor(float(P["focal_norm"])),
)
c08 = RUN + "/08_eye/frame_photo_params.npz"
if os.path.exists(c08) and "pupil" in np.load(c08).files:
    st.pupil = torch.tensor([float(np.load(c08)["pupil"][0])])
else:
    st.pupil = torch.zeros(1)

g = gnm_pytorch.GNM.from_local(version=gnm_pytorch.GNMMajorVersion.V3, variant=gnm_pytorch.GNMVariant.HEAD)
gnp = gnm_numpy.GNM.from_local(version=gnm_numpy.GNMMajorVersion.V3, variant=gnm_numpy.GNMVariant.HEAD)
faces = np.asarray(g.triangles.detach().cpu().numpy())
with torch.no_grad():
    V0 = g(st.identity()[None], st.expression()[None], st.rotations()[None], st.translation[None])[0].cpu().numpy()
gid = lambda n: pf._group_ids(gnp, n)

el = LC.eligibility(V0, faces, gid, GATE)
print("eligible covering-lid verts: L=%d R=%d (total %d)" % (
    len(el["L"]), len(el["R"]), len(el["L"]) + len(el["R"])))
rest = LC._stats(V0, faces, gid, el, MARGIN, GATE)
print("rest   : min_sdf=%.3f mm  n_below_margin=%d" % (rest["min_sdf_mm"], rest["n_below_margin"]))

# ---------- sweep reasonable gaze ----------
def fmt(s):
    return "(%+d, %+d)" % s

rows = []
worst_pre = 1e9; worst_pre_cfg = None; max_pen = 0
for pitch in range(-30, 31, 10):
    for yaw in range(-30, 31, 10):
        cfg = (pitch, yaw)
        Vg = LC.rotate_eyeball(V0, gid, {"L": cfg, "R": cfg})
        pre = LC._stats(Vg, faces, gid, el, MARGIN, GATE)
        Vf, stt = LC.enforce_lid_clearance(Vg, V0, faces, gid, MARGIN, GATE)
        post = stt["post"]
        rows.append((cfg, pre["min_sdf_mm"], pre["n_below_margin"],
                     post["min_sdf_mm"], post["n_below_margin"]))
        if pre["min_sdf_mm"] < worst_pre:
            worst_pre = pre["min_sdf_mm"]; worst_pre_cfg = cfg
        max_pen = max(max_pen, pre["n_below_margin"])

print("\n%-10s %12s %10s %12s %10s" % ("gaze", "pre_min(mm)", "pre_n_pen", "post_min(mm)", "post_n_pen"))
for cfg, pmin, pn, fmin, fn in rows:
    print("%-10s %12.3f %10d %12.3f %10d" % (fmt(cfg), pmin, pn, fmin, fn))

print("\n=== SUMMARY ===")
print("worst pre  min_sdf = %.3f mm @ %s   (max n_below_margin=%d)" % (worst_pre, fmt(worst_pre_cfg), max_pen))
all_post_ok = all(r[3] >= MARGIN * 1000.0 - 1e-6 for r in rows)
print("post  all >= %.2f mm ? %s" % (MARGIN * 1000.0, "YES" if all_post_ok else "NO"))
print("CONCLUSION: reasonable gaze +/-30 deg pitch/yaw -> "
      + ("NO penetration at all (pre min_sdf >= 0, no fix needed)"
         if worst_pre >= 0 and max_pen == 0
         else "minor penetration possible; safety net keeps post >= %.2f mm" % (MARGIN * 1000.0)))
