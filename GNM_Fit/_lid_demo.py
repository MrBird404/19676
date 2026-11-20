import numpy as np, torch, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import photo_fit as pf
import lid_clearance as LC
from gnm.shape import gnm_pytorch, gnm_numpy
from fit_gnm import FitState
from nvdiffrast_render import NVDiffRenderer

EV3 = "/root/autodl-tmp/gnm_fit_out/expression_v3"
RUN = EV3 + "/pipeline_run_reorder"
OUT = "/root/autodl-tmp/notes/gnm_uv/测试用图片"
MARGIN = 1e-4   # 0.1 mm safe distance
GATE = 2e-3     # 2 mm NN gate
dev = "cuda" if torch.cuda.is_available() else "cpu"

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
st.pupil = torch.tensor([float(np.load(c08)["pupil"][0])]) if os.path.exists(c08) and "pupil" in np.load(c08).files else torch.zeros(1)

g = gnm_pytorch.GNM.from_local(version=gnm_pytorch.GNMMajorVersion.V3, variant=gnm_pytorch.GNMVariant.HEAD)
gnp = gnm_numpy.GNM.from_local(version=gnm_numpy.GNMMajorVersion.V3, variant=gnm_numpy.GNMVariant.HEAD)
faces = np.asarray(g.triangles.detach().cpu().numpy())
with torch.no_grad():
    V0 = g(st.identity()[None], st.expression()[None], st.rotations()[None], st.translation[None])[0].cpu().numpy()
gid = lambda n: pf._group_ids(gnp, n)

el = LC.eligibility(V0, faces, gid, GATE)
print("rest:", LC._stats(V0, faces, gid, el, MARGIN, GATE), "eligible per side:", {k: len(v) for k, v in el.items()})

worst_pre = 1e9; worst_post = 1e9; worst_cfg = None; npre = 0
for pitch in (-40, -25, -10, 0, 10, 25, 40):
    for yaw in (-40, -25, -10, 0, 10, 25, 40):
        Vg = LC.rotate_eyeball(V0, gid, {"L": (pitch, yaw), "R": (pitch, yaw)})
        pre = LC._stats(Vg, faces, gid, el, MARGIN, GATE)
        Vf, stt = LC.enforce_lid_clearance(Vg, V0, faces, gid, MARGIN, GATE)
        post = stt["post"]
        if pre["min_sdf_mm"] < worst_pre:
            worst_pre = pre["min_sdf_mm"]; worst_cfg = (pitch, yaw)
        npre = max(npre, pre["n_below_margin"])
        worst_post = min(worst_post, post["min_sdf_mm"])
print("SWEEP +-25deg (25 gazes): worst pre min_sdf=%.3f mm (max n_pen=%d) @ %s ; worst post min_sdf=%.3f mm (margin=%.3f)"
      % (worst_pre, npre, worst_cfg, worst_post, MARGIN * 1000))

W, H = [int(x) for x in np.load(EV3 + "/facepoints68.npz")["image_size"]]
rt = st.camera_rt().to(dev)[None]; K = st.camera_k(W, H).to(dev)[None]
ren = NVDiffRenderer(use_opengl=False, lighting_type="front", lighting_space="camera", flip_y_raster=False).to(dev)
faces_t = torch.as_tensor(faces, dtype=torch.long, device=dev)

def vcol():
    c = np.tile(np.array([0.72, 0.58, 0.48], np.float32), (V0.shape[0], 1))
    c[LC.groups(gid)[0]] = (0.85, 0.85, 0.83)
    c[np.asarray(gid("irises"), np.int64)] = (0.28, 0.33, 0.42)
    c[np.asarray(gid("pupils"), np.int64)] = (0.04, 0.04, 0.04)
    return c
VC = vcol()

def render(V):
    vc = torch.as_tensor(VC, dtype=torch.float32, device=dev)[None]
    with torch.no_grad():
        out = ren.render_rgba_vis(torch.as_tensor(V, dtype=torch.float32, device=dev)[None],
                                  faces_t, rt, K, (H, W), background_color=[1, 1, 1], v_color=vc)
    return out["rgba"][0, ..., :3].detach().cpu().numpy()

# strong gaze for the visual
pitch0, yaw0 = 25, 0
Vg = LC.rotate_eyeball(V0, gid, {"L": (pitch0, yaw0), "R": (pitch0, yaw0)})
Vf, stt = LC.enforce_lid_clearance(Vg, V0, faces, gid, MARGIN, GATE)
print("demo gaze", (pitch0, yaw0), "pre", stt["pre"], "post", stt["post"])

import cv2
def crop(im):
    x0, x1, y0, y1 = 440, 620, 215, 320
    return (np.clip(im, 0, 1) * 255).astype(np.uint8)[y0:y1, x0:x1]
r0 = crop(render(V0)); rg = crop(render(Vg)); rf = crop(render(Vf))
w1 = r0.shape[1]
row = np.hstack([r0, rg, rf])
for txt, x in [("base", 6), ("gaze up (no fix)", w1 + 6), ("lid safe-dist", 2 * w1 + 6)]:
    cv2.putText(row, txt, (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
row = cv2.resize(row, (row.shape[1] * 3, row.shape[0] * 3), interpolation=cv2.INTER_NEAREST)
cv2.imwrite(OUT + "/眼皮_眼球_安全距离_前后.png", row[:, :, ::-1])
print("saved", OUT + "/眼皮_眼球_安全距离_前后.png")
