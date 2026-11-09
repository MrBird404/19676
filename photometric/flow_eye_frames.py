"""Rotate each frame's eyeball onto that frame's SAM3 pupil center.

The eye joint only rotates. Vertices are not translated. Head pose, identity,
expression and the static offset stay at the global-tracking result.
"""
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/root/autodl-tmp/GNM-main")
sys.path.insert(0, "/root/autodl-tmp/GNM_Fit")

from gnm.shape import gnm_pytorch
from nvdiffrast_render.camera import project_world_to_pixels
from fit_gnm import FitState
import photo_fit

PRIOR = Path("/root/autodl-tmp/gnm_fit_out/expression_v3/flow_frame0/track_global/track_params.npz")
BACKUP = PRIOR.with_name("track_params_before_sam_eyes.npz")
SAM = Path("/root/autodl-tmp/gnm_fit_out/expression_v3/flow_frame0/sam_eyes/centers.npz")
LMK = Path("/root/autodl-tmp/gnm_fit_out/expression_v3/facepoints68.npz")
UV_VERTS = (17303, 17689)
STEPS = 40


def main() -> None:
    z = np.load(PRIOR)
    base = np.load(BACKUP if BACKUP.is_file() else PRIOR)
    sam = np.load(SAM)
    centers = np.asarray(sam["centers"], dtype=np.float32)
    lmk = np.load(LMK, allow_pickle=True)
    width, height = [int(x) for x in lmk["image_size"]]
    ph, pw, scale = photo_fit.scale_for_photo(height, width, 1024)
    n = centers.shape[0]

    def tensor(v):
        return torch.tensor(np.asarray(v), dtype=torch.float32)

    gnm = gnm_pytorch.GNM.from_local(version=gnm_pytorch.GNMMajorVersion.V3, variant=gnm_pytorch.GNMVariant.HEAD)
    state = FitState(
        id_head=tensor(z["id_head"]), id_eyes=tensor(z["id_eyes"]),
        ex_lower=tensor(z["expression_lower_face"][0]),
        rot_neck=tensor(z["rot_neck"][0]), rot_head=tensor(z["rot_head"][0]),
        rot_leye=tensor(base["rot_leye"][0]), rot_reye=tensor(base["rot_reye"][0]),
        translation=tensor(z["translation"][0]),
        cam_aa=tensor(z["cam_aa"]), cam_t=tensor(z["cam_t"]),
        focal_norm=tensor(z["focal_norm"]),
    )
    static_off = tensor(z["offset"])
    ex_lower = np.asarray(z["expression_lower_face"], dtype=np.float32)
    ex_leye = np.asarray(z["expression_left_eyelid"], dtype=np.float32)
    ex_reye = np.asarray(z["expression_right_eyelid"], dtype=np.float32)
    rot_neck = np.asarray(z["rot_neck"], dtype=np.float32)
    rot_head = np.asarray(z["rot_head"], dtype=np.float32)
    translation = np.asarray(z["translation"], dtype=np.float32)
    leye_out = np.zeros((n, 3), np.float32)
    reye_out = np.zeros((n, 3), np.float32)
    err_out = np.zeros((n, 2), np.float32)
    uv = torch.tensor(UV_VERTS, dtype=torch.long)

    def expression():
        return torch.cat([
            torch.nn.functional.pad(tensor(ex_leye[t]), (0, 88)),
            torch.nn.functional.pad(tensor(ex_reye[t]), (0, 88)),
            state.ex_lower,
            torch.zeros(33),
        ])

    def posed():
        verts = gnm.vertex_positions_bind_pose(state.identity()[None], expression()[None])
        joints = gnm.joint_positions_bind_pose(state.identity()[None])
        verts = verts + gnm.compute_pose_correctives(state.rotations()[None])
        verts = verts + static_off[None]
        return gnm.vertex_positions_world(
            verts, joints, state.rotations()[None], state.translation[None],
        )[0]

    def project(pts):
        k = state.camera_k(width, height)
        if scale != 1.0:
            k = k * scale
        return project_world_to_pixels(pts, state.camera_rt(), k, pw, ph, flip_y=True)

    for t in range(n):
        state.ex_lower = tensor(ex_lower[t])
        state.rot_neck = tensor(rot_neck[t])
        state.rot_head = tensor(rot_head[t])
        state.translation = tensor(translation[t])
        anchor_l = state.rot_leye.detach().clone()
        anchor_r = state.rot_reye.detach().clone()
        state.rot_leye = anchor_l.clone().requires_grad_(True)
        state.rot_reye = anchor_r.clone().requires_grad_(True)
        opt = torch.optim.Adam([state.rot_leye, state.rot_reye], lr=2e-2)
        with torch.no_grad():
            pred0 = project(posed()[uv])
        order = torch.argsort(pred0[:, 0])
        # 图像上更靠左的黑圆圆心配给投影更靠左的那只眼。
        target = torch.zeros(2, 2)
        target[order] = torch.tensor(centers[t][np.argsort(centers[t, :, 0])])
        for _ in range(STEPS):
            opt.zero_grad()
            pred = project(posed()[uv])
            loss = (pred - target).pow(2).sum()
            # 绕视线的转角不改变这个点的投影，把它拴在上一帧，避免眼球翻面。
            loss = loss + 40.0 * ((state.rot_leye - anchor_l).pow(2).sum() + (state.rot_reye - anchor_r).pow(2).sum())
            loss.backward()
            opt.step()
        with torch.no_grad():
            pred = project(posed()[uv])
            err = torch.linalg.norm(pred - target, dim=-1)
        leye_out[t] = state.rot_leye.detach().numpy()
        reye_out[t] = state.rot_reye.detach().numpy()
        err_out[t] = err.numpy()
        state.rot_leye = state.rot_leye.detach()
        state.rot_reye = state.rot_reye.detach()
        if t % 20 == 0 or t + 1 == n:
            print(
                f"[eye {t:03d}] err {err_out[t, 0]:.2f}/{err_out[t, 1]:.2f}px "
                f"target {target.numpy().tolist()}",
                flush=True,
            )
    backup = PRIOR.with_name("track_params_before_sam_eyes.npz")
    if not backup.exists():
        backup.write_bytes(PRIOR.read_bytes())
    data = {k: z[k] for k in z.files}
    data["rot_leye"] = leye_out
    data["rot_reye"] = reye_out
    data["sam_eye_centers"] = centers
    data["sam_eye_err"] = err_out
    np.savez(PRIOR, **data)
    print(
        f"[eye] mean err {err_out.mean():.3f}px max {err_out.max():.3f}px wrote {PRIOR}",
        flush=True,
    )


if __name__ == "__main__":
    main()
