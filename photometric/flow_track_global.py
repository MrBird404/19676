"""Global tracking, VHAP rgb_global_tracking.

Starts from track_eyelid_gaze. Thirty epochs, learning rates at 0.1 of the
VHAP rates. Shared across frames: shape, camera, lights, texture residual and
the static offset. Per frame: pose, eye joints, lower face and the leading
eyelid components. Dynamic offset stays off. The eyeball is not translated.
Outer-lid collision stays on. Added offset is regularized from the adopted
neck tuck, and eye vertices stay pinned.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(_ROOT / "MediaPipe_备份"))
sys.path.insert(0, "/root/autodl-tmp/MediaPipe_V3")
sys.path.insert(0, "/root/autodl-tmp/GNM-main")

from gnm.shape import gnm_numpy, gnm_pytorch
import gnm_mediapipe_landmarks as mp_lmk
from nvdiffrast_render import NVDiffRenderer
from nvdiffrast_render.camera import project_world_to_pixels
from fit_gnm import FitState, ibug68_vhap_weights
import photo_fit
from flow_photo import group, load_base, raster_mask
from uvmask import tensor_mask

OUT = Path("/root/autodl-tmp/gnm_fit_out/expression_v3/flow_frame0/track_global")
PRIOR = Path("/root/autodl-tmp/gnm_fit_out/expression_v3/flow_frame0/track_eyelid_gaze/track_params.npz")
_UV = Path(__file__).resolve().parent / "uv"
PAINTED = _UV / "gnm_uv_from_flame_lmk68.png"
EYE_TEX = _UV / "gnm_eye_uv.png"
EYE_UV = _UV / "gnm_eye_uv_repacked.npz"
APPEAR = Path("/root/autodl-tmp/gnm_fit_out/expression_v3/flow_frame0/photo_contour_only/photo_appearance.npz")
LMK = Path("/root/autodl-tmp/gnm_fit_out/expression_v3/facepoints68.npz")
LMK_FILE = _ROOT / "MediaPipe_备份" / "head_sparse_68_v3_改.txt"
EPOCHS = 30
N_EYE = 12


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    saved = np.load(APPEAR)
    prior = np.load(PRIOR)
    lmk_npz = np.load(LMK, allow_pickle=True)
    gt_all = np.asarray(lmk_npz["landmarks"], dtype=np.float32)
    paths = [Path(str(p)) for p in lmk_npz["image_paths"]]
    n_frames = len(paths)
    width, height = [int(x) for x in lmk_npz["image_size"]]
    ph, pw, scale = photo_fit.scale_for_photo(height, width, 1024)

    base_face = load_base(PAINTED)
    base_eye = load_base(EYE_TEX)
    eye_pack = np.load(EYE_UV)
    eye_ids = np.asarray(eye_pack["eye_faces"], dtype=np.int64)
    eye_tri_uv = np.asarray(eye_pack["eye_triangle_uvs"], dtype=np.float32)
    gnm = gnm_pytorch.GNM.from_local(version=gnm_pytorch.GNMMajorVersion.V3, variant=gnm_pytorch.GNMVariant.HEAD)
    gnm_np = gnm_numpy.GNM.from_local(version=gnm_numpy.GNMMajorVersion.V3, variant=gnm_numpy.GNMVariant.HEAD)
    faces = np.asarray(gnm.triangles.detach().cpu().numpy())
    n_verts = int(faces.max()) + 1
    tri_uv = np.asarray(gnm_np.triangle_uvs, dtype=np.float32)
    regions = photo_fit.build_photo_regions(gnm_np, faces)

    def leaf(v):
        return torch.nn.Parameter(torch.tensor(np.asarray(v), dtype=torch.float32))

    def sequence(array):
        return leaf(array)

    ex_lower = sequence(prior["expression_lower_face"])
    ex_leye = sequence(prior["expression_left_eyelid"])
    ex_reye = sequence(prior["expression_right_eyelid"])
    rot_neck = sequence(prior["rot_neck"])
    rot_head = sequence(prior["rot_head"])
    rot_leye = sequence(prior["rot_leye"])
    rot_reye = sequence(prior["rot_reye"])
    translation = sequence(prior["translation"])
    state = FitState(
        id_head=leaf(prior["id_head"]), id_eyes=leaf(prior["id_eyes"]),
        ex_lower=ex_lower[0],
        rot_neck=rot_neck[0], rot_head=rot_head[0],
        rot_leye=rot_leye[0], rot_reye=rot_reye[0],
        translation=translation[0],
        cam_aa=leaf(prior["cam_aa"]), cam_t=leaf(prior["cam_t"]),
        focal_norm=leaf(prior["focal_norm"]),
    )
    static_off = leaf(prior["offset"])
    static_off.requires_grad_(False)
    offset0 = static_off.detach().clone()
    res = leaf(prior["residual"])
    lights = leaf(prior["lights"])
    eye_col = torch.tensor(np.asarray(prior["eye_col"]), dtype=torch.float32)
    named = {
        "ex_lower": ex_lower, "ex_leye": ex_leye, "ex_reye": ex_reye,
        "rot_neck": rot_neck, "rot_head": rot_head,
        "rot_leye": rot_leye, "rot_reye": rot_reye, "translation": translation,
        "id_head": state.id_head, "id_eyes": state.id_eyes,
        "cam_aa": state.cam_aa, "cam_t": state.cam_t, "focal_norm": state.focal_norm,
        "static_off": static_off, "res": res, "lights": lights,
    }
    cur = {"leye": ex_leye[0], "reye": ex_reye[0], "t": 0}

    h, w = base_face.shape[:2]
    eh, ew = base_eye.shape[:2]
    eye_masks = [torch.tensor(raster_mask(
        faces[eye_ids], eye_tri_uv, group(gnm_np, name), n_verts, (eh, ew),
    )) for name in ("scleras", "irises", "pupils")]
    lip_ids = np.unique(np.concatenate([
        group(gnm_np, "upper_lip"), group(gnm_np, "lower_lip"),
    ]))
    brow = np.unique(np.concatenate([group(gnm_np, n) for n in (
        "left_brow_region", "right_brow_region", "middle_brow_region")]))
    keep_verts = np.unique(np.concatenate([group(gnm_np, "hockey_mask"), brow, lip_ids]))
    face_keep_t = torch.tensor(tensor_mask(faces, tri_uv, keep_verts, n_verts, (h, w)))

    device = torch.device("cuda")
    color_lut = torch.zeros(faces.shape[0] + 1, dtype=torch.bool, device=device)
    color_lut[torch.as_tensor(regions["color_except"], dtype=torch.long, device=device) + 1] = True
    face_ids = np.setdiff1d(np.arange(len(faces), dtype=np.int64), eye_ids)
    full_cid = np.asarray(regions["fid2cid"], dtype=np.int64)
    bound_t = torch.as_tensor(regions["bound_except"], dtype=torch.long, device=device)
    lmk_idx, lmk_bary = mp_lmk.load_landmarks(LMK_FILE)
    idx_t = torch.as_tensor(lmk_idx, dtype=torch.long)
    bary_t = torch.as_tensor(lmk_bary, dtype=torch.float32)
    w_lmk = ibug68_vhap_weights(nose_boost=10.0)
    w_lmk[:17] = 1.0
    w_lmk[36:48] = 8.0
    w_lmk_t = torch.as_tensor(w_lmk, dtype=torch.float32)
    with torch.no_grad():
        rest_v = gnm.vertex_positions_bind_pose(
            state.identity()[None], torch.zeros(1, 383),
        )[0]
        joints0 = gnm.joint_positions_bind_pose(state.identity()[None])[0]
    rest_np = rest_v.numpy()

    def lid_shell(center, ball_ids, lid_ids):
        c = center.numpy()
        ball = rest_np[ball_ids] - c
        lid = rest_np[lid_ids] - c
        dist = np.linalg.norm(lid, axis=1)
        direc = lid / np.clip(dist[:, None], 1e-8, None)
        support = (ball @ direc.T).max(axis=0)
        keep = dist > support + 2.0e-4
        return lid_ids[keep], direc[keep], support[keep], c

    shells = [
        lid_shell(joints0[2], group(gnm_np, "left_eye"), group(gnm_np, "expression_basis_left_eye")),
        lid_shell(joints0[3], group(gnm_np, "right_eye"), group(gnm_np, "expression_basis_right_eye")),
    ]
    lid_idx = torch.as_tensor(np.concatenate([s[0] for s in shells]), dtype=torch.long)
    lid_dir = torch.tensor(np.concatenate([s[1] for s in shells]), dtype=torch.float32)
    lid_sup = torch.tensor(np.concatenate([s[2] for s in shells]), dtype=torch.float32)
    lid_center = torch.tensor(
        np.concatenate([np.repeat(s[3][None], len(s[0]), 0) for s in shells]),
        dtype=torch.float32,
    )
    print(f"[lid] outer verts {len(lid_idx)}", flush=True)
    pen_mm = {"v": 0.0}
    gaze_px = {"v": 0.0}

    renderer = NVDiffRenderer(
        use_opengl=False, lighting_type="SH", lighting_space="world",
        disturb_rate_fg=0.0, disturb_rate_bg=0.0,
        fid2cid=torch.as_tensor(full_cid, dtype=torch.long),
        shade_smooth=True, flip_y_raster=False,
    ).to(device)

    def set_cid(part_ids):
        cid = torch.as_tensor(full_cid[part_ids], dtype=torch.long, device=device)
        renderer.fid2cid = F.pad(cid, (1, 0), value=0)

    def albedo_face():
        up = F.interpolate(res, size=(h, w), mode="bilinear", align_corners=True)[0].permute(1, 2, 0)
        return (base_face + up * face_keep_t[..., None]).clamp(1e-4, 1.0 - 1e-4)

    def albedo_eye():
        add = torch.zeros_like(base_eye)
        for msk, col in zip(eye_masks, eye_col):
            add = add + msk[..., None] * col
        return (base_eye + add).clamp(1e-4, 1.0 - 1e-4)

    eye_tex = albedo_eye()
    tri = faces.astype(np.int64)
    edge = np.unique(np.sort(np.concatenate([
        tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]],
    ], 0), axis=1), axis=0)
    edge0 = torch.tensor(edge[:, 0], dtype=torch.long)
    edge1 = torch.tensor(edge[:, 1], dtype=torch.long)
    eye_pin = torch.tensor(np.unique(np.concatenate([
        group(gnm_np, "left_eye"), group(gnm_np, "right_eye"),
    ])), dtype=torch.long)

    def expression_vec():
        return torch.cat([
            F.pad(cur["leye"], (0, 100 - N_EYE)),
            F.pad(cur["reye"], (0, 100 - N_EYE)),
            state.ex_lower,
            torch.zeros(33),
        ])

    def bind_verts():
        verts = gnm.vertex_positions_bind_pose(state.identity()[None], expression_vec()[None])
        verts = verts + gnm.compute_pose_correctives(state.rotations()[None])
        return (verts + static_off[None])[0]

    def posed():
        joints = gnm.joint_positions_bind_pose(state.identity()[None])
        return gnm.vertex_positions_world(
            bind_verts()[None], joints, state.rotations()[None], state.translation[None],
        )[0]

    def lid_clash(bind):
        radial = ((bind[lid_idx] - lid_center) * lid_dir).sum(-1)
        pen = F.relu(lid_sup + 3.0e-4 - radial)
        pen_mm["v"] = float(pen.detach().max()) * 1e3
        return pen.pow(2).sum()

    def project(pts):
        k = state.camera_k(width, height)
        if scale != 1.0:
            k = k * scale
        return project_world_to_pixels(pts, state.camera_rt(), k, pw, ph, flip_y=True)

    def raster(part_ids, part_uv, tex):
        set_cid(part_ids)
        part_faces = torch.as_tensor(faces[part_ids], dtype=torch.int32, device=device)
        verts_uv_t = torch.as_tensor(part_uv.reshape(-1, 2), dtype=torch.float32, device=device)
        faces_uv_t = torch.arange(len(part_ids) * 3, device=device, dtype=torch.int32).reshape(-1, 3)
        verts = posed().to(device)[None].contiguous()
        rt = state.camera_rt().to(device)[None]
        k4 = state.camera_k(width, height)
        if scale != 1.0:
            k4 = k4 * scale
        rast = renderer.rasterize(
            verts, part_faces, rt, k4.to(device)[None], (ph, pw),
            use_cache=False, require_grad=True,
        )
        out = renderer.render_rgba(
            rast, verts[0], part_faces, verts_uv_t, faces_uv_t,
            tex.permute(2, 0, 1)[None].to(device), lights.to(device),
            background_color=[1.0, 1.0, 1.0], enable_disturbance=False,
            align_boundary_except_vid=bound_t,
        )
        depth = renderer._to_image_space(rast["rast_out"][..., 2:3])
        fid = renderer._to_image_space(rast["rast_out"][..., 3:4].long())
        return out["rgba"], depth, fid

    def render():
        eye_out, eye_z, _ = raster(eye_ids, eye_tri_uv, eye_tex)
        face_out, face_z, face_fid = raster(face_ids, tri_uv[face_ids], albedo_face())
        eye_on = eye_out[..., 3:] > 0.5
        face_on = face_out[..., 3:] > 0.5
        face_front = face_on & (~eye_on | (face_z <= eye_z))
        rgb = torch.where(
            face_front, face_out[..., :3],
            torch.where(eye_on, eye_out[..., :3], torch.ones_like(face_out[..., :3])),
        )
        alpha = (face_front | eye_on).float()
        ids = torch.as_tensor(face_ids, dtype=torch.long, device=device)
        flat = face_fid.reshape(-1).clamp(min=0)
        glob = torch.zeros_like(flat)
        hit = flat > 0
        glob[hit] = ids[flat[hit] - 1] + 1
        return torch.cat([rgb, alpha], -1), glob.view_as(face_fid)

    def load_rgb(path: Path) -> torch.Tensor:
        bgr = cv2.imread(str(path))
        if scale != 1.0:
            bgr = cv2.resize(bgr, (pw, ph), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return torch.as_tensor(rgb, dtype=torch.float32, device=device)[None]

    def visible_pupil(rgba, targets):
        luma = rgba[0, ..., :3].mean(-1)
        height_i, width_i = luma.shape
        gy, gx = torch.meshgrid(
            torch.arange(height_i, device=luma.device),
            torch.arange(width_i, device=luma.device),
            indexing="ij",
        )
        parts = []
        targets = targets.to(luma.device)
        for xy in targets:
            win = (gx - xy[0]).pow(2) + (gy - xy[1]).pow(2) < 18.0 ** 2
            vals = luma[win]
            if int(vals.numel()) < 20:
                continue
            thr = torch.quantile(vals.detach(), 0.12)
            weight = torch.relu(thr - luma) * win.float()
            mass = weight.sum()
            if float(mass.detach()) < 1.0:
                continue
            center = torch.stack([
                (weight * gx).sum() / mass,
                (weight * gy).sum() / mass,
            ])
            parts.append((center - xy).abs().mean())
        if not parts:
            return luma.sum() * 0.0
        return torch.stack(parts).mean()

    def pair(cur_v, seq, t):
        loss = cur_v.sum() * 0.0
        if t > 0:
            loss = loss + (cur_v - seq[t - 1].detach()).pow(2).mean()
        if t + 1 < n_frames:
            loss = loss + (cur_v - seq[t + 1].detach()).pow(2).mean()
        return loss

    def energy(gt_rgb, gt_lmk, pupil):
        t = cur["t"]
        rgba, global_fid = render()
        covered = (rgba[..., 3:] > 0) & ~color_lut[global_fid.clamp(0, color_lut.shape[0] - 1)]
        blue_cap = (gt_rgb[..., 2:] > gt_rgb[..., 0:1] + 0.08) & (gt_rgb[..., 2:] > 0.25)
        dark_cloth = gt_rgb.mean(-1, keepdim=True) < 0.22
        vis = (covered & ~blue_cap & ~dark_cloth).detach()
        pred = rgba[..., :3]
        rgb_loss = ((pred - gt_rgb).abs() * vis).sum() / (vis.sum() * 3).clamp(min=1.0)
        pred_xy = project((posed()[idx_t] * bary_t.unsqueeze(-1)).sum(1))
        lmk = torch.linalg.norm(pred_xy - gt_lmk, dim=-1)
        lmk_w = (lmk * w_lmk_t).sum() / w_lmk_t.sum()
        open_l = pred_xy[[40, 41], 1].mean() - pred_xy[[37, 38], 1].mean()
        open_r = pred_xy[[46, 47], 1].mean() - pred_xy[[43, 44], 1].mean()
        gt_l = gt_lmk[[40, 41], 1].mean() - gt_lmk[[37, 38], 1].mean()
        gt_r = gt_lmk[[46, 47], 1].mean() - gt_lmk[[43, 44], 1].mean()
        open_err = (torch.stack([open_l, open_r]) - torch.stack([gt_l, gt_r])).pow(2).mean()
        dev = rgb_loss.device
        smooth = (
            3.0e2 * pair(state.translation, translation, t)
            + 3.0e1 * pair(state.rot_neck, rot_neck, t)
            + 3.0e1 * pair(state.rot_head, rot_head, t)
            + pair(state.ex_lower, ex_lower, t)
            + 0.05 * pair(cur["leye"], ex_leye, t)
            + 0.05 * pair(cur["reye"], ex_reye, t)
            + pair(state.rot_leye, rot_leye, t)
            + pair(state.rot_reye, rot_reye, t)
        )
        clash = lid_clash(bind_verts())
        gaze = visible_pupil(rgba, pupil)
        gaze_px["v"] = float(gaze.detach())
        delta = static_off - offset0
        reg = (
            0.3 * state.id_head.pow(2).mean()
            + 0.3 * state.id_eyes.pow(2).mean()
            + 0.03 * state.ex_lower.pow(2).mean()
            + 0.03 * cur["leye"].pow(2).mean()
            + 0.03 * cur["reye"].pow(2).mean()
            + 3.0e2 * delta.pow(2).mean()
            + 1.0e6 * (delta[edge0] - delta[edge1]).pow(2).mean()
            + 1.0e4 * (res[:, :, :-1, :] - res[:, :, 1:, :]).pow(2).mean()
            + 1.0e4 * (res[:, :, :, :-1] - res[:, :, :, 1:]).pow(2).mean()
        )
        loss = (
            30.0 * rgb_loss
            + 10.0 * lmk_w.to(dev)
            + 4.0 * open_err.to(dev)
            + 40.0 * gaze.to(dev)
            + 2.0e7 * clash.to(dev)
            + smooth.to(dev)
            + reg.to(dev)
        )
        return loss, rgb_loss, lmk, rgba

    def apply_frame(t):
        state.ex_lower = ex_lower[t]
        state.rot_neck = rot_neck[t]
        state.rot_head = rot_head[t]
        state.rot_leye = rot_leye[t]
        state.rot_reye = rot_reye[t]
        state.translation = translation[t]
        cur["leye"] = ex_leye[t]
        cur["reye"] = ex_reye[t]
        cur["t"] = t

    def pupil_circle(bgr, pts):
        mask = np.zeros(bgr.shape[:2], np.uint8)
        hull = cv2.convexHull(np.round(pts).astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 1)
        luma = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
        ys, xs = np.nonzero(mask)
        vals = luma[ys, xs]
        sel = vals <= np.percentile(vals, 15)
        dark = np.stack([xs[sel], ys[sel]], 1).astype(np.float32)
        (cx, cy), _ = cv2.minEnclosingCircle(dark)
        return np.array([cx, cy], np.float32)

    pupils = np.zeros((n_frames, 2, 2), np.float32)
    pupils[0] = np.array([[588.36664, 264.5], [467.51962, 265.32352]], np.float32)
    for t in range(1, n_frames):
        bgr = cv2.imread(str(paths[t]))
        pupils[t, 0] = pupil_circle(bgr, gt_all[t, 42:48])
        pupils[t, 1] = pupil_circle(bgr, gt_all[t, 36:42])

    opt = torch.optim.Adam([
        {"params": [ex_lower, ex_leye, ex_reye], "lr": 5.0e-3},
        {"params": [
            rot_neck, rot_head, rot_leye, rot_reye, state.id_head, state.id_eyes,
        ], "lr": 5.0e-4},
        {"params": [translation], "lr": 1.0e-4},
        {"params": [state.cam_aa, state.cam_t, state.focal_norm, lights, res], "lr": 5.0e-4},
    ])
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9)
    order = np.arange(n_frames)
    init_score = (
        30.0 * float(np.asarray(prior["rgb"]).mean())
        + 10.0 * float(np.asarray(prior["lmk"]).mean())
    )
    best_score = init_score
    best = {k: v.detach().clone() for k, v in named.items()}
    print(f"[global] frames {n_frames} epochs {EPOCHS} batch 16 vhAP lr x0.1", flush=True)

    for epoch in range(EPOCHS):
        opening = {k: v.detach().clone() for k, v in named.items()}
        rng = np.random.default_rng(epoch)
        perm = rng.permutation(order)
        rgb_sum = lmk_sum = gaze_sum = 0.0
        worst = (-1.0, -1)
        for start in range(0, n_frames, 16):
            batch = [int(t) for t in perm[start:start + 16]]
            opt.zero_grad()
            for t in batch:
                apply_frame(t)
                gt_rgb = load_rgb(paths[t])
                gt_lmk = torch.as_tensor(gt_all[t], dtype=torch.float32)
                pupil = torch.tensor(pupils[t])
                loss, rgb_loss, lmk, _ = energy(gt_rgb, gt_lmk, pupil)
                (loss / len(batch)).backward()
                rgb_v = float(rgb_loss.detach())
                lmk_v = float(lmk.detach().mean())
                rgb_sum += rgb_v
                lmk_sum += lmk_v
                gaze_sum += gaze_px["v"]
                if lmk_v > worst[0]:
                    worst = (lmk_v, t)
            torch.nn.utils.clip_grad_norm_([
                ex_lower, ex_leye, ex_reye, rot_neck, rot_head, translation,
                state.id_head, state.id_eyes, state.cam_aa, state.cam_t,
                state.focal_norm, lights, res,
            ], 0.5)
            torch.nn.utils.clip_grad_norm_([rot_leye, rot_reye], 1.0)
            opt.step()
            with torch.no_grad():
                static_off[eye_pin] = offset0[eye_pin]
        sched.step()
        n_fit = float(len(order))
        score = 30.0 * rgb_sum / n_fit + 10.0 * lmk_sum / n_fit
        if score < best_score:
            best_score = score
            best = opening
            mark = " *"
        else:
            mark = ""
        print(
            f"[epoch {epoch + 1:02d}/{EPOCHS}] rgb={rgb_sum / n_fit:.4f} "
            f"lmk={lmk_sum / n_fit:.2f} gaze={gaze_sum / n_fit:.2f}px "
            f"worst={worst[0]:.2f}@{worst[1]}{mark}",
            flush=True,
        )

    with torch.no_grad():
        for k, v in best.items():
            named[k].copy_(v)
    expr = np.zeros((n_frames, 150), np.float32)
    elid = np.zeros((n_frames, N_EYE), np.float32)
    erid = np.zeros((n_frames, N_EYE), np.float32)
    neck = np.zeros((n_frames, 3), np.float32)
    head = np.zeros((n_frames, 3), np.float32)
    leye = np.zeros((n_frames, 3), np.float32)
    reye = np.zeros((n_frames, 3), np.float32)
    trans = np.zeros((n_frames, 3), np.float32)
    rgb_log = np.zeros(n_frames, np.float32)
    lmk_log = np.zeros(n_frames, np.float32)
    nose_log = np.zeros(n_frames, np.float32)
    for t in range(n_frames):
        apply_frame(t)
        gt_rgb = load_rgb(paths[t])
        gt_lmk = torch.as_tensor(gt_all[t], dtype=torch.float32)
        with torch.no_grad():
            _, rgb_loss, lmk, rgba = energy(gt_rgb, gt_lmk, torch.tensor(pupils[t]))
        expr[t] = ex_lower[t].detach().numpy()
        elid[t] = ex_leye[t].detach().numpy()
        erid[t] = ex_reye[t].detach().numpy()
        neck[t] = rot_neck[t].detach().numpy()
        head[t] = rot_head[t].detach().numpy()
        leye[t] = rot_leye[t].detach().numpy()
        reye[t] = rot_reye[t].detach().numpy()
        trans[t] = translation[t].detach().numpy()
        rgb_log[t] = float(rgb_loss)
        lmk_log[t] = float(lmk.mean())
        nose_log[t] = float(lmk[27:36].mean())
        bgr = cv2.cvtColor(
            (rgba[0, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8),
            cv2.COLOR_RGB2BGR,
        )
        cv2.imwrite(str(OUT / f"frame_{t:04d}.jpg"), bgr)
        if t % 40 == 0 or t + 1 == n_frames:
            photo_fit.dump_photo_preview(
                OUT, f"f{t:04d}",
                gt_rgb[0].detach().float().cpu().numpy(),
                rgba[0].detach().float().cpu().numpy(),
            )
        print(
            f"[write {t:03d}/{n_frames}] rgb={rgb_log[t]:.4f} lmk={lmk_log[t]:.2f} "
            f"nose={nose_log[t]:.2f} pen={pen_mm['v']:.2f}mm gaze={gaze_px['v']:.2f}px",
            flush=True,
        )

    np.savez(
        OUT / "track_params.npz",
        expression_lower_face=expr,
        expression_left_eyelid=elid,
        expression_right_eyelid=erid,
        rot_neck=neck,
        rot_head=head,
        rot_leye=leye,
        rot_reye=reye,
        translation=trans,
        rgb=rgb_log,
        lmk=lmk_log,
        nose=nose_log,
        id_head=state.id_head.detach().numpy(),
        id_eyes=state.id_eyes.detach().numpy(),
        cam_aa=state.cam_aa.detach().numpy(),
        cam_t=state.cam_t.detach().numpy(),
        focal_norm=state.focal_norm.detach().numpy(),
        offset=static_off.detach().numpy(),
        residual=res.detach().numpy(),
        lights=lights.detach().numpy(),
        eye_col=eye_col.detach().numpy(),
        image_paths=np.array([str(p) for p in paths]),
    )
    base_lmk = np.asarray(prior["lmk"], dtype=np.float32)
    print(
        f"[global] rgb {rgb_log.mean():.4f} lmk {lmk_log.mean():.2f} "
        f"max_lmk {lmk_log.max():.2f} at {int(lmk_log.argmax())} "
        f"was {base_lmk.mean():.2f} max {base_lmk.max():.2f}",
        flush=True,
    )
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
