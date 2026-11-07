"""Sequential tracking after the adopted frame-0 fit.

Shape, camera, lights, texture and the static offset stay at photo_contour_only.
Each later frame starts from the previous one and fits pose, expression and eye
rotation. Dynamic offset stays off: VHAP leaves it off unless asked, and a free
per-frame offset wrinkles this mesh. Eye vertices are not translated.
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

OUT = Path("/root/autodl-tmp/gnm_fit_out/expression_v3/flow_frame0/track")
_UV = Path(__file__).resolve().parent / "uv"
PAINTED = _UV / "gnm_uv_from_flame_lmk68.png"
EYE_TEX = _UV / "gnm_eye_uv.png"
EYE_UV = _UV / "gnm_eye_uv_repacked.npz"
APPEAR = Path("/root/autodl-tmp/gnm_fit_out/expression_v3/flow_frame0/photo_contour_only/photo_appearance.npz")
LMK = Path("/root/autodl-tmp/gnm_fit_out/expression_v3/facepoints68.npz")
LMK_FILE = _ROOT / "MediaPipe_备份" / "head_sparse_68_v3_改.txt"
STEPS = 50


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    saved = np.load(APPEAR)
    lmk_npz = np.load(LMK, allow_pickle=True)
    gt_all = np.asarray(lmk_npz["landmarks"], dtype=np.float32)
    paths = [Path(str(p)) for p in lmk_npz["image_paths"]]
    n_frames = len(paths)
    width, height = [int(x) for x in lmk_npz["image_size"]]
    stage0 = cv2.imread(str(paths[0]))
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

    def p(v, grad=False):
        t = torch.tensor(np.asarray(v), dtype=torch.float32)
        t.requires_grad_(grad)
        return t

    rot0 = np.asarray(saved["rotations"], dtype=np.float32).copy()
    state = FitState(
        id_head=p(saved["id_head"]), id_eyes=p(saved["id_eyes"]),
        ex_lower=p(saved["expression_lower_face"], True),
        rot_neck=p(rot0[0], True), rot_head=p(rot0[1], True),
        rot_leye=p(rot0[2], True), rot_reye=p(rot0[3], True),
        translation=p(saved["translation"], True),
        cam_aa=p(saved["cam_aa"]), cam_t=p(saved["cam_t"]),
        focal_norm=p(saved["focal_norm"]),
    )
    static_off = torch.tensor(np.asarray(saved["offset"]), dtype=torch.float32)
    res = torch.tensor(np.asarray(saved["residual"]), dtype=torch.float32)
    lights = torch.tensor(saved["lights"], dtype=torch.float32)
    eye_col = torch.tensor(saved["eye_col"], dtype=torch.float32)

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
    lmk_idx, _lmk_bary = mp_lmk.load_landmarks(LMK_FILE)
    idx_t = torch.as_tensor(lmk_idx, dtype=torch.long)
    bary_t = torch.as_tensor(_lmk_bary, dtype=torch.float32)
    w_lmk = ibug68_vhap_weights(nose_boost=10.0)
    w_lmk[:17] = 1.0
    w_lmk_t = torch.as_tensor(w_lmk, dtype=torch.float32)

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

    face_tex = albedo_face()
    eye_tex = albedo_eye()

    def posed():
        verts = gnm.vertex_positions_bind_pose(state.identity()[None], state.expression()[None])
        joints = gnm.joint_positions_bind_pose(state.identity()[None])
        rotations = state.rotations()[None]
        verts = verts + gnm.compute_pose_correctives(rotations)
        verts = verts + static_off[None]
        return gnm.vertex_positions_world(verts, joints, rotations, state.translation[None])[0]

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
        face_out, face_z, face_fid = raster(face_ids, tri_uv[face_ids], face_tex)
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

    def energy(gt_rgb, gt_lmk, prev):
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
        dev = rgb_loss.device
        smooth = (
            3.0e2 * (state.translation - prev["translation"]).pow(2).mean()
            + 3.0e1 * (state.rot_neck - prev["rot_neck"]).pow(2).mean()
            + 3.0e1 * (state.rot_head - prev["rot_head"]).pow(2).mean()
            + (state.ex_lower - prev["ex_lower"]).pow(2).mean()
        )
        loss = 30.0 * rgb_loss + 10.0 * lmk_w.to(dev) + smooth.to(dev)
        return loss, rgb_loss, lmk, rgba

    def snapshot():
        return {
            "ex_lower": state.ex_lower.detach().clone(),
            "rot_neck": state.rot_neck.detach().clone(),
            "rot_head": state.rot_head.detach().clone(),
            "rot_leye": state.rot_leye.detach().clone(),
            "rot_reye": state.rot_reye.detach().clone(),
            "translation": state.translation.detach().clone(),
        }

    def restore(s):
        with torch.no_grad():
            state.ex_lower.copy_(s["ex_lower"])
            state.rot_neck.copy_(s["rot_neck"])
            state.rot_head.copy_(s["rot_head"])
            state.rot_leye.copy_(s["rot_leye"])
            state.rot_reye.copy_(s["rot_reye"])
            state.translation.copy_(s["translation"])

    expr = np.zeros((n_frames, 150), np.float32)
    neck = np.zeros((n_frames, 3), np.float32)
    head = np.zeros((n_frames, 3), np.float32)
    leye = np.zeros((n_frames, 3), np.float32)
    reye = np.zeros((n_frames, 3), np.float32)
    trans = np.zeros((n_frames, 3), np.float32)
    rgb_log = np.zeros(n_frames, np.float32)
    lmk_log = np.zeros(n_frames, np.float32)
    nose_log = np.zeros(n_frames, np.float32)

    opt = torch.optim.Adam([
        {"params": [state.ex_lower], "lr": 1e-2},
        {"params": [state.rot_neck, state.rot_head], "lr": 5e-3},
        {"params": [state.rot_leye, state.rot_reye], "lr": 5e-3},
        {"params": [state.translation], "lr": 1e-3},
    ])
    for g in opt.param_groups:
        g["base_lr"] = g["lr"]

    print(f"[track] frames {n_frames} steps {STEPS}", flush=True)
    for t in range(n_frames):
        gt_rgb = load_rgb(paths[t])
        gt_lmk = torch.as_tensor(gt_all[t], dtype=torch.float32)
        prev = snapshot()
        if t == 0:
            with torch.no_grad():
                _, rgb_c, err_c, rgba_c = energy(gt_rgb, gt_lmk, prev)
            best = prev
            rgb_v, err_v = float(rgb_c), err_c.detach()
            rgba_best = rgba_c.detach()
        else:
            best = prev
            best_score = float("inf")
            rgba_best = None
            rgb_v, err_v = 0.0, None
            opt.state.clear()
            for i in range(STEPS):
                warm = min((i + 1) / 10.0, 1.0)
                tcos = i / max(STEPS - 1, 1)
                lr_scale = warm * (0.4 + 0.6 * (0.5 * (1.0 + np.cos(np.pi * tcos))))
                for g in opt.param_groups:
                    g["lr"] = g["base_lr"] * lr_scale
                opt.zero_grad()
                loss, _, _, _ = energy(gt_rgb, gt_lmk, prev)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([
                    state.ex_lower, state.rot_neck, state.rot_head,
                    state.rot_leye, state.rot_reye, state.translation,
                ], 0.5)
                opt.step()
                if i == 0 or i == STEPS - 1 or (i + 1) % 10 == 0:
                    with torch.no_grad():
                        _, rgb_c, err_c, rgba_c = energy(gt_rgb, gt_lmk, prev)
                    nose = float(err_c[27:36].mean())
                    active = float(err_c.mean())
                    score = 30.0 * float(rgb_c) + 10.0 * active
                    if nose < 3.0 and active < 8.0 and score < best_score:
                        best, best_score = snapshot(), score
                        rgb_v, err_v = float(rgb_c), err_c.detach()
                        rgba_best = rgba_c.detach()
            if err_v is None:
                restore(prev)
                with torch.no_grad():
                    _, rgb_c, err_c, rgba_c = energy(gt_rgb, gt_lmk, prev)
                best = prev
                rgb_v, err_v = float(rgb_c), err_c.detach()
                rgba_best = rgba_c.detach()
                print(f"[track {t:03d}] reverted", flush=True)
            else:
                restore(best)
        expr[t] = best["ex_lower"].numpy()
        neck[t] = best["rot_neck"].numpy()
        head[t] = best["rot_head"].numpy()
        leye[t] = best["rot_leye"].numpy()
        reye[t] = best["rot_reye"].numpy()
        trans[t] = best["translation"].numpy()
        rgb_log[t] = rgb_v
        lmk_log[t] = float(err_v.mean())
        nose_log[t] = float(err_v[27:36].mean())
        bgr = cv2.cvtColor(
            (rgba_best[0, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8),
            cv2.COLOR_RGB2BGR,
        )
        cv2.imwrite(str(OUT / f"frame_{t:04d}.jpg"), bgr)
        if t % 20 == 0 or t + 1 == n_frames:
            photo_fit.dump_photo_preview(
                OUT, f"f{t:04d}",
                gt_rgb[0].detach().float().cpu().numpy(),
                rgba_best[0].detach().float().cpu().numpy(),
            )
        print(
            f"[track {t:03d}/{n_frames}] rgb={rgb_v:.4f} "
            f"lmk={lmk_log[t]:.2f} nose={nose_log[t]:.2f} jaw={float(err_v[:17].mean()):.2f}",
            flush=True,
        )
        opt.zero_grad(set_to_none=True)

    np.savez(
        OUT / "track_params.npz",
        expression_lower_face=expr,
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
        offset=static_off.numpy(),
        residual=res.numpy(),
        lights=lights.numpy(),
        eye_col=eye_col.numpy(),
        image_paths=np.array([str(p) for p in paths]),
    )
    print(
        f"[track] rgb {rgb_log.mean():.4f} lmk {lmk_log.mean():.2f} "
        f"max_lmk {lmk_log.max():.2f} at {int(lmk_log.argmax())}",
        flush=True,
    )
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
