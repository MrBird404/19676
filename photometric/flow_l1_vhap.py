"""Photometric layer 1, VHAP rgb_init_texture.

Camera, shape, texture and lights move. Pose, expression and the eyeball
stay fixed. The residual is smoothed by total variation so it separates
skin, lips and brows instead of painting fine detail. Eye color can move.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/root/autodl-tmp/GNM-main")
sys.path.insert(0, "/root/autodl-tmp/GNM_Fit")

from gnm.shape import gnm_numpy, gnm_pytorch
import gnm_mediapipe_landmarks as mp_lmk
from nvdiffrast_render import NVDiffRenderer
from nvdiffrast_render.camera import project_world_to_pixels
from fit_gnm import FitState, ibug68_vhap_weights
import photo_fit
from flow_photo import group, load_base, raster_mask
from uvmask import tensor_mask

OUT = Path("/root/autodl-tmp/gnm_fit_out/expression_v3/flow_frame0/photo_l1_vhap")
_UV = Path(__file__).resolve().parent / "uv"
PAINTED = _UV / "gnm_uv_from_flame_lmk68.png"
EYE_TEX = _UV / "gnm_eye_uv.png"
EYE_UV = _UV / "gnm_eye_uv_repacked.npz"
PARAMS = Path("/root/autodl-tmp/gnm_fit_out/expression_v3/flow_frame0/frame_000000_params.npz")
APPEAR = Path("/root/autodl-tmp/gnm_fit_out/expression_v3/flow_frame0/photo/photo_appearance.npz")
LMK = Path("/root/autodl-tmp/gnm_fit_out/expression_v3/facepoints68.npz")
LMK_FILE = Path("/root/autodl-tmp/GNM_Fit/head_sparse_68_v3_改.txt")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    params = np.load(PARAMS)
    saved = np.load(APPEAR)
    lmk_npz = np.load(LMK, allow_pickle=True)
    gt_xy = np.asarray(lmk_npz["landmarks"][0], dtype=np.float32)
    width, height = [int(x) for x in lmk_npz["image_size"]]
    image = Path(str(lmk_npz["image_paths"][0]))
    stage_bgr = cv2.imread(str(image))
    ph, pw, scale = photo_fit.scale_for_photo(height, width, 1024)
    bgr = stage_bgr if scale == 1.0 else cv2.resize(stage_bgr, (pw, ph), interpolation=cv2.INTER_AREA)
    rgb_np = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

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

    def p(v, grad=False):
        t = torch.tensor(np.asarray(v), dtype=torch.float32)
        t.requires_grad_(grad)
        return t

    rot = np.asarray(params["rotations"], dtype=np.float32).copy()
    rot[2:] = np.asarray(saved["eye_rot"], dtype=np.float32)
    state = FitState(
        id_head=p(params["id_head"], True), id_eyes=p(params["id_eyes"]),
        ex_lower=p(params["expression_lower_face"]),
        rot_neck=p(rot[0]), rot_head=p(rot[1]),
        rot_leye=p(rot[2]), rot_reye=p(rot[3]),
        translation=p(params["translation"]),
        cam_aa=p(params["cam_aa"], True), cam_t=p(params["cam_t"], True),
        focal_norm=p(params["focal_norm"]),
    )
    id_ref = state.id_head.detach().clone()

    def posed():
        return gnm(
            state.identity()[None], state.expression()[None],
            state.rotations()[None], state.translation[None],
        )[0]

    h, w = base_face.shape[:2]
    eh, ew = base_eye.shape[:2]
    eye_masks = [torch.tensor(raster_mask(
        faces[eye_ids], eye_tri_uv, group(gnm_np, name), n_verts, (eh, ew),
    )) for name in ("scleras", "irises", "pupils")]
    eye_col = torch.nn.Parameter(torch.tensor(saved["eye_col"], dtype=torch.float32))
    regions = photo_fit.build_photo_regions(gnm_np, faces)
    brow = np.unique(np.concatenate([group(gnm_np, n) for n in (
        "left_brow_region", "right_brow_region", "middle_brow_region")]))
    lip = np.unique(np.concatenate([group(gnm_np, "upper_lip"), group(gnm_np, "lower_lip")]))
    keep_verts = np.unique(np.concatenate([group(gnm_np, "hockey_mask"), brow, lip]))
    face_keep_t = torch.tensor(tensor_mask(faces, tri_uv, keep_verts, n_verts, (h, w)))
    res = torch.nn.Parameter(torch.zeros(1, 3, 128, 128))
    lights = torch.nn.Parameter(torch.zeros(1, 9, 3))

    device = torch.device("cuda")
    gt_rgb = torch.as_tensor(rgb_np, dtype=torch.float32, device=device)[None]
    color_lut = torch.zeros(faces.shape[0] + 1, dtype=torch.bool, device=device)
    color_lut[torch.as_tensor(regions["color_except"], dtype=torch.long, device=device) + 1] = True
    face_ids = np.setdiff1d(np.arange(len(faces), dtype=np.int64), eye_ids)
    full_cid = np.asarray(regions["fid2cid"], dtype=np.int64)
    renderer = NVDiffRenderer(
        use_opengl=False, lighting_type="SH", lighting_space="world",
        disturb_rate_fg=0.3, disturb_rate_bg=0.0,
        fid2cid=torch.as_tensor(full_cid, dtype=torch.long),
        shade_smooth=True, flip_y_raster=False,
    ).to(device)
    with torch.no_grad():
        lights[:, 0, :] = 1.0 / renderer.sh_const[0].detach().cpu()

    def set_cid(part_ids):
        cid = torch.as_tensor(full_cid[part_ids], dtype=torch.long, device=device)
        renderer.fid2cid = F.pad(cid, (1, 0), value=0)

    def project(pts):
        k = state.camera_k(width, height)
        if scale != 1.0:
            k = k * scale
        return project_world_to_pixels(pts, state.camera_rt(), k, pw, ph, flip_y=True)

    lmk_idx, lmk_bary = mp_lmk.load_landmarks(LMK_FILE)
    idx_t = torch.as_tensor(lmk_idx, dtype=torch.long)
    bary_t = torch.as_tensor(lmk_bary, dtype=torch.float32)
    w_lmk = ibug68_vhap_weights(nose_boost=10.0)
    w_lmk_t = torch.as_tensor(w_lmk, dtype=torch.float32)
    gt_lmk = torch.as_tensor(gt_xy, dtype=torch.float32)
    uv_center_t = torch.tensor([17303, 17689], dtype=torch.long)
    target_t = torch.tensor(np.asarray(saved["target_centers"], dtype=np.float32))

    def face_albedo():
        up = F.interpolate(res, size=(h, w), mode="bilinear", align_corners=True)[0].permute(1, 2, 0)
        return (base_face + up * face_keep_t[..., None]).clamp(1e-4, 1.0 - 1e-4)

    def eye_albedo():
        add = torch.zeros_like(base_eye)
        for m, c in zip(eye_masks, eye_col):
            add = add + m[..., None] * c
        return (base_eye + add).clamp(1e-4, 1.0 - 1e-4)

    def render(disturb):
        def one(part_ids, part_uv, tex):
            set_cid(part_ids)
            part_faces = torch.as_tensor(faces[part_ids], dtype=torch.int32, device=device)
            verts_uv_t = torch.as_tensor(part_uv.reshape(-1, 2), dtype=torch.float32, device=device)
            faces_uv_t = torch.arange(len(part_ids) * 3, device=device, dtype=torch.int32).reshape(-1, 3)
            verts = posed().to(device)[None].contiguous()
            rt = state.camera_rt().to(device)[None]
            k4 = state.camera_k(width, height).detach()
            if scale != 1.0:
                k4 = k4 * scale
            rast = renderer.rasterize(
                verts, part_faces, rt, k4.to(device)[None], (ph, pw), use_cache=False, require_grad=True,
            )
            out = renderer.render_rgba(
                rast, verts[0], part_faces, verts_uv_t, faces_uv_t,
                tex.permute(2, 0, 1)[None].to(device), lights.to(device),
                background_color=[1.0, 1.0, 1.0], enable_disturbance=disturb,
            )
            depth = renderer._to_image_space(rast["rast_out"][..., 2:3])
            fid = renderer._to_image_space(rast["rast_out"][..., 3:4].long())
            return out, depth, fid

        eye_out, eye_z, _ = one(eye_ids, eye_tri_uv, eye_albedo())
        face_out, face_z, face_fid = one(face_ids, tri_uv[face_ids], face_albedo())
        eye_on = eye_out["rgba"][..., 3:] > 0.5
        face_on = face_out["rgba"][..., 3:] > 0.5
        face_front = face_on & (~eye_on | (face_z <= eye_z))
        rgb = torch.where(
            face_front, face_out["rgba"][..., :3],
            torch.where(eye_on, eye_out["rgba"][..., :3], torch.ones_like(face_out["rgba"][..., :3])),
        )
        alpha = (face_front | eye_on).float()
        rgba = torch.cat([rgb, alpha], -1)
        ids = torch.as_tensor(face_ids, dtype=torch.long, device=device)
        flat = face_fid.reshape(-1).clamp(min=0)
        glob = torch.zeros_like(flat)
        hit = flat > 0
        glob[hit] = ids[flat[hit] - 1] + 1
        return rgba, glob.view_as(face_fid), face_out["diffuse_detach_normal"]

    def energy(disturb):
        rgba, global_fid, diffuse = render(disturb)
        covered = (rgba[..., 3:] > 0) & ~color_lut[global_fid.clamp(0, color_lut.shape[0] - 1)]
        blue_cap = (gt_rgb[..., 2:] > gt_rgb[..., 0:1] + 0.08) & (gt_rgb[..., 2:] > 0.25)
        dark_cloth = gt_rgb.mean(-1, keepdim=True) < 0.22
        vis = (covered & ~blue_cap & ~dark_cloth).detach()
        pred = rgba[..., :3]
        rgb_loss = ((pred - gt_rgb).abs() * vis).sum() / (vis.sum() * 3).clamp(min=1.0)
        verts = posed()
        pred_xy = project((verts[idx_t] * bary_t.unsqueeze(-1)).sum(1))
        lmk = torch.linalg.norm(pred_xy - gt_lmk, dim=-1)
        lmk_w = (lmk * w_lmk_t).sum() / w_lmk_t.sum()
        eye = (project(verts[uv_center_t]) - target_t).abs().mean()
        tv = (res[:, :, :-1, :] - res[:, :, 1:, :]).pow(2).mean()
        tv = tv + (res[:, :, :, :-1] - res[:, :, :, 1:]).pow(2).mean()
        reg_diffuse = torch.relu(diffuse.max() - 1.0)
        dev = rgb_loss.device
        loss = (
            30.0 * rgb_loss
            + 10.0 * lmk_w.to(dev)
            + 10.0 * eye.to(dev)
            + 10.0 * reg_diffuse
            + 1.0e4 * tv
            + 0.002 * (lights[:, 1:] ** 2).mean()
            + 0.02 * eye_col.pow(2).mean()
            + (0.3 * state.id_head.pow(2).mean()).to(dev)
        )
        return loss, rgb_loss, lmk, eye, rgba

    opt = torch.optim.Adam([
        {"params": [state.id_head], "lr": 5e-3},
        {"params": [state.cam_aa, state.cam_t], "lr": 5e-3},
        {"params": [res], "lr": 5e-3},
        {"params": [lights], "lr": 5e-3},
        {"params": [eye_col], "lr": 5e-3},
    ])
    for g in opt.param_groups:
        g["base_lr"] = g["lr"]

    def snap():
        return (
            state.id_head.detach().clone(),
            state.cam_aa.detach().clone(),
            state.cam_t.detach().clone(),
            res.detach().clone(),
            lights.detach().clone(),
            eye_col.detach().clone(),
        )

    def restore(s):
        with torch.no_grad():
            state.id_head.copy_(s[0])
            state.cam_aa.copy_(s[1])
            state.cam_t.copy_(s[2])
            res.copy_(s[3])
            lights.copy_(s[4])
            eye_col.copy_(s[5])

    with torch.no_grad():
        _, rgb0, err0, eye0, rgba0 = energy(False)
    photo_fit.dump_photo_preview(OUT, "before", rgb_np, rgba0[0].detach().float().cpu().numpy())
    print(
        f"[before] rgb={float(rgb0):.4f} lmk={float(err0.mean()):.3f} "
        f"nose={float(err0[27:36].mean()):.3f} jaw={float(err0[:17].mean()):.3f} "
        f"eye={float(eye0):.3f}",
        flush=True,
    )
    best, best_score, best_i = snap(), float("inf"), 0
    steps = 400
    for i in range(steps):
        warm = min((i + 1) / 40.0, 1.0)
        tcos = i / max(steps - 1, 1)
        lr_scale = warm * (0.35 + 0.65 * (0.5 * (1.0 + np.cos(np.pi * tcos))))
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * lr_scale
        opt.zero_grad()
        loss, _, _, _, _ = energy(True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([state.id_head, state.cam_aa, state.cam_t], 0.5)
        torch.nn.utils.clip_grad_norm_([res], 4.0)
        torch.nn.utils.clip_grad_norm_([lights, eye_col], 1.0)
        opt.step()
        if i == 0 or (i + 1) % 50 == 0 or i + 1 == steps:
            with torch.no_grad():
                _, rgb_c, err_c, eye_c, _ = energy(False)
                did = float((state.id_head - id_ref).abs().mean())
            score = 30.0 * float(rgb_c) + 10.0 * float((err_c * w_lmk_t).sum() / w_lmk_t.sum())
            ok = float(err_c.mean()) < 3.5 and float(err_c[27:36].mean()) < 1.0
            if ok and score < best_score:
                best, best_score, best_i = snap(), score, i + 1
            print(
                f"[l1v {i+1:3d}/{steps}] rgb={float(rgb_c):.4f} "
                f"lmk={float(err_c.mean()):.3f} nose={float(err_c[27:36].mean()):.3f} "
                f"jaw={float(err_c[:17].mean()):.3f} eye={float(eye_c):.3f} "
                f"|did|={did:.4f} best@{best_i}",
                flush=True,
            )
    restore(best)
    with torch.no_grad():
        _, rgb1, err1, eye1, rgba1 = energy(False)
        did = float((state.id_head - id_ref).abs().mean())
    photo_fit.dump_photo_preview(OUT, "tex", rgb_np, rgba1[0].detach().float().cpu().numpy())
    face_png = (face_albedo().detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    cv2.imwrite(str(OUT / "face_albedo.png"), cv2.cvtColor(np.flip(face_png, 0), cv2.COLOR_RGB2BGR))
    np.savez(
        OUT / "photo_appearance.npz",
        residual=res.detach().cpu().numpy(),
        lights=lights.detach().cpu().numpy(),
        eye_col=eye_col.detach().cpu().numpy(),
        id_head=state.id_head.detach().numpy(),
        id_eyes=state.id_eyes.detach().numpy(),
        expression_lower_face=state.ex_lower.detach().numpy(),
        rotations=state.rotations().detach().numpy(),
        translation=state.translation.detach().numpy(),
        cam_aa=state.cam_aa.detach().numpy(),
        cam_t=state.cam_t.detach().numpy(),
        focal_norm=state.focal_norm.detach().numpy(),
    )
    print(
        f"[after] rgb={float(rgb1):.4f} lmk={float(err1.mean()):.3f} "
        f"nose={float(err1[27:36].mean()):.3f} jaw={float(err1[:17].mean()):.3f} "
        f"eye={float(eye1):.3f} |did|={did:.4f}",
        flush=True,
    )
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
