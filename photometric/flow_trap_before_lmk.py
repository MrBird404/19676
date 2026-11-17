"""Drop the trapezius in bind pose before the rigid landmarks are finished.

The 68-point vertices stay put, so the rigid face fit does not move. Only the
lateral neck, the trapezius, shifts down onto the black shirt.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/root/autodl-tmp/GNM_Fit")
sys.path.insert(0, "/root/autodl-tmp/GNM-main")

from gnm.shape import gnm_numpy, gnm_pytorch
import gnm_mediapipe_landmarks as mp_lmk
from nvdiffrast_render import NVDiffRenderer
from nvdiffrast_render.camera import project_world_to_pixels
from fit_gnm import FitState
import photo_fit
from flow_photo import group, load_base, raster_mask
from uvmask import tensor_mask

OUT = Path("/root/autodl-tmp/gnm_fit_out/expression_v3/flow_frame0/trap_before_lmk")
_UV = Path(__file__).resolve().parent / "uv"
PAINTED = _UV / "gnm_uv_from_flame_lmk68.png"
EYE_TEX = _UV / "gnm_eye_uv.png"
EYE_UV = _UV / "gnm_eye_uv_repacked.npz"
LMK_PARAMS = Path("/root/autodl-tmp/gnm_fit_out/expression_v3/lmk_init_frame0_v3改/frame_000000_params.npz")
APPEAR = Path("/root/autodl-tmp/gnm_fit_out/expression_v3/flow_frame0/photo_contour_only/photo_appearance.npz")
PHOTO = Path("/root/autodl-tmp/gnm_fit_out/expression_v3/facepoints68.npz")
LMK_FILE = Path("/root/autodl-tmp/GNM_Fit/head_sparse_68_v3_改.txt")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    saved = np.load(LMK_PARAMS)
    color = np.load(APPEAR)
    photo_npz = np.load(PHOTO, allow_pickle=True)
    gt = np.asarray(photo_npz["landmarks"][0], dtype=np.float32)
    width, height = [int(x) for x in photo_npz["image_size"]]
    bgr = cv2.imread(str(photo_npz["image_paths"][0]))
    ph, pw, scale = photo_fit.scale_for_photo(height, width, 1024)
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
    rest = np.asarray(gnm_np(None, None, None, None), dtype=np.float32)
    regions = photo_fit.build_photo_regions(gnm_np, faces)
    edges_np = np.asarray(regions["edges"], dtype=np.int64)
    lmk_idx, lmk_bary = mp_lmk.load_landmarks(LMK_FILE)
    lmk_verts = np.unique(lmk_idx)

    ears = group(gnm_np, "ears")
    eyes_v = group(gnm_np, "eyes")
    teeth = np.unique(np.concatenate([
        group(gnm_np, "teeth"), group(gnm_np, "gums"), group(gnm_np, "tongue"),
    ]))
    y = rest[:, 1]
    ax = np.abs(rest[:, 0])
    pinned = np.zeros(n_verts, dtype=bool)
    pinned[lmk_verts] = True
    pinned[ears] = True
    pinned[eyes_v] = True
    pinned[teeth] = True
    pinned[y > 0.17] = True

    def p(v):
        return torch.tensor(np.asarray(v, dtype=np.float32))

    rot = np.asarray(saved["rotations"], dtype=np.float32)
    state = FitState(
        id_head=p(saved["id_head"]), id_eyes=p(saved["id_eyes"]),
        ex_lower=p(saved["expression_lower_face"]),
        rot_neck=p(rot[0]), rot_head=p(rot[1]),
        rot_leye=p(rot[2]), rot_reye=p(rot[3]),
        translation=p(saved["translation"]),
        cam_aa=p(saved["cam_aa"]), cam_t=p(saved["cam_t"]),
        focal_norm=p(saved["focal_norm"]),
    )
    res = torch.tensor(np.asarray(color["residual"], dtype=np.float32))
    lights = torch.tensor(color["lights"], dtype=torch.float32)
    eye_col = torch.tensor(color["eye_col"], dtype=torch.float32)
    extra = torch.zeros(n_verts, 3)

    h, w = base_face.shape[:2]
    eh, ew = base_eye.shape[:2]
    eye_masks = [torch.tensor(raster_mask(
        faces[eye_ids], eye_tri_uv, group(gnm_np, name), n_verts, (eh, ew),
    )) for name in ("scleras", "irises", "pupils")]
    lip_ids = np.unique(np.concatenate([group(gnm_np, "upper_lip"), group(gnm_np, "lower_lip")]))
    brow = np.unique(np.concatenate([group(gnm_np, n) for n in (
        "left_brow_region", "right_brow_region", "middle_brow_region")]))
    keep_verts = np.unique(np.concatenate([group(gnm_np, "hockey_mask"), brow, lip_ids]))
    face_keep_t = torch.tensor(tensor_mask(faces, tri_uv, keep_verts, n_verts, (h, w)))
    device = torch.device("cuda")
    full_cid = np.asarray(regions["fid2cid"], dtype=np.int64)
    face_ids = np.setdiff1d(np.arange(len(faces), dtype=np.int64), eye_ids)
    idx_t = torch.as_tensor(lmk_idx, dtype=torch.long)
    bary_t = torch.as_tensor(lmk_bary, dtype=torch.float32)

    def posed():
        verts = gnm.vertex_positions_bind_pose(state.identity()[None], state.expression()[None])
        joints = gnm.joint_positions_bind_pose(state.identity()[None])
        rotations = state.rotations()[None]
        verts = verts + gnm.compute_pose_correctives(rotations)
        verts = verts + extra[None]
        return gnm.vertex_positions_world(verts, joints, rotations, state.translation[None])[0].clone()

    def project(pts):
        k = state.camera_k(width, height).detach()
        if scale != 1.0:
            k = k * scale
        return project_world_to_pixels(pts, state.camera_rt().detach(), k, pw, ph, flip_y=True)

    with torch.no_grad():
        xy = project(posed()).numpy()
        probe = extra.clone()
        probe[:, 1] = -0.005
        probe[pinned] = 0
        extra.copy_(probe)
        xy_down = project(posed()).numpy()
        extra.zero_()
    scale_px = (xy_down[:, 1] - xy[:, 1]) / 5.0
    scale_px = np.where(np.abs(scale_px) < 0.2, 4.0, scale_px)

    lum = rgb_np.mean(-1)
    shirt_top = np.full(pw, -1, dtype=np.int32)
    for x in range(pw):
        run = 0
        for row in range(490, ph):
            pix = rgb_np[row, x]
            if lum[row, x] < 0.30 and abs(float(pix[0] - pix[2])) < 0.10 and (x < 490 or x > 640):
                run += 1
                if run >= 5:
                    shirt_top[x] = row - 4
                    break
            else:
                run = 0
    # Fill gaps from the nearest known shoulder column.
    known = np.nonzero(shirt_top > 0)[0]
    if known.size:
        for x in range(pw):
            if shirt_top[x] < 0:
                shirt_top[x] = shirt_top[known[np.argmin(np.abs(known - x))]]

    lat = (ax > 0.055) & (y < 0.16) & (y > 0.08) & (rest[:, 2] > -0.02) & ~pinned
    need = np.zeros(n_verts, dtype=np.float32)
    for i in np.nonzero(lat)[0]:
        px, py = xy[i]
        if not (0 <= px < pw and 0 <= py < ph):
            continue
        target = float(shirt_top[int(np.clip(px, 0, pw - 1))])
        gap = target - py
        if gap > 4:
            need[i] = np.clip(gap / scale_px[i], 0.0, 18.0) * 1e-3
    # Spread the downward move, then pin the face and the landmarks back to zero.
    acc = np.zeros(n_verts, dtype=np.float64)
    cnt = np.zeros(n_verts, dtype=np.float64)
    for i, j in edges_np:
        acc[i] += need[j]
        acc[j] += need[i]
        cnt[i] += 1
        cnt[j] += 1
    disp = need.copy()
    for _ in range(20):
        sm = acc / np.maximum(cnt, 1.0)
        disp = np.where(pinned, 0.0, 0.40 * disp + 0.60 * sm).astype(np.float32)
        acc[:] = 0.0
        for i, j in edges_np:
            acc[i] += disp[j]
            acc[j] += disp[i]
    disp[pinned] = 0.0
    extra[:, 1] = torch.tensor(-disp)
    jump = np.linalg.norm(extra.numpy()[edges_np[:, 0]] - extra.numpy()[edges_np[:, 1]], axis=1).max()
    print(
        f"[trap] moved {int((disp > 1e-4).sum())} max {disp.max()*1e3:.1f}mm "
        f"edge {jump*1e3:.2f}mm lmk_disp {float(np.linalg.norm(extra.numpy()[lmk_verts], axis=1).max())*1e3:.3f}mm",
        flush=True,
    )

    def face_err():
        pred = project((posed()[idx_t] * bary_t.unsqueeze(-1)).sum(1))
        return float(torch.linalg.norm(pred - torch.as_tensor(gt), dim=-1).mean())

    renderer = NVDiffRenderer(
        use_opengl=False, lighting_type="SH", lighting_space="world",
        disturb_rate_fg=0.0, disturb_rate_bg=0.0,
        fid2cid=torch.as_tensor(full_cid, dtype=torch.long),
        shade_smooth=True, flip_y_raster=False,
    ).to(device)
    bound_t = torch.as_tensor(regions["bound_except"], dtype=torch.long, device=device)

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

    def set_cid(part_ids):
        cid = torch.as_tensor(full_cid[part_ids], dtype=torch.long, device=device)
        renderer.fid2cid = F.pad(cid, (1, 0), value=0)

    def beauty():
        def one(part_ids, part_uv, tex):
            set_cid(part_ids)
            part_faces = torch.as_tensor(faces[part_ids], dtype=torch.int32, device=device)
            verts_uv_t = torch.as_tensor(part_uv.reshape(-1, 2), dtype=torch.float32, device=device)
            faces_uv_t = torch.arange(len(part_ids) * 3, device=device, dtype=torch.int32).reshape(-1, 3)
            verts = posed().to(device)[None].contiguous()
            rt = state.camera_rt().detach().to(device)[None]
            k4 = state.camera_k(width, height).detach()
            if scale != 1.0:
                k4 = k4 * scale
            rast = renderer.rasterize(
                verts, part_faces, rt, k4.to(device)[None], (ph, pw),
                use_cache=False, require_grad=False,
            )
            out = renderer.render_rgba(
                rast, verts[0], part_faces, verts_uv_t, faces_uv_t,
                tex.permute(2, 0, 1)[None].to(device), lights.to(device),
                background_color=[1.0, 1.0, 1.0], enable_disturbance=False,
                align_boundary_except_vid=bound_t,
            )
            depth = renderer._to_image_space(rast["rast_out"][..., 2:3])
            return out["rgba"], depth

        eye_out, eye_z = one(eye_ids, eye_tri_uv, eye_tex)
        face_out, face_z = one(face_ids, tri_uv[face_ids], face_tex)
        eye_on = eye_out[..., 3:] > 0.5
        face_on = face_out[..., 3:] > 0.5
        front = face_on & (~eye_on | (face_z <= eye_z))
        rgb = torch.where(
            front, face_out[..., :3],
            torch.where(eye_on, eye_out[..., :3], torch.ones_like(face_out[..., :3])),
        )
        return torch.cat([rgb, (front | eye_on).float()], -1)

    with torch.no_grad():
        held = extra.clone()
        extra.zero_()
        err0 = face_err()
        photo_fit.dump_photo_preview(OUT, "before", rgb_np, beauty()[0].detach().float().cpu().numpy())
        extra.copy_(held)
        err1 = face_err()
        rgba = beauty()
        photo_fit.dump_photo_preview(OUT, "trap", rgb_np, rgba[0].detach().float().cpu().numpy())
    print(f"[lmk] face before {err0:.2f}px after {err1:.2f}px", flush=True)
    np.savez(
        OUT / "trap_offset.npz",
        offset=extra.numpy(),
        id_head=state.id_head.numpy(),
        id_eyes=state.id_eyes.numpy(),
        expression_lower_face=state.ex_lower.numpy(),
        rotations=state.rotations().numpy(),
        translation=state.translation.numpy(),
        cam_aa=state.cam_aa.numpy(),
        cam_t=state.cam_t.numpy(),
        focal_norm=state.focal_norm.numpy(),
    )
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
