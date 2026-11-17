"""Align the neck after the face is locked.

Frame-0 pose, expression, identity and camera stay at photo_contour_only.
Only neck vertices move, with a smooth thickness and length. The target is
the visible neck skin in the photo, bounded by the face contour and the shirt.
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

OUT = Path("/root/autodl-tmp/gnm_fit_out/expression_v3/flow_frame0/neck0")
_UV = Path(__file__).resolve().parent / "uv"
PAINTED = _UV / "gnm_uv_from_flame_lmk68.png"
EYE_TEX = _UV / "gnm_eye_uv.png"
EYE_UV = _UV / "gnm_eye_uv_repacked.npz"
APPEAR = Path("/root/autodl-tmp/gnm_fit_out/expression_v3/flow_frame0/photo_contour_only/photo_appearance.npz")
LMK = Path("/root/autodl-tmp/gnm_fit_out/expression_v3/facepoints68.npz")
LMK_FILE = Path("/root/autodl-tmp/GNM_Fit/head_sparse_68_v3_改.txt")
MASK = Path(__file__).resolve().parent / "face_mask.npz"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    saved = np.load(APPEAR)
    lmk_npz = np.load(LMK, allow_pickle=True)
    gt = np.asarray(lmk_npz["landmarks"][0], dtype=np.float32)
    image = Path(str(lmk_npz["image_paths"][0]))
    width, height = [int(x) for x in lmk_npz["image_size"]]
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
    rest = np.asarray(gnm_np(None, None, None, None), dtype=np.float32)
    regions = photo_fit.build_photo_regions(gnm_np, faces)
    edges_np = np.asarray(regions["edges"], dtype=np.int64)

    ears = group(gnm_np, "ears")
    eyes_v = group(gnm_np, "eyes")
    teeth = np.unique(np.concatenate([
        group(gnm_np, "teeth"), group(gnm_np, "gums"), group(gnm_np, "tongue"),
    ]))
    y = rest[:, 1]
    pinned = np.zeros(n_verts, dtype=bool)
    pinned[ears] = True
    pinned[eyes_v] = True
    pinned[teeth] = True
    pinned[y > 0.165] = True
    t = np.clip((0.165 - y) / (0.165 - 0.07), 0.0, 1.0)
    fall = (t * t * (3.0 - 2.0 * t)).astype(np.float32)
    fall[pinned] = 0.0
    acc = np.zeros(n_verts, dtype=np.float64)
    cnt = np.zeros(n_verts, dtype=np.float64)
    for i, j in edges_np:
        acc[i] += fall[j]
        acc[j] += fall[i]
        cnt[i] += 1
        cnt[j] += 1
    for _ in range(16):
        sm = acc / np.maximum(cnt, 1.0)
        fall = np.where(pinned, 0.0, 0.45 * fall + 0.55 * sm).astype(np.float32)
        acc[:] = 0.0
        for i, j in edges_np:
            acc[i] += fall[j]
            acc[j] += fall[i]
    peak = float(fall.max())
    if peak > 1e-6:
        fall /= peak
        fall[pinned] = 0.0
    steep = float(np.abs(fall[edges_np[:, 0]] - fall[edges_np[:, 1]]).max())
    print(
        f"[neck] pinned {int(pinned.sum())} fall>0.2 {int((fall > 0.2).sum())} "
        f"steep {steep:.4f} budget_mm {1.0 / max(steep, 1e-6):.1f}",
        flush=True,
    )

    def p(v):
        return torch.tensor(np.asarray(v, dtype=np.float32))

    rot = np.asarray(saved["rotations"], dtype=np.float32).copy()
    state = FitState(
        id_head=p(saved["id_head"]), id_eyes=p(saved["id_eyes"]),
        ex_lower=p(saved["expression_lower_face"]),
        rot_neck=p(rot[0]), rot_head=p(rot[1]),
        rot_leye=p(rot[2]), rot_reye=p(rot[3]),
        translation=p(saved["translation"]),
        cam_aa=p(saved["cam_aa"]), cam_t=p(saved["cam_t"]),
        focal_norm=p(saved["focal_norm"]),
    )
    base_off = torch.tensor(np.asarray(saved["offset"], dtype=np.float32))
    res = torch.tensor(np.asarray(saved["residual"], dtype=np.float32))
    lights = torch.tensor(saved["lights"], dtype=torch.float32)
    eye_col = torch.tensor(saved["eye_col"], dtype=torch.float32)
    amp = torch.zeros(2)

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
    full_cid = np.asarray(regions["fid2cid"], dtype=np.int64)
    neck_ids = np.nonzero((fall[faces] > 1e-4).all(1))[0]
    neck_ids = np.setdiff1d(neck_ids, eye_ids)
    face_ids = np.setdiff1d(np.arange(len(faces), dtype=np.int64), eye_ids)
    lmk_idx, lmk_bary = mp_lmk.load_landmarks(LMK_FILE)
    idx_t = torch.as_tensor(lmk_idx, dtype=torch.long)
    bary_t = torch.as_tensor(lmk_bary, dtype=torch.float32)
    lmk_fall = float(fall[np.unique(lmk_idx)].max())
    print(f"[neck] neck faces {len(neck_ids)} landmark fall {lmk_fall:.4f}", flush=True)

    contour = np.load(MASK)["lower_contour_smooth"]
    order = np.argsort(contour[:, 0])
    jaw_y = np.interp(
        np.arange(pw), contour[order, 0], contour[order, 1],
        left=np.nan, right=np.nan,
    )
    x0, x1 = int(contour[:, 0].min()), int(contour[:, 0].max())
    lum = rgb_np.mean(-1)
    blue = (rgb_np[..., 2] > rgb_np[..., 0] + 0.08) & (rgb_np[..., 2] > 0.25)
    skin = np.zeros((ph, pw), dtype=bool)
    for x in range(max(0, x0 - 10), min(pw, x1 + 10)):
        yb = jaw_y[x]
        if not np.isfinite(yb):
            continue
        skin[int(np.ceil(yb)):min(ph, int(np.ceil(yb)) + 110), x] = True
    skin &= (lum > 0.18) & (lum < 0.50) & ~blue & (rgb_np[..., 0] > rgb_np[..., 2] + 0.03)
    skin = cv2.morphologyEx(skin.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    nlab, labels, stats, _ = cv2.connectedComponentsWithStats(skin.astype(np.uint8), 8)
    if nlab > 1:
        skin = labels == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))
    mask_bgr = np.zeros((ph, pw, 3), np.uint8)
    mask_bgr[skin] = (0, 180, 255)
    cv2.imwrite(str(OUT / "neck_skin_mask.jpg"), mask_bgr)
    rows = []
    for yy in range(ph):
        cols = np.nonzero(skin[yy])[0]
        if cols.size >= 8:
            rows.append((yy, int(cols.min()), int(cols.max())))
    skin_bottom = rows[-1][0] if rows else 0
    print(f"[neck] skin px {int(skin.sum())} rows {len(rows)} bottom {skin_bottom}", flush=True)

    renderer = NVDiffRenderer(
        use_opengl=False, lighting_type="SH", lighting_space="world",
        disturb_rate_fg=0.0, disturb_rate_bg=0.0,
        fid2cid=torch.as_tensor(full_cid, dtype=torch.long),
        shade_smooth=True, flip_y_raster=False,
    ).to(device)
    bound_t = torch.as_tensor(regions["bound_except"], dtype=torch.long, device=device)

    def set_cid(part_ids):
        cid = torch.as_tensor(full_cid[part_ids], dtype=torch.long, device=device)
        renderer.fid2cid = F.pad(cid, (1, 0), value=0)

    fall_t = torch.tensor(fall)
    rest_x = torch.tensor(rest[:, 0])
    rest_z = torch.tensor(rest[:, 2])

    def neck_field():
        thick, length = amp[0], amp[1]
        f = fall_t
        return torch.stack([-thick * rest_x * f, length * f, -thick * rest_z * f], dim=-1)

    def posed():
        verts = gnm.vertex_positions_bind_pose(state.identity()[None], state.expression()[None])
        joints = gnm.joint_positions_bind_pose(state.identity()[None])
        rotations = state.rotations()[None]
        verts = verts + gnm.compute_pose_correctives(rotations)
        verts = verts + (base_off + neck_field())[None]
        return gnm.vertex_positions_world(
            verts, joints, rotations, state.translation[None],
        )[0].clone()

    def project(pts):
        k = state.camera_k(width, height).detach()
        if scale != 1.0:
            k = k * scale
        return project_world_to_pixels(pts, state.camera_rt().detach(), k, pw, ph, flip_y=True)

    def face_err():
        pred = project((posed()[idx_t] * bary_t.unsqueeze(-1)).sum(1))
        err = torch.linalg.norm(pred - torch.as_tensor(gt), dim=-1)
        return float(err.mean()), float(err[:17].mean())

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

    def raster(part_ids, part_uv, tex):
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
        return out["rgba"]

    def neck_measure():
        alpha = raster(neck_ids, tri_uv[neck_ids], face_tex)[0, ..., 3].detach().cpu().numpy()
        on = alpha > 0.5
        edge = 0.0
        n = 0
        bottom = 0
        for yy, left, right in rows:
            cols = np.nonzero(on[yy])[0]
            if cols.size < 4:
                edge += abs(left - right)
                n += 1
                continue
            edge += abs(int(cols.min()) - left) + abs(int(cols.max()) - right)
            n += 1
            bottom = max(bottom, yy)
        edge /= max(n, 1)
        return edge, abs(bottom - skin_bottom), alpha

    def field_edge_mm():
        field = neck_field()
        jump = torch.linalg.norm(field[edges_np[:, 0]] - field[edges_np[:, 1]], dim=-1)
        return float(jump.max()) * 1e3

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
        face_front = face_on & (~eye_on | (face_z <= eye_z))
        rgb = torch.where(
            face_front, face_out[..., :3],
            torch.where(eye_on, eye_out[..., :3], torch.ones_like(face_out[..., :3])),
        )
        alpha = (face_front | eye_on).float()
        return torch.cat([rgb, alpha], -1)

    with torch.no_grad():
        base_lmk, base_jaw = face_err()
        base_edge, base_len, _ = neck_measure()
        photo_fit.dump_photo_preview(OUT, "before", rgb_np, beauty()[0].detach().float().cpu().numpy())
    print(
        f"[before] face={base_lmk:.2f}px jaw={base_jaw:.2f}px "
        f"width_err={base_edge:.1f}px len_err={base_len:.0f}px",
        flush=True,
    )
    if base_lmk > 3.0:
        raise RuntimeError(f"locked face is already off: {base_lmk:.1f}px")

    best = amp.clone()
    best_score = base_edge + 0.5 * base_len
    with torch.no_grad():
        for thick in (0.0, 0.04, 0.08, 0.12, 0.18, 0.24):
            for length_mm in (0.0, 3.0, 6.0, 9.0, 12.0):
                amp.copy_(torch.tensor([thick, length_mm * 1e-3]))
                em = field_edge_mm()
                if em >= 1.0:
                    print(f"  thick={thick:.2f} len={length_mm:.0f} edge {em:.2f} skip", flush=True)
                    continue
                lmk, _jaw = face_err()
                if lmk > base_lmk + 0.3:
                    print(f"  thick={thick:.2f} len={length_mm:.0f} face {lmk:.2f} skip", flush=True)
                    continue
                width_err, len_err, _ = neck_measure()
                score = width_err + 0.5 * len_err
                print(
                    f"  thick={thick:.2f} len={length_mm:.0f} "
                    f"width={width_err:.1f} len_err={len_err:.0f} "
                    f"face={lmk:.2f} edge={em:.2f}",
                    flush=True,
                )
                if score < best_score:
                    best = amp.clone()
                    best_score = score
        amp.copy_(best)
        lmk, jaw = face_err()
        width_err, len_err, _ = neck_measure()
        rgba = beauty()
        photo_fit.dump_photo_preview(OUT, "neck", rgb_np, rgba[0].detach().float().cpu().numpy())
        field = neck_field()
        move = float(field.norm(dim=-1).max()) * 1e3
        print(
            f"[after] thick={float(amp[0]):.3f} len={float(amp[1])*1e3:.1f}mm "
            f"move={move:.1f}mm face={lmk:.2f}px jaw={jaw:.2f}px "
            f"width_err={width_err:.1f}px len_err={len_err:.0f}px edge={field_edge_mm():.2f}mm",
            flush=True,
        )
        if lmk <= base_lmk + 0.3:
            np.savez(
                OUT / "photo_appearance.npz",
                residual=res.numpy(),
                lights=lights.numpy(),
                eye_col=eye_col.numpy(),
                offset=(base_off + field).numpy(),
                id_head=state.id_head.numpy(),
                id_eyes=state.id_eyes.numpy(),
                expression_lower_face=state.ex_lower.numpy(),
                rotations=state.rotations().numpy(),
                translation=state.translation.numpy(),
                cam_aa=state.cam_aa.numpy(),
                cam_t=state.cam_t.numpy(),
                focal_norm=state.focal_norm.numpy(),
                neck_thick=np.float32(amp[0]),
                neck_len_m=np.float32(amp[1]),
            )
            print(f"wrote {OUT}", flush=True)
        else:
            print("[after] face moved, appearance not saved", flush=True)


if __name__ == "__main__":
    main()
