"""Posed mesh and a 9-panel evaluation sheet, every 20 frames.

Panels, left to right: photo, render over the photo, photo-minus-render,
albedo, normals, shading, antialiased edges, coverage, 68 points.
Reads the global-tracking result.
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
from fit_gnm import FitState
import photo_fit
from flow_photo import group, load_base, raster_mask
from uvmask import tensor_mask

PRIOR = Path(__import__("os").environ.get(
    "TRACK_NPZ",
    "/root/autodl-tmp/gnm_fit_out/expression_v3/flow_frame0/track_global/track_params.npz",
))
OUT = Path(__import__("os").environ.get(
    "EXPORT_OUT",
    "/root/autodl-tmp/gnm_fit_out/expression_v3/flow_frame0/every20",
))
_UV = Path(__file__).resolve().parent / "uv"
PAINTED = _UV / "gnm_uv_from_flame_lmk68.png"
EYE_TEX = _UV / "gnm_eye_uv.png"
EYE_UV = _UV / "gnm_eye_uv_repacked.npz"
LMK = Path("/root/autodl-tmp/gnm_fit_out/expression_v3/facepoints68.npz")
LMK_FILE = Path("/root/autodl-tmp/GNM_Fit/head_sparse_68_v3_改.txt")
N_EYE = 12


def seismic(gray: np.ndarray) -> np.ndarray:
    g = np.clip(gray, 0.0, 1.0)
    red = np.clip(g * 2.0, 0.0, 1.0)
    blue = np.clip((1.0 - g) * 2.0, 0.0, 1.0)
    green = 1.0 - np.abs(g - 0.5) * 2.0
    return np.stack([red, green, blue], axis=-1).astype(np.float32)


def u8(img: np.ndarray) -> np.ndarray:
    return np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)


def dots(img: np.ndarray, pts: np.ndarray, color, radius: int) -> None:
    for x, y in pts:
        cv2.circle(img, (int(round(float(x))), int(round(float(y)))), radius, color, -1, lineType=cv2.LINE_AA)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(PRIOR)
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

    def tensor(v):
        return torch.tensor(np.asarray(v), dtype=torch.float32)

    rot0 = np.asarray(z["rot_neck"][0], dtype=np.float32)
    state = FitState(
        id_head=tensor(z["id_head"]), id_eyes=tensor(z["id_eyes"]),
        ex_lower=tensor(z["expression_lower_face"][0]),
        rot_neck=tensor(rot0), rot_head=tensor(z["rot_head"][0]),
        rot_leye=tensor(z["rot_leye"][0]), rot_reye=tensor(z["rot_reye"][0]),
        translation=tensor(z["translation"][0]),
        cam_aa=tensor(z["cam_aa"]), cam_t=tensor(z["cam_t"]),
        focal_norm=tensor(z["focal_norm"]),
    )
    ex_lower = np.asarray(z["expression_lower_face"], dtype=np.float32)
    ex_leye = np.asarray(z["expression_left_eyelid"], dtype=np.float32)
    ex_reye = np.asarray(z["expression_right_eyelid"], dtype=np.float32)
    rot_neck = np.asarray(z["rot_neck"], dtype=np.float32)
    rot_head = np.asarray(z["rot_head"], dtype=np.float32)
    rot_leye = np.asarray(z["rot_leye"], dtype=np.float32)
    rot_reye = np.asarray(z["rot_reye"], dtype=np.float32)
    translation = np.asarray(z["translation"], dtype=np.float32)
    static_off = tensor(z["offset"])
    res = tensor(z["residual"])
    lights = tensor(z["lights"])
    eye_col = tensor(z["eye_col"])
    cur = {"leye": tensor(ex_leye[0]), "reye": tensor(ex_reye[0])}

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
    face_ids = np.setdiff1d(np.arange(len(faces), dtype=np.int64), eye_ids)
    full_cid = np.asarray(regions["fid2cid"], dtype=np.int64)
    bound_t = torch.as_tensor(regions["bound_except"], dtype=torch.long, device=device)
    lmk_idx, lmk_bary = mp_lmk.load_landmarks(LMK_FILE)
    idx_t = torch.as_tensor(lmk_idx, dtype=torch.long)
    bary_t = torch.as_tensor(lmk_bary, dtype=torch.float32)

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
    face_tex = albedo_face()

    def region(v, n):
        if v.shape[-1] >= n:
            return v[..., :n]
        return F.pad(v, (0, n - v.shape[-1]))

    def expression_vec():
        tongue = cur["tongue"] if "tongue" in cur else torch.zeros(32)
        pupil = cur["pupil"] if "pupil" in cur else torch.zeros(1)
        return torch.cat([
            region(cur["leye"], 100), region(cur["reye"], 100),
            state.ex_lower, tongue, pupil,
        ])

    def posed():
        verts = gnm.vertex_positions_bind_pose(state.identity()[None], expression_vec()[None])
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
            use_cache=False, require_grad=False,
        )
        out = renderer.render_rgba(
            rast, verts[0], part_faces, verts_uv_t, faces_uv_t,
            tex.permute(2, 0, 1)[None].to(device), lights.to(device),
            background_color=[0.0, 0.0, 0.0], enable_disturbance=False,
            align_boundary_except_vid=bound_t,
        )
        depth = renderer._to_image_space(rast["rast_out"][..., 2:3])
        return out, depth

    def paint(gt, pred_on_photo, err, albedo, normal, diffuse, aa, cover, lmk):
        panels = [gt, pred_on_photo, err, albedo, normal, diffuse, aa, cover, lmk]
        return np.concatenate([u8(p) for p in panels], axis=1)

    frames = list(range(0, n_frames, 20))
    print(f"[export] frames {frames}", flush=True)
    for t in frames:
        state.ex_lower = tensor(ex_lower[t])
        state.rot_neck = tensor(rot_neck[t])
        state.rot_head = tensor(rot_head[t])
        state.rot_leye = tensor(rot_leye[t])
        state.rot_reye = tensor(rot_reye[t])
        state.translation = tensor(translation[t])
        cur["leye"] = tensor(ex_leye[t])
        cur["reye"] = tensor(ex_reye[t])
        if "expression_tongue" in z.files:
            cur["tongue"] = tensor(z["expression_tongue"][t])
            cur["pupil"] = tensor(z["expression_pupil"][t])
        with torch.no_grad():
            verts = posed().detach().cpu().numpy()
            eye_out, eye_z = raster(eye_ids, eye_tri_uv, eye_tex)
            face_out, face_z = raster(face_ids, tri_uv[face_ids], face_tex)
            eye_on = eye_out["rgba"][..., 3:] > 0.5
            face_on = face_out["rgba"][..., 3:] > 0.5
            front = face_on & (~eye_on | (face_z <= eye_z))
            on = front | eye_on

            def take(key, white):
                face_b = face_out[key][..., :3]
                eye_b = eye_out[key][..., :3]
                bg = torch.ones_like(face_b) if white else torch.zeros_like(face_b)
                return torch.where(front, face_b, torch.where(eye_on, eye_b, bg))[0]

            rgb = take("rgba", False)
            albedo = take("albedo", True).clamp(0, 1)
            normal = (take("normal", False) / 2 + 0.5).clamp(0, 1)
            normal = torch.where(on[0], normal, torch.ones_like(normal))
            diffuse = take("diffuse", True).clamp(0, 1)
            if diffuse.shape[-1] == 3:
                diffuse = diffuse.mean(-1, keepdim=True).expand_as(rgb)
            aa = take("aa", False).clamp(0, 1)
            pred_xy = project((posed()[idx_t] * bary_t.unsqueeze(-1)).sum(1)).detach().cpu().numpy()

        bgr = cv2.imread(str(paths[t]))
        if scale != 1.0:
            bgr = cv2.resize(bgr, (pw, ph), interpolation=cv2.INTER_AREA)
        gt = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb_np = rgb.detach().cpu().numpy()
        on_np = on[0].detach().cpu().numpy().astype(np.float32)
        pred = rgb_np * on_np + gt * (1.0 - on_np)
        err = (gt - pred).mean(-1)
        err = seismic(err / 2.0 + 0.5)
        blue = np.array([0.2, 0.5, 1.0], dtype=np.float32)
        cover = on_np * (0.45 * blue + 0.55 * gt) + (1.0 - on_np) * gt
        lmk = u8(gt)
        dots(lmk, gt_all[t], (0, 170, 0), 3)
        dots(lmk, pred_xy, (220, 30, 30), 2)
        sheet = paint(
            gt, pred, err,
            albedo.detach().cpu().numpy(),
            normal.detach().cpu().numpy(),
            diffuse.detach().cpu().numpy(),
            aa.detach().cpu().numpy(),
            cover,
            lmk.astype(np.float32) / 255.0,
        )
        stem = f"frame_{t:04d}"
        obj = OUT / f"{stem}.obj"
        with obj.open("w") as f:
            f.write(f"# frame {t} posed mesh, meters\n")
            for v in verts:
                f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
            for tri in faces:
                f.write(f"f {int(tri[0])+1} {int(tri[1])+1} {int(tri[2])+1}\n")
        jpg = OUT / f"{stem}.jpg"
        cv2.imwrite(str(jpg), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"[export] {stem} verts {verts.shape[0]} sheet {sheet.shape[1]}x{sheet.shape[0]}", flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
