"""Step 1, mesh frozen at the landmark fit.

Skin: one chromaticity offset each for cheek, brow, and lip. Brightness is SH.
Eyes: SAM3 dark-disk center is the target. The eyeball stays on the head joint
and only rotates, so the pupil projects onto that center. Then the same color
offset for sclera, iris, and pupil.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "MediaPipe_备份"))
sys.path.insert(0, "/root/autodl-tmp/GNM-main")
sys.path.insert(0, "/root/autodl-tmp/sam3-main")
sys.path.insert(0, "/root/autodl-tmp/colmap_video_jamin_static")

from gnm.shape import gnm_numpy, gnm_pytorch
import gnm_mediapipe_landmarks as mp_lmk
from nvdiffrast_render import NVDiffRenderer
from nvdiffrast_render.camera import project_world_to_pixels
from fit_gnm import FitState, ibug68_vhap_weights
import photo_fit

OUT = Path("/root/autodl-tmp/gnm_fit_out/expression_v3/flow_frame0/photo")
PAINTED = Path("/root/autodl-tmp/gnm_fit_out/expression_v3/gnm_uv_from_flame_lmk68.png")
EYE_TEX = Path("/root/autodl-tmp/gnm_fit_out/expression_v3/gnm_eye_uv.png")
EYE_UV = Path("/root/autodl-tmp/gnm_fit_out/expression_v3/gnm_eye_uv_repacked.npz")
PARAMS = Path("/root/autodl-tmp/gnm_fit_out/expression_v3/flow_frame0/frame_000000_params.npz")
LMK = Path("/root/autodl-tmp/gnm_fit_out/expression_v3/facepoints68.npz")
LMK_FILE = Path(__file__).resolve().parents[1] / "MediaPipe_备份" / "head_sparse_68_v3_改.txt"
CKPT = "/root/autodl-tmp/sam3-main/checkpoints/sam3.pt"


def load_base(path: Path) -> torch.Tensor:
    albedo = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    albedo = np.flip(albedo, axis=0).copy()
    return torch.tensor(np.clip(albedo, 0.0, 1.0))


def group(gnm_np, name: str) -> np.ndarray:
    if name not in set(gnm_np.vertex_group_names):
        return np.zeros(0, dtype=np.int64)
    return np.asarray(gnm_np.vertex_group_indices(name), dtype=np.int64)


def raster_mask(faces, tri_uv, vert_ids, n_verts, shape) -> np.ndarray:
    h, w = shape
    flag = np.zeros(n_verts, dtype=np.uint8)
    if vert_ids.size:
        flag[vert_ids] = 1
    use = np.nonzero(flag[faces].all(1))[0]
    img = np.zeros((h, w), np.uint8)
    for fi in use:
        uv = tri_uv[fi]
        if np.linalg.norm(np.roll(uv, -1, 0) - uv, axis=1).max() > 0.25:
            continue
        px = np.round(np.stack([uv[:, 0] * (w - 1), (1.0 - uv[:, 1]) * (h - 1)], 1)).astype(np.int32)
        cv2.fillConvexPoly(img, px, 1)
    return img.astype(np.float32)


def sam_eyeballs(image_path: Path) -> np.ndarray:
    # SAM3 依赖在 base conda（Python 3.12）。光度用 gnm 环境，两边分开跑，先读缓存。
    cache = OUT / "eye_mask.npz"
    if not cache.is_file():
        import subprocess

        subprocess.check_call(["/root/miniconda3/bin/python", "-u", "/tmp/sam_eyes.py"])
    data = np.load(cache)
    print(f"[sam] loaded {cache.name} prompt {str(data['prompt'])} area {int(data['mask'].sum())}", flush=True)
    return data["mask"].astype(bool)


def circle_centers(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    circles = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 40:
            continue
        ys, xs = np.nonzero(labels == i)
        pts = np.stack([xs, ys], 1).astype(np.float32)
        (cx, cy), radius = cv2.minEnclosingCircle(pts)
        circles.append((radius, cx, cy))
    circles.sort(reverse=True)
    if len(circles) < 2:
        raise RuntimeError(f"need two eyeballs, got {len(circles)}")
    return np.array([[circles[0][1], circles[0][2]], [circles[1][1], circles[1][2]]], np.float32)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    params = np.load(PARAMS)
    lmk_npz = np.load(LMK, allow_pickle=True)
    gt_xy = np.asarray(lmk_npz["landmarks"][0], dtype=np.float32)
    width, height = [int(x) for x in lmk_npz["image_size"]]
    image = Path(str(lmk_npz["image_paths"][0]))
    stage_bgr = cv2.imread(str(image))
    ph, pw, scale = photo_fit.scale_for_photo(height, width, 1024)
    bgr = stage_bgr if scale == 1.0 else cv2.resize(stage_bgr, (pw, ph), interpolation=cv2.INTER_AREA)
    rgb_np = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    mask = sam_eyeballs(image)
    if scale != 1.0:
        mask = cv2.resize(mask.astype(np.uint8), (pw, ph), interpolation=cv2.INTER_NEAREST).astype(bool)
    centers = circle_centers(mask)
    print(f"[eye] sam centers {centers.tolist()}", flush=True)

    base_face = load_base(PAINTED)
    base_eye = load_base(EYE_TEX)
    eye_pack = np.load(EYE_UV)
    eye_ids = np.asarray(eye_pack["eye_faces"], dtype=np.int64)
    eye_tri_uv = np.asarray(eye_pack["eye_triangle_uvs"], dtype=np.float32)

    gnm = gnm_pytorch.GNM.from_local(version=gnm_pytorch.GNMMajorVersion.V3, variant=gnm_pytorch.GNMVariant.HEAD)
    gnm_np = gnm_numpy.GNM.from_local(version=gnm_numpy.GNMMajorVersion.V3, variant=gnm_numpy.GNMVariant.HEAD)
    faces = np.asarray(gnm.triangles.detach().cpu().numpy())
    n_verts = int(faces.max()) + 1

    def p(v):
        t = torch.tensor(np.asarray(v), dtype=torch.float32)
        t.requires_grad_(False)
        return t

    rot = np.asarray(params["rotations"], dtype=np.float32)
    state = FitState(
        id_head=p(params["id_head"]), id_eyes=p(params["id_eyes"]),
        ex_lower=p(params["expression_lower_face"]),
        rot_neck=p(rot[0]), rot_head=p(rot[1]), rot_leye=p(rot[2]), rot_reye=p(rot[3]),
        translation=p(params["translation"]), cam_aa=p(params["cam_aa"]),
        cam_t=p(params["cam_t"]), focal_norm=p(params["focal_norm"]),
    )

    def posed():
        return gnm(
            state.identity()[None], state.expression()[None],
            state.rotations()[None], state.translation[None],
        )[0]

    verts0 = posed().detach()
    left_ids = group(gnm_np, "left_eye")
    right_ids = group(gnm_np, "right_eye")
    pupil_all = group(gnm_np, "pupils")
    pupil_sets = [np.intersect1d(pupil_all, ids) for ids in (left_ids, right_ids)]
    # 先在眼珠 UV 上找黑色圆的圆心，再找贴到网格上时占住这个 UV 的外表面顶点。
    tri_uv_eye = eye_tri_uv
    uv_img = cv2.imread(str(EYE_TEX))
    uh, uw = uv_img.shape[:2]
    uv_gray = cv2.cvtColor(uv_img, cv2.COLOR_BGR2GRAY)
    _, uv_labels, uv_stats, _ = cv2.connectedComponentsWithStats((uv_gray <= 40).astype(np.uint8), 8)
    uv_queries = []
    for i in range(1, int(uv_stats.shape[0])):
        if int(uv_stats[i, cv2.CC_STAT_AREA]) < 20:
            continue
        ys, xs = np.nonzero(uv_labels == i)
        (cx, cy), _ = cv2.minEnclosingCircle(np.stack([xs, ys], 1).astype(np.float32))
        uv_queries.append(np.array([cx / (uw - 1), 1.0 - cy / (uh - 1)], np.float32))
    uv_queries.sort(key=lambda q: float(q[0]))
    cam_r = state.camera_rt()[:3, :3]
    cam_t = state.camera_rt()[:3, 3]
    a = tri_uv_eye[:, 0]
    b = tri_uv_eye[:, 1]
    c = tri_uv_eye[:, 2]
    uv_center_ids = []
    for uv_q in uv_queries:
        v0, v1 = b - a, c - a
        v2 = uv_q - a
        den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
        good = np.abs(den) > 1e-12
        inv = np.zeros_like(den)
        inv[good] = 1.0 / den[good]
        bu = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) * inv
        bv = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) * inv
        bw = 1.0 - bu - bv
        inside = np.nonzero(good & (bu >= -1e-4) & (bv >= -1e-4) & (bw >= -1e-4))[0]
        best_vid, best_z = None, -1e9
        for j in inside:
            face = faces[eye_ids[int(j)]]
            bary = np.array([bw[j], bu[j], bv[j]], np.float32)
            vid = int(face[int(np.argmax(bary))])
            if float(np.max(bary)) < 0.5:
                continue
            xyz = verts0[vid]
            cam_z = float((cam_r @ xyz + cam_t)[2])
            if cam_z > best_z:
                best_vid, best_z = vid, cam_z
        uv_center_ids.append(best_vid)
        print(f"[uv] black center {uv_q.tolist()} -> vert {best_vid}", flush=True)
    uv_center_t = torch.tensor(uv_center_ids, dtype=torch.long)

    def project(pts):
        k = state.camera_k(width, height)
        if scale != 1.0:
            k = k * scale
        return project_world_to_pixels(pts, state.camera_rt(), k, pw, ph, flip_y=True)

    def black_points(verts):
        return verts[uv_center_t]

    def photo_black_centers(image_bgr, seeds):
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        found, radii = [], []
        for sx, sy in seeds:
            x0, y0 = max(int(sx) - 36, 0), max(int(sy) - 28, 0)
            x1, y1 = min(int(sx) + 36, gray.shape[1]), min(int(sy) + 28, gray.shape[0])
            crop = gray[y0:y1, x0:x1]
            # 反光点会把黑圆咬掉一块，先合上再取外接圆，圆心才是黑圆的圆心。
            mask = (crop <= 58).astype(np.uint8) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
            best = None
            for i in range(1, n):
                if int(stats[i, cv2.CC_STAT_AREA]) < 15:
                    continue
                ys, xs = np.nonzero(labels == i)
                pts = np.stack([xs, ys], 1).astype(np.float32)
                (cx, cy), radius = cv2.minEnclosingCircle(pts)
                dist = (xs.mean() + x0 - sx) ** 2 + (ys.mean() + y0 - sy) ** 2
                rec = (dist, radius, cx + x0, cy + y0)
                if best is None or rec[0] < best[0]:
                    best = rec
            if best is None:
                raise RuntimeError(f"no black pupil near {(sx, sy)}")
            found.append([best[2], best[3]])
            radii.append(best[1])
        return np.asarray(found, np.float32), np.asarray(radii, np.float32)

    before_xy = project(black_points(verts0)).detach()
    # 照片目标用最初 SAM 圆的圆心，就是那张图里蓝色小点的位置。
    order = np.argsort(before_xy[:, 0].numpy())
    tgt_order = np.argsort(centers[:, 0])
    target = np.zeros_like(centers)
    target[order] = centers[tgt_order]
    target_t = torch.tensor(target)
    print(f"[eye] photo target {target.tolist()}", flush=True)
    # 眼珠蒙在头关节上，只转轴角，不平移顶点。
    state.rot_leye.requires_grad_(True)
    state.rot_reye.requires_grad_(True)
    rot_opt = torch.optim.Adam([state.rot_leye, state.rot_reye], lr=5e-2)
    for i in range(80):
        rot_opt.zero_grad()
        pred = project(black_points(posed()))
        loss = (pred - target_t).pow(2).sum()
        loss = loss + 1e-2 * (state.rot_leye.pow(2).sum() + state.rot_reye.pow(2).sum())
        loss.backward()
        rot_opt.step()
    state.rot_leye.requires_grad_(False)
    state.rot_reye.requires_grad_(False)
    after_xy = project(black_points(posed().detach())).detach()
    for name, rot, xy0, xy1, tgt in zip(
        ("left", "right"),
        (state.rot_leye, state.rot_reye),
        before_xy, after_xy, target,
    ):
        deg = float(rot.norm()) * 180.0 / np.pi
        err = float(torch.linalg.norm(xy1 - torch.tensor(tgt)))
        print(
            f"[eye] {name} uv-black {xy0.tolist()} -> {xy1.tolist()}  "
            f"target {tgt.tolist()}  rot {deg:.1f} deg  err {err:.2f}px",
            flush=True,
        )

    dbg = cv2.cvtColor((rgb_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    for x, y in target:
        cv2.circle(dbg, (int(x), int(y)), 8, (0, 255, 0), 2)
    for x, y in before_xy.numpy():
        cv2.circle(dbg, (int(x), int(y)), 4, (0, 0, 255), -1)
    for x, y in after_xy.numpy():
        cv2.circle(dbg, (int(x), int(y)), 3, (255, 0, 0), -1)
    cv2.imwrite(str(OUT / "eye_centers.jpg"), dbg)

    h, w = base_face.shape[:2]
    tri_uv = np.asarray(gnm_np.triangle_uvs, dtype=np.float32)
    brow = np.unique(np.concatenate([group(gnm_np, n) for n in (
        "left_brow_region", "right_brow_region", "middle_brow_region")]))
    lip = np.unique(np.concatenate([group(gnm_np, "upper_lip"), group(gnm_np, "lower_lip")]))
    cheek = group(gnm_np, "hockey_mask")
    mask_lip = raster_mask(faces, tri_uv, lip, n_verts, (h, w))
    mask_brow = raster_mask(faces, tri_uv, brow, n_verts, (h, w))
    mask_cheek = raster_mask(faces, tri_uv, cheek, n_verts, (h, w))
    mask_brow = mask_brow * (1.0 - mask_lip)
    mask_cheek = mask_cheek * (1.0 - np.clip(mask_lip + mask_brow, 0, 1))
    eh, ew = base_eye.shape[:2]
    mask_sclera = raster_mask(faces[eye_ids], eye_tri_uv, group(gnm_np, "scleras"), n_verts, (eh, ew))
    mask_iris = raster_mask(faces[eye_ids], eye_tri_uv, group(gnm_np, "irises"), n_verts, (eh, ew))
    mask_pupil = raster_mask(faces[eye_ids], eye_tri_uv, group(gnm_np, "pupils"), n_verts, (eh, ew))
    print(
        f"[mask] cheek {mask_cheek.mean():.3f} brow {mask_brow.mean():.3f} lip {mask_lip.mean():.3f} "
        f"sclera {mask_sclera.mean():.3f} iris {mask_iris.mean():.3f} pupil {mask_pupil.mean():.3f}",
        flush=True,
    )

    colors = torch.nn.Parameter(torch.zeros(6, 3))
    face_masks = [torch.tensor(m) for m in (mask_cheek, mask_brow, mask_lip)]
    eye_masks = [torch.tensor(m) for m in (mask_sclera, mask_iris, mask_pupil)]
    device = torch.device("cuda")
    gt_rgb = torch.as_tensor(rgb_np, dtype=torch.float32, device=device)[None]
    regions = photo_fit.build_photo_regions(gnm_np, faces)
    color_lut = torch.zeros(faces.shape[0] + 1, dtype=torch.bool, device=device)
    color_except = torch.as_tensor(regions["color_except"], dtype=torch.long, device=device)
    color_lut[color_except + 1] = True
    face_ids = np.setdiff1d(np.arange(len(faces), dtype=np.int64), eye_ids)
    lights = torch.nn.Parameter(torch.zeros(1, 9, 3))
    renderer = NVDiffRenderer(
        use_opengl=False, lighting_type="SH", lighting_space="world",
        disturb_rate_fg=0.0, disturb_rate_bg=0.0,
        fid2cid=torch.as_tensor(regions["fid2cid"], dtype=torch.long),
        shade_smooth=True, flip_y_raster=False,
    ).to(device)
    with torch.no_grad():
        lights[:, 0, :] = 1.0 / renderer.sh_const[0].detach().cpu()

    def world_verts():
        return posed()

    def apply_colors(base, masks, offset):
        add = torch.zeros_like(base)
        for m, c in zip(masks, offset):
            add = add + m[..., None] * c
        return (base + add).clamp(1e-4, 1.0 - 1e-4)

    def render_pass(part_ids, part_uv, tex):
        part_faces = torch.as_tensor(faces[part_ids], dtype=torch.int32, device=device)
        verts_uv_t = torch.as_tensor(part_uv.reshape(-1, 2), dtype=torch.float32, device=device)
        faces_uv_t = torch.arange(len(part_ids) * 3, device=device, dtype=torch.int32).reshape(-1, 3)
        verts = world_verts().to(device)[None].contiguous()
        rt = state.camera_rt().to(device)[None]
        k4 = state.camera_k(width, height).detach()
        if scale != 1.0:
            k4 = k4 * scale
        rast = renderer.rasterize(
            verts, part_faces, rt, k4.to(device)[None], (ph, pw), use_cache=False, require_grad=True,
        )
        tex_nchw = tex.permute(2, 0, 1)[None].to(device)
        out = renderer.render_rgba(
            rast, verts[0], part_faces, verts_uv_t, faces_uv_t, tex_nchw, lights.to(device),
            background_color=[1.0, 1.0, 1.0], enable_disturbance=False,
        )
        depth = renderer._to_image_space(rast["rast_out"][..., 2:3])
        fid = renderer._to_image_space(rast["rast_out"][..., 3:4].long())
        return out, depth, fid

    def composite(eye_out, eye_z, face_out, face_z, face_fid):
        eye_on = eye_out["rgba"][..., 3:] > 0.5
        face_on = face_out["rgba"][..., 3:] > 0.5
        face_front = face_on & (~eye_on | (face_z <= eye_z))
        rgb = torch.where(
            face_front, face_out["rgba"][..., :3],
            torch.where(eye_on, eye_out["rgba"][..., :3], torch.ones_like(face_out["rgba"][..., :3])),
        )
        alpha = (face_front | eye_on).float()
        ids = torch.as_tensor(face_ids, dtype=torch.long, device=device)
        flat = face_fid.reshape(-1).clamp(min=0)
        glob = torch.zeros_like(flat)
        m = flat > 0
        glob[m] = ids[flat[m] - 1] + 1
        return torch.cat([rgb, alpha], -1), glob.view_as(face_fid), face_out["diffuse_detach_normal"]

    def energy():
        face_al = apply_colors(base_face, face_masks, colors[:3])
        eye_al = apply_colors(base_eye, eye_masks, colors[3:])
        eye_out, eye_z, _ = render_pass(eye_ids, eye_tri_uv, eye_al)
        face_out, face_z, face_fid = render_pass(face_ids, tri_uv[face_ids], face_al)
        rgba, global_fid, diffuse = composite(eye_out, eye_z, face_out, face_z, face_fid)
        vis = ((rgba[..., 3:] > 0) & ~color_lut[global_fid.clamp(0, color_lut.shape[0] - 1)]).detach()
        pred = rgba[..., :3]
        pred_c = pred / pred.sum(-1, keepdim=True).clamp(min=1e-3)
        gt_c = gt_rgb / gt_rgb.sum(-1, keepdim=True).clamp(min=1e-3)
        chroma = ((pred_c - gt_c).abs() * vis).sum() / (vis.sum() * 3).clamp(min=1.0)
        luma = ((pred.mean(-1, keepdim=True) - gt_rgb.mean(-1, keepdim=True)).abs() * vis).sum()
        luma = luma / vis.sum().clamp(min=1.0)
        rgb_loss = ((pred - gt_rgb).abs() * vis).sum() / (vis.sum() * 3).clamp(min=1.0)
        reg_diffuse = torch.relu(diffuse.max() - 1.0) + diffuse.var()
        loss = 30.0 * chroma + 15.0 * luma + 10.0 * reg_diffuse + 0.01 * (lights[:, 1:] ** 2).mean()
        loss = loss + 1.0 * colors.pow(2).mean()
        return loss, rgb_loss, rgba

    opt = torch.optim.Adam([
        {"params": [colors], "lr": 2e-2},
        {"params": [lights], "lr": 5e-3},
    ])
    for g in opt.param_groups:
        g["base_lr"] = g["lr"]
    best = (colors.detach().clone(), lights.detach().clone())
    best_score = float("inf")
    best_i = 0
    steps = 200
    for i in range(steps):
        tcos = i / max(steps - 1, 1)
        lr_scale = 0.15 + 0.85 * (0.5 * (1.0 + np.cos(np.pi * tcos)))
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * lr_scale
        opt.zero_grad()
        loss, rgb_loss, _ = energy()
        loss.backward()
        if colors.grad is not None:
            g = colors.grad
            colors.grad = g - g.mean(dim=-1, keepdim=True)
        torch.nn.utils.clip_grad_norm_([colors, lights], 1.0)
        score = float(loss.detach())
        if score < best_score:
            best_score = score
            best = (colors.detach().clone(), lights.detach().clone())
            best_i = i + 1
        opt.step()
        if i == 0 or (i + 1) % 40 == 0 or i + 1 == steps:
            print(
                f"[chroma {i+1:3d}/{steps}] rgb={float(rgb_loss.detach()):.4f} "
                f"|c|={float(colors.detach().abs().mean()):.3f} best={best_score:.3f}@{best_i}",
                flush=True,
            )
    with torch.no_grad():
        colors.copy_(best[0])
        lights.copy_(best[1])
        colors[3:].zero_()

    # 位置已经对上。眼珠只改巩膜、虹膜、瞳孔三块的颜色，圆的形状不动，明暗也让它学。
    eye_col = torch.nn.Parameter(torch.zeros(3, 3))

    def eye_albedo():
        return apply_colors(base_eye, eye_masks, eye_col)

    def eye_energy():
        face_al = apply_colors(base_face, face_masks, colors[:3])
        eye_out, eye_z, _ = render_pass(eye_ids, eye_tri_uv, eye_albedo())
        face_out, face_z, face_fid = render_pass(face_ids, tri_uv[face_ids], face_al)
        rgba, _, _ = composite(eye_out, eye_z, face_out, face_z, face_fid)
        eye_on = eye_out["rgba"][..., 3:] > 0.5
        face_on = face_out["rgba"][..., 3:] > 0.5
        eye_vis = (eye_on & ~(face_on & (face_z <= eye_z))).detach()
        pred = rgba[..., :3]
        rgb_loss = ((pred - gt_rgb).abs() * eye_vis).sum() / (eye_vis.sum() * 3).clamp(min=1.0)
        loss = rgb_loss + 0.02 * eye_col.pow(2).mean()
        return loss, rgb_loss, rgba

    eye_opt = torch.optim.Adam([eye_col], lr=5e-2)
    best_eye = eye_col.detach().clone()
    best_eye_score = float("inf")
    eye_steps = 150
    for i in range(eye_steps):
        eye_opt.zero_grad()
        loss, rgb_loss, _ = eye_energy()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([eye_col], 1.0)
        if float(loss.detach()) < best_eye_score:
            best_eye_score = float(loss.detach())
            best_eye = eye_col.detach().clone()
        eye_opt.step()
        if i == 0 or (i + 1) % 50 == 0 or i + 1 == eye_steps:
            print(
                f"[eye-tex {i+1:3d}/{eye_steps}] eye_rgb={float(rgb_loss.detach()):.4f} "
                f"|c|={float(eye_col.detach().abs().mean()):.3f}",
                flush=True,
            )
    with torch.no_grad():
        eye_col.copy_(best_eye)
        _, rgb_loss, rgba = eye_energy()
    eye_png = (eye_albedo().detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    eye_png = cv2.cvtColor(np.flip(eye_png, 0), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(OUT / "eye_albedo.png"), eye_png)
    photo_fit.dump_photo_preview(OUT, "tex", rgb_np, rgba[0].detach().float().cpu().numpy())
    names = ("cheek", "brow", "lip", "sclera", "iris", "pupil")
    for name, c in zip(names, colors.detach().cpu().numpy()):
        print(f"[color] {name} {np.round(c, 3).tolist()}", flush=True)
    albedo_np = eye_albedo().detach().cpu().numpy()
    for name, m in (("sclera", mask_sclera), ("iris", mask_iris), ("pupil", mask_pupil)):
        sel = m > 0.5
        mean = albedo_np[sel].mean(0) if sel.any() else np.zeros(3)
        print(f"[eye-tex] {name} {np.round(mean, 3).tolist()}", flush=True)
    print(f"eye rgb={float(rgb_loss):.4f}", flush=True)

    # 二层：身份、姿态、表情、分区颜色和光照一起微调。没有静态偏移。
    # 下颌轮廓权重为 0，鼻子 ×10。眼珠圆心用重投影拴在 SAM 蓝点上。
    for name in (
        "id_head", "id_eyes", "ex_lower", "rot_neck", "rot_head",
        "rot_leye", "rot_reye", "translation", "cam_aa", "cam_t",
    ):
        getattr(state, name).requires_grad_(True)
    lmk_idx, lmk_bary = mp_lmk.load_landmarks(LMK_FILE)
    idx_t = torch.as_tensor(lmk_idx, dtype=torch.long)
    bary_t = torch.as_tensor(lmk_bary, dtype=torch.float32)
    w_lmk = ibug68_vhap_weights(nose_boost=10.0)
    w_lmk[:17] = 0.0
    w_lmk_t = torch.as_tensor(w_lmk, dtype=torch.float32)
    gt_lmk = torch.as_tensor(gt_xy, dtype=torch.float32)

    def landmark_xy(verts):
        pts = (verts[idx_t] * bary_t.unsqueeze(-1)).sum(1)
        return project(pts)

    def report_lmk(tag):
        with torch.no_grad():
            err = (landmark_xy(posed()) - gt_lmk).norm(dim=-1).cpu().numpy()
        active = w_lmk > 0
        print(
            f"[lmk {tag}] all {err.mean():.3f}px  nose {err[27:36].mean():.3f}  "
            f"jaw {err[:17].mean():.3f}  active {err[active].mean():.3f}",
            flush=True,
        )

    def joint_energy():
        face_al = apply_colors(base_face, face_masks, colors[:3])
        eye_out, eye_z, _ = render_pass(eye_ids, eye_tri_uv, eye_albedo())
        face_out, face_z, face_fid = render_pass(face_ids, tri_uv[face_ids], face_al)
        rgba, global_fid, diffuse = composite(eye_out, eye_z, face_out, face_z, face_fid)
        vis = ((rgba[..., 3:] > 0) & ~color_lut[global_fid.clamp(0, color_lut.shape[0] - 1)]).detach()
        pred = rgba[..., :3]
        pred_c = pred / pred.sum(-1, keepdim=True).clamp(min=1e-3)
        gt_c = gt_rgb / gt_rgb.sum(-1, keepdim=True).clamp(min=1e-3)
        chroma = ((pred_c - gt_c).abs() * vis).sum() / (vis.sum() * 3).clamp(min=1.0)
        luma = ((pred.mean(-1, keepdim=True) - gt_rgb.mean(-1, keepdim=True)).abs() * vis).sum()
        luma = luma / vis.sum().clamp(min=1.0)
        rgb_loss = ((pred - gt_rgb).abs() * vis).sum() / (vis.sum() * 3).clamp(min=1.0)
        reg_diffuse = torch.relu(diffuse.max() - 1.0) + diffuse.var()
        verts = posed()
        lmk = ((landmark_xy(verts) - gt_lmk).abs().sum(-1) * w_lmk_t).sum() / w_lmk_t.sum()
        eye_err = (project(black_points(verts)) - target_t).abs().mean()
        reg_id = state.id_head.pow(2).mean() + state.id_eyes.pow(2).mean()
        reg_ex = state.ex_lower.pow(2).mean()
        loss = (
            30.0 * chroma + 15.0 * luma + 10.0 * reg_diffuse
            + 0.01 * (lights[:, 1:] ** 2).mean()
            + 1.0 * colors[:3].pow(2).mean() + 0.02 * eye_col.pow(2).mean()
            + 2.0 * lmk.to(chroma.device) + 4.0 * eye_err.to(chroma.device)
            + 0.02 * reg_id.to(chroma.device) + 0.02 * reg_ex.to(chroma.device)
        )
        return loss, rgb_loss, lmk, eye_err, rgba

    report_lmk("before-joint")
    joint_params = [
        state.id_head, state.id_eyes, state.ex_lower,
        state.rot_neck, state.rot_head, state.rot_leye, state.rot_reye,
        state.translation, state.cam_aa, state.cam_t,
        colors, lights, eye_col,
    ]
    joint_opt = torch.optim.Adam([
        {"params": [state.id_head, state.id_eyes], "lr": 1e-3},
        {"params": [state.ex_lower], "lr": 5e-3},
        {"params": [state.rot_neck, state.rot_head], "lr": 1e-3},
        {"params": [state.rot_leye, state.rot_reye], "lr": 5e-4},
        {"params": [state.translation, state.cam_t], "lr": 2e-4},
        {"params": [state.cam_aa], "lr": 5e-4},
        {"params": [colors], "lr": 5e-3},
        {"params": [eye_col], "lr": 1e-2},
        {"params": [lights], "lr": 2e-3},
    ])
    best_snap = state.snapshot()
    best_tex = (colors.detach().clone(), lights.detach().clone(), eye_col.detach().clone())
    best_joint = float("inf")
    joint_steps = 120
    for i in range(joint_steps):
        joint_opt.zero_grad()
        loss, rgb_loss, lmk, eye_err, _ = joint_energy()
        loss.backward()
        if colors.grad is not None:
            g = colors.grad
            colors.grad = g - g.mean(dim=-1, keepdim=True)
        torch.nn.utils.clip_grad_norm_(joint_params, 0.5)
        score = float(loss.detach())
        if score < best_joint:
            best_joint = score
            best_snap = state.snapshot()
            best_tex = (colors.detach().clone(), lights.detach().clone(), eye_col.detach().clone())
        joint_opt.step()
        if i == 0 or (i + 1) % 30 == 0 or i + 1 == joint_steps:
            print(
                f"[joint {i+1:3d}/{joint_steps}] rgb={float(rgb_loss.detach()):.4f} "
                f"lmk={float(lmk.detach()):.3f} eye={float(eye_err.detach()):.3f} "
                f"best={best_joint:.3f}",
                flush=True,
            )
    state.restore(best_snap)
    with torch.no_grad():
        colors.copy_(best_tex[0])
        lights.copy_(best_tex[1])
        eye_col.copy_(best_tex[2])
        colors[3:].zero_()
        _, rgb_loss, lmk, eye_err, rgba = joint_energy()
    report_lmk("after-joint")
    photo_fit.dump_photo_preview(OUT, "joint", rgb_np, rgba[0].detach().float().cpu().numpy())
    print(
        f"joint rgb={float(rgb_loss):.4f} lmk={float(lmk):.3f} eye={float(eye_err):.3f}",
        flush=True,
    )
    np.savez(
        OUT / "photo_appearance.npz",
        colors=colors.detach().cpu().numpy(),
        lights=lights.detach().cpu().numpy(),
        eye_col=eye_col.detach().cpu().numpy(),
        eye_rot=torch.stack([state.rot_leye, state.rot_reye]).detach().numpy(),
        id_head=state.id_head.detach().numpy(),
        id_eyes=state.id_eyes.detach().numpy(),
        expression_lower_face=state.ex_lower.detach().numpy(),
        rotations=state.rotations().detach().numpy(),
        translation=state.translation.detach().numpy(),
        cam_aa=state.cam_aa.detach().numpy(),
        cam_t=state.cam_t.detach().numpy(),
        focal_norm=state.focal_norm.detach().numpy(),
        sam_centers=centers,
        target_centers=target,
    )
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
