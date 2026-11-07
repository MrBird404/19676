"""Pull the neck with the face contour only.

No 68-point term. The neck stays in the picture. Its offset is one smooth
falloff times three amplitudes, so neighboring vertices cannot sawtooth.
Eyes, ears, teeth, and the cap stay pinned.
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
sys.path.insert(0, "/root/autodl-tmp/GNM-main")

from gnm.shape import gnm_numpy, gnm_pytorch
from nvdiffrast_render import NVDiffRenderer
from fit_gnm import FitState
import photo_fit
from flow_photo import group, load_base, raster_mask
from uvmask import tensor_mask

OUT = Path("/root/autodl-tmp/gnm_fit_out/expression_v3/flow_frame0/photo_contour_only")
_UV = Path(__file__).resolve().parent / "uv"
PAINTED = _UV / "gnm_uv_from_flame_lmk68.png"
EYE_TEX = _UV / "gnm_eye_uv.png"
EYE_UV = _UV / "gnm_eye_uv_repacked.npz"
APPEAR = Path("/root/autodl-tmp/gnm_fit_out/expression_v3/flow_frame0/photo_contour/photo_appearance.npz")
LMK = Path("/root/autodl-tmp/gnm_fit_out/expression_v3/facepoints68.npz")
MASK = Path(__file__).resolve().parent / "face_mask.npz"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    saved = np.load(APPEAR)
    lmk_npz = np.load(LMK, allow_pickle=True)
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
    rest = np.asarray(gnm_np(None, None, None, None), dtype=np.float32)
    regions = photo_fit.build_photo_regions(gnm_np, faces)
    edges_np = np.asarray(regions["edges"], dtype=np.int64)

    skin = group(gnm_np, "skin")
    ears = group(gnm_np, "ears")
    eyes_v = group(gnm_np, "eyes")
    teeth = np.unique(np.concatenate([
        group(gnm_np, "teeth"), group(gnm_np, "gums"), group(gnm_np, "tongue"),
    ]))
    occluded = np.zeros(n_verts, dtype=bool)
    occluded[np.asarray(regions["pin"], dtype=np.int64)] = True
    bottom = np.zeros(n_verts, dtype=bool)
    # pin includes the neck cut. Those vertices are allowed to lift.
    boundary_bottom = rest[:, 1] < 0.12
    occluded[boundary_bottom] = False
    occluded[ears] = True
    occluded[eyes_v] = True
    occluded[teeth] = True

    free = np.zeros(n_verts, dtype=bool)
    free[skin] = True
    free[occluded] = False
    free &= rest[:, 1] < 0.24
    free &= rest[:, 2] > -0.01
    # Distance along the mesh to the nearest pinned vertex, so the tuck
    # fades out before it meets the ears, the jaw, or the cap.
    adj = [[] for _ in range(n_verts)]
    for i, j in edges_np:
        adj[int(i)].append(int(j))
        adj[int(j)].append(int(i))
    dist = np.full(n_verts, np.inf, dtype=np.float64)
    heap = []
    for i in np.nonzero(occluded)[0]:
        dist[i] = 0.0
        heap.append((0.0, int(i)))
    import heapq
    heapq.heapify(heap)
    while heap:
        d, i = heapq.heappop(heap)
        if d > dist[i]:
            continue
        for j in adj[i]:
            nd = d + float(np.linalg.norm(rest[i] - rest[j]))
            if nd < dist[j]:
                dist[j] = nd
                heapq.heappush(heap, (nd, j))
    y = rest[:, 1]
    y_ramp = np.clip((0.22 - y) / (0.22 - 0.09), 0.0, 1.0)
    d_ramp = np.clip((np.nan_to_num(dist, nan=0.0, posinf=0.0) - 0.012) / (0.050 - 0.012), 0.0, 1.0)
    # No hard mask. A hard cut next to a moving vertex is what made the edge jump.
    fall = (y_ramp * d_ramp).astype(np.float32)
    fall[occluded] = 0.0
    # Diffuse the weight so a long tuck does not become a step at one edge.
    acc = np.zeros(n_verts, dtype=np.float64)
    cnt = np.zeros(n_verts, dtype=np.float64)
    for i, j in edges_np:
        acc[i] += fall[j]
        acc[j] += fall[i]
        cnt[i] += 1
        cnt[j] += 1
    for _ in range(30):
        sm = acc / np.maximum(cnt, 1.0)
        fall = (0.35 * fall + 0.65 * sm).astype(np.float32)
        fall[occluded] = 0.0
        acc[:] = 0.0
        for i, j in edges_np:
            acc[i] += fall[j]
            acc[j] += fall[i]
    peak = float(fall.max())
    if peak > 1e-6:
        fall /= peak
        fall[occluded] = 0.0
    df = np.abs(fall[edges_np[:, 0]] - fall[edges_np[:, 1]])
    steep = float(df.max())
    print(
        f"[neck] free {int((fall > 0.05).sum())} fall>0.2 {int((fall > 0.2).sum())} "
        f"steep {steep:.4f} mm_at_1mm_edge {1.0/max(steep, 1e-6):.1f}",
        flush=True,
    )

    base_off = np.asarray(saved["offset"], dtype=np.float32).copy()
    base_off[occluded] = 0
    fall_t = torch.tensor(fall)
    rest_x = torch.tensor(rest[:, 0])
    x_scale = float(np.abs(rest[fall > 0, 0]).max()) if (fall > 0).any() else 1.0
    amp = torch.nn.Parameter(torch.zeros(3))

    def neck_offset():
        # Three inward amplitudes, in meters: narrow, lift, push back.
        mag = amp.clamp(min=0)
        f = fall_t
        dx = -mag[0] * (rest_x / x_scale) * f
        dy = mag[1] * f
        dz = -mag[2] * f
        return torch.stack([dx, dy, dz], dim=-1)

    keep_t = torch.tensor(~occluded, dtype=torch.float32)[:, None]

    def total_offset():
        return (torch.tensor(base_off) + neck_offset()) * keep_t

    def edge_mm(off):
        jump = torch.linalg.norm(off[edges_np[:, 0]] - off[edges_np[:, 1]], dim=-1)
        return float(jump.max()) * 1e3

    with torch.no_grad():
        unit = torch.zeros(3)
        for axis, name in ((1, "up10mm"), (2, "back10mm"), (0, "narrow10mm")):
            trial = unit.clone()
            trial[axis] = 0.010 if axis else 0.010
            # temporary
        print(f"[neck] base edge {edge_mm(torch.tensor(base_off)):.3f}mm", flush=True)

    def p(v):
        return torch.tensor(np.asarray(v), dtype=torch.float32)

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
    res = torch.tensor(np.asarray(saved["residual"]), dtype=torch.float32)
    lights = torch.tensor(saved["lights"], dtype=torch.float32)
    eye_col = torch.tensor(saved["eye_col"], dtype=torch.float32)

    h, w = base_face.shape[:2]
    eh, ew = base_eye.shape[:2]
    eye_masks = [torch.tensor(raster_mask(
        faces[eye_ids], eye_tri_uv, group(gnm_np, name), n_verts, (eh, ew),
    )) for name in ("scleras", "irises", "pupils")]

    device = torch.device("cuda")
    full_cid = np.asarray(regions["fid2cid"], dtype=np.int64)
    neck_flag = fall > 1e-4
    neck_ids = np.nonzero(neck_flag[faces].all(1))[0]
    neck_ids = np.setdiff1d(neck_ids, eye_ids)
    face_ids = np.setdiff1d(np.arange(len(faces), dtype=np.int64), eye_ids)
    print(f"[neck] neck faces {len(neck_ids)}", flush=True)

    contour = np.load(MASK)["lower_contour_smooth"]
    xs = np.arange(pw, dtype=np.float32)
    order = np.argsort(contour[:, 0])
    jaw_y = np.interp(
        xs, contour[order, 0].astype(np.float32), contour[order, 1].astype(np.float32),
        left=np.nan, right=np.nan,
    )
    x0 = int(np.floor(contour[:, 0].min()))
    x1 = int(np.ceil(contour[:, 0].max()))
    forbidden = np.zeros((ph, pw), dtype=bool)
    y_left = float(jaw_y[x0])
    y_right = float(jaw_y[x1])
    for x in range(pw):
        if x0 <= x <= x1:
            yb = jaw_y[x]
        elif x0 - 80 <= x < x0:
            yb = y_left
        elif x1 < x <= x1 + 80:
            yb = y_right
        else:
            continue
        yb = int(np.ceil(yb))
        if yb < ph:
            forbidden[yb:, x] = True
    forb_t = torch.tensor(forbidden, device=device)
    print(f"[neck] forbidden px {int(forbidden.sum())}", flush=True)

    renderer = NVDiffRenderer(
        use_opengl=False, lighting_type="SH", lighting_space="world",
        disturb_rate_fg=0.0, disturb_rate_bg=0.0,
        fid2cid=torch.as_tensor(full_cid, dtype=torch.long),
        shade_smooth=True, flip_y_raster=False,
    ).to(device)

    def set_cid(part_ids):
        cid = torch.as_tensor(full_cid[part_ids], dtype=torch.long, device=device)
        renderer.fid2cid = F.pad(cid, (1, 0), value=0)

    def posed():
        identity = state.identity()[None]
        expression = state.expression()[None]
        rotations = state.rotations()[None]
        translation = state.translation[None]
        verts = gnm.vertex_positions_bind_pose(identity, expression)
        joints = gnm.joint_positions_bind_pose(identity)
        verts = verts + gnm.compute_pose_correctives(rotations)
        verts = verts + total_offset()[None].to(verts.device)
        return gnm.vertex_positions_world(verts, joints, rotations, translation)[0]

    def face_albedo():
        up = F.interpolate(res, size=(h, w), mode="bilinear", align_corners=True)[0].permute(1, 2, 0)
        return base_face + up * 0  # residual is applied below with the face mask

    # Keep the saved residual exactly. face_keep is rebuilt the same way as the contour run.
    lip_ids = np.unique(np.concatenate([
        group(gnm_np, "upper_lip"), group(gnm_np, "lower_lip"),
    ]))
    brow = np.unique(np.concatenate([group(gnm_np, n) for n in (
        "left_brow_region", "right_brow_region", "middle_brow_region")]))
    keep_verts = np.unique(np.concatenate([group(gnm_np, "hockey_mask"), brow, lip_ids]))
    face_keep_t = torch.tensor(tensor_mask(faces, tri_uv, keep_verts, n_verts, (h, w)))

    def albedo_face():
        up = F.interpolate(res, size=(h, w), mode="bilinear", align_corners=True)[0].permute(1, 2, 0)
        return (base_face + up * face_keep_t[..., None]).clamp(1e-4, 1.0 - 1e-4)

    def albedo_eye():
        add = torch.zeros_like(base_eye)
        for m, c in zip(eye_masks, eye_col):
            add = add + m[..., None] * c
        return (base_eye + add).clamp(1e-4, 1.0 - 1e-4)

    def raster(part_ids, part_uv, tex, detach_vid):
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
            background_color=[1.0, 1.0, 1.0], enable_disturbance=False,
            align_boundary_except_vid=detach_vid,
        )
        return out["rgba"], verts

    bound_t = torch.as_tensor(regions["bound_except"], dtype=torch.long, device=device)

    def neck_alpha():
        rgba, _ = raster(neck_ids, tri_uv[neck_ids], albedo_face(), None)
        return rgba[..., 3]

    def beauty():
        eye_rgba, eye_z_pack = raster(eye_ids, eye_tri_uv, albedo_eye(), bound_t)
        # depth is inside rgba path only; recompute composite with a second return
        return eye_rgba

    # Composite properly, including depth. Rewritten here so the saved picture
    # still has the neck.
    def beauty_rgba():
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
                verts, part_faces, rt, k4.to(device)[None], (ph, pw), use_cache=False, require_grad=False,
            )
            out = renderer.render_rgba(
                rast, verts[0], part_faces, verts_uv_t, faces_uv_t,
                tex.permute(2, 0, 1)[None].to(device), lights.to(device),
                background_color=[1.0, 1.0, 1.0], enable_disturbance=False,
                align_boundary_except_vid=bound_t,
            )
            depth = renderer._to_image_space(rast["rast_out"][..., 2:3])
            return out["rgba"], depth

        eye_out, eye_z = one(eye_ids, eye_tri_uv, albedo_eye())
        face_out, face_z = one(face_ids, tri_uv[face_ids], albedo_face())
        eye_on = eye_out[..., 3:] > 0.5
        face_on = face_out[..., 3:] > 0.5
        face_front = face_on & (~eye_on | (face_z <= eye_z))
        rgb = torch.where(
            face_front, face_out[..., :3],
            torch.where(eye_on, eye_out[..., :3], torch.ones_like(face_out[..., :3])),
        )
        alpha = (face_front | eye_on).float()
        return torch.cat([rgb, alpha], -1)

    def hang_of(alpha):
        return (alpha * forb_t[None]).sum() / forb_t.sum().clamp(min=1)

    opt = torch.optim.Adam([amp], lr=5.0e-4)
    with torch.no_grad():
        photo_fit.dump_photo_preview(OUT, "before", rgb_np, beauty_rgba()[0].detach().float().cpu().numpy())
        h0 = float(hang_of(neck_alpha()))
        print(f"[before] hang={h0:.4f} amp={amp.detach().tolist()} edge={edge_mm(total_offset()):.3f}mm", flush=True)
    best = amp.detach().clone()
    best_hang = h0

    def field_edge_mm():
        field = neck_offset()
        jump = torch.linalg.norm(field[edges_np[:, 0]] - field[edges_np[:, 1]], dim=-1)
        return float(jump.max()) * 1e3

    with torch.no_grad():
        budget = 0.00090 / max(steep, 1e-6) / np.sqrt(3.0)
        print(f"[search] budget {budget*1e3:.1f}mm  lift narrow back -> hang field_edge", flush=True)
        levels = (0.0, 0.35 * budget, 0.7 * budget, budget)
        for lift in levels:
            for narrow in levels:
                for back in levels:
                    amp.copy_(torch.tensor([narrow, lift, back], dtype=torch.float32))
                    em = field_edge_mm()
                    if em >= 1.0:
                        print(
                            f"  {lift*1e3:.0f} {narrow*1e3:.0f} {back*1e3:.0f} "
                            f"edge {em:.2f} skip",
                            flush=True,
                        )
                        continue
                    hv = float(hang_of(neck_alpha()))
                    print(
                        f"  {lift*1e3:.0f} {narrow*1e3:.0f} {back*1e3:.0f} "
                        f"hang {hv:.4f} edge {em:.3f}",
                        flush=True,
                    )
                    if hv < best_hang:
                        best = amp.detach().clone()
                        best_hang = hv
        amp.copy_(best)
        print(
            f"[search] picked amp_mm={[round(float(v)*1e3, 2) for v in amp]} "
            f"hang={best_hang:.4f}",
            flush=True,
        )
    steps = 80
    for i in range(steps):
        opt.zero_grad()
        alpha = neck_alpha()
        hang = hang_of(alpha)
        # 20 mm on any axis costs about the same as a leftover hang of 0.05.
        loss = 40.0 * hang + 80.0 * amp.pow(2).sum().to(hang.device)
        loss.backward()
        opt.step()
        with torch.no_grad():
            amp.clamp_(min=0)
            field = neck_offset()
            jump = torch.linalg.norm(field[edges_np[:, 0]] - field[edges_np[:, 1]], dim=-1)
            em = float(jump.max()) * 1e3
            total_em = edge_mm(total_offset())
            hv = float(hang_of(neck_alpha()))
            ok = em < 1.0
            if ok and hv < best_hang:
                best = amp.detach().clone()
                best_hang = hv
            if i == 0 or (i + 1) % 25 == 0 or i + 1 == steps:
                length = torch.linalg.norm(total_offset(), dim=-1)
                print(
                    f"[neck {i+1:3d}/{steps}] hang={hv:.4f} field_edge={em:.3f}mm "
                    f"total_edge={total_em:.3f}mm "
                    f"amp_mm={[round(float(v) * 1e3, 2) for v in amp.detach()]} "
                    f"off {float(length.mean())*1e3:.2f}/{float(length.max())*1e3:.2f}mm "
                    f"best_hang={best_hang:.4f}",
                    flush=True,
                )
            if not ok:
                amp.copy_(best)
                for g in opt.param_groups:
                    g["lr"] *= 0.5
    with torch.no_grad():
        amp.copy_(best)
        rgba = beauty_rgba()
        photo_fit.dump_photo_preview(OUT, "offset", rgb_np, rgba[0].detach().float().cpu().numpy())
        alpha = neck_alpha()
        length = torch.linalg.norm(total_offset(), dim=-1)
        print(
            f"[after] hang={float(hang_of(alpha)):.4f} edge={edge_mm(total_offset()):.3f}mm "
            f"amp_mm={[round(float(v) * 1e3, 2) for v in amp.detach()]} "
            f"off {float(length.mean())*1e3:.2f}/{float(length.max())*1e3:.2f}mm",
            flush=True,
        )
        np.savez(
            OUT / "photo_appearance.npz",
            residual=res.numpy(),
            lights=lights.numpy(),
            eye_col=eye_col.numpy(),
            offset=total_offset().numpy(),
            id_head=state.id_head.numpy(),
            id_eyes=state.id_eyes.numpy(),
            expression_lower_face=state.ex_lower.numpy(),
            rotations=state.rotations().numpy(),
            translation=state.translation.numpy(),
            cam_aa=state.cam_aa.numpy(),
            cam_t=state.cam_t.numpy(),
            focal_norm=state.focal_norm.numpy(),
            neck_amp=amp.detach().numpy(),
        )
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
