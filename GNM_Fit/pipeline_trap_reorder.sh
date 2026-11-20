#!/usr/bin/env bash
# REORDERED pipeline: rigid -> lmk+SAM eyes (camera fixed here) -> neck/shoulder
# trap-PCA (now at the FINAL camera) -> ear -> ear coef -> L1 -> L2.
# One run, one directory tree ($RUN). Each stage reads the previous npz inside $RUN.
set -euo pipefail
# All paths are env-overridable so the script runs on another machine/data layout.
EV3=${EV3:-/root/autodl-tmp/gnm_fit_out/expression_v3}
REPO=${REPO:-/root/autodl-tmp/19676}
RUN=${RUN:-$EV3/pipeline_run_reorder}
EAR_MASKS=${EAR_MASKS:-/root/autodl-tmp/notes/gnm_uv/sam_ears/ear_masks.npz}
PY=${PY:-conda run -n gnm python}
mkdir -p "$RUN"/{01_rigid,02_lmk_eye,03_neck_trapPCA,04_ear_semantic,05_ear_coef,06_l1,07_l2}

echo "== 01 rigid =="
$PY "$REPO/GNM_Fit/fit_gnm.py" --init "$EV3/frame0_init_v3改.npz" \
  --landmarks "$EV3/facepoints68.npz" --landmark_file "$REPO/GNM_Fit/head_sparse_68_v3_改.txt" \
  --preset vhap_monocular --stop_after rigid --out_dir "$RUN/01_rigid"

echo "== 02 lmk + SAM eyes (camera fixed here) =="
$PY "$REPO/GNM_Fit/fit_gnm.py" --init "$EV3/frame0_init_v3改.npz" \
  --warm_params "$RUN/01_rigid/frame_000000_params.npz" \
  --landmarks "$EV3/facepoints68.npz" --landmark_file "$REPO/GNM_Fit/head_sparse_68_v3_改.txt" \
  --preset vhap_monocular --expr_pca_dim 150 \
  --sam_eye_centers "$EV3/flow_frame0/sam_eyes/centers.npz" --out_dir "$RUN/02_lmk_eye"

echo "== 03 neck/shoulder trap-PCA (K7, at lmk camera) =="
$PY "$REPO/GNM_Fit/fit_neck_trap_pca.py" \
  --params "$RUN/02_lmk_eye/frame_000000_params.npz" \
  --shoulder_lines "$EV3/neck_shoulder_preview/neck_shoulder_line_000001.npz" \
  --image "$EV3/images/000001.jpg" --mode_set k7 --rounds 12 --out_dir "$RUN/03_neck_trapPCA"

echo "== 04 ear semantic (vertv3) =="
$PY "$REPO/GNM_Fit/fit_ear_semantic.py" \
  --params "$RUN/03_neck_trapPCA/frame_000000_params.npz" \
  --params_now "$RUN/03_neck_trapPCA/frame_000000_params.npz" \
  --ear_mask_npz "$EAR_MASKS" \
  --image "$EV3/images/000001.jpg" \
  --iters 500 --lr 0.5 --trans_scale 1.0 --w_reg 1.0 --margin_px 6 \
  --coef_max 40 --size_bias 2 --rigid_axes vert --vert_max 10 --prot_max 1.5 --sig_y_up 0.06 \
  --out_dir "$RUN/04_ear_semantic"

echo "== 05 ear coef =="
$PY "$REPO/GNM_Fit/semantic_coef.py" \
  --base_params "$RUN/04_ear_semantic/frame_000000_params.npz" \
  --out_dir "$RUN/05_ear_coef" --trans_scale 1.0

echo "== 06 L1 (tex) =="
$PY "$REPO/GNM_Fit/photo_fit.py" \
  --params "$RUN/05_ear_coef/frame_000000_params.npz" \
  --landmarks "$EV3/facepoints68.npz" --landmark_file "$REPO/GNM_Fit/head_sparse_68_v3_改.txt" \
  --image "$EV3/images/000001.jpg" \
  --stop_after tex --photo_tex_iters 500 --photo_tex_res 2048 --reg_tex_tv 5e3 \
  --w_jaw_outer 20 --expr_pca_dim 0 --freeze_cam_neck --freeze_shape --out_dir "$RUN/06_l1"

echo "== 07 L2 (all, fully open, neck tiny lr) =="
$PY "$REPO/GNM_Fit/photo_fit.py" \
  --params "$RUN/06_l1/frame_photo_params.npz" \
  --init_appearance "$RUN/06_l1/photo_appearance.npz" \
  --landmarks "$EV3/facepoints68.npz" --landmark_file "$REPO/GNM_Fit/head_sparse_68_v3_改.txt" \
  --image "$EV3/images/000001.jpg" \
  --stop_after rgb --photo_tex_iters 0 \
  --expr_pca "$RUN/02_lmk_eye/gnm_face_expr_pca150.npz" --expr_lr 2e-2 \
  --neck_lr 1e-4 \
  --eye_centers "$EV3/flow_frame0/sam_eyes/centers.npz" --out_dir "$RUN/07_l2"

echo "== 08 eye (眼球专属: L1 眼色/眼白 + L2 SAM3 瞳孔半径 r; rot_leye/reye 不动, 无静态偏移) =="
mkdir -p "$RUN/08_eye"
$PY "$REPO/GNM_Fit/photo_fit.py" \
  --params "$RUN/07_l2/frame_photo_params.npz" \
  --init_appearance "$RUN/07_l2/photo_appearance.npz" \
  --landmarks "$EV3/facepoints68.npz" --landmark_file "$REPO/GNM_Fit/head_sparse_68_v3_改.txt" \
  --image "$EV3/images/000001.jpg" \
  --stop_after eye --photo_tex_iters 0 \
  --expr_pca "$RUN/02_lmk_eye/gnm_face_expr_pca150.npz" \
  --eye_tex_iters 200 --eye_pupil_iters 0 --eye_opt_id 1 \
  --eye_pupil_mask "$EV3/flow_frame0/photo/eye_mask.npz" \
  --out_dir "$RUN/08_eye"

echo "== stage npz =="
find "$RUN" -maxdepth 2 -name 'frame_*_params.npz' | sort
echo "DONE"
