# 19676

单目流程。逐步说明见 [流程.md](流程.md)。第 0 帧的偏移 mesh 在 [mesh/frame_0000.obj](mesh/frame_0000.obj)。原先采用的第 0 帧仍是 `photo_contour_only`，跟踪结果在 `track_global/`。

1. `MediaPipe_备份/fit_gnm.py`：刚体和形状，预设 `vhap_monocular`，3D 锚点 `head_sparse_68_v3_改.txt`。刚体各点权重一样。形状阶段鼻子 27–36 权重 10，下颌 0–16 权重 1。
2. `photometric/flow_photo.py`：眼球只旋转，不平移。
3. `photometric/flow_l1_vhap.py`：光度第一层，VHAP `rgb_init_texture`。输出 `photo_l1_vhap/`。
4. `photometric/flow_l2_vhap.py`：光度第二层。残差锁在第一层，脸颊保持暖色。输出 `photo_l2_hue/`。
5. `photometric/flow_offset_vhap.py`：静态偏移加在蒙皮前，眼睛偏移为 0。只动偏移和光照，并约束摆好后的嘴唇边长。输出 `photo_offset_vhap/`。
6. `photometric/flow_offset_detail.py`：大块肤色锁住，嘴唇附近可以局部变暗。输出 `photo_offset_detail/`。
7. `photometric/flow_contour.py`：下颌点贴到照片，并用平滑下轮廓收下颌。输出 `photo_contour/`。
8. `photometric/flow_contour_only.py`：第 0 帧结果。这一步不用 68 点，按 `photometric/face_mask.npz` 的 `lower_contour_smooth` 把脖子平滑往里收。脖子仍渲染。输出 `photo_contour_only/`。
9. `photometric/flow_track.py`：逐帧跟踪。身份、相机、灯光、颜色和静态偏移锁在第 0 帧。后面每帧只动姿态和下脸表情。输出 `track/`。
10. `photometric/flow_track_eyelid.py`：左右眼皮各用前 12 个分量，外层眼皮有碰撞约束。眼球只旋转，可见瞳孔对准该帧的瞳孔圆。输出 `track_eyelid_gaze/`。
11. `photometric/flow_track_global.py`：全局跟踪。静态偏移冻结。输出 `track_global/`。

脸和眼睛的 UV 在 `photometric/uv/`。拟合结果写到 `gnm_fit_out/expression_v3/flow_frame0/`。
