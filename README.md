# 19676

单目流程到光度一层。光度一层用的是 VHAP `rgb_init_texture` 这一档：相机、形状、贴图、光照一起优化，姿态、表情、关节和顶点偏移不动。

入口是 `photometric/flow_l1_vhap.py`。它前面依赖：

- `MediaPipe_备份/fit_gnm.py`：刚体和形状，预设 `vhap_monocular`，3D 锚点 `head_sparse_68_v3_改.txt`
- `photometric/flow_photo.py`：眼球只旋转，不对齐时不平移
- `MediaPipe_备份/photo_fit.py`：脸部区域和预览
- `photometric/uv/`：按 68 点贴好的脸 UV、眼球 UV，以及 GNM 头部 UV

残差仍然只改颜色、不改明暗。输出目录是 `gnm_fit_out/expression_v3/flow_frame0/photo_l1_vhap/`。