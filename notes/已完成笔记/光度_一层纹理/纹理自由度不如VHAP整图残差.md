# 一层色彩：VHAP 底图+残差

状态：已完成。光度色彩路径已按 VHAP + 后置约定重构；landmark 代码未改。

## 改法（`photo_fit.py`）

| | 旧 GNM | 现（VHAP 式） |
|---|---|---|
| 皮肤 | 整张 `sigmoid(logit)` 自由拧 | authored `skin.png` **底 + `tex_res` 残差**，clamp |
| 眼/牙/龈/舌 | 固定片 | 仍固定片（**后置**，本层不学） |
| 灯光 | `1/sh_const[0]` | VHAP `lights_uniform`：`L0=√(4π)` |
| except | 试过有/无 hair | **hair/boundary/neck**（VHAP） |
| 扰动 bg | 0.3 | **0.5** |
| 对照中格 | 曾误用白底 render | **overlay** |

结构差（有意保留）：GNM 只学 **skin 瓦片**残差；VHAP 是整张 FLAME UV 残差。眼色 4a 不在本层。

## 结果

起点：`lmk_align_vhap_frame0_v4_sam_eyes`  
输出：`gnm_fit_out/expression_v3/photo_l1_sam_eyes_vhap_residual/`（1024）→ 定稿并入 2048 版  
六格（残差 1024）：`notes/gnm_uv/测试用图片/GNM v4 SAM眼 一层VHAP残差贴图 六格.jpg`

| | 脸区偏 R | 训练末 rgb |
|---|---:|---:|
| 旧自由 logit | ~136 | ~0.107 |
| 底+残差 | ~119 | ~0.087 |

占位眼 UV（authored）：瞳孔黑半径 **37**、虹膜灰 **100**（1024 片；左右 `notes/gnm_uv/UV/eye_{left,right}.png`，已同步 `19676/notes/...`）。
