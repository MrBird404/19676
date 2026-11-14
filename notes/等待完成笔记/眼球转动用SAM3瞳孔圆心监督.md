# 眼球转动用 SAM3 瞳孔圆心监督

状态：等待处理（未归档）。方案已定；**第 0 帧 landmark 后单独拧眼关节**已接入 `fit_gnm.py`，全序列跟踪 / 瞳孔大小与颜色仍待做。

相关模块：`流程与方法笔记/眼睛相关模块详解以及对应功能.md`、`流程与方法笔记/关节姿态模块.md`。

预览（查效果用）：`gnm_fit_out/expression_v3/sam_eyes_lmk_preview/`（`overlays/`、`crops/`、`centers.npz`）。

---

## 已有检测与后处理（`19676/photometric/`）

| 脚本 | 作用 |
|---|---|
| `sam_eyes_frames.py` | SAM3 文本提示 → mask → `minEnclosingCircle` → `centers.npz` |
| `sam_eyes_fill.py` | 漏检帧用 68 眼眶框 + `pupil` 补圆心 |
| `flow_eye_frames.py` | 跟踪结果上只优化 `rot_leye/reye` 贴圆心 |
| `flow_track_full_expr.py` | 后续可冻住 SAM 对准的眼关节 |

圆心数值：`gnm_fit_out/expression_v3/flow_frame0/sam_eyes/centers.npz`（189×2×2）。

---

## 方案（讨论结论）

**注视 / 脸皮 / 眼色 / 瞳孔大小 四段分开关**，不要塞进总光度一起拧。

### 第 0 帧具体步骤

```text
1. lmk（刚体 + shape）
2. 只拧眼关节贴 SAM 圆心
3. 脸光度（冻眼关节；不学眼色 / 不开瞳孔维）
4a. 只开眼睛分区色（光度）
4b. 只开瞳孔 1 维（半径 / 面积）
```

| 段 | 开 | 冻 | 监督 |
|---|---|---|---|
| **1. Landmark** | 刚体、`id_head` / `id_eyes`、脸表情 PCA | 瞳孔 1 维、眼贴图色、眼关节瞄准 | iBUG 68 |
| **2. 注视**（已接） | 只 `rot_leye` / `rot_reye` | 身份、表情、头姿、相机 | SAM 圆心 ↔ 固定眼球顶点投影（非光度） |
| **3. 脸光度** | 皮肤贴图 / 光照；再按阶段加姿态、表情、静态偏移 | **眼关节**；本段不学眼色、不开瞳孔维。可选也冻 `id_eyes` | 脸皮像素光度（眼睛继续用现成 UV 片） |
| **4a. 眼色** | 巩膜 / 虹膜 / 瞳孔分区常数色（或眼睛贴图残差） | 眼关节、头姿、皮肤贴图、表情、相机 | 眼睛像素光度 |
| **4b. 瞳孔大小** | 表情最后 1 维瞳孔 | 上面全部 + 颜色参数 | SAM 外接圆半径（或暗区面积）↔ mesh 瞳孔投影；可加点眼睛光度 |

对应代码落点：

- 段 2：`fit_gnm.py`（已做）
- 段 3：`photo_fit.py` 的 `rgb_init_texture → rgb_init_all → rgb_init_offset`（眼关节已 `requires_grad_(False)`）
- 段 4：待接到主光度后；本地雏形在 `flow_track_full_expr.py` 的 `albedo_eye()`（sclera/iris/pupil 色偏）+ `expression_pupil`

**不是**「脸光度结束后再冻脸、把眼关节 + `id_eyes` + 瞳孔维一起开光度」。

逐帧跟踪（仍待做）：每帧重复段 2；眼色一般序列共用；瞳孔可跟帧变。`id_eyes`（3）可在段 1/3 带着；68 点几乎无信号，有眼睛外观监督时再细调。

---

## 已做：第 0 帧 lmk 后眼步骤

`19676/GNM_Fit/fit_gnm.py`（`vhap_monocular`）：

- shape 结束后，若存在 `--sam_eye_centers`（默认尝试 `landmarks` 同级 `flow_frame0/sam_eyes/centers.npz`），则：
  - 冻结身份 / 表情 / 头姿 / 相机
  - 只优化 `rot_leye`、`rot_reye`（默认 40 步）
  - 顶点默认 `17303, 17689`；左右按图像 x 排序与圆心配对
  - 写出 `stage_3_sam_eyes_*.jpg`，params 里带 `sam_eye_centers` / `sam_eye_err`

试跑输出：`gnm_fit_out/expression_v3/lmk_align_vhap_frame0_v4_sam_eyes/`  
第 0 帧圆心对齐约 **0.33 / 0.74 px**，眼转约 **23° / 12°**。

对照图：

- 八格：`notes/gnm_uv/测试用图片/GNM v4 + PCA150 SAM眼睛对齐.jpg`
- 眼周前后：`notes/gnm_uv/测试用图片/SAM眼睛对齐_眼周对比.jpg`（左 shape 眼关节=0，右 SAM 对齐后）

---

## 眼皮–眼球穿模检查（第 0 帧）

现有防穿模：`photometric/flow_track_eyelid.py` 外层眼皮相对眼球的**径向碰撞**，在 **bind 空间**计算，主要挡**眼皮表情往里挤**，不是专门挡 `rot_leye/reye` 转动。

本帧（睁眼、眼皮表情未开）数值：

| | bind 碰撞（现有指标） | 世界系眼皮–眼球间隙（近似） |
|---|---|---|
| 眼关节 = 0 | max 0.59 mm，21/3199 点 | L/R 最深约 −0.97 / −0.69 mm |
| SAM 对齐后 | **同样** 0.59 mm（眼表情没动） | 略加深到约 −1.10 / −0.89 mm |

肉眼：八格 / 眼周对比里没有明显「眼球捅穿眼皮」；贴得紧，有亚毫米级接触，拧完注视后略差一点。

### 要不要在第 0 帧眼步骤用上

**第 0 帧「只拧眼关节」：先不用现有防穿模。**  
它几乎不随眼关节变化；硬塞也管不住注视旋转。

**该用的时候：** 开眼皮表情 / 闭眼跟踪（`flow_track_eyelid` 那条）——本来就是为防眼皮打进眼球写的。若以后注视拧得很猛仍穿模，再加**姿态后世界系**碰撞，而不是原样搬 bind 版。

---

## 仍待做

- 逐帧跟踪里注视高权或每帧眼步骤；圆心时序平滑、侧脸左右配对。  
- 段 4a / 4b：眼睛分区光度（颜色）与瞳孔 1 维（大小），接到脸光度之后。  
- 可选：`centers.npz` 同时存外接圆半径，预览绿圈与检测一致。  
- 眼皮表情阶段启用现有径向碰撞；必要时再补世界系防穿模。  
- 归档条件：第 0 帧眼步骤稳定 + 跟踪策略敲定后再移入已完成。
