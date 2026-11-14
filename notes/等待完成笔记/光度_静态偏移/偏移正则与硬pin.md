# 偏移正则与硬 pin

状态：等待处理。阶段：`rgb_init_offset`（静态偏移）。

代码：`photo_fit.py` offset 段的 lap / \|δ\| / `pin`。

## 差距

| | VHAP | GNM |
|---|---|---|
| `reg_offset`（\|δ\|） | 3e2 | 3e2 |
| `reg_offset_lap` | **1e6** | **1e5**（弱 10×） |
| `reg_offset_rigid` | 3e2（耳/颈/眼/唇紧） | **无** |
| 放松 | hair、ears | 无 |
| 硬约束 | 靠 mask/软正则 | **eyes / teeth+gums+tongue / ears / bottom 硬 pin=0** |

钉死眼球和牙避免偏移拽歪——合理，但和 VHAP 软刚体不是同一套。Laplacian 弱一档，细皱褶更容易进网格。

## 要做

评估 lap 提到 1e6、是否补 rigid；硬 pin 哪些保留为有意。
