"""UV mask used by photometric layer 1.

Row 0 of the mask is texture v=0, matching the tensor fed to the renderer.
"""
import cv2
import numpy as np

def tensor_mask(faces, tri_uv, vert_ids, n_verts, shape) -> np.ndarray:
    """Mask in the flipped texture tensor, where row 0 is uv v=0."""
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
        px = np.round(np.stack([uv[:, 0] * (w - 1), uv[:, 1] * (h - 1)], 1)).astype(np.int32)
        cv2.fillConvexPoly(img, px, 1)
    return img.astype(np.float32)

