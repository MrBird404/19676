"""SAM3 pupil centers for every frame. Runs in the base conda env, not gnm."""
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/root/autodl-tmp/sam3-main")
if "pkg_resources" not in sys.modules:
    stub = types.ModuleType("pkg_resources")

    def resource_filename(package, resource):
        return str(Path("/root/autodl-tmp/sam3-main") / package.replace(".", "/") / resource)

    stub.resource_filename = resource_filename
    sys.modules["pkg_resources"] = stub

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

CKPT = "/root/autodl-tmp/sam3-main/checkpoints/sam3.pt"
LMK = Path("/root/autodl-tmp/gnm_fit_out/expression_v3/facepoints68.npz")
OUT = Path("/root/autodl-tmp/gnm_fit_out/expression_v3/flow_frame0/sam_eyes/centers.npz")
PROMPTS = ("black part inside the eye", "眼睛内部的黑色部分", "pupil")


def two_centers(mask: np.ndarray):
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
        return None
    return np.array([[circles[0][1], circles[0][2]], [circles[1][1], circles[1][2]]], np.float32)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    paths = [Path(str(p)) for p in np.load(LMK, allow_pickle=True)["image_paths"]]
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    model = build_sam3_image_model(checkpoint_path=CKPT, load_from_HF=False, device="cuda")
    processor = Sam3Processor(model, device="cuda", confidence_threshold=0.25)
    centers = np.zeros((len(paths), 2, 2), np.float32)
    prompts = []
    prev = None
    for t, path in enumerate(paths):
        image = Image.open(path).convert("RGB")
        state = processor.set_image(image)
        found, used = None, ""
        for prompt in PROMPTS:
            processor.reset_all_prompts(state)
            state = processor.set_text_prompt(prompt=prompt, state=state)
            masks = state["masks"]
            if masks is None or int(masks.shape[0]) == 0:
                continue
            m = masks.detach().float().cpu().numpy()
            if m.ndim == 4:
                m = m[:, 0]
            scores = state["scores"].detach().float().cpu().numpy().reshape(-1)
            keep = m[scores > 0.25] > 0.5
            if keep.size == 0:
                keep = m[int(scores.argmax())][None] > 0.5
            merged = keep.any(0)
            found = two_centers(merged)
            used = prompt
            if found is not None:
                break
        if found is None:
            if prev is None:
                raise RuntimeError(f"frame {t} has no two pupils")
            found = prev.copy()
            used = "prev"
        centers[t] = found
        prompts.append(used)
        prev = found
        print(f"[sam {t:03d}] {used} {found.tolist()}", flush=True)
    del model, processor
    torch.cuda.empty_cache()
    np.savez(OUT, centers=centers, prompts=np.array(prompts), image_paths=np.array([str(p) for p in paths]))
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
