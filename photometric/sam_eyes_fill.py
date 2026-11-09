"""Fill frames where the text prompt found no two pupils.

Each missed eye gets its own iBUG box plus the text "pupil". The saved point
is the mask's enclosing-circle center, not the box center.
"""
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


def circle(mask: np.ndarray, anchor):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    best = None
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 12 or area > 2200:
            continue
        ys, xs = np.nonzero(labels == i)
        pts = np.stack([xs, ys], 1).astype(np.float32)
        (cx, cy), radius = cv2.minEnclosingCircle(pts)
        if radius < 2.0 or radius > 20.0:
            continue
        dist = float(np.hypot(cx - anchor[0], cy - anchor[1]))
        if dist > 22.0:
            continue
        if best is None or dist < best[0]:
            best = (dist, area, cx, cy, radius)
    return None if best is None else best


def main() -> None:
    saved = np.load(OUT, allow_pickle=True)
    centers = np.array(saved["centers"], dtype=np.float32)
    prompts = list(saved["prompts"])
    lmk = np.load(LMK, allow_pickle=True)
    paths = [Path(str(p)) for p in lmk["image_paths"]]
    landmarks = np.asarray(lmk["landmarks"], dtype=np.float32)
    width, height = [int(x) for x in lmk["image_size"]]
    missed = [i for i, p in enumerate(prompts) if p == "prev"]
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    model = build_sam3_image_model(checkpoint_path=CKPT, load_from_HF=False, device="cuda")
    processor = Sam3Processor(model, device="cuda", confidence_threshold=0.05)
    for t in missed:
        image = Image.open(paths[t]).convert("RGB")
        state = processor.set_image(image)
        found = []
        for eye in (landmarks[t, 36:42], landmarks[t, 42:48]):
            anchor = eye.mean(0)
            x0, y0 = eye.min(0)
            x1, y1 = eye.max(0)
            box = [
                float((x0 + x1) * 0.5 / width),
                float((y0 + y1) * 0.5 / height),
                float((x1 - x0 + 24) / width),
                float((y1 - y0 + 16) / height),
            ]
            hit = None
            for prompt in ("pupil", "black part inside the eye"):
                processor.reset_all_prompts(state)
                state = processor.set_text_prompt(prompt=prompt, state=state)
                state = processor.add_geometric_prompt(box=box, label=True, state=state)
                masks = state["masks"]
                if masks is None or int(masks.shape[0]) == 0:
                    continue
                m = masks.detach().float().cpu().numpy()
                if m.ndim == 4:
                    m = m[:, 0]
                scores = state["scores"].detach().float().cpu().numpy().reshape(-1)
                order = np.argsort(-scores)
                for j in order:
                    hit = circle(m[j] > 0.5, anchor)
                    if hit is not None:
                        break
                if hit is not None:
                    break
            found.append(hit)
        if found[0] is None or found[1] is None:
            prompts[t] = "prev"
            print(f"[fill {t:03d}] still missing {found[0] is not None}/{found[1] is not None}", flush=True)
            continue
        centers[t, 0] = [found[0][2], found[0][3]]
        centers[t, 1] = [found[1][2], found[1][3]]
        prompts[t] = "box-pupil"
        print(f"[fill {t:03d}] {centers[t].tolist()} r {found[0][4]:.1f}/{found[1][4]:.1f}", flush=True)
    del model, processor
    torch.cuda.empty_cache()
    np.savez(OUT, centers=centers, prompts=np.array(prompts), image_paths=saved["image_paths"])
    print(f"still prev {(np.array(prompts) == 'prev').sum()} wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
