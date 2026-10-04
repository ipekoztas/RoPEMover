"""
Reproduce the RoPEMover results on ObjMove-A (ObjectMover-Benchmark).

Expects each ``<benchmark_root>/ObjMove-A/real_XXX/`` folder to contain
``src_input.jpg``, ``src_mask_hr.png`` and ``src_depth_moge.npy``. If the depth
file is missing it is computed with MoGe-2 (and cached there unless
--no_cache_depth is set).

    python scripts/infer_objmove_a.py \
        --benchmark_root /path/to/ObjectMover-Benchmark \
        --output_dir results/objmove_a
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ropemover import MoGeDepthEstimator, load_pipeline, move_object  # noqa: E402
from ropemover.depth import DEPTH_RES, depth_to_uint8  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "objmove_a"


def load_prompts(csv_path):
    with open(csv_path, newline="") as f:
        return {int(r["image_index"]): r["prompt"] for r in csv.DictReader(f)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark_root", required=True, help="path to ObjectMover-Benchmark")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--prompts", default=str(DATA / "prompts.csv"))
    p.add_argument("--drags", default=str(DATA / "drags.json"),
                   help="per-image x/y (px), z (MoGe depth offset) and scale_value")
    p.add_argument("--lora", default=None)
    p.add_argument("--model_dir", default=os.environ.get("DIFFSYNTH_MODEL_BASE_PATH", "./models"))
    p.add_argument("--download_source", default="huggingface", choices=["huggingface", "modelscope"])
    p.add_argument("--start", type=int, default=1)
    p.add_argument("--end", type=int, default=200)
    p.add_argument("--no_cache_depth", action="store_true")
    args = p.parse_args()

    torch.set_grad_enabled(False)
    root = Path(args.benchmark_root) / "ObjMove-A"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(args.prompts)
    drags = json.loads(Path(args.drags).read_text())

    estimator = None
    pipe = load_pipeline(lora=args.lora, model_dir=args.model_dir, download_source=args.download_source)

    for i in range(args.start, args.end + 1):
        img_id = f"{i:03d}"
        d = root / f"real_{img_id}"
        drag = drags.get(f"{i}.png")
        if drag is None:
            print(f"[{img_id}] no drag data, skipping")
            continue

        image = Image.open(d / "src_input.jpg").convert("RGB")
        mask = Image.open(d / "src_mask_hr.png")

        depth_path = d / "src_depth_moge.npy"
        if depth_path.exists():
            source_depth = np.load(depth_path).astype(np.float32)
        else:
            estimator = estimator or MoGeDepthEstimator()
            source_depth = estimator(image, DEPTH_RES)
            if not args.no_cache_depth:
                np.save(depth_path, source_depth)

        dx, dy = float(drag.get("x", 0.0)), float(drag.get("y", 0.0))
        dz, scale = float(drag.get("z", 0.0)), float(drag.get("scale_value", 1.0))
        print(f"\n[{img_id}] drag=({dx:.1f},{dy:.1f}) scale={scale:.4f} delta_d={dz:+.4f}")

        result = move_object(pipe, image, mask, source_depth, prompts.get(i, ""),
                             dx=dx, dy=dy, dz=dz, scale=scale)
        result.image.save(out / f"real_{img_id}.png")
        Image.fromarray(depth_to_uint8(result.target_depth), mode="L").save(
            out / f"real_{img_id}_synth_depth.png")
        print(f"  saved → {out / f'real_{img_id}.png'}")

    print("\nDone!")


if __name__ == "__main__":
    main()
