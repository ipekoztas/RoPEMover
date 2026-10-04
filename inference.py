"""
RoPEMover single-image inference.

Examples
--------
# explicit displacement (pixels of the input image) + depth offset (meters)
python inference.py --image examples/captured_vase/image.jpg --mask examples/captured_vase/mask.png \
    --prompt "Move the vase to the left." --dx -110 --dy 20 --dz -0.1 --output out.png

# derive displacement and scale from a target mask / box (as in the ObjectMover benchmark)
python inference.py --image img.jpg --mask src_mask.png --target_mask tgt_mask.png \
    --prompt "Move the mug to the right." --output out.png
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image

from ropemover import MoGeDepthEstimator, load_pipeline, move_object, PAPER_SETTINGS
from ropemover.depth import DEPTH_RES, depth_to_uint8


def drag_from_target_mask(src_mask: Image.Image, tgt_mask: Image.Image, same_threshold=0.15):
    """Centroid displacement (original pixels) and linear scale between two masks."""
    src = np.array(src_mask.convert("L")) > 127
    tgt = np.array(tgt_mask.convert("L").resize(src_mask.size, Image.NEAREST)) > 127
    sy, sx = np.where(src)
    ty, tx = np.where(tgt)
    dx, dy = tx.mean() - sx.mean(), ty.mean() - sy.mean()

    area = lambda m: (np.array(m.convert("L").resize((512, 512), Image.NEAREST)) > 127).sum()
    ratio = float(np.sqrt(area(tgt_mask) / max(area(src_mask), 1)))
    if abs(ratio - 1.0) <= same_threshold:
        ratio = 1.0
    return float(dx), float(dy), ratio


def main():
    p = argparse.ArgumentParser(description="RoPEMover: depth-aware object relocation")
    p.add_argument("--image", required=True, help="source RGB image")
    p.add_argument("--mask", required=True, help="binary mask of the object to move (white = object)")
    p.add_argument("--prompt", required=True, help='edit instruction, e.g. "Move the cup to the left."')
    p.add_argument("--output", default="output.png")

    g = p.add_argument_group("motion")
    g.add_argument("--dx", type=float, help="horizontal displacement in input-image pixels (+ = right)")
    g.add_argument("--dy", type=float, help="vertical displacement in input-image pixels (+ = down)")
    g.add_argument("--dz", type=float, default=0.0, help="depth offset in meters (+ = away from camera)")
    g.add_argument("--scale", type=float, default=None, help="object scale factor (default 1.0)")
    g.add_argument("--target_mask", help="optional target mask/box; derives --dx/--dy/--scale from it")

    g = p.add_argument_group("depth")
    g.add_argument("--depth", help="optional precomputed raw MoGe-2 depth (.npy, 512x512 on the 512² resized image)")
    g.add_argument("--save_depth", action="store_true", help="also save the synthesized target depth")

    g = p.add_argument_group("model")
    g.add_argument("--lora", default=None,
                   help="LoRA path or HF '<repo>[:<file>]' (default: weights/ or ipekoztas/RoPEMover)")
    g.add_argument("--model_dir", default=os.environ.get("DIFFSYNTH_MODEL_BASE_PATH", "./models"),
                   help="where Qwen-Image base weights are stored / downloaded")
    g.add_argument("--download_source", default="huggingface", choices=["huggingface", "modelscope"])

    g = p.add_argument_group("sampling")
    g.add_argument("--seed", type=int, default=PAPER_SETTINGS["seed"])
    g.add_argument("--steps", type=int, default=PAPER_SETTINGS["num_inference_steps"])
    g.add_argument("--warp_end_step", type=int, default=PAPER_SETTINGS["warp_end_step"],
                   help="RoPE warping is applied for the first N denoising steps")
    args = p.parse_args()

    image = Image.open(args.image).convert("RGB")
    mask = Image.open(args.mask)

    dx, dy, scale = args.dx, args.dy, args.scale
    if args.target_mask:
        tdx, tdy, tscale = drag_from_target_mask(mask, Image.open(args.target_mask))
        dx = tdx if dx is None else dx
        dy = tdy if dy is None else dy
        scale = tscale if scale is None else scale
    if dx is None or dy is None:
        p.error("provide --dx and --dy, or --target_mask")
    scale = 1.0 if scale is None else scale

    if args.depth:
        source_depth = np.load(args.depth).astype(np.float32)
    else:
        print("Estimating source depth with MoGe-2 ...")
        estimator = MoGeDepthEstimator()
        source_depth = estimator(image, DEPTH_RES)
        del estimator
        torch.cuda.empty_cache()

    print("Loading Qwen-Image-Edit-2511 + RoPEMover LoRA ...")
    torch.set_grad_enabled(False)
    pipe = load_pipeline(lora=args.lora, model_dir=args.model_dir, download_source=args.download_source)

    print(f"Moving object: dx={dx:.1f}px dy={dy:.1f}px dz={args.dz:+.4f} scale={scale:.4f}")
    result = move_object(
        pipe, image, mask, source_depth, args.prompt,
        dx=dx, dy=dy, dz=args.dz, scale=scale,
        seed=args.seed, num_inference_steps=args.steps, warp_end_step=args.warp_end_step,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    result.image.save(args.output)
    print(f"Saved → {args.output}")
    if args.save_depth:
        depth_path = os.path.splitext(args.output)[0] + "_target_depth.png"
        Image.fromarray(depth_to_uint8(result.target_depth), mode="L").save(depth_path)
        print(f"Saved → {depth_path}")


if __name__ == "__main__":
    main()
