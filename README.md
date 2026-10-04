<div align="center">

<h2>RoPEMover: Depth-Aware Object Relocation via Positional Embeddings</h2>

[İpek Öztaş](https://ipekoztas.github.io/)<sup>1,3</sup> |
[Duygu Ceylan](https://www.duygu-ceylan.com/)<sup>2</sup> |
[Aybars Buğra Aksoy](https://www.linkedin.com/in/aybars-bugra-aksoy/)<sup>1</sup> |
[Ayşegül Dündar](https://www.cs.bilkent.edu.tr/~adundar/)<sup>1</sup>

<sup>1</sup>Bilkent University, <sup>2</sup>Adobe Research, <sup>3</sup>Brown University

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://ipekoztas.github.io/RoPEMover/)&nbsp;
[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2606.27332)&nbsp;
[![Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-demo-blue)](#)

</div>

---

## 📰 Abstract

Moving an object in a single image requires geometry-consistent spatial
rearrangement, including handling occlusions, revealing previously unseen
regions, and maintaining coherent shadows and reflections. Existing
approaches are not well suited to this setting and often fail to preserve
such scene-level consistency.

We address this problem by introducing a geometry-aware object motion
method that operates directly on the positional representations of
diffusion transformers. Our key insight is that rotary positional
embeddings (RoPE) define a structured spatial field that can be
explicitly manipulated to induce controlled motion. We extend 2D RoPE
into a depth-aware formulation that encodes 3D spatial structure,
enabling consistent object displacement and scene-aware updates.

Our model is trained using synthetic data combined with a small set of
real images via parameter-efficient fine-tuning. Despite minimal real
supervision, it preserves object identity under large spatial
displacements, generates plausible content in newly revealed regions,
and consistently updates scene-dependent effects such as shadows and
illumination.

Experimental results on standard object motion benchmarks demonstrate
state-of-the-art performance across all evaluation metrics.

---

## ⏰ Updates

- [x] **2026.6.25**: Project page released at
  [`ipekoztas.github.io/RoPEMover`](https://ipekoztas.github.io/RoPEMover/).
- [ ] Release arXiv preprint.
- [x] **2026.10.4**: Released inference code and pretrained weights.
- [ ] Release interactive Hugging Face Space demo.

---

## 🛠️ Installation

```bash
git clone https://github.com/ipekoztas/RoPEMover.git
cd RoPEMover
python -m venv .venv && source .venv/bin/activate   # Python >= 3.10
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

The code ships a modified copy of
[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) (`diffsynth/`)
that implements depth-aware 3D RoPE and RoPE warping for Qwen-Image-Edit.
Run all commands from the repository root.

## 📦 Pretrained weights

RoPEMover is a LoRA (rank 16, ~236 MB) on top of
[Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511).

| Weight | Base model | Training data |
|---|---|---|
| [`ropemover_qwen_image_edit_2511_lora.safetensors`](https://huggingface.co/ipekoztas/RoPEMover) | Qwen-Image-Edit-2511 | 5k synthetic CLEVR pairs → real captured pairs |

Everything is downloaded automatically on first use:

- the LoRA from [`ipekoztas/RoPEMover`](https://huggingface.co/ipekoztas/RoPEMover)
  (or place it at `weights/ropemover_qwen_image_edit_2511_lora.safetensors`),
- the Qwen-Image base weights (transformer, text encoder, VAE, tokenizer,
  processor; ~55 GB) into `--model_dir` (default `./models`),
- [MoGe-2](https://huggingface.co/Ruicheng/moge-2-vitl-normal) for monocular depth.

For offline machines, download once, then set
`DIFFSYNTH_SKIP_DOWNLOAD=true HF_HUB_OFFLINE=1` and point `--model_dir` to the
folder containing `Qwen/Qwen-Image`, `Qwen/Qwen-Image-Edit` and
`Qwen/Qwen-Image-Edit-2511`.

All our experiments run on a single NVIDIA A100 (64 GB) at 512×512.

## 🚀 Inference

You need a source image, a binary mask of the object (white = object), an
edit prompt and the desired motion:

```bash
python inference.py \
    --image  path/to/image.jpg \
    --mask   path/to/object_mask.png \
    --prompt "Move the mug to the right side of the table." \
    --dx 300 --dy -40 --dz 0.1 \
    --output results/mug_moved.png --save_depth
```

| Argument | Meaning |
|---|---|
| `--dx`, `--dy` | 2D displacement in pixels **of the input image** (+x right, +y down) |
| `--dz` | depth offset in meters (MoGe-2 metric depth); `> 0` moves the object away from the camera |
| `--scale` | object scale factor (default `1.0`) |
| `--target_mask` | alternatively, a mask/box at the target location; `dx`, `dy` and `scale` are derived from the centroid shift and area ratio |
| `--depth` | optional precomputed MoGe-2 depth `.npy` (512×512, computed on the 512² resized image) |
| `--seed`, `--steps`, `--warp_end_step` | sampling settings; paper defaults are `123`, `40`, `20` |

### Example inputs

[`examples/`](examples) contains six images with object masks to try,
two each from ObjMove-A, ObjMove-B and our captured dataset. Each folder has
an `image.jpg` and a `mask.png`. You choose the prompt and the motion, for example:

```bash
python inference.py \
    --image  examples/captured_vase/image.jpg \
    --mask   examples/captured_vase/mask.png \
    --prompt "Move the vase to the left." \
    --dx -110 --dy 20 --dz -0.1 \
    --output results/vase.png
```

`dx`/`dy` are in pixels of the given image. The ObjMove-A/B examples are
downscaled to 1024 px and the captured examples are 512×512.

How it works: the source depth is estimated with MoGe-2. The object is then
removed from the depth map (Laplacian inpainting) and re-inserted at
`(dx, dy)` with offset `dz`, with a z-test for occlusions. The result is the
target depth map, which serves as the depth axis of the 3D RoPE. During the
first `warp_end_step` denoising steps, the positional embeddings of the
object's tokens are warped to the target location.

Python API:

```python
from PIL import Image
from ropemover import MoGeDepthEstimator, load_pipeline, move_object

image, mask = Image.open("image.jpg"), Image.open("mask.png")
depth = MoGeDepthEstimator()(image)            # 512x512 metric depth
pipe = load_pipeline()                         # Qwen-Image-Edit-2511 + RoPEMover LoRA
out = move_object(pipe, image, mask, depth, "Move the mug to the right.",
                  dx=300, dy=-40, dz=0.1)
out.image.save("moved.png")
```

## 📊 Reproducing ObjMove-A results

Download the [ObjectMover benchmark](https://huggingface.co/datasets/Andyx/ObjectMover-Benchmark)
and run:

```bash
python scripts/infer_objmove_a.py \
    --benchmark_root /path/to/ObjectMover-Benchmark \
    --output_dir results/objmove_a
```

The per-image prompts and motions (`x`, `y`, `z`, `scale_value`) we used
are in [`data/objmove_a/`](data/objmove_a). Source depth (`src_depth_moge.npy`) is
computed with MoGe-2 and cached in each sample folder if it is missing.

---

## 🔗 Citation

If you find our work useful for your research, please consider citing:

```bibtex
@article{oztas2026ropemover,
  title   = {RoPEMover: Depth-Aware Object Relocation via Positional Embeddings},
  author  = {Oztas, Ipek and Ceylan, Duygu and Aksoy, Aybars Bugra and Dundar, Aysegul},
  journal = {arXiv preprint},
  year    = {2026}
}
```

---

## ©️ License

This project is released under the [Apache 2.0 license](LICENSE).
Use of the pretrained weights is also subject to the license of
[Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511).

## 🙏 Acknowledgements

Built on [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
(Apache 2.0), [Qwen-Image](https://github.com/QwenLM/Qwen-Image) and
[MoGe](https://github.com/microsoft/MoGe). We evaluate on the
[ObjectMover benchmark](https://huggingface.co/datasets/Andyx/ObjectMover-Benchmark)
(Apache 2.0); the `objmove_a_*` and `objmove_b_*` example inputs are taken from it.

## 📧 Contact

For questions, please reach out to
[ipek.oztas@bilkent.edu.tr](mailto:ipek.oztas@bilkent.edu.tr).
