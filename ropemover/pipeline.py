"""
RoPEMover inference on top of Qwen-Image-Edit-2511 (DiffSynth-Studio backend).
"""
import os
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig

from .depth import DEPTH_RES, synthesize_target_depth, to_pefield

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DEFAULT_LORA_REPO = "ipekoztas/RoPEMover"
DEFAULT_LORA_FILE = "ropemover_qwen_image_edit_2511_lora.safetensors"

# Inference hyper-parameters used for all results in the paper.
PAPER_SETTINGS = dict(
    num_inference_steps=40,
    warp_end_step=20,
    seed=123,
    resolution=512,
)


def resolve_lora(lora: str = None) -> str:
    """Local path if given/existing, otherwise download from the Hugging Face Hub."""
    if lora is not None and os.path.isfile(lora):
        return lora
    local_default = os.path.join(os.path.dirname(__file__), "..", "weights", DEFAULT_LORA_FILE)
    if lora is None and os.path.isfile(local_default):
        return os.path.abspath(local_default)
    from huggingface_hub import hf_hub_download
    repo_id, filename = DEFAULT_LORA_REPO, DEFAULT_LORA_FILE
    if lora is not None:  # "<repo_id>" or "<repo_id>:<filename>"
        repo_id, _, fn = lora.partition(":")
        filename = fn or DEFAULT_LORA_FILE
    return hf_hub_download(repo_id=repo_id, filename=filename)


def load_pipeline(lora: str = None, model_dir: str = "./models",
                  download_source: str = "huggingface", device: str = "cuda",
                  torch_dtype=torch.bfloat16) -> QwenImagePipeline:
    """
    Builds the Qwen-Image-Edit-2511 pipeline and attaches the RoPEMover LoRA.

    Base weights are looked up under ``model_dir/<model_id>/...`` and
    downloaded there if missing (set ``DIFFSYNTH_SKIP_DOWNLOAD=true`` to forbid
    downloads on offline machines).
    """
    kw = dict(local_model_path=model_dir, download_source=download_source)
    if os.environ.get("DIFFSYNTH_MODEL_BASE_PATH") is None:
        os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = model_dir
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch_dtype,
        device=device,
        model_configs=[
            ModelConfig(model_id="Qwen/Qwen-Image-Edit-2511", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors", **kw),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors", **kw),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **kw),
        ],
        tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/", **kw),
        processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/", **kw),
    )
    pipe.load_lora(pipe.dit, resolve_lora(lora))
    return pipe


@dataclass
class MoveResult:
    image: Image.Image
    target_depth: np.ndarray   # synthesized raw target depth (DEPTH_RES²)
    target_mask: np.ndarray    # bool target mask (DEPTH_RES²)


@torch.no_grad()
def move_object(
    pipe: QwenImagePipeline,
    image: Image.Image,
    mask: Image.Image,
    source_depth: np.ndarray,
    prompt: str,
    dx: float,
    dy: float,
    dz: float = 0.0,
    scale: float = 1.0,
    seed: int = PAPER_SETTINGS["seed"],
    num_inference_steps: int = PAPER_SETTINGS["num_inference_steps"],
    warp_end_step: int = PAPER_SETTINGS["warp_end_step"],
    resolution: int = PAPER_SETTINGS["resolution"],
) -> MoveResult:
    """
    Move the masked object in ``image``.

    Args:
        image:        source RGB image (any resolution).
        mask:         binary object mask, same size as ``image`` (white = object).
        source_depth: raw MoGe-2 depth of ``image`` resized to DEPTH_RES²
                      (see ``MoGeDepthEstimator``).
        prompt:       edit instruction, e.g. "Move the cup to the left."
        dx, dy:       displacement in pixels of the ORIGINAL image resolution
                      (+x = right, +y = down).
        dz:           depth offset in MoGe units (meters); + = away from camera.
        scale:        object scale factor (1.0 = unchanged).
    """
    image = image.convert("RGB")
    mask = mask.convert("RGB")
    orig_w, orig_h = image.size

    depth = np.asarray(source_depth, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth.squeeze(-1)
    assert depth.shape == (DEPTH_RES, DEPTH_RES), \
        f"source_depth must be {DEPTH_RES}x{DEPTH_RES}, got {depth.shape}"

    M_src = np.array(mask.convert("L").resize((DEPTH_RES, DEPTH_RES), Image.NEAREST)) > 127
    if M_src.sum() == 0:
        raise ValueError("Empty object mask.")

    dx_depth = int(round(dx * (DEPTH_RES / orig_w)))
    dy_depth = int(round(dy * (DEPTH_RES / orig_h)))
    D_tgt, M_tgt = synthesize_target_depth(depth, M_src, dx_depth, dy_depth, dz)
    depth_tensor = torch.from_numpy(to_pefield(D_tgt)).to(device=pipe.device, dtype=torch.float32)

    W = H = resolution
    mask_r = mask.resize((W, H))
    out = pipe(
        prompt,
        edit_image=[image.resize((W, H)), mask_r],
        depth=depth_tensor,
        source_depth=depth_tensor,  # the synthesized target depth drives both source & target tokens
        seed=seed,
        num_inference_steps=num_inference_steps,
        height=H,
        width=W,
        drag_bx=dx * W / orig_w,
        drag_by=dy * H / orig_h,
        drag_scale=scale,
        drag_mask=mask_r,
        inject_noise_in_vacated_positions=False,
        warp_end_step=warp_end_step,
        zero_cond_t=True,
        edit_image_auto_resize=False,
    )
    return MoveResult(image=out, target_depth=D_tgt, target_mask=M_tgt)
