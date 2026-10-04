import torch, math
from PIL import Image
from typing import Union
from tqdm import tqdm
from einops import rearrange
import numpy as np
from math import prod

from ..diffusion import FlowMatchScheduler
from ..core import ModelConfig, gradient_checkpoint_forward
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit, ControlNetInput
from ..utils.lora.merge import merge_lora

from ..models.qwen_image_dit import QwenImageDiT
from ..models.qwen_image_text_encoder import QwenImageTextEncoder
from ..models.qwen_image_vae import QwenImageVAE
from ..models.qwen_image_controlnet import QwenImageBlockWiseControlNet
from ..models.siglip2_image_encoder import Siglip2ImageEncoder
from ..models.dinov3_image_encoder import DINOv3ImageEncoder
from ..models.qwen_image_image2lora import QwenImageImage2LoRAModel


class QwenImagePipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16,
        )
        from transformers import Qwen2Tokenizer, Qwen2VLProcessor
        
        self.scheduler = FlowMatchScheduler("Qwen-Image")
        self.text_encoder: QwenImageTextEncoder = None
        self.dit: QwenImageDiT = None
        self.vae: QwenImageVAE = None
        self.blockwise_controlnet: QwenImageBlockwiseMultiControlNet = None
        self.tokenizer: Qwen2Tokenizer = None
        self.siglip2_image_encoder: Siglip2ImageEncoder = None
        self.dinov3_image_encoder: DINOv3ImageEncoder = None
        self.image2lora_style: QwenImageImage2LoRAModel = None
        self.image2lora_coarse: QwenImageImage2LoRAModel = None
        self.image2lora_fine: QwenImageImage2LoRAModel = None
        self.processor: Qwen2VLProcessor = None
        self.in_iteration_models = ("dit", "blockwise_controlnet")
        self.units = [
            QwenImageUnit_ShapeChecker(),
            QwenImageUnit_NoiseInitializer(),
            QwenImageUnit_InputImageEmbedder(),
            QwenImageUnit_Inpaint(),
            QwenImageUnit_EditImageEmbedder(),
            QwenImageUnit_LayerInputImageEmbedder(),
            QwenImageUnit_ContextImageEmbedder(),
            QwenImageUnit_DragMaskToTokens(),
            QwenImageUnit_PromptEmbedder(),
            QwenImageUnit_EntityControl(),
            QwenImageUnit_BlockwiseControlNet(),
        ]
        self.model_fn = model_fn_qwen_image
    
    
    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
        processor_config: ModelConfig = None,
        vram_limit: float = None,
    ):
        # Initialize pipeline
        pipe = QwenImagePipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)
        
        # Fetch models
        pipe.text_encoder = model_pool.fetch_model("qwen_image_text_encoder")
        pipe.dit = model_pool.fetch_model("qwen_image_dit")
        pipe.vae = model_pool.fetch_model("qwen_image_vae")
        pipe.blockwise_controlnet = QwenImageBlockwiseMultiControlNet(model_pool.fetch_model("qwen_image_blockwise_controlnet", index="all"))
        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            from transformers import Qwen2Tokenizer
            pipe.tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_config.path)
        if processor_config is not None:
            processor_config.download_if_necessary()
            from transformers import Qwen2VLProcessor
            pipe.processor = Qwen2VLProcessor.from_pretrained(processor_config.path)
        pipe.siglip2_image_encoder = model_pool.fetch_model("siglip2_image_encoder")
        pipe.dinov3_image_encoder = model_pool.fetch_model("dinov3_image_encoder")
        pipe.image2lora_style = model_pool.fetch_model("qwen_image_image2lora_style")
        pipe.image2lora_coarse = model_pool.fetch_model("qwen_image_image2lora_coarse")
        pipe.image2lora_fine = model_pool.fetch_model("qwen_image_image2lora_fine")
        
        # VRAM Management
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe
    
    
    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: str = "",
        cfg_scale: float = 4.0,
        # Image
        input_image: Image.Image = None,
        denoising_strength: float = 1.0,
        # Inpaint
        inpaint_mask: Image.Image = None,
        inpaint_blur_size: int = None,
        inpaint_blur_sigma: float = None,
        # Shape
        height: int = 1328,
        width: int = 1328,
        # Randomness
        seed: int = None,
        rand_device: str = "cpu",
        # Steps
        num_inference_steps: int = 30,
        exponential_shift_mu: float = None,
        # Blockwise ControlNet
        blockwise_controlnet_inputs: list[ControlNetInput] = None,
        # EliGen
        eligen_entity_prompts: list[str] = None,
        eligen_entity_masks: list[Image.Image] = None,
        eligen_enable_on_negative: bool = False,
        # Qwen-Image-Edit
        edit_image: Image.Image = None,
        edit_image_auto_resize: bool = True,
        edit_rope_interpolation: bool = False,
        # Qwen-Image-Edit-2511
        zero_cond_t: bool = False,
        # Qwen-Image-Layered
        layer_input_image: Image.Image = None,
        layer_num: int = None,
        # In-context control
        context_image: Image.Image = None,
        # Tile
        tiled: bool = False,
        tile_size: int = 128,
        tile_stride: int = 64,
        # Progress bar
        progress_bar_cmd = tqdm,
        # Drag
        drag_mask: Image.Image = None,
        drag_bx: float = 0.0,
        drag_by: float = 0.0,
        drag_scale: float = 1.0,
        drag_rotate=None,  # "left" | "right" | "180", or a numeric degree value (in-place RoPE rotation of the masked object)
        depth: torch.Tensor = None,
        source_depth: torch.Tensor = None,
        separate_source_depth: bool = False,
        inject_noise_in_vacated_positions: bool = True,  # ← NEW
        scheduled_noise: bool = False,  # ← NEW
        noise_timestep: int = None,  # ← NEW PARAMETER
        warp_end_step: int = None,  # stop warping after this step (None = warp all steps)
        drag_objects: list = None,  # multi-object warp: list of {"mask","bx","by","scale","theta"}; None = existing single-mask behavior
        depth_before_rope: bool = False,  # inject depth before RoPE rotation is computed, vs legacy post-hoc overwrite (default)
    ):
        drag_theta = _drag_rotate_to_theta(drag_rotate)
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, dynamic_shift_len=(height // 16) * (width // 16), exponential_shift_mu=exponential_shift_mu)
        print("Timesteps: ", num_inference_steps)
        # Parameters
        inputs_posi = {
            "prompt": prompt,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
        }
        inputs_shared = {
            "cfg_scale": cfg_scale,
            "input_image": input_image, "denoising_strength": denoising_strength,
            "inpaint_mask": inpaint_mask, "inpaint_blur_size": inpaint_blur_size, "inpaint_blur_sigma": inpaint_blur_sigma,
            "height": height, "width": width,
            "seed": seed, "rand_device": rand_device,
            "num_inference_steps": num_inference_steps,
            "blockwise_controlnet_inputs": blockwise_controlnet_inputs,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            "eligen_entity_prompts": eligen_entity_prompts, "eligen_entity_masks": eligen_entity_masks, "eligen_enable_on_negative": eligen_enable_on_negative,
            "edit_image": edit_image, "edit_image_auto_resize": edit_image_auto_resize, "edit_rope_interpolation": edit_rope_interpolation, 
            "context_image": context_image,
            "zero_cond_t": zero_cond_t,
            "layer_input_image": layer_input_image,
            "layer_num": layer_num,
            "drag_mask": drag_mask, "drag_bx": drag_bx, "drag_by": drag_by, "drag_scale": drag_scale, "drag_theta": drag_theta, "depth": depth, "source_depth": source_depth, "separate_source_depth": separate_source_depth,
            "inject_noise_in_vacated_positions": inject_noise_in_vacated_positions,  # ← NEW
            "scheduled_noise": scheduled_noise,  # ← NEW
            "noise_timestep": noise_timestep,  # ← ADD THIS
            "warp_end_step": warp_end_step,
            "drag_objects": drag_objects,
            "depth_before_rope": depth_before_rope,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            noise_pred = self.cfg_guided_model_fn(
                self.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = self.step(self.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs_shared)
        
        # Decode
        self.load_models_to_device(['vae'])
        image = self.vae.decode(inputs_shared["latents"], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if layer_num is None:
            image = self.vae_output_to_image(image)
        else:
            image = [self.vae_output_to_image(i, pattern="C H W") for i in image]
        self.load_models_to_device([])

        return image


class QwenImageBlockwiseMultiControlNet(torch.nn.Module):
    def __init__(self, models: list[QwenImageBlockWiseControlNet]):
        super().__init__()
        if not isinstance(models, list):
            models = [models]
        self.models = torch.nn.ModuleList(models)
        for model in models:
            if hasattr(model, "vram_management_enabled") and getattr(model, "vram_management_enabled"):
                self.vram_management_enabled = True

    def preprocess(self, controlnet_inputs: list[ControlNetInput], conditionings: list[torch.Tensor], **kwargs):
        processed_conditionings = []
        for controlnet_input, conditioning in zip(controlnet_inputs, conditionings):
            conditioning = rearrange(conditioning, "B C (H P) (W Q) -> B (H W) (C P Q)", P=2, Q=2)
            model_output = self.models[controlnet_input.controlnet_id].process_controlnet_conditioning(conditioning)
            processed_conditionings.append(model_output)
        return processed_conditionings

    def blockwise_forward(self, image, conditionings: list[torch.Tensor], controlnet_inputs: list[ControlNetInput], progress_id, num_inference_steps, block_id, **kwargs):
        res = 0
        for controlnet_input, conditioning in zip(controlnet_inputs, conditionings):
            progress = (num_inference_steps - 1 - progress_id) / max(num_inference_steps - 1, 1)
            if progress > controlnet_input.start + (1e-4) or progress < controlnet_input.end - (1e-4):
                continue
            model_output = self.models[controlnet_input.controlnet_id].blockwise_forward(image, conditioning, block_id)
            res = res + model_output * controlnet_input.scale
        return res


class QwenImageUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width"),
            output_params=("height", "width"),
        )

    def process(self, pipe: QwenImagePipeline, height, width):
        h_before, w_before = height, width
        height, width = pipe.check_resize_height_width(height, width)
        print(f"[DEBUG pipeline] check_resize: ({h_before}×{w_before}) → ({height}×{width})")
        return {"height": height, "width": width}



class QwenImageUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "seed", "rand_device", "layer_num"),
            output_params=("noise",),
        )

    def process(self, pipe: QwenImagePipeline, height, width, seed, rand_device, layer_num):
        if layer_num is None:
            noise = pipe.generate_noise((1, 16, height//8, width//8), seed=seed, rand_device=rand_device, rand_torch_dtype=pipe.torch_dtype)
        else:
            noise = pipe.generate_noise((layer_num + 1, 16, height//8, width//8), seed=seed, rand_device=rand_device, rand_torch_dtype=pipe.torch_dtype)
        return {"noise": noise}



class QwenImageUnit_InputImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "noise", "tiled", "tile_size", "tile_stride"),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: QwenImagePipeline, input_image, noise, tiled, tile_size, tile_stride):
        if input_image is None:
            return {"latents": noise, "input_latents": None}
        pipe.load_models_to_device(['vae'])
        if isinstance(input_image, list):
            input_latents = []
            for image in input_image:
                image = pipe.preprocess_image(image).to(device=pipe.device, dtype=pipe.torch_dtype)
                input_latents.append(pipe.vae.encode(image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride))
            input_latents = torch.concat(input_latents, dim=0)
        else:
            image = pipe.preprocess_image(input_image).to(device=pipe.device, dtype=pipe.torch_dtype)
            input_latents = pipe.vae.encode(image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents, "input_latents": input_latents}


class QwenImageUnit_LayerInputImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("layer_input_image", "tiled", "tile_size", "tile_stride"),
            output_params=("layer_input_latents",),
            onload_model_names=("vae",)
        )

    def process(self, pipe: QwenImagePipeline, layer_input_image, tiled, tile_size, tile_stride):
        if layer_input_image is None:
            return {}
        pipe.load_models_to_device(['vae'])
        image = pipe.preprocess_image(layer_input_image).to(device=pipe.device, dtype=pipe.torch_dtype)
        latents = pipe.vae.encode(image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return {"layer_input_latents": latents}


class QwenImageUnit_Inpaint(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("inpaint_mask", "height", "width", "inpaint_blur_size", "inpaint_blur_sigma"),
            output_params=("inpaint_mask",),
        )

    def process(self, pipe: QwenImagePipeline, inpaint_mask, height, width, inpaint_blur_size, inpaint_blur_sigma):
        if inpaint_mask is None:
            return {}
        inpaint_mask = pipe.preprocess_image(inpaint_mask.convert("RGB").resize((width // 8, height // 8)), min_value=0, max_value=1)
        inpaint_mask = inpaint_mask.mean(dim=1, keepdim=True)
        if inpaint_blur_size is not None and inpaint_blur_sigma is not None:
            from torchvision.transforms import GaussianBlur
            blur = GaussianBlur(kernel_size=inpaint_blur_size * 2 + 1, sigma=inpaint_blur_sigma)
            inpaint_mask = blur(inpaint_mask)
        return {"inpaint_mask": inpaint_mask}


class QwenImageUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt"},
            input_params_nega={"prompt": "negative_prompt"},
            input_params=("edit_image",),
            output_params=("prompt_emb", "prompt_emb_mask"),
            onload_model_names=("text_encoder",)
        )
        
    def extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)
        return split_result
    
    def calculate_dimensions(self, target_area, ratio):
        width = math.sqrt(target_area * ratio)
        height = width / ratio
        width = round(width / 32) * 32
        height = round(height / 32) * 32
        return width, height
    
    def resize_image(self, image, target_area=384*384):
        width, height = self.calculate_dimensions(target_area, image.size[0] / image.size[1])
        return image.resize((width, height))
    
    def encode_prompt(self, pipe: QwenImagePipeline, prompt):
        template = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        drop_idx = 34
        txt = [template.format(e) for e in prompt]
        model_inputs = pipe.tokenizer(txt, max_length=4096+drop_idx, padding=True, truncation=True, return_tensors="pt").to(pipe.device)
        if model_inputs.input_ids.shape[1] >= 1024:
            print(f"Warning!!! QwenImage model was trained on prompts up to 512 tokens. Current prompt requires {model_inputs['input_ids'].shape[1] - drop_idx} tokens, which may lead to unpredictable behavior.")
        hidden_states = pipe.text_encoder(input_ids=model_inputs.input_ids, attention_mask=model_inputs.attention_mask, output_hidden_states=True,)[-1]
        split_hidden_states = self.extract_masked_hidden(hidden_states, model_inputs.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        return split_hidden_states
        
    def encode_prompt_edit(self, pipe: QwenImagePipeline, prompt, edit_image):
        template =  "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"
        drop_idx = 64
        txt = [template.format(e) for e in prompt]
        model_inputs = pipe.processor(text=txt, images=edit_image, padding=True, return_tensors="pt").to(pipe.device)
        hidden_states = pipe.text_encoder(input_ids=model_inputs.input_ids, attention_mask=model_inputs.attention_mask, pixel_values=model_inputs.pixel_values, image_grid_thw=model_inputs.image_grid_thw, output_hidden_states=True,)[-1]
        split_hidden_states = self.extract_masked_hidden(hidden_states, model_inputs.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        return split_hidden_states
    
    def encode_prompt_edit_multi(self, pipe: QwenImagePipeline, prompt, edit_image):
        template =  "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        drop_idx = 64
        img_prompt_template = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
        base_img_prompt = "".join([img_prompt_template.format(i + 1) for i in range(len(edit_image))])
        txt = [template.format(base_img_prompt + e) for e in prompt]
        edit_image = [self.resize_image(image) for image in edit_image]
        model_inputs = pipe.processor(text=txt, images=edit_image, padding=True, return_tensors="pt").to(pipe.device)
        hidden_states = pipe.text_encoder(input_ids=model_inputs.input_ids, attention_mask=model_inputs.attention_mask, pixel_values=model_inputs.pixel_values, image_grid_thw=model_inputs.image_grid_thw, output_hidden_states=True,)[-1]
        split_hidden_states = self.extract_masked_hidden(hidden_states, model_inputs.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        return split_hidden_states

    def process(self, pipe: QwenImagePipeline, prompt, edit_image=None) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        if pipe.text_encoder is not None:
            prompt = [prompt]
            if edit_image is None:
                split_hidden_states = self.encode_prompt(pipe, prompt)
            elif isinstance(edit_image, Image.Image):
                split_hidden_states = self.encode_prompt_edit(pipe, prompt, edit_image)
            else:
                split_hidden_states = self.encode_prompt_edit_multi(pipe, prompt, edit_image)
            attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
            max_seq_len = max([e.size(0) for e in split_hidden_states])
            prompt_embeds = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states])
            encoder_attention_mask = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list])
            prompt_embeds = prompt_embeds.to(dtype=pipe.torch_dtype, device=pipe.device)
            return {"prompt_emb": prompt_embeds, "prompt_emb_mask": encoder_attention_mask}
        else:
            return {}


class QwenImageUnit_EntityControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            take_over=True,
            input_params=("eligen_entity_prompts", "width", "height", "eligen_enable_on_negative", "cfg_scale"),
            output_params=("entity_prompt_emb", "entity_masks", "entity_prompt_emb_mask"),
            onload_model_names=("text_encoder",)
        )

    def extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)
        return split_result

    def get_prompt_emb(self, pipe: QwenImagePipeline, prompt) -> dict:
        if pipe.text_encoder is not None:
            prompt = [prompt]
            template = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
            drop_idx = 34
            txt = [template.format(e) for e in prompt]
            txt_tokens = pipe.tokenizer(txt, max_length=1024+drop_idx, padding=True, truncation=True, return_tensors="pt").to(pipe.device)
            hidden_states = pipe.text_encoder(input_ids=txt_tokens.input_ids, attention_mask=txt_tokens.attention_mask, output_hidden_states=True,)[-1]
            
            split_hidden_states = self.extract_masked_hidden(hidden_states, txt_tokens.attention_mask)
            split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
            attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
            max_seq_len = max([e.size(0) for e in split_hidden_states])
            prompt_embeds = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states])
            encoder_attention_mask = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list])
            prompt_embeds = prompt_embeds.to(dtype=pipe.torch_dtype, device=pipe.device)
            return {"prompt_emb": prompt_embeds, "prompt_emb_mask": encoder_attention_mask}
        else:
            return {}

    def preprocess_masks(self, pipe, masks, height, width, dim):
        out_masks = []
        for mask in masks:
            mask = pipe.preprocess_image(mask.resize((width, height), resample=Image.NEAREST)).mean(dim=1, keepdim=True) > 0
            mask = mask.repeat(1, dim, 1, 1).to(device=pipe.device, dtype=pipe.torch_dtype)
            out_masks.append(mask)
        return out_masks

    def prepare_entity_inputs(self, pipe, entity_prompts, entity_masks, width, height):
        entity_masks = self.preprocess_masks(pipe, entity_masks, height//8, width//8, 1)
        entity_masks = torch.cat(entity_masks, dim=0).unsqueeze(0) # b, n_mask, c, h, w
        prompt_embs, prompt_emb_masks = [], []
        for entity_prompt in entity_prompts:
            prompt_emb_dict = self.get_prompt_emb(pipe, entity_prompt)
            prompt_embs.append(prompt_emb_dict['prompt_emb'])
            prompt_emb_masks.append(prompt_emb_dict['prompt_emb_mask'])
        return prompt_embs, prompt_emb_masks, entity_masks

    def prepare_eligen(self, pipe, prompt_emb_nega, eligen_entity_prompts, eligen_entity_masks, width, height, enable_eligen_on_negative, cfg_scale):
        entity_prompt_emb_posi, entity_prompt_emb_posi_mask, entity_masks_posi = self.prepare_entity_inputs(pipe, eligen_entity_prompts, eligen_entity_masks, width, height)
        if enable_eligen_on_negative and cfg_scale != 1.0:
            entity_prompt_emb_nega = [prompt_emb_nega['prompt_emb']] * len(entity_prompt_emb_posi)
            entity_prompt_emb_nega_mask = [prompt_emb_nega['prompt_emb_mask']] * len(entity_prompt_emb_posi)
            entity_masks_nega = entity_masks_posi
        else:
            entity_prompt_emb_nega, entity_prompt_emb_nega_mask, entity_masks_nega = None, None, None
        eligen_kwargs_posi = {"entity_prompt_emb": entity_prompt_emb_posi, "entity_masks": entity_masks_posi, "entity_prompt_emb_mask": entity_prompt_emb_posi_mask}
        eligen_kwargs_nega = {"entity_prompt_emb": entity_prompt_emb_nega, "entity_masks": entity_masks_nega, "entity_prompt_emb_mask": entity_prompt_emb_nega_mask}
        return eligen_kwargs_posi, eligen_kwargs_nega

    def process(self, pipe: QwenImagePipeline, inputs_shared, inputs_posi, inputs_nega):
        eligen_entity_prompts, eligen_entity_masks = inputs_shared.get("eligen_entity_prompts", None), inputs_shared.get("eligen_entity_masks", None)
        if eligen_entity_prompts is None or eligen_entity_masks is None or len(eligen_entity_prompts) == 0 or len(eligen_entity_masks) == 0:
            return inputs_shared, inputs_posi, inputs_nega
        pipe.load_models_to_device(self.onload_model_names)
        eligen_enable_on_negative = inputs_shared.get("eligen_enable_on_negative", False)
        eligen_kwargs_posi, eligen_kwargs_nega = self.prepare_eligen(pipe, inputs_nega,
            eligen_entity_prompts, eligen_entity_masks, inputs_shared["width"], inputs_shared["height"],
            eligen_enable_on_negative, inputs_shared["cfg_scale"])
        inputs_posi.update(eligen_kwargs_posi)
        if inputs_shared.get("cfg_scale", 1.0) != 1.0:
            inputs_nega.update(eligen_kwargs_nega)
        return inputs_shared, inputs_posi, inputs_nega



class QwenImageUnit_BlockwiseControlNet(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("blockwise_controlnet_inputs", "tiled", "tile_size", "tile_stride"),
            output_params=("blockwise_controlnet_conditioning",),
            onload_model_names=("vae",)
        )

    def apply_controlnet_mask_on_latents(self, pipe, latents, mask):
        mask = (pipe.preprocess_image(mask) + 1) / 2
        mask = mask.mean(dim=1, keepdim=True)
        mask = 1 - torch.nn.functional.interpolate(mask, size=latents.shape[-2:])
        latents = torch.concat([latents, mask], dim=1)
        return latents

    def apply_controlnet_mask_on_image(self, pipe, image, mask):
        mask = mask.resize(image.size)
        mask = pipe.preprocess_image(mask).mean(dim=[0, 1]).cpu()
        image = np.array(image)
        image[mask > 0] = 0
        image = Image.fromarray(image)
        return image

    def process(self, pipe: QwenImagePipeline, blockwise_controlnet_inputs: list[ControlNetInput], tiled, tile_size, tile_stride):
        if blockwise_controlnet_inputs is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        conditionings = []
        for controlnet_input in blockwise_controlnet_inputs:
            image = controlnet_input.image
            if controlnet_input.inpaint_mask is not None:
                image = self.apply_controlnet_mask_on_image(pipe, image, controlnet_input.inpaint_mask)

            image = pipe.preprocess_image(image).to(device=pipe.device, dtype=pipe.torch_dtype)
            image = pipe.vae.encode(image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)

            if controlnet_input.inpaint_mask is not None:
                image = self.apply_controlnet_mask_on_latents(pipe, image, controlnet_input.inpaint_mask)
            conditionings.append(image)
            
        return {"blockwise_controlnet_conditioning": conditionings}


class QwenImageUnit_EditImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("edit_image", "tiled", "tile_size", "tile_stride", "edit_image_auto_resize"),
            output_params=("edit_latents", "edit_image"),
            onload_model_names=("vae",)
        )


    def calculate_dimensions(self, target_area, ratio):
        import math
        target_area = 512 * 512
        width = math.sqrt(target_area * ratio)
        height = width / ratio
        width = round(width / 32) * 32
        height = round(height / 32) * 32
        return width, height


    def edit_image_auto_resize(self, edit_image):
        calculated_width, calculated_height = self.calculate_dimensions(1024 * 1024, edit_image.size[0] / edit_image.size[1])
        print(f"[DEBUG pipeline] edit_image_auto_resize: {edit_image.size} → ({calculated_width}×{calculated_height})  target_area=512*512")
        return edit_image.resize((calculated_width, calculated_height))


    def process(self, pipe: QwenImagePipeline, edit_image, tiled, tile_size, tile_stride,
                edit_image_auto_resize=False):
        if edit_image is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        if isinstance(edit_image, Image.Image):
            resized_edit_image = self.edit_image_auto_resize(edit_image) if edit_image_auto_resize else edit_image
            edit_image = pipe.preprocess_image(resized_edit_image).to(device=pipe.device, dtype=pipe.torch_dtype)
            edit_latents = pipe.vae.encode(edit_image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        else:
            resized_edit_image, edit_latents = [], []
            for image in edit_image:
                if edit_image_auto_resize:
                    image = self.edit_image_auto_resize(image)
                resized_edit_image.append(image)
                image = pipe.preprocess_image(image).to(device=pipe.device, dtype=pipe.torch_dtype)
                latents = pipe.vae.encode(image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
                edit_latents.append(latents)
                del image
                torch.cuda.empty_cache()

        return {"edit_latents": edit_latents, "edit_image": resized_edit_image}


class QwenImageUnit_Image2LoRAEncode(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("image2lora_images",),
            output_params=("image2lora_x", "image2lora_residual", "image2lora_residual_highres"),
            onload_model_names=("siglip2_image_encoder", "dinov3_image_encoder", "text_encoder"),
        )
        from ..core.data.operators import ImageCropAndResize
        self.processor_lowres = ImageCropAndResize(height=28*8, width=28*8)
        self.processor_highres = ImageCropAndResize(height=1024, width=1024)

    def extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)
        return split_result

    def encode_prompt_edit(self, pipe: QwenImagePipeline, prompt, edit_image):
        prompt = [prompt]
        template =  "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"
        drop_idx = 64
        txt = [template.format(e) for e in prompt]
        model_inputs = pipe.processor(text=txt, images=edit_image, padding=True, return_tensors="pt").to(pipe.device)
        hidden_states = pipe.text_encoder(input_ids=model_inputs.input_ids, attention_mask=model_inputs.attention_mask, pixel_values=model_inputs.pixel_values, image_grid_thw=model_inputs.image_grid_thw, output_hidden_states=True,)[-1]
        split_hidden_states = self.extract_masked_hidden(hidden_states, model_inputs.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states])
        prompt_embeds = prompt_embeds.to(dtype=pipe.torch_dtype, device=pipe.device)
        return prompt_embeds.view(1, -1)
    
    def encode_images_using_siglip2(self, pipe: QwenImagePipeline, images: list[Image.Image]):
        pipe.load_models_to_device(["siglip2_image_encoder"])
        embs = []
        for image in images:
            image = self.processor_highres(image)
            embs.append(pipe.siglip2_image_encoder(image).to(pipe.torch_dtype))
        embs = torch.stack(embs)
        return embs
    
    def encode_images_using_dinov3(self, pipe: QwenImagePipeline, images: list[Image.Image]):
        pipe.load_models_to_device(["dinov3_image_encoder"])
        embs = []
        for image in images:
            image = self.processor_highres(image)
            embs.append(pipe.dinov3_image_encoder(image).to(pipe.torch_dtype))
        embs = torch.stack(embs)
        return embs
    
    def encode_images_using_qwenvl(self, pipe: QwenImagePipeline, images: list[Image.Image], highres=False):
        pipe.load_models_to_device(["text_encoder"])
        embs = []
        for image in images:
            image = self.processor_highres(image) if highres else self.processor_lowres(image)
            embs.append(self.encode_prompt_edit(pipe, prompt="", edit_image=image))
        embs = torch.stack(embs)
        return embs

    def encode_images(self, pipe: QwenImagePipeline, images: list[Image.Image]):
        if images is None:
            return {}
        if not isinstance(images, list):
            images = [images]
        embs_siglip2 = self.encode_images_using_siglip2(pipe, images)
        embs_dinov3 = self.encode_images_using_dinov3(pipe, images)
        x = torch.concat([embs_siglip2, embs_dinov3], dim=-1)
        residual = None
        residual_highres = None
        if pipe.image2lora_coarse is not None:
            residual = self.encode_images_using_qwenvl(pipe, images, highres=False)
        if pipe.image2lora_fine is not None:
            residual_highres = self.encode_images_using_qwenvl(pipe, images, highres=True)
        return x, residual, residual_highres

    def process(self, pipe: QwenImagePipeline, image2lora_images):
        if image2lora_images is None:
            return {}
        x, residual, residual_highres = self.encode_images(pipe, image2lora_images)
        return {"image2lora_x": x, "image2lora_residual": residual, "image2lora_residual_highres": residual_highres}


class QwenImageUnit_Image2LoRADecode(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("image2lora_x", "image2lora_residual", "image2lora_residual_highres"),
            output_params=("lora",),
            onload_model_names=("image2lora_coarse", "image2lora_fine", "image2lora_style"),
        )
    
    def process(self, pipe: QwenImagePipeline, image2lora_x, image2lora_residual, image2lora_residual_highres):
        if image2lora_x is None:
            return {}
        loras = []
        if pipe.image2lora_style is not None:
            pipe.load_models_to_device(["image2lora_style"])
            for x in image2lora_x:
                loras.append(pipe.image2lora_style(x=x, residual=None))
        if pipe.image2lora_coarse is not None:
            pipe.load_models_to_device(["image2lora_coarse"])
            for x, residual in zip(image2lora_x, image2lora_residual):
                loras.append(pipe.image2lora_coarse(x=x, residual=residual))
        if pipe.image2lora_fine is not None:
            pipe.load_models_to_device(["image2lora_fine"])
            for x, residual in zip(image2lora_x, image2lora_residual_highres):
                loras.append(pipe.image2lora_fine(x=x, residual=residual))
        lora = merge_lora(loras, alpha=1 / len(image2lora_x))
        return {"lora": lora}


class QwenImageUnit_ContextImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("context_image", "height", "width", "tiled", "tile_size", "tile_stride"),
            output_params=("context_latents",),
            onload_model_names=("vae",)
        )

    def process(self, pipe: QwenImagePipeline, context_image, height, width, tiled, tile_size, tile_stride):
        if context_image is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        context_image = pipe.preprocess_image(context_image.resize((width, height))).to(device=pipe.device, dtype=pipe.torch_dtype)
        context_latents = pipe.vae.encode(context_image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return {"context_latents": context_latents}
    

class QwenImageUnit_DragMaskToTokens(PipelineUnit):
    """
    Convert image-space drag mask to token-space mask on the DiT grid.
    The token grid for Qwen-Image is (height//16, width//16), matching model_fn_qwen_image.
    """
    def __init__(self):
        super().__init__(
            input_params=("drag_mask", "drag_bx", "drag_by", "drag_scale", "drag_theta", "height", "width", "depth", "source_depth", "separate_source_depth", "inject_noise_in_vacated_positions", "drag_objects",),
            output_params=("drag_token_mask", "drag_object_token_masks",),
        )

    def _tokenize_mask(self, pipe: QwenImagePipeline, mask_img, Ht, Wt):
        mask_resized = mask_img.resize((Wt, Ht), Image.NEAREST)
        mask_tensor = pipe.preprocess_image(mask_resized, min_value=0.0, max_value=1.0)
        mask_tensor = mask_tensor.mean(dim=1, keepdim=True)
        return (mask_tensor > 0.5).to(dtype=pipe.torch_dtype, device=pipe.device)

    def process(self, pipe: QwenImagePipeline, drag_mask, drag_bx, drag_by, drag_scale, drag_theta, height, width, depth, source_depth, separate_source_depth, inject_noise_in_vacated_positions, drag_objects):
        Ht, Wt = height // 16, width // 16
        outputs = {}

        # Multi-object path: tokenize each object's own mask, carry its own bx/by/scale/theta through.
        if drag_objects:
            resolved = []
            for i, obj in enumerate(drag_objects):
                obj_theta = _drag_rotate_to_theta(obj.get("theta", 0.0))
                token_mask = self._tokenize_mask(pipe, obj["mask"], Ht, Wt)
                print(f"[DEBUG pipeline] DragObjects[{i}]: token grid={Ht}×{Wt}  bx={obj['bx']:.2f}  by={obj['by']:.2f}  "
                      f"scale={obj.get('scale', 1.0):.4f}  theta={obj_theta:.2f}  token_mask_px={int(token_mask.sum())}")
                resolved.append({
                    "token_mask": token_mask,
                    "bx": obj["bx"], "by": obj["by"],
                    "scale": obj.get("scale", 1.0), "theta": obj_theta,
                })
            outputs["drag_object_token_masks"] = resolved

        # Existing single-mask path -- untouched.
        if drag_mask is None or (drag_bx == 0.0 and drag_by == 0.0 and drag_theta == 0.0):
            return outputs

        print(f"[DEBUG pipeline] DragMask: input mask size={drag_mask.size}  height={height}  width={width}")
        print(f"[DEBUG pipeline] DragMask: token grid={Ht}×{Wt}  drag_bx={drag_bx:.2f}  drag_by={drag_by:.2f}  drag_scale={drag_scale:.4f}  drag_theta={drag_theta:.2f}")
        drag_token_mask = self._tokenize_mask(pipe, drag_mask, Ht, Wt)
        print(f"[DEBUG pipeline] DragMask: token mask px={int(drag_token_mask.sum())}  shape={drag_token_mask.shape}")
        if depth is not None:
            print(f"[DEBUG pipeline] DragMask: depth shape={depth.shape}  range=[{depth.min():.4f}, {depth.max():.4f}]")
        if source_depth is not None:
            print(f"[DEBUG pipeline] DragMask: source_depth shape={source_depth.shape}  range=[{source_depth.min():.4f}, {source_depth.max():.4f}]")
        outputs["drag_token_mask"] = drag_token_mask
        return outputs


# In-place rotation presets for the masked object, expressed as a RoPE-grid rotation
# angle (degrees) around the drag mask's centroid. Positive angle = counterclockwise
# (matches the usual "rotate left" convention in image editors); see _warp_single_qwen_rope.
_DRAG_ROTATE_PRESETS = {"left": 90.0, "right": -90.0, "180": 180.0, "opposite": 180.0}


def _drag_rotate_to_theta(drag_rotate) -> float:
    if drag_rotate is None:
        return 0.0
    if isinstance(drag_rotate, (int, float)):
        return float(drag_rotate)
    key = str(drag_rotate).strip().lower()
    if key in _DRAG_ROTATE_PRESETS:
        return _DRAG_ROTATE_PRESETS[key]
    try:
        # Arbitrary angle in degrees (e.g. "10", "-10"). Note: unlike the 90/180
        # presets, non-multiples of 90 don't land on exact token-grid positions,
        # so the nearest-neighbor RoPE gather will alias more at small angles.
        return float(key)
    except ValueError:
        raise ValueError(
            f"drag_rotate must be one of {sorted(_DRAG_ROTATE_PRESETS)}, a numeric degree value, or None, got {drag_rotate!r}"
        )


def _warp_single_qwen_rope(
    emb: torch.Tensor,
    drag_token_mask: torch.Tensor,
    drag_bx: float,
    drag_by: float,
    drag_scale: float,
    progress_id: int,
    num_inference_steps: int,
    inject_noise_in_vacated_positions: bool = True,  # ← NEW
    warp_end_step: int = None,
    drag_theta: float = 0.0,  # ← NEW: in-place rotation (degrees, + = counterclockwise) around the mask centroid
) -> torch.Tensor:
    """
    Warp a single complex RoPE tensor `emb` (image or text) inside `drag_token_mask`.

    New behavior (image case):
      * Treat `drag_bx`, `drag_by` as pixel offsets.
      * Convert to token offsets using a 16px patch size.
      * For tokens inside the mask, copy RoPE from the location shifted by
        (dy_tok, dx_tok) on the H_t x W_t token grid, so masked tokens
        "act as if" they lived at the dragged location.
        * If inject_noise_in_vacated_positions=True, source positions that are
        vacated are filled with noise RoPE tokens.
      * `drag_theta` additionally rotates the masked tokens' sampling positions
        around the mask centroid before the translate/scale above is applied.

    Supports shapes:
      * [L, D]   (no batch)
      * [B, L, D]
    """
    # Only warp complex RoPE tensors
    if not torch.is_complex(emb):
        return emb

    # Only warp during the first half of the diffusion steps
    # half = max(1, num_inference_steps // 2)
    # if progress_id >= half:
    #     return emb

    # Stop warping after warp_end_step (cumulative-from-start experiment)
    if warp_end_step is not None and progress_id >= warp_end_step:
        return emb

    # No-op if there is no drag and no rotation
    if (drag_bx == 0.0) and (drag_by == 0.0) and (drag_theta == 0.0):
        return emb

    # Token grid from mask: drag_token_mask is [B_mask, 1, Ht, Wt]
    if drag_token_mask is None or drag_token_mask.numel() == 0:
        return emb

    B_mask, _, Ht, Wt = drag_token_mask.shape
    L_grid = Ht * Wt

    # Convert pixel delta to token delta (Qwen mask is at H/16 x W/16)
    patch_px = 16.0
    dx_tok = int(round(drag_bx / patch_px))
    dy_tok = int(round(drag_by / patch_px))
    if progress_id == 0:
        print(f"[DEBUG pipeline] RoPE warp: drag_bx={drag_bx:.2f}  drag_by={drag_by:.2f}  "
              f"dx_tok={dx_tok}  dy_tok={dy_tok}  scale={drag_scale:.4f}  theta={drag_theta:.2f}  "
              f"grid={Ht}×{Wt}  step={progress_id}/{num_inference_steps}")
    if dx_tok == 0 and dy_tok == 0 and drag_theta == 0.0:
        return emb

    device = emb.device

    # Normalize emb to [B, L, D]
    squeezed = False
    if emb.ndim == 2:
        L, D = emb.shape
        B = 1
        emb_b = emb.unsqueeze(0)  # [1, L, D]
        squeezed = True
    elif emb.ndim == 3:
        B, L, D = emb.shape
        emb_b = emb
    else:
        # Unsupported rank; leave unchanged
        return emb

    # We expect image RoPE length to match the Ht*Wt grid
    if L < L_grid:
        raise ValueError(
            f"RoPE length L={L} is smaller than Ht*Wt={L_grid}. "
            "Directional warp assumes at least one full HxW image token grid."
        )

    # Split into: head = main image tokens, tail = extra image tokens
    emb_head = emb_b[:, :L_grid, :]   # [B, L_grid, D]
    emb_tail = emb_b[:, L_grid:, :]   # [B, L - L_grid, D]

    # Build 2D → 1D index mapping for the shifted coordinates
    if drag_scale != 1.0 or drag_theta != 0.0:
        rows = torch.arange(Ht, device=device, dtype=torch.float32)
        cols = torch.arange(Wt, device=device, dtype=torch.float32)
    else:
        rows = torch.arange(Ht, device=device)
        cols = torch.arange(Wt, device=device)
    rr, cc = torch.meshgrid(rows, cols, indexing="ij")  # [Ht, Wt]


    mask_2d = drag_token_mask[:, 0].to(device=device)

    # Flatten/broadcast mask to [B, L_grid]
    mask_flat = (drag_token_mask[:, 0] > 0.5).to(device=device)  # [B_mask, Ht, Wt]
    mask_flat = mask_flat.view(B_mask, L_grid)                   # [B_mask, L_grid]

    if B_mask == 1 and B > 1:
        mask_flat = mask_flat.expand(B, -1)
    elif B_mask != B:
        raise ValueError(
            f"drag_token_mask batch={B_mask} does not match RoPE batch={B}"
        )
    print("[Scaling] scaling factor:", drag_scale)
    # rr_dst = (rr + dy_tok).clamp(0, Ht - 1)
    # cc_dst = (cc + dx_tok).clamp(0, Wt - 1)
    # idx_src = (rr_dst * Wt + cc_dst).reshape(-1).long()  # [L_grid]
        # ===== APPLY TRANSFORMATION =====
    if drag_scale != 1.0 or drag_theta != 0.0:
        # SCALE + ROTATE MODE: rotate(scale(pos - center)) + center + drag

        # Calculate mask center for scaling/rotation
        if mask_2d[0].any():
            mask_indices = torch.nonzero(mask_2d[0], as_tuple=False)
            center_coords = mask_indices.float().mean(dim=0)
            center_h = center_coords[0].item()
            center_w = center_coords[1].item()
            print(f"[Scaling] Center: ({center_h:.2f}, {center_w:.2f}), Scale: {drag_scale:.4f}x, Theta: {drag_theta:.2f}°")
        else:
            # No mask - use image center
            center_h = (Ht - 1) / 2.0
            center_w = (Wt - 1) / 2.0

        # Apply scaling around center
        dr = (rr - center_h) * drag_scale
        dc = (cc - center_w) * drag_scale

        # Apply rotation around center (+theta = counterclockwise)
        if drag_theta != 0.0:
            theta_rad = math.radians(drag_theta)
            cos_t, sin_t = math.cos(theta_rad), math.sin(theta_rad)
            dr, dc = dr * cos_t - dc * sin_t, dr * sin_t + dc * cos_t

        # Apply translation
        transformed_rr = center_h + dr + dy_tok
        transformed_cc = center_w + dc + dx_tok

        # Clamp and convert to indices
        transformed_rr = transformed_rr.clamp(0, Ht - 1)
        transformed_cc = transformed_cc.clamp(0, Wt - 1)
        idx_dst_rr = transformed_rr.round().long()
        idx_dst_cc = transformed_cc.round().long()
        idx_dst = (idx_dst_rr * Wt + idx_dst_cc).reshape(-1).long()
    else:
        # TRANSLATION ONLY MODE: pos + drag (original behavior)
        rr_dst = (rr + dy_tok).clamp(0, Ht - 1)
        cc_dst = (cc + dx_tok).clamp(0, Wt - 1)
        idx_dst = (rr_dst * Wt + cc_dst).reshape(-1).long()
        # For noise injection compatibility
        idx_dst_rr = rr_dst.long()
        idx_dst_cc = cc_dst.long()

    # Build per-batch source indices [B, L_grid]
    idx_src_b = idx_dst.unsqueeze(0).expand(B, -1)  # [B, L_grid]

    # Gather shifted RoPE from the destination positions, only for the head
    batch_idx = torch.arange(B, device=device).unsqueeze(-1).expand(B, L_grid)
    emb_shifted = emb_head[batch_idx, idx_src_b]  # [B, L_grid, D]

    # Apply only inside the mask, only on the head
    mask_bc = mask_flat.view(B, L_grid, 1)
    emb_head_warped = torch.where(mask_bc, emb_shifted, emb_head)  # [B, L_grid, D]

    # if inject_noise_in_vacated_positions:
    #     print("Injecting noise into vacated positions...")
        
    #     # Create destination mask
    #     destination_mask = torch.zeros_like(mask_2d, dtype=torch.bool)
    #     original_mask_positions = mask_2d[0] > 0.5
    #     destination_mask[0, idx_dst_rr[original_mask_positions], idx_dst_cc[original_mask_positions]] = True
        
    #     # Vacated = original mask positions NOT covered by destination
    #     source_mask = (mask_2d > 0.5) & ~destination_mask
        
    #     # Flatten and apply
    #     source_mask_flat = source_mask.view(B_mask, L_grid)
    #     if B_mask == 1 and B > 1:
    #         source_mask_flat = source_mask_flat.expand(B, -1)
        
    #     # Generate noise
    #     noise = torch.randn_like(emb_head_warped) * 0.1
    #     if torch.is_complex(emb_head_warped):
    #         noise = noise.to(dtype=emb_head_warped.dtype)
        
    #     # Apply noise to vacated positions
    #     source_mask_bc = source_mask_flat.view(B, L_grid, 1)
    #     num_vacated = source_mask_flat[0].sum().item()
    #     if num_vacated > 0:
    #         print(f"[Noise Injection] Filled {num_vacated} vacated tokens")
        
    #     # Save visualization (one-liner)
    #     Image.fromarray((source_mask_flat[0].view(Ht, Wt).float().cpu().numpy() * 255).astype(np.uint8)).save("source_mask.png")
        
    #     emb_head_out = torch.where(source_mask_bc, noise, emb_head_warped)
    # else:
    emb_head_out = emb_head_warped
    # Concatenate warped head with untouched tail
    emb_out = torch.cat([emb_head_out, emb_tail], dim=1)  # [B, L, D]

    if squeezed:
        return emb_out[0]
    return emb_out

def warp_qwen_image_rope(
    image_rotary_emb,
    drag_token_mask: torch.Tensor,
    drag_bx: float,
    drag_by: float,
    drag_scale: float,
    progress_id: int,
    num_inference_steps: int,
    inject_noise_in_vacated_positions: bool = True,  # ← NEW
    warp_end_step: int = None,
    drag_theta: float = 0.0,  # ← NEW
):
    """
    Apply a constant local RoPE warp (inside drag_token_mask)
    only for the first half of the diffusion steps.

    `image_rotary_emb` can be either:
      * a single complex tensor [L, D] or [B, L, D], or
      * a tuple (img_freqs, txt_freqs) as returned by QwenEmbedRope / QwenEmbedLayer3DRope.
    """
    # Tuple case: (img_freqs, txt_freqs)
    if isinstance(image_rotary_emb, (tuple, list)) and len(image_rotary_emb) == 2:
        img_freqs, txt_freqs = image_rotary_emb
        img_freqs_warped = _warp_single_qwen_rope(
            img_freqs, drag_token_mask, drag_bx, drag_by, drag_scale, progress_id, num_inference_steps, inject_noise_in_vacated_positions, warp_end_step, drag_theta
        )
        return (img_freqs_warped, txt_freqs)

    # Single tensor case: keep old behavior but routed through helper
    if isinstance(image_rotary_emb, torch.Tensor):
        return _warp_single_qwen_rope(
            image_rotary_emb, drag_token_mask, drag_bx, drag_by, drag_scale, progress_id, num_inference_steps, inject_noise_in_vacated_positions, warp_end_step, drag_theta
        )

    # Unknown type: no-op
    return image_rotary_emb


def warp_qwen_image_rope_edit_segments(
    image_rotary_emb,
    img_shapes,
    edit_segment_indices,
    height: int,
    width: int,
    drag_token_mask: torch.Tensor,
    drag_bx: float,
    drag_by: float,
    drag_scale: float,
    progress_id: int,
    num_inference_steps: int,
    inject_noise_in_vacated_positions: bool = True,  # ← NEW
    warp_end_step: int = None,
    drag_theta: float = 0.0,  # ← NEW
    drag_objects_resolved: list = None,  # multi-object: [{"token_mask","bx","by","scale","theta"}, ...]
):
    """
    Warp only the RoPE segments corresponding to edit-image latents.

    If shapes/format are inconsistent, falls back to `warp_qwen_image_rope`
    (which warps the main image segment).

    Multi-object mode (`drag_objects_resolved` provided): the first edit
    segment (source, containing all objects) gets every object's warp
    applied in sequence; each subsequent edit segment (assumed to be that
    object's own reference mask, in the same order as `drag_objects_resolved`)
    gets only its own corresponding object's warp. Falls back to the single
    shared drag_mask/params for every segment when `drag_objects_resolved`
    is None (existing behavior, unchanged).
    """
    # we expect (img_freqs, txt_freqs)
    if not (isinstance(image_rotary_emb, (tuple, list)) and len(image_rotary_emb) == 2):
        return warp_qwen_image_rope(
            image_rotary_emb=image_rotary_emb,
            drag_token_mask=drag_token_mask,
            drag_bx=drag_bx,
            drag_by=drag_by,
            drag_scale=drag_scale,
            progress_id=progress_id,
            num_inference_steps=num_inference_steps,
            inject_noise_in_vacated_positions=inject_noise_in_vacated_positions,
            warp_end_step=warp_end_step,
            drag_theta=drag_theta,
        )

    img_freqs, txt_freqs = image_rotary_emb

    # normalize to [B, L, D]
    if img_freqs.ndim == 2:
        img_freqs_b = img_freqs.unsqueeze(0)
        squeezed = True
    elif img_freqs.ndim == 3:
        img_freqs_b = img_freqs
        squeezed = False
    else:
        # unexpected rank; fall back to main-image warp
        return warp_qwen_image_rope(
            image_rotary_emb=image_rotary_emb,
            drag_token_mask=drag_token_mask,
            drag_bx=drag_bx,
            drag_by=drag_by,
            drag_scale=drag_scale,
            progress_id=progress_id,
            num_inference_steps=num_inference_steps,
            inject_noise_in_vacated_positions=inject_noise_in_vacated_positions,
            warp_end_step=warp_end_step,
            drag_theta=drag_theta,
        )

    B, L_total, _ = img_freqs_b.shape

    # segment lengths and offsets from img_shapes
    seg_lens = [s[1] * s[2] for s in img_shapes]  # H*W per segment
    offsets = [0]
    for L_seg in seg_lens:
        offsets.append(offsets[-1] + L_seg)
    total_L = offsets[-1]

    if L_total < total_L:
        # shapes inconsistent; fall back to main-image warp
        return warp_qwen_image_rope(
            image_rotary_emb=image_rotary_emb,
            drag_token_mask=drag_token_mask,
            drag_bx=drag_bx,
            drag_by=drag_by,
            drag_scale=drag_scale,
            progress_id=progress_id,
            num_inference_steps=num_inference_steps,
            inject_noise_in_vacated_positions=inject_noise_in_vacated_positions,
            warp_end_step=warp_end_step,
            drag_theta=drag_theta,
        )

    H_main, W_main = height // 16, width // 16

    def _resize_token_mask(mask, H_seg, W_seg):
        if mask.shape[-2:] == (H_seg, W_seg):
            return mask
        return torch.nn.functional.interpolate(
            mask.float(), size=(H_seg, W_seg), mode="nearest",
        ).to(dtype=mask.dtype)

    # warp only edit segments in-place
    for i, seg_idx in enumerate(edit_segment_indices):
        if seg_idx < 0 or seg_idx + 1 >= len(offsets):
            continue
        start = offsets[seg_idx]
        end = offsets[seg_idx + 1]
        _, H_seg, W_seg = img_shapes[seg_idx]

        seg = img_freqs_b[:, start:end, :]  # [B, L_seg, D]

        if drag_objects_resolved:
            if i == 0:
                # source segment: contains every object. Each object's warp must
                # gather from the PRISTINE original (not from another object's
                # already-warped output) -- otherwise if object B's gather-source
                # coordinate happens to land inside object A's mask footprint,
                # B would pick up A's already-shifted content instead of the
                # true original background/object content. So: warp the
                # original independently per object, then combine via
                # torch.where on each object's own (disjoint) mask.
                seg_original = seg
                seg_combined = seg
                for obj in drag_objects_resolved:
                    mask_seg = _resize_token_mask(obj["token_mask"], H_seg, W_seg)
                    seg_obj_warped = _warp_single_qwen_rope(
                        seg_original, mask_seg, obj["bx"], obj["by"], obj.get("scale", 1.0),
                        progress_id, num_inference_steps,
                        inject_noise_in_vacated_positions=inject_noise_in_vacated_positions,
                        warp_end_step=warp_end_step, drag_theta=obj.get("theta", 0.0),
                    )
                    mask_bc = mask_seg.view(mask_seg.shape[0], -1, 1).to(dtype=torch.bool)
                    if mask_bc.shape[0] == 1 and seg_combined.shape[0] > 1:
                        mask_bc = mask_bc.expand(seg_combined.shape[0], -1, -1)
                    seg_combined = torch.where(mask_bc, seg_obj_warped, seg_combined)
                seg = seg_combined
            else:
                # subsequent edit segment = that object's own reference mask
                obj_idx = i - 1
                if obj_idx < len(drag_objects_resolved):
                    obj = drag_objects_resolved[obj_idx]
                    mask_seg = _resize_token_mask(obj["token_mask"], H_seg, W_seg)
                    seg = _warp_single_qwen_rope(
                        seg, mask_seg, obj["bx"], obj["by"], obj.get("scale", 1.0),
                        progress_id, num_inference_steps,
                        inject_noise_in_vacated_positions=inject_noise_in_vacated_positions,
                        warp_end_step=warp_end_step, drag_theta=obj.get("theta", 0.0),
                    )
            seg_warped = seg
        else:
            # existing single-mask behavior, unchanged
            mask_seg = drag_token_mask
            if H_seg != H_main or W_seg != W_main:
                mask_seg = _resize_token_mask(drag_token_mask, H_seg, W_seg)
            seg_warped = _warp_single_qwen_rope(
                seg,
                mask_seg,
                drag_bx,
                drag_by,
                drag_scale,
                progress_id,
                num_inference_steps,
                inject_noise_in_vacated_positions=inject_noise_in_vacated_positions,
                warp_end_step=warp_end_step,
                drag_theta=drag_theta,
            )
        img_freqs_b[:, start:end, :] = seg_warped

    img_freqs_out = img_freqs_b[0] if squeezed else img_freqs_b
    return (img_freqs_out, txt_freqs)


def model_fn_qwen_image(
    dit: QwenImageDiT = None,
    blockwise_controlnet: QwenImageBlockwiseMultiControlNet = None,
    latents=None,
    timestep=None,
    prompt_emb=None,
    prompt_emb_mask=None,
    height=None,
    width=None,
    blockwise_controlnet_conditioning=None,
    blockwise_controlnet_inputs=None,
    progress_id=0,
    num_inference_steps=1,
    entity_prompt_emb=None,
    entity_prompt_emb_mask=None,
    entity_masks=None,
    edit_latents=None,
    layer_input_latents=None,
    layer_num=None,
    context_latents=None,
    enable_fp8_attention=False,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    edit_rope_interpolation=False,
    zero_cond_t=False,
    drag_token_mask=None,
    drag_bx: float = 0.0,
    drag_by: float = 0.0,
    drag_scale: float = 1.0,
    drag_theta: float = 0.0,
    depth: torch.Tensor = None,
    source_depth: torch.Tensor = None,
    separate_source_depth: bool = False,
    inject_noise_in_vacated_positions: bool = True,  # ← NEW
    scheduled_noise: bool = False,
    noise_timestep: int = None,
    warp_end_step: int = None,
    drag_object_token_masks: list = None,  # multi-object warp: list of {"token_mask","bx","by","scale","theta"}
    depth_before_rope: bool = False,  # inject depth before RoPE rotation is computed (genuine unit-magnitude rotation), vs legacy post-hoc overwrite
    **kwargs
):
    # track which img_shapes entries correspond to edit images
    edit_segment_indices: list[int] = []

    if layer_num is None:
        layer_num = 1
        img_shapes = [(1, latents.shape[2]//2, latents.shape[3]//2)]
    else:
        layer_num = layer_num + 1
        img_shapes = [(1, latents.shape[2]//2, latents.shape[3]//2)] * layer_num
    txt_seq_lens = prompt_emb_mask.sum(dim=1).tolist()
    timestep = timestep / 1000
    
    image = rearrange(latents, "(B N) C (H P) (W Q) -> B (N H W) (C P Q)", H=height//16, W=width//16, P=2, Q=2, N=layer_num)
    image_seq_len = image.shape[1]

    if context_latents is not None:
        img_shapes += [(context_latents.shape[0], context_latents.shape[2]//2, context_latents.shape[3]//2)]
        context_image = rearrange(context_latents, "B C (H P) (W Q) -> B (H W) (C P Q)", H=context_latents.shape[2]//2, W=context_latents.shape[3]//2, P=2, Q=2)
        image = torch.cat([image, context_image], dim=1)
    if edit_latents is not None:
        edit_latents_list = edit_latents if isinstance(edit_latents, list) else [edit_latents]
        for e in edit_latents_list:
            img_shapes.append((e.shape[0], e.shape[2]//2, e.shape[3]//2))
            edit_segment_indices.append(len(img_shapes) - 1)
        edit_image = [rearrange(e, "B C (H P) (W Q) -> B (H W) (C P Q)", H=e.shape[2]//2, W=e.shape[3]//2, P=2, Q=2) for e in edit_latents_list]
        image = torch.cat([image] + edit_image, dim=1)
    if layer_input_latents is not None:
        layer_num = layer_num + 1
        img_shapes += [(layer_input_latents.shape[0], layer_input_latents.shape[2]//2, layer_input_latents.shape[3]//2)]
        layer_input_latents = rearrange(layer_input_latents, "B C (H P) (W Q) -> B (H W) (C P Q)", P=2, Q=2)
        image = torch.cat([image, layer_input_latents], dim=1)

    image = dit.img_in(image)
    if zero_cond_t:
        timestep = torch.cat([timestep, timestep * 0], dim=0)
        modulate_index = torch.tensor(
            [[0] * prod(sample[0]) + [1] * sum([prod(s) for s in sample[1:]]) for sample in [img_shapes]],
            device=timestep.device,
            dtype=torch.int,
        )
    else:
        modulate_index = None
    conditioning = dit.time_text_embed(
        timestep,
        image.dtype,
        addition_t_cond=None if not dit.time_text_embed.use_additional_t_cond else torch.tensor([0]).to(device=image.device, dtype=torch.long)
    )

    if entity_prompt_emb is not None:
        text, image_rotary_emb, attention_mask = dit.process_entity_masks(
            latents, prompt_emb, prompt_emb_mask, entity_prompt_emb, entity_prompt_emb_mask,
            entity_masks, height, width, image, img_shapes,
        )
    else:
        text = dit.txt_in(dit.txt_norm(prompt_emb))

        # ===== PRE-ROTATION DEPTH INJECTION (depth_before_rope=True) =====
        # Builds depth into the frame-axis rotation itself (genuine unit-magnitude
        # e^{i*depth*omega}, matching Flux/PE-Field), instead of overwriting an
        # already-computed rotation afterward. Legacy path (depth_before_rope=False,
        # default) is fully unchanged below.
        depth_maps_for_rope = None
        if depth_before_rope and (depth is not None or source_depth is not None):
            depth_maps_for_rope = {}
            if depth is not None:
                depth_maps_for_rope[0] = depth.to(device=latents.device, dtype=torch.float32)
            if len(edit_segment_indices) > 0:
                first_edit_idx = edit_segment_indices[0]
                if separate_source_depth and source_depth is not None:
                    depth_maps_for_rope[first_edit_idx] = source_depth.to(device=latents.device, dtype=torch.float32)
                elif depth is not None:
                    depth_maps_for_rope[first_edit_idx] = depth.to(device=latents.device, dtype=torch.float32)

        if edit_rope_interpolation:
            image_rotary_emb = dit.pos_embed.forward_sampling(img_shapes, txt_seq_lens, device=latents.device)
        else:
            image_rotary_emb = dit.pos_embed(img_shapes, txt_seq_lens, device=latents.device, depth_maps=depth_maps_for_rope)

        # ===== INJECT DEPTHS INTO ROPE (legacy post-hoc overwrite) =====
        if not depth_before_rope and (depth is not None or source_depth is not None):
            # Extract img_freqs from tuple
            if isinstance(image_rotary_emb, (tuple, list)) and len(image_rotary_emb) == 2:
                img_freqs, txt_freqs = image_rotary_emb
            else:
                img_freqs = image_rotary_emb
                txt_freqs = None
            
            Ht, Wt = height // 16, width // 16
            L_grid = Ht * Wt
            
            # Calculate segment offsets from img_shapes
            seg_lens = [s[1] * s[2] for s in img_shapes]  # H*W per segment
            offsets = [0]
            for L_seg in seg_lens:
                offsets.append(offsets[-1] + L_seg)
            
            # ===== 1. INJECT TARGET DEPTH (main image tokens) =====
            if depth is not None:
                depth_device = depth.to(device=latents.device, dtype=latents.dtype)
                
                # Handle different depth shapes
                if depth_device.dim() == 2:
                    depth_for_resize = depth_device.unsqueeze(0).unsqueeze(0)
                elif depth_device.dim() == 3:
                    depth_for_resize = depth_device.unsqueeze(0)
                elif depth_device.dim() == 4:
                    depth_for_resize = depth_device
                else:
                    raise ValueError(f"Unsupported target depth shape: {depth_device.shape}")
                
                # Resize to token grid
                if depth_for_resize.shape[-2:] != (Ht, Wt):
                    depth_resized = torch.nn.functional.interpolate(
                        depth_for_resize,
                        size=(Ht, Wt),
                        mode='bilinear',
                        align_corners=False
                    )
                else:
                    depth_resized = depth_for_resize
                
                depth_flat = depth_resized.squeeze(0).squeeze(0).flatten()
                
                print(f"[Target Depth] Range: [{depth_flat.min():.4f}, {depth_flat.max():.4f}]")
                
                # Inject into MAIN IMAGE tokens (img_shapes[0])
                target_start = offsets[0]
                target_end = offsets[1]
                
                if img_freqs.ndim == 2:  # [L, D]
                    if torch.is_complex(img_freqs):
                        img_freqs[target_start:target_end, 0] = depth_flat.to(dtype=img_freqs.dtype) #before source_depth_flat.to(dtype=img_freqs.dtype)
                    else:
                        img_freqs[target_start:target_end, 0] = depth_flat
                    print(f"[Target Depth] Injected into img_freqs[{target_start}:{target_end}, 0]")
                elif img_freqs.ndim == 3:  # [B, L, D]
                    if torch.is_complex(img_freqs):
                        img_freqs[:, target_start:target_end, 0] = depth_flat.unsqueeze(0).to(dtype=img_freqs.dtype)
                    else:
                        img_freqs[:, target_start:target_end, 0] = depth_flat.unsqueeze(0)
                    print(f"[Target Depth] Injected into img_freqs[:, {target_start}:{target_end}, 0]")
            
            # ===== 2. INJECT DEPTH into source tokens (first edit image tokens) =====
            # separate_source_depth=True  → use source_depth PE-field for source tokens
            # separate_source_depth=False → use target depth for source tokens (original behaviour)
            if depth is not None and len(edit_segment_indices) > 0:
                first_edit_idx = edit_segment_indices[0]
                _, H_src, W_src = img_shapes[first_edit_idx]
                source_start = offsets[first_edit_idx]
                source_end = offsets[first_edit_idx + 1]

                if separate_source_depth and source_depth is not None:
                    src_d = source_depth.to(device=latents.device, dtype=latents.dtype)
                    if src_d.dim() == 2:
                        src_d = src_d.unsqueeze(0).unsqueeze(0)
                    elif src_d.dim() == 3:
                        src_d = src_d.unsqueeze(0)
                    src_for_resize = src_d
                    print("[Source Depth] Using separate source PE-field for source tokens")
                else:
                    src_for_resize = depth_for_resize

                depth_src_flat = torch.nn.functional.interpolate(
                    src_for_resize, size=(H_src, W_src), mode='bilinear', align_corners=False
                ).squeeze(0).squeeze(0).flatten()

                if img_freqs.ndim == 2:  # [L, D]
                    img_freqs[source_start:source_end, 0] = depth_src_flat.to(dtype=img_freqs.dtype) if torch.is_complex(img_freqs) else depth_src_flat
                    print(f"[Source Depth] Injected into img_freqs[{source_start}:{source_end}, 0]")
                elif img_freqs.ndim == 3:  # [B, L, D]
                    img_freqs[:, source_start:source_end, 0] = depth_src_flat.unsqueeze(0).to(dtype=img_freqs.dtype) if torch.is_complex(img_freqs) else depth_src_flat.unsqueeze(0)
                    print(f"[Source Depth] Injected into img_freqs[:, {source_start}:{source_end}, 0]")
            
            # Reconstruct image_rotary_emb
            if txt_freqs is not None:
                image_rotary_emb = (img_freqs, txt_freqs)
            else:
                image_rotary_emb = img_freqs
        
        attention_mask = None

        if drag_object_token_masks and len(edit_segment_indices) > 0:
            # multi-object warp: independent per-object mask+transform, only meaningful
            # when there are edit segments to distribute objects across.
            image_rotary_emb = warp_qwen_image_rope_edit_segments(
                image_rotary_emb=image_rotary_emb,
                img_shapes=img_shapes,
                edit_segment_indices=edit_segment_indices,
                height=height,
                width=width,
                drag_token_mask=drag_token_mask,
                drag_bx=drag_bx,
                drag_by=drag_by,
                drag_scale=drag_scale,
                progress_id=progress_id,
                num_inference_steps=num_inference_steps,
                inject_noise_in_vacated_positions=inject_noise_in_vacated_positions,
                warp_end_step=warp_end_step,
                drag_theta=drag_theta,
                drag_objects_resolved=drag_object_token_masks,
            )
        elif (
            drag_token_mask is not None
            and (drag_bx != 0.0 or drag_by != 0.0 or drag_theta != 0.0)
        ):
            if len(edit_segment_indices) > 0:
                # warp only edit-image RoPE segments
                image_rotary_emb = warp_qwen_image_rope_edit_segments(
                    image_rotary_emb=image_rotary_emb,
                    img_shapes=img_shapes,
                    edit_segment_indices=edit_segment_indices,
                    height=height,
                    width=width,
                    drag_token_mask=drag_token_mask,
                    drag_bx=drag_bx,
                    drag_by=drag_by,
                    drag_scale=drag_scale,
                    progress_id=progress_id,
                    num_inference_steps=num_inference_steps,
                    inject_noise_in_vacated_positions=inject_noise_in_vacated_positions,  # ← NEW
                    warp_end_step=warp_end_step,
                    drag_theta=drag_theta,
                )
            else:
                # default: warp main image RoPEs
                image_rotary_emb = warp_qwen_image_rope(
                    image_rotary_emb=image_rotary_emb,
                    drag_token_mask=drag_token_mask,
                    drag_bx=drag_bx,
                    drag_by=drag_by,
                    drag_scale=drag_scale,
                    progress_id=progress_id,
                    num_inference_steps=num_inference_steps,
                    inject_noise_in_vacated_positions=inject_noise_in_vacated_positions,  # ← NEW
                    warp_end_step=warp_end_step,
                    drag_theta=drag_theta,
                )

        # ===== MASK SOURCE IMAGE TOKENS AFTER WARPING =====
        if (drag_token_mask is not None
            and (drag_bx != 0.0 or drag_by != 0.0 or drag_theta != 0.0)
            and inject_noise_in_vacated_positions
            and len(edit_segment_indices) > 0):
            
            
            if inject_noise_in_vacated_positions:
                print(f"[Source Token Masking] Step {progress_id}/{num_inference_steps}")
                
                # Get source segment info
                first_edit_idx = edit_segment_indices[0]
                _, H_src, W_src = img_shapes[first_edit_idx]
                L_src = H_src * W_src
                
                # Calculate segment offsets
                seg_lens = [s[1] * s[2] for s in img_shapes]
                offsets = [0]
                for L_seg in seg_lens:
                    offsets.append(offsets[-1] + L_seg)
                
                # Source segment token range
                source_start = offsets[first_edit_idx]
                source_end = offsets[first_edit_idx + 1]
                
                # Resize mask to source resolution if needed
                mask_2d = drag_token_mask[:, 0]  # [B, Ht, Wt]
                Ht_main, Wt_main = mask_2d.shape[1], mask_2d.shape[2]
                
                if H_src != Ht_main or W_src != Wt_main:
                    mask_src = torch.nn.functional.interpolate(
                        mask_2d.unsqueeze(1).float(),
                        size=(H_src, W_src),
                        mode='nearest'
                    ) > 0.5
                    mask_src = mask_src.squeeze(1)
                else:
                    mask_src = mask_2d
                
                # Calculate destination (same logic as RoPE)
                device = image.device
                dx_tok = int(round(drag_bx / 16))
                dy_tok = int(round(drag_by / 16))

                if drag_scale != 1.0 or drag_theta != 0.0:
                    rows = torch.arange(H_src, device=device, dtype=torch.float32)
                    cols = torch.arange(W_src, device=device, dtype=torch.float32)
                else:
                    rows = torch.arange(H_src, device=device)
                    cols = torch.arange(W_src, device=device)
                rr, cc = torch.meshgrid(rows, cols, indexing="ij")

                if drag_scale != 1.0 or drag_theta != 0.0:
                    if mask_src[0].any():
                        mask_indices = torch.nonzero(mask_src[0], as_tuple=False)
                        center_coords = mask_indices.float().mean(dim=0)
                        center_h, center_w = center_coords[0].item(), center_coords[1].item()
                    else:
                        center_h, center_w = (H_src - 1) / 2.0, (W_src - 1) / 2.0

                    scaled_dr = (rr - center_h) * drag_scale
                    scaled_dc = (cc - center_w) * drag_scale
                    if drag_theta != 0.0:
                        theta_rad = math.radians(drag_theta)
                        cos_t, sin_t = math.cos(theta_rad), math.sin(theta_rad)
                        scaled_dr, scaled_dc = scaled_dr * cos_t - scaled_dc * sin_t, scaled_dr * sin_t + scaled_dc * cos_t
                    transformed_rr = (center_h + scaled_dr + dy_tok).clamp(0, H_src - 1).round().long()
                    transformed_cc = (center_w + scaled_dc + dx_tok).clamp(0, W_src - 1).round().long()
                else:
                    transformed_rr = (rr + dy_tok).clamp(0, H_src - 1).long()
                    transformed_cc = (cc + dx_tok).clamp(0, W_src - 1).long()
                
                # Destination mask
                destination_mask = torch.zeros_like(mask_src, dtype=torch.bool)
                original_positions = mask_src[0] > 0.5
                destination_mask[0, transformed_rr[original_positions], transformed_cc[original_positions]] = True
                
                # Vacated mask (original - destination)
                vacated_mask_2d = (mask_src > 0.5) & ~destination_mask  # [B, H_src, W_src]
                
                num_vacated = vacated_mask_2d[0].sum().item()
                if num_vacated > 0:
                    print(f"  Vacated tokens in source: {num_vacated}")
                    
                    # Flatten to 1D mask for token sequence
                    vacated_mask_flat = vacated_mask_2d.view(-1, L_src)  # [B, L_src]
                    
                    # Get source tokens from image
                    source_tokens = image[:, source_start:source_end, :]  # [B, L_src, D]
                    
                    # Generate noise
                    noise_tokens = torch.randn_like(source_tokens)

                    # Scale noise based on flag
                    if scheduled_noise:
                        # SCHEDULED: Scale by timestep
                        current_t = float(timestep[0])
                        noise_scale = max(current_t, 0.01)
                        print(f"  Using SCHEDULED noise (scale: {noise_scale:.3f})")
                    else:
                        # RANDOM: Full scale
                        noise_scale = 1.0
                        print(f"  Using RANDOM noise (scale: {noise_scale:.3f})")
                    noise_tokens = noise_tokens * noise_scale
                    
                    # Apply mask
                    vacated_mask_bc = vacated_mask_flat.unsqueeze(-1)  # [B, L_src, 1]
                    Image.fromarray((vacated_mask_2d[0].float().cpu().numpy() * 255).astype(np.uint8)).save(f"source_vacated_mask_step_{progress_id}.png")

                    source_tokens_masked = torch.where(
                        vacated_mask_bc,
                        noise_tokens,
                        source_tokens
                    )
                    
                    # Update image in-place
                    image[:, source_start:source_end, :] = source_tokens_masked
                    
                    print(f"  Masked source tokens at {num_vacated} vacated positions")
                    print(f"  Noise scale: {noise_scale:.3f}")
        
    if blockwise_controlnet_conditioning is not None:
        blockwise_controlnet_conditioning = blockwise_controlnet.preprocess(
            blockwise_controlnet_inputs, blockwise_controlnet_conditioning)

    for block_id, block in enumerate(dit.transformer_blocks):
        text, image = gradient_checkpoint_forward(
            block,
            use_gradient_checkpointing,
            use_gradient_checkpointing_offload,
            image=image,
            text=text,
            temb=conditioning,
            image_rotary_emb=image_rotary_emb,
            attention_mask=attention_mask,
            enable_fp8_attention=enable_fp8_attention,
            modulate_index=modulate_index,
        )
        if blockwise_controlnet_conditioning is not None:
            image_slice = image[:, :image_seq_len].clone()
            controlnet_output = blockwise_controlnet.blockwise_forward(
                image=image_slice, conditionings=blockwise_controlnet_conditioning,
                controlnet_inputs=blockwise_controlnet_inputs, block_id=block_id,
                progress_id=progress_id, num_inference_steps=num_inference_steps,
            )
            image[:, :image_seq_len] = image_slice + controlnet_output
    
    if zero_cond_t:
        conditioning = conditioning.chunk(2, dim=0)[0]
    image = dit.norm_out(image, conditioning)
    image = dit.proj_out(image)
    image = image[:, :image_seq_len]
    
    latents = rearrange(image, "B (N H W) (C P Q) -> (B N) C (H P) (W Q)", H=height//16, W=width//16, P=2, Q=2, B=1)
    return latents
