import json, os, subprocess, torch
from accelerate import Accelerator

EVAL_SLURM_SCRIPT = os.environ.get("ROPEMOVER_EVAL_SLURM_SCRIPT", "eval_checkpoint.sh")

class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x,
                 infer_script=None, eval_results_dir=None):
        self.output_path           = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter  = state_dict_converter
        self.num_steps             = 0
        self.infer_script          = infer_script
        self.eval_results_dir      = eval_results_dir

    def _submit_eval_job(self, step, ckpt_path):
        """Submit a SLURM job that runs inference + eval for this checkpoint."""
        if not (self.infer_script and self.eval_results_dir):
            return
        out_json = os.path.join(self.output_path, f"eval_step_{step}.json")
        export = (
            f"ALL,"
            f"INFER_SCRIPT={self.infer_script},"
            f"LORA_CKPT={ckpt_path},"
            f"EVAL_RESULTS_DIR={self.eval_results_dir},"
            f"EVAL_OUTPUT_JSON={out_json},"
            f"STEP={step}"
        )
        result = subprocess.run(
            ["sbatch", f"--export={export}", EVAL_SLURM_SCRIPT],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            print(f"[Eval] step={step} — submitted SLURM job {job_id} "
                  f"(results → {out_json})", flush=True)
        else:
            print(f"[Eval] step={step} — sbatch failed: {result.stderr.strip()}", flush=True)

    def on_step_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None):
        self.num_steps += 1
        if save_steps is not None and self.num_steps % save_steps == 0:
            ckpt_name = f"step-{self.num_steps}.safetensors"
            self.save_model(accelerator, model, ckpt_name)
            if accelerator.is_main_process:
                ckpt_path = os.path.join(self.output_path, ckpt_name)
                self._submit_eval_job(self.num_steps, ckpt_path)

    def on_epoch_end(self, accelerator: Accelerator, model: torch.nn.Module, epoch_id):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)

    def on_training_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")

    def save_model(self, accelerator: Accelerator, model: torch.nn.Module, file_name):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)
