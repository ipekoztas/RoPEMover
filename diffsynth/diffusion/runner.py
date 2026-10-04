import json, os, torch
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger


def _read_training_log(log_path):
    if os.path.exists(log_path):
        with open(log_path) as f:
            return json.load(f)
    return {"step_log": [], "epoch_log": []}


def _write_training_log(log_path, log):
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)


def _run_val_loop(val_dataloader, model, accelerator):
    """Run one pass over val_dataloader with no_grad. Returns avg loss (float)."""
    model.eval()
    val_losses = []
    with torch.no_grad():
        for val_data in val_dataloader:
            loss = model(val_data)
            # Gather loss from all processes and take mean
            gathered = accelerator.gather(loss.detach().unsqueeze(0))
            val_losses.append(gathered.mean().item())
    model.train()
    return sum(val_losses) / len(val_losses) if val_losses else 0.0


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    val_dataset: torch.utils.data.Dataset = None,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs

    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)

    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers
        )
        model, optimizer, dataloader, scheduler, val_dataloader = accelerator.prepare(
            model, optimizer, dataloader, scheduler, val_dataloader
        )
    else:
        model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    log_path = os.path.join(model_logger.output_path, "training_log.json")
    if accelerator.is_main_process:
        os.makedirs(model_logger.output_path, exist_ok=True)

    for epoch_id in range(num_epochs):
        epoch_losses = []

        for data in tqdm(dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                accelerator.backward(loss)
                optimizer.step()

                # Accumulate loss for this epoch
                loss_val = accelerator.gather(loss.detach().unsqueeze(0)).mean().item()
                epoch_losses.append(loss_val)

                model_logger.on_step_end(accelerator, model, save_steps)
                scheduler.step()

                # Log every save_steps steps
                if (save_steps is not None
                        and model_logger.num_steps % save_steps == 0
                        and accelerator.is_main_process):
                    log = _read_training_log(log_path)
                    log["step_log"].append({
                        "step": model_logger.num_steps,
                        "loss": round(loss_val, 6),
                    })
                    _write_training_log(log_path, log)

        # End of epoch: compute averages and run val
        avg_train = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        avg_val = None
        if val_dataloader is not None:
            avg_val = _run_val_loop(val_dataloader, model, accelerator)

        if accelerator.is_main_process:
            log = _read_training_log(log_path)
            entry = {
                "epoch": epoch_id + 1,
                "step": model_logger.num_steps,
                "avg_train_loss": round(avg_train, 6),
            }
            if avg_val is not None:
                entry["avg_val_loss"] = round(avg_val, 6)
            log["epoch_log"].append(entry)
            _write_training_log(log_path, log)
            print(
                f"[Epoch {epoch_id + 1}] step={model_logger.num_steps} "
                f"train_loss={avg_train:.6f}"
                + (f"  val_loss={avg_val:.6f}" if avg_val is not None else ""),
                flush=True,
            )

        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)

    model_logger.on_training_end(accelerator, model, save_steps)


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
    **kwargs,
):
    if args is not None:
        num_workers = args.dataset_num_workers

    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    model, dataloader = accelerator.prepare(model, dataloader)

    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)
