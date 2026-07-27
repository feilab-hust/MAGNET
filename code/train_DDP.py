"""Unified DDP training loop with one task/dataset selected per epoch."""
from __future__ import annotations

import csv
import datetime
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from skimage.metrics import normalized_root_mse, peak_signal_noise_ratio, structural_similarity

from models import MultiModel_Trainner
from utils.dataloader_MultiTask import get_multi_data
from utils.checkpoint import load_checkpoint, make_checkpoint
from utils.utility import (
    choose_task, cleanup_dataloader, mkdir_or_exsist, save_settings,
    parse_int_sequence, task_head_mapping,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAGNET_FAMILIES = {"magnet", "magnet_old"}


def _is_magnet_family(family):
    return str(family).lower() in MAGNET_FAMILIES


def _move(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    return value


def _normalise_magnet_batch(batch, data_norm, data_dim, prompt_enable, device):
    inp = data_norm["inp"]
    gt = data_norm["gt"]
    inp_sub = torch.as_tensor(inp["sub"], dtype=torch.float32, device=device).view(1, -1, *([1] * data_dim))
    inp_div = torch.as_tensor(inp["div"], dtype=torch.float32, device=device).view(1, -1, *([1] * data_dim))
    gt_sub = torch.as_tensor(gt["sub"], dtype=torch.float32, device=device).view(1, 1, -1)
    gt_div = torch.as_tensor(gt["div"], dtype=torch.float32, device=device).view(1, 1, -1)
    batch["inp"] = (batch["inp"] - inp_sub) / inp_div
    batch["gt"] = (batch["gt"] - gt_sub) / gt_div
    if prompt_enable:
        batch["prompt"][0] = (batch["prompt"][0] - inp_sub) / inp_div
        image_sub = gt_sub.view(1, -1, *([1] * data_dim))
        image_div = gt_div.view(1, -1, *([1] * data_dim))
        batch["prompt"][1] = (batch["prompt"][1] - image_sub) / image_div
    return batch


def _load_checkpoint(path, trainer, optimizer, scheduler, device, mode="pretrained"):
    epoch, _ = load_checkpoint(
        path, trainer, device, optimizer, scheduler, mode=mode
    )
    return epoch


def _latest_epoch_checkpoint(checkpoint_dir):
    """Prefer the every-epoch rolling checkpoint, then fall back to snapshots."""
    rolling = checkpoint_dir / "MultiModel_CKPT.tar"
    if rolling.is_file():
        return rolling
    checkpoints = sorted(checkpoint_dir.glob("MultiModel_CKPT_epoch_*.tar"))
    return checkpoints[-1] if checkpoints else None


def _prepare_batch(batch, family, data_norm, data_dim, prompt_enable, model_task, device):
    batch = _move(batch, device)
    if _is_magnet_family(family):
        batch = _normalise_magnet_batch(batch, data_norm, data_dim, prompt_enable, device)
    elif family == "unifmir":
        batch["task_id"] = model_task
    return batch


def _metric_values(prediction, target):
    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if target.ndim == 2:
        ssim = structural_similarity(target, prediction, data_range=1.0)
    else:
        pred_slices = prediction.reshape((-1, *prediction.shape[-2:]))
        target_slices = target.reshape((-1, *target.shape[-2:]))
        ssim = np.mean([
            structural_similarity(gt_slice, pred_slice, data_range=1.0)
            for pred_slice, gt_slice in zip(pred_slices, target_slices)
        ])
    return {
        "psnr": peak_signal_noise_ratio(target, prediction, data_range=1.0),
        "ssim": ssim,
        "nrmse": normalized_root_mse(target, prediction),
    }


def _write_test_results(rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["name", "label", "psnr", "ssim", "nrmse"]
    with open(output_dir / "metrics.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [f"num_samples: {len(rows)}", "data_range: 1.0"]
    for key in ("psnr", "ssim", "nrmse"):
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        if values.size:
            lines.extend((f"{key}_mean: {values.mean():.6f}",
                          f"{key}_std: {values.std():.6f}"))
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _limit_done(count, limit):
    return limit is not None and count >= limit


def _under_limit(count, limit):
    return limit is None or count < limit


def _shape_for_sample(collated_shape, sample_index):
    return tuple(int(axis[sample_index]) if torch.is_tensor(axis) else int(axis)
                 for axis in collated_shape)


def _test_tensors(result, batch, family, data_norm):
    prediction = result["prediction"]
    if not _is_magnet_family(family):
        input_image = batch.get("scaled_inp")
        if input_image is None:
            input_image = batch["inp"]
        target = batch.get("scaled_tar")
        if target is None:
            target = batch.get("gtc", batch.get("hr", batch.get("gts")))
        return prediction, input_image.clamp(0.0, 1.0), target

    input_image = batch["inp"]
    target = batch["gt"]
    inp_norm = data_norm["inp"]
    gt_norm = data_norm["gt"]
    inp_offset = torch.as_tensor(
        inp_norm["sub"], device=prediction.device
    ).view(1, -1, *([1] * (input_image.ndim - 2)))
    inp_divisor = torch.as_tensor(
        inp_norm["div"], device=prediction.device
    ).view(1, -1, *([1] * (input_image.ndim - 2)))
    offset = torch.as_tensor(gt_norm["sub"], device=prediction.device).view(1, 1, -1)
    divisor = torch.as_tensor(gt_norm["div"], device=prediction.device).view(1, 1, -1)
    input_image = input_image * inp_divisor + inp_offset
    prediction = prediction * divisor + offset
    target = target * divisor + offset
    restored_predictions, restored_targets = [], []
    for sample_index in range(prediction.shape[0]):
        shape = _shape_for_sample(batch["sample_shape"], sample_index)
        restored_predictions.append(prediction[sample_index].permute(1, 0).reshape(shape))
        restored_targets.append(target[sample_index].permute(1, 0).reshape(shape))
    return (
        torch.stack(restored_predictions),
        input_image.clamp(0.0, 1.0),
        torch.stack(restored_targets),
    )


def _tile_starts(length, tile, overlap):
    stride = tile - overlap
    starts = list(range(0, max(length - tile, 0) + 1, stride))
    last = max(length - tile, 0)
    if not starts or starts[-1] != last:
        starts.append(last)
    return starts


def _tile_shape(value, spatial_dims):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        values = tuple(int(item) for item in value)
        if len(values) == 1:
            return values * spatial_dims
        if len(values) != spatial_dims:
            raise ValueError(
                f"eval_patch_size needs 1 or {spatial_dims} values, got {values}"
            )
        return values
    return (int(value),) * spatial_dims


def _feather_weight(shape, overlap, has_before, has_after, device):
    weight = torch.ones(shape, dtype=torch.float32, device=device)
    for dim, size in enumerate(shape):
        axis = torch.ones(size, dtype=torch.float32, device=device)
        width = min(int(overlap[dim]), max(0, size // 2))
        if width:
            ramp = torch.linspace(0, 1, width + 2, device=device)[1:-1]
            if has_before[dim]:
                axis[:width] = ramp
            if has_after[dim]:
                axis[-width:] = ramp.flip(0)
        view = [1] * len(shape)
        view[dim] = size
        weight *= axis.view(view)
    return weight


def _pad_spatial(tensor, target_shape):
    padding = []
    for actual, target in reversed(list(zip(tensor.shape[2:], target_shape))):
        padding.extend((0, max(0, target - actual)))
    return F.pad(tensor, tuple(padding), mode="replicate") if any(padding) else tensor


def _magnet_patch_prediction(trainer, patch, batch, output_shape):
    spatial_dims = len(output_shape)
    axes = [
        torch.linspace(
            -1 + 1 / size, 1 - 1 / size, size,
            device=patch.device, dtype=patch.dtype,
        ) for size in output_shape
    ]
    grids = torch.meshgrid(*axes, indexing="ij")
    coord = torch.stack(grids, dim=-1).reshape(1, -1, spatial_dims)
    coord = coord.expand(patch.shape[0], -1, -1)
    cell = torch.ones_like(coord)
    for dim, size in enumerate(output_shape):
        cell[..., dim] *= 2 / size
    result = trainer.forward_batch({
        "inp": patch,
        "coord": coord,
        "cell": cell,
        "prompt": batch.get("prompt", ["None", "None"]),
        "promptPD": batch.get("promptPD"),
        "s_factor": batch.get(
            "s_factor", torch.ones(patch.shape[0], device=patch.device)
        ),
    })["prediction"]
    return result.permute(0, 2, 1).reshape(
        patch.shape[0], -1, *output_shape
    )


def _square_patch_prediction(trainer, patch, batch, output_shape):
    patch_batch = {
        "task_id": batch.get("task_id", trainer.config.get("task", 1)),
        "s": batch.get("s"),
        "txt_emb": batch.get("txt_emb"),
        "timesteps": batch.get("timesteps"),
    }
    return trainer.forward_image(
        patch, batch=patch_batch, output_size=output_shape
    )


def _tiled_test_result(trainer, batch, family, data_dim, eval_patch_size):
    is_magnet = _is_magnet_family(family)
    source = batch["inp"] if is_magnet else batch.get(
        "scaled_inp", batch["inp"]
    )
    spatial_dims = source.ndim - 2
    tile_shape = _tile_shape(eval_patch_size, spatial_dims)
    if tile_shape is None or all(
        actual <= tile for actual, tile in zip(source.shape[2:], tile_shape)
    ):
        return trainer.training_step(batch)
    if spatial_dims not in (2, 3):
        raise ValueError(f"Tiled test supports 2D/3D, got {spatial_dims}D")
    if not is_magnet and spatial_dims == 3:
        raise NotImplementedError(
            f"{family} is a 2D model and cannot infer 3D volumes directly"
        )

    target = batch["gt"] if is_magnet else batch.get(
        "scaled_tar", batch.get("gtc", batch.get("gts"))
    )
    target_spatial = (
        tuple(int(axis[0]) if torch.is_tensor(axis) else int(axis)
              for axis in batch["sample_shape"][-spatial_dims:])
        if is_magnet else tuple(target.shape[-spatial_dims:])
    )
    source_spatial = tuple(source.shape[-spatial_dims:])
    ratios = tuple(out_size / in_size for out_size, in_size in zip(
        target_spatial, source_spatial
    ))
    overlap = tuple(max(1, tile // 4) for tile in tile_shape)
    starts = [
        _tile_starts(length, tile, ov)
        for length, tile, ov in zip(source_spatial, tile_shape, overlap)
    ]

    output = None
    weights = None
    import itertools
    for indices in itertools.product(*[range(len(axis)) for axis in starts]):
        positions = tuple(starts[dim][index] for dim, index in enumerate(indices))
        actual_shape = tuple(
            min(tile_shape[dim], source_spatial[dim] - positions[dim])
            for dim in range(spatial_dims)
        )
        slices = tuple(
            slice(position, position + actual)
            for position, actual in zip(positions, actual_shape)
        )
        patch = _pad_spatial(source[(..., *slices)], tile_shape)
        padded_output_shape = tuple(
            max(1, int(round(tile * ratio)))
            for tile, ratio in zip(tile_shape, ratios)
        )
        wanted_shape = tuple(
            max(1, int(round(actual * ratio)))
            for actual, ratio in zip(actual_shape, ratios)
        )
        prediction = (
            _magnet_patch_prediction(trainer, patch, batch, padded_output_shape)
            if is_magnet else
            _square_patch_prediction(trainer, patch, batch, padded_output_shape)
        )
        prediction = prediction[(..., *tuple(slice(0, size) for size in wanted_shape))]
        if output is None:
            output = torch.zeros(
                (source.shape[0], prediction.shape[1], *target_spatial),
                dtype=torch.float32, device=source.device,
            )
            weights = torch.zeros_like(output)
        output_starts = tuple(
            int(round(position * ratio))
            for position, ratio in zip(positions, ratios)
        )
        output_slices = tuple(
            slice(position, min(position + size, target_spatial[dim]))
            for dim, (position, size) in enumerate(zip(output_starts, wanted_shape))
        )
        clipped_shape = tuple(item.stop - item.start for item in output_slices)
        prediction = prediction[(..., *tuple(slice(0, size) for size in clipped_shape))]
        weight = _feather_weight(
            clipped_shape,
            tuple(max(1, int(round(ov * ratio))) for ov, ratio in zip(overlap, ratios)),
            tuple(index > 0 for index in indices),
            tuple(index < len(starts[dim]) - 1 for dim, index in enumerate(indices)),
            source.device,
        )
        output[(..., *output_slices)] += prediction.float() * weight
        weights[(..., *output_slices)] += weight

    image_prediction = output / weights.clamp_min(1e-8)
    prediction = (
        image_prediction.flatten(2).permute(0, 2, 1)
        if is_magnet else image_prediction
    )
    loss_target = target
    loss_name = str(trainer.config.get("loss", "l1")).lower()
    loss = (
        F.mse_loss(prediction, loss_target)
        if loss_name == "mse" else F.l1_loss(prediction, loss_target)
    )
    return {"prediction": prediction, "raw_output": prediction, "loss": loss}


@torch.no_grad()
def run_large_test(trainer, loader, family, data_norm, data_dim, prompt_enable,
                   model_task, device, output_dir, test_num, save_num,
                   eval_patch_size=None):
    trainer.eval()
    if eval_patch_size is not None:
        print(
            f"Test tiled inference | family={family} | "
            f"eval_patch_size={eval_patch_size} | overlap=25%"
        )
    metric_counts, save_counts = defaultdict(int), defaultdict(int)
    rows, losses = [], []
    for iteration, batch in enumerate(loader, start=1):
        print(
            f"\rTest | iter={iteration}/{len(loader)}",
            end="",
            flush=True,
        )
        labels = [str(label) for label in batch["sample_label"]]
        if all(_limit_done(metric_counts[label], test_num) and
               _limit_done(save_counts[label], save_num)
               for label in labels):
            continue
        batch = _prepare_batch(batch, family, data_norm, data_dim,
                               prompt_enable, model_task, device)
        # Validation uses mixed precision by default to reduce activation
        # memory while keeping metrics and saved TIFFs in float32 below.
        with torch.amp.autocast("cuda"):
            result = _tiled_test_result(
                trainer, batch, family, data_dim, eval_patch_size
            )
        losses.append(float(result["loss"].detach().cpu()))
        predictions, inputs, targets = _test_tensors(result, batch, family, data_norm)

        for sample_index, label in enumerate(labels):
            safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
            prediction = np.squeeze(predictions[sample_index].detach().float().cpu().numpy())
            input_image = np.squeeze(inputs[sample_index].detach().float().cpu().numpy())
            target = np.squeeze(targets[sample_index].detach().float().cpu().numpy())
            if _under_limit(save_counts[label], save_num):
                number = f"{save_counts[label]:05d}"
                for folder, suffix, image in (
                    ("predictions", "pred", prediction),
                    ("inputs", "input", input_image),
                    ("targets", "target", target),
                ):
                    directory = output_dir / safe_label / folder
                    directory.mkdir(parents=True, exist_ok=True)
                    tifffile.imwrite(
                        directory / f"{number}_{suffix}.tif",
                        image.astype(np.float32),
                    )
                save_counts[label] += 1
            if _under_limit(metric_counts[label], test_num):
                metrics = _metric_values(prediction, target)
                rows.append({"name": f"{metric_counts[label]:05d}", "label": label, **metrics})
                metric_counts[label] += 1

    print()
    _write_test_results(rows, output_dir)
    for label in sorted(set(metric_counts) | set(save_counts)):
        label_rows = [row for row in rows if row["label"] == label]
        if label_rows:
            _write_test_results(label_rows, output_dir / re.sub(r"[^A-Za-z0-9_.-]+", "_", label))
    print("Test metric counts:", dict(metric_counts))
    print("Saved prediction counts:", dict(save_counts))
    return float(np.mean(losses)) if losses else float("nan")


def TRAIN_DDP(local_rank, world_size, args):
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(
        backend=args.DDP_backend,
        rank=local_rank,
        world_size=world_size,
        timeout=datetime.timedelta(minutes=30),
    )

    # Anchor outputs to the repository, not to the shell's current directory.
    output_dir = PROJECT_ROOT / "log" / args.expname
    checkpoint_dir = output_dir / "ckpt" / args.model_family.lower()
    if local_rank == 0:
        mkdir_or_exsist([str(output_dir), str(checkpoint_dir)], verbose=True)
        save_settings(save_path=str(output_dir), args=args)

    trainer = MultiModel_Trainner.from_args(args).to(device)
    optimizer = torch.optim.AdamW((p for p in trainer.parameters() if p.requires_grad), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.n_steps], gamma=args.gamma)

    resume_path = (_latest_epoch_checkpoint(checkpoint_dir)
                   if args.resume else args.loading_MT_ckpt_path)
    # Every rank restores optimizer/scheduler state. DDP will synchronize model
    # parameters, but it does not synchronize optimizer momentum buffers.
    load_mode = "resume" if args.resume else "pretrained"
    start_epoch = _load_checkpoint(
        resume_path, trainer, optimizer, scheduler, device, mode=load_mode
    )
    epoch_tensor = torch.tensor(start_epoch, device=device)
    dist.all_reduce(epoch_tensor, op=dist.ReduceOp.MIN)
    start_epoch = int(epoch_tensor.item())

    trainer = DDP(
        trainer,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=args.find_unused_parameters,
    )
    scaler = torch.amp.GradScaler("cuda")
    family = args.model_family.lower()
    all_iter_times = []
    unifmir_heads = None
    if family == "unifmir":
        _, unifmir_heads = task_head_mapping(
            args.task_idx, args.task_list, args.model_task
        )

    for epoch in range(start_epoch, args.n_epochs):
        # This is intentionally inside the epoch loop: task_list is traversed
        # round-robin, and the corresponding dataset is rebuilt every epoch.
        task = choose_task(args.task_idx, args.task_list, epoch)
        model_task = (
            unifmir_heads[epoch % len(unifmir_heads)]
            if unifmir_heads is not None
            else parse_int_sequence(args.model_task, "model_task")[0]
        )
        loaded = get_multi_data(
            task_idx=task,
            data_config=args.MT_data_config,
            num_work=args.num_workers,
            seed=args.seed or 42,
            local_rank=local_rank,
            world_size=world_size,
            require_text_embedding=(family == "fluoresfm"),
        )
        train_loader, train_sampler, test_loader = loaded[:3]
        (data_norm, task_name, data_dim, prompt_enable, post_SR,
         eval_patch_size) = loaded[3:]
        train_sampler.set_epoch(epoch)
        trainer.module.config["prompt_enable"] = prompt_enable
        trainer.module.config["post_SR"] = post_SR
        trainer.train()
        epoch_losses = []

        for iteration, batch in enumerate(train_loader, start=1):
            torch.cuda.synchronize(device)
            iter_start = time.perf_counter()
            batch = _prepare_batch(batch, family, data_norm, data_dim,
                                   prompt_enable, model_task, device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                result = trainer(batch, True)
            scaler.scale(result["loss"]).backward()
            if args.gclip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainer.parameters(), args.gclip)
            scaler.step(optimizer)
            scaler.update()

            reduced = result["loss"].detach().clone()
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
            reduced /= world_size
            torch.cuda.synchronize(device)
            iter_time = torch.tensor(
                time.perf_counter() - iter_start,
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(iter_time, op=dist.ReduceOp.MAX)
            iter_seconds = float(iter_time.cpu())
            epoch_losses.append(float(reduced.cpu()))
            if local_rank == 0:
                all_iter_times.append(iter_seconds)
                print(f"\rEpoch {epoch + 1}/{args.n_epochs} | task={task_name} | "
                      f"iter={iteration}/{len(train_loader)} | loss={reduced.item():.6f} "
                      f"| model_task={model_task} "
                      f"| iter_time={iter_seconds:.4f}s", end="")

        # Finish the carriage-return training status before eval starts its
        # own carriage-return progress line.
        if local_rank == 0:
            print()
        dist.barrier()
        if local_rank == 0:
            validation_loss = float("nan")
            is_sample_epoch = epoch % args.sample_interval == 0
            if is_sample_epoch:
                test_dir = (
                    output_dir / "test" / task_name
                    / f"{family}_epoch_{epoch + 1:05d}"
                )
                validation_loss = run_large_test(
                    trainer=trainer.module,
                    loader=test_loader,
                    family=family,
                    data_norm=data_norm,
                    data_dim=data_dim,
                    prompt_enable=prompt_enable,
                    model_task=model_task,
                    device=device,
                    output_dir=test_dir,
                    test_num=args.test_num,
                    save_num=args.save_num,
                    eval_patch_size=eval_patch_size,
                )
            train_l1 = float(np.mean(epoch_losses))
            checkpoint = make_checkpoint(
                trainer=trainer.module,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                train_l1=train_l1,
                eval_metrics={"loss": validation_loss},
                task=task_name,
            )
            # Rolling state is overwritten every epoch so resume always starts
            # from the most recently completed epoch.
            torch.save(checkpoint, checkpoint_dir / "MultiModel_CKPT.tar")
            if is_sample_epoch:
                checkpoint_name = f"MultiModel_CKPT_epoch_{epoch + 1:05d}.tar"
                torch.save(checkpoint, checkpoint_dir / checkpoint_name)
            print(f"train={train_l1:.6f} | validation={validation_loss:.6f}")
        dist.barrier()
        scheduler.step()
        cleanup_dataloader(train_loader)
        cleanup_dataloader(test_loader)
        del train_loader, train_sampler, test_loader

    if local_rank == 0:
        average = float(np.mean(all_iter_times)) if all_iter_times else float("nan")
        print(f"Average training iter time: {average:.4f}s")
    dist.destroy_process_group()
