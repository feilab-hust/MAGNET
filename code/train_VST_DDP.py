"""Dedicated FluoResFM virtual-staining DDP training branch."""
from __future__ import annotations

import datetime
import time
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from models import MultiModel_Trainner
from utils.checkpoint import extract_model_state, make_checkpoint
from utils.dataloader_VST import get_vst_loaders


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_weights(path, trainer, optimizer, scheduler, device, resume):
    if not path or not Path(path).is_file():
        print("[VST checkpoint] No checkpoint; training from scratch.")
        return 0
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = extract_model_state(checkpoint)
    state = {key.removeprefix("module.").removeprefix("model."): value
             for key, value in state.items()}
    current = trainer.model.state_dict()
    if resume:
        trainer.model.load_state_dict(state, strict=True)
        if checkpoint.get("optimizer_state_dict"):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print(f"[VST checkpoint] resumed: {path}")
        return int(checkpoint.get("epoch", 0))
    compatible = {key: value for key, value in state.items()
                  if key in current and current[key].shape == value.shape}
    skipped = sorted(key for key, value in state.items()
                     if key in current and current[key].shape != value.shape)
    trainer.model.load_state_dict(compatible, strict=False)
    print(f"[VST checkpoint] pretrained={path} | loaded={len(compatible)} | "
          f"reinitialized={skipped}")
    return 0


def _starts(length, tile, overlap):
    values = list(range(0, max(length - tile, 0) + 1, tile - overlap))
    last = max(0, length - tile)
    if not values or values[-1] != last:
        values.append(last)
    return values


@torch.no_grad()
def _tiled_predict(trainer, image, embedding, tile, device):
    overlap = tile // 4
    _, _, height, width = image.shape
    output = torch.zeros((1, 3, height, width), device=device)
    weights = torch.zeros_like(output)
    ramp = torch.linspace(0, 1, overlap + 2, device=device)[1:-1]
    ys, xs = _starts(height, tile, overlap), _starts(width, tile, overlap)
    for yi, top in enumerate(ys):
        for xi, left in enumerate(xs):
            patch = image[..., top:top + tile, left:left + tile]
            with torch.amp.autocast("cuda"):
                pred = trainer.forward_image(patch, {"txt_emb": embedding})
            wy = torch.ones(tile, device=device)
            wx = torch.ones(tile, device=device)
            if yi: wy[:overlap] = ramp
            if yi < len(ys)-1: wy[-overlap:] = ramp.flip(0)
            if xi: wx[:overlap] = ramp
            if xi < len(xs)-1: wx[-overlap:] = ramp.flip(0)
            weight = (wy[:, None] * wx[None, :])[None, None]
            output[..., top:top + tile, left:left + tile] += pred.float() * weight
            weights[..., top:top + tile, left:left + tile] += weight
    return output / weights.clamp_min(1e-8)


@torch.no_grad()
def _validate(trainer, loader, tile, device, output_dir, save_num):
    trainer.eval()
    losses = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, batch in enumerate(loader):
        image = batch["inp"].to(device)
        target = batch["target"].to(device)
        embedding = batch["txt_emb"].to(device)
        prediction = _tiled_predict(trainer, image, embedding, tile, device)
        losses.append(float(F.l1_loss(prediction, target).cpu()))
        if save_num is None or index < save_num:
            pred = prediction[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            gt = target[0].permute(1, 2, 0).cpu().numpy()
            tifffile.imwrite(output_dir / f"{index:05d}_pred.tif", (pred*255).astype(np.uint8))
            tifffile.imwrite(output_dir / f"{index:05d}_target.tif", (gt*255).astype(np.uint8))
    return float(np.mean(losses)) if losses else float("nan")


def VST_DDP(local_rank, world_size, args):
    if args.model_family.lower() != "fluoresfm":
        raise ValueError("VST_DDP currently supports model_family=fluoresfm only")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(args.DDP_backend, rank=local_rank, world_size=world_size,
                            timeout=datetime.timedelta(minutes=30))
    output_dir = PROJECT_ROOT / "log" / args.expname
    checkpoint_dir = output_dir / "ckpt" / "fluoresfm_vst"
    if local_rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    trainer = MultiModel_Trainner.from_args(args).to(device)
    if trainer.model.out[-1].out_channels != 3:
        raise ValueError("VST_DDP requires model_parameters={'out_channels': 3}")
    optimizer = torch.optim.AdamW(trainer.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[args.n_steps], gamma=args.gamma)
    resume_path = checkpoint_dir / "VST_CKPT.tar" if args.resume else Path(args.loading_MT_ckpt_path)
    start_epoch = _load_weights(resume_path, trainer, optimizer, scheduler, device, args.resume)
    trainer = DDP(trainer, device_ids=[local_rank], output_device=local_rank,
                  find_unused_parameters=args.find_unused_parameters)
    scaler = torch.amp.GradScaler("cuda")
    train_loader, sampler, test_loader, label, eval_tile = get_vst_loaders(
        args.MT_data_config, args.num_workers, local_rank, world_size)
    all_times = []

    for epoch in range(start_epoch, args.n_epochs):
        sampler.set_epoch(epoch)
        trainer.train()
        losses = []
        for iteration, batch in enumerate(train_loader, 1):
            start = time.perf_counter()
            image = batch["inp"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            embedding = batch["txt_emb"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                # Go through DDP.forward so gradient synchronization hooks run.
                prediction = trainer({"scaled_inp": image, "txt_emb": embedding})
                loss = F.l1_loss(prediction, target)
            scaler.scale(loss).backward()
            if args.gclip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainer.parameters(), args.gclip)
            scaler.step(optimizer)
            scaler.update()
            reduced = loss.detach().clone()
            dist.all_reduce(reduced); reduced /= world_size
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            losses.append(float(reduced.cpu()))
            if local_rank == 0:
                all_times.append(elapsed)
                print(f"\rVST epoch={epoch+1}/{args.n_epochs} iter={iteration}/{len(train_loader)} "
                      f"loss={reduced.item():.6f} time={elapsed:.3f}s", end="")
        if local_rank == 0:
            print()
        dist.barrier()
        validation = float("nan")
        sample_epoch = (epoch + 1) % args.sample_interval == 0
        if local_rank == 0:
            if sample_epoch:
                validation = _validate(trainer.module, test_loader, eval_tile, device,
                                       output_dir / "test" / f"epoch_{epoch+1:05d}", args.save_num)
            checkpoint = make_checkpoint(trainer.module, optimizer, scheduler, epoch + 1,
                                         float(np.mean(losses)), {"loss": validation}, label)
            torch.save(checkpoint, checkpoint_dir / "VST_CKPT.tar")
            if sample_epoch:
                torch.save(checkpoint, checkpoint_dir / f"VST_CKPT_epoch_{epoch+1:05d}.tar")
            print(f"VST train={np.mean(losses):.6f} validation={validation:.6f}")
        dist.barrier()
        scheduler.step()
    if local_rank == 0:
        print(f"Average VST iter time: {np.mean(all_times):.4f}s")
    dist.destroy_process_group()
