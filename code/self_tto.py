"""PSF-guided self-supervised training shared by all model families."""
from __future__ import annotations

import ast
import csv
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F

import Configs
from coord_fre_loss import coordinate_frequency_loss
from models import MultiModel_Trainner
from models.magnet.utils.criterions import fre_loss, mae_loss, mse_loss, tv_loss
from models.magnet.utils.fft_conv import fft_conv
from train_DDP import (
    _is_magnet_family, _limit_done, _metric_values, _under_limit,
    _write_test_results,
)
from utils.checkpoint import load_checkpoint, make_checkpoint
from utils.dataloader_zs import get_multi_data
from utils.utility import cleanup_dataloader, mkdir_or_exsist, save_settings


# Background-only high-frequency regularization for TTO.
#
# This suppresses artifacts that disappear after PSF convolution and therefore
# are almost invisible to the reconstruction loss.  The mask is estimated from
# the current network prediction: low-intensity pixels are treated as background,
# then the Laplacian energy of the prediction is penalized only there.  The BL
# loss weight is intentionally read from config/args, not fixed in this file.
TTO_BG_PERCENTILE = 0.60
TTO_BG_MASK_SOFTNESS = 0.05

# Coordinate-query carrier suppression for MAGNET SR/deconvolution. The loss
# detects narrow spectral peaks at the aliased input-pixel lattice frequency;
# it does not impose a general high-frequency smoothing penalty.  The CG loss
# weight is intentionally read from config/args, not fixed in this file.
TTO_COORD_FRE_BAND_WIDTH = 0.035
TTO_COORD_FRE_RING_INNER_RATIO = 1.5
TTO_COORD_FRE_RING_OUTER_RATIO = 3.0
TTO_COORD_FRE_PEAK_RATIO = 1.25
TTO_COORD_FRE_HARMONICS = 1
TTO_COORD_FRE_INCLUDE_AXES = True
TTO_COORD_FRE_INCLUDE_DIAGONALS = True
TTO_COORD_FRE_MIN_FREQUENCY = 0.08


def _tto_constraint_weights(args):
    """Return BL/CG weights controlled by Configs.py or the config file."""
    return (
        float(getattr(args, "tto_bl_weight", 0.0)),
        float(getattr(args, "tto_cg_weight", 0.0)),
    )


def _build_psf(args, device):
    psf = torch.as_tensor(tifffile.imread(args.psf_dir), dtype=torch.float32)
    psf = psf.reshape(1, 1, *psf.shape[-2:]).to(device)
    psf /= psf.sum()
    return psf


def _degrade(image, psf):
    """Apply the original-scale PSF with numerically stable float32 FFTs.

    The 127x127 PSF creates considerably larger FFT reductions than the old
    resized kernel.  Letting the surrounding AMP context cast this operation
    to float16 can overflow the frequency-domain tensors and poison training
    with NaNs.  Only the degradation operator is kept in float32; the network
    forward pass remains under AMP.
    """
    device_type = image.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        return fft_conv(
            signal=image.float(),
            kernel=psf.float(),
            padding_mode="reflect",
        )


def _gaussian_kernel(kernel_size, sigma, channels, device, dtype):
    """Create a depthwise 2D Gaussian kernel."""
    coordinates = torch.arange(kernel_size, device=device, dtype=dtype)
    coordinates = coordinates - (kernel_size - 1) / 2
    gaussian = torch.exp(-(coordinates ** 2) / (2 * sigma ** 2))
    gaussian = gaussian / gaussian.sum()
    kernel = gaussian[:, None] * gaussian[None, :]
    return kernel.view(1, 1, kernel_size, kernel_size).expand(
        channels, 1, kernel_size, kernel_size
    )


def _low_frequency_mask(image, kernel_size=31, sigma=8.0):
    """Estimate spatially varying signal strength with a strong blur."""
    kernel_size = min(kernel_size, 2 * min(image.shape[-2:]) - 1)
    kernel_size = max(3, kernel_size if kernel_size % 2 else kernel_size - 1)
    padding = kernel_size // 2
    kernel = _gaussian_kernel(
        kernel_size, sigma, image.shape[1], image.device, image.dtype
    )
    pad_mode = "reflect" if min(image.shape[-2:]) > padding else "replicate"
    padded = F.pad(image, (padding,) * 4, mode=pad_mode)
    return F.conv2d(padded, kernel, groups=image.shape[1]).clamp_min(0.0)


def _noise_minus_one_to_one(reference):
    """Generate Gaussian noise and independently rescale it to [-1, 1]."""
    noise = torch.randn_like(reference)
    flat = noise.flatten(2)
    minimum = flat.amin(dim=-1, keepdim=True).unsqueeze(-1)
    maximum = flat.amax(dim=-1, keepdim=True).unsqueeze(-1)
    return 2.0 * (noise - minimum) / (maximum - minimum).clamp_min(1e-6) - 1.0


def _make_noisy_pair(lr):
    """Create two independent signal-dependent noisy views of one image."""
    mask = _low_frequency_mask(lr)
    p = (lr + _noise_minus_one_to_one(lr) * mask).clamp_min(0.0)
    n = (lr + _noise_minus_one_to_one(lr) * mask).clamp_min(0.0)
    return p, n


def _background_mask(reference, output_size):
    """Detached soft mask for low-signal regions at prediction resolution."""
    reference = reference.detach().float()
    if reference.shape[-2:] != tuple(output_size):
        reference = F.interpolate(
            reference, size=output_size, mode="bicubic", align_corners=False
        )
    flat = reference.flatten(2)
    threshold = torch.quantile(
        flat, float(TTO_BG_PERCENTILE), dim=-1, keepdim=True
    ).unsqueeze(-1)
    if TTO_BG_MASK_SOFTNESS > 0:
        mask = torch.sigmoid((threshold - reference) / float(TTO_BG_MASK_SOFTNESS))
    else:
        mask = (reference <= threshold).to(reference.dtype)
    return mask.detach()


def _masked_laplacian_loss(image, mask):
    """Laplacian high-frequency energy restricted to a background mask."""
    if mask is None:
        return image.new_zeros(())
    lap = _laplacian_abs(image)
    weighted = lap * mask.float()
    return weighted.sum() / mask.float().sum().clamp_min(1.0)


def _laplacian_abs(image):
    """Absolute 2D Laplacian response per pixel."""
    image_float = image.float()
    channels = image_float.shape[1]
    kernel = image_float.new_tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
    ).view(1, 1, 3, 3).expand(channels, 1, 3, 3)
    padded = F.pad(image_float, (1, 1, 1, 1), mode="reflect")
    return F.conv2d(padded, kernel, groups=channels).abs()


def _to_debug_np(tensor):
    """Convert a tensor image to float32 numpy without display normalization."""
    return np.squeeze(tensor.detach().float().cpu().numpy()).astype(np.float32)


def _normalizer(data_norm, device):
    inp = data_norm["inp"]
    gt = data_norm["gt"]
    inp_sub = torch.as_tensor(inp["sub"], device=device).view(1, -1, 1, 1)
    inp_div = torch.as_tensor(inp["div"], device=device).view(1, -1, 1, 1)
    gt_sub = torch.as_tensor(gt["sub"], device=device).view(1, -1, 1, 1)
    gt_div = torch.as_tensor(gt["div"], device=device).view(1, -1, 1, 1)
    return inp_sub, inp_div, gt_sub, gt_div


def _optimizer(trainer, args):
    if not _is_magnet_family(trainer.family):
        return torch.optim.AdamW(
            (parameter for parameter in trainer.parameters() if parameter.requires_grad),
            lr=args.lr,
        )
    train_names = ("imnet", "Linear_2d_input")
    active, frozen = [], []
    for name, parameter in trainer.model.named_parameters():
        (active if any(token in name for token in train_names) else frozen).append(parameter)
    return torch.optim.AdamW([
        {"params": active, "lr": args.lr},
        {"params": frozen, "lr": 0.0},
    ])


def _forward_tto(trainer, batch, image, output_size, args):
    return trainer.forward_image(
        image,
        batch={
            "txt_emb": batch.get("txt_emb"),
            "task_id": args.model_task,
            "s_factor": batch.get("s"),
        },
        output_size=output_size,
    )


def _reconstruction_loss(loss_config, convolved, observation, reconstruction,
                         trainer, device):
    total = convolved.new_zeros(())
    details = {}
    for name, weight in loss_config.items():
        if name == "fre_loss":
            value = fre_loss(reconstruction)
        elif name == "tv_loss":
            feature = (
                trainer.model.features
                if _is_magnet_family(trainer.family) else reconstruction
            )
            value = tv_loss(feature)
        elif name == "mae_loss":
            value = mae_loss(pred=convolved, target=observation, device=device)
        elif name == "mse_loss":
            value = mse_loss(pred=convolved, target=observation, device=device)
        else:
            raise ValueError(f"Unsupported SSL loss: {name}")
        value = float(weight) * value
        total = total + value
        details[name] = float(value.detach().cpu())
    return total, details


@torch.no_grad()
def _run_tto_test(trainer, loader, data_norm, psf, args, device,
                   output_dir):
    """Evaluate only the clean-input PSF deconvolution path.

    Testing is intentionally independent of the alternating training mode:
    it never calls ``_make_noisy_pair`` and never adds synthetic noise.
    """
    trainer.eval()
    inp_sub, inp_div, gt_sub, gt_div = _normalizer(data_norm, device)
    metric_counts, save_counts = defaultdict(int), defaultdict(int)
    rows = []
    artifact_rows = []
    bl_weight, _ = _tto_constraint_weights(args)
    for iteration, batch in enumerate(loader, start=1):
        print(
            f"\rTest | iter={iteration}/{len(loader)}",
            end="",
            flush=True,
        )
        labels = [str(label) for label in batch["sample_label"]]
        if all(_limit_done(metric_counts[label], args.test_num) and
               _limit_done(save_counts[label], args.save_num) for label in labels):
            continue
        # Always use the clean dataloader image during evaluation.
        p = (batch["p"].to(device) - inp_sub) / inp_div
        n = (batch["n"].to(device) - inp_sub) / inp_div
        if "txt_emb" in batch:
            batch["txt_emb"] = batch["txt_emb"].to(device)
        reconstruction = _forward_tto(trainer, batch, p, args.output_size, args)
        convolved = _degrade(
            F.interpolate(reconstruction, size=n.shape[-2:], mode="bicubic", align_corners=False),
            psf,
        )
        restored = reconstruction * gt_div + gt_sub
        restored_input = (p * inp_div + inp_sub).clamp(0.0, 1.0)
        restored_convolved = (convolved * inp_div + inp_sub).clamp(0.0, 1.0)
        bg_mask = None
        bg_laplacian = None
        if bl_weight > 0:
            bg_mask = _background_mask(reconstruction, reconstruction.shape[-2:])
            bg_laplacian = _laplacian_abs(reconstruction) * bg_mask.float()
        for index, label in enumerate(labels):
            safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
            pred_np = np.squeeze(restored[index].float().cpu().numpy()).astype(np.float32)
            conv_np = np.squeeze(
                restored_convolved[index].float().cpu().numpy()
            ).astype(np.float32)
            conv_metric_np = np.squeeze(
                convolved[index].float().cpu().numpy()
            ).astype(np.float32)
            input_np = np.squeeze(restored_input[index].float().cpu().numpy()).astype(np.float32)
            target_np = np.squeeze(n[index].float().cpu().numpy()).astype(np.float32)
            visual_target_np = None
            if "target" in batch:
                visual_target_np = np.squeeze(
                    batch["target"][index].float().cpu().numpy()
                ).astype(np.float32).clip(0.0, 1.0)
            if _under_limit(save_counts[label], args.save_num):
                number = f"{save_counts[label]:05d}"
                save_images = [
                    ("predictions", "pred", pred_np),
                    ("inputs", "input", input_np),
                    ("convolved", "conv", conv_np),
                ]
                if bg_mask is not None and bg_laplacian is not None:
                    save_images.extend([
                        ("bg_masks", "bg_mask", _to_debug_np(bg_mask[index])),
                        ("bg_laplacian", "bg_lap", _to_debug_np(bg_laplacian[index])),
                    ])
                for folder, suffix, image in save_images:
                    directory = output_dir / safe_label / folder
                    directory.mkdir(parents=True, exist_ok=True)
                    tifffile.imwrite(directory / f"{number}_{suffix}.tif", image)
                if visual_target_np is not None:
                    target_dir = output_dir / safe_label / "target"
                    target_dir.mkdir(parents=True, exist_ok=True)
                    tifffile.imwrite(
                        target_dir / f"{number}_target.tif", visual_target_np
                    )
                save_counts[label] += 1
            if _under_limit(metric_counts[label], args.test_num):
                if visual_target_np is not None:
                    # TTO is self-supervised during training, but model
                    # selection must measure the restored prediction against
                    # the held-out visual target when one is available.
                    metrics = _metric_values(
                        pred_np.clip(0.0, 1.0), visual_target_np
                    )
                else:
                    metrics = _metric_values(conv_metric_np, target_np)
                rows.append({"name": f"{metric_counts[label]:05d}",
                             "label": label, **metrics})
                scale_y = reconstruction.shape[-2] / p.shape[-2]
                scale_x = reconstruction.shape[-1] / p.shape[-1]
                coordinate_scale = float((scale_y * scale_x) ** 0.5)
                cg_score = coordinate_frequency_loss(
                    reconstruction[index:index + 1],
                    scale_factor=coordinate_scale,
                    band_width=TTO_COORD_FRE_BAND_WIDTH,
                    ring_inner_ratio=TTO_COORD_FRE_RING_INNER_RATIO,
                    ring_outer_ratio=TTO_COORD_FRE_RING_OUTER_RATIO,
                    peak_ratio=TTO_COORD_FRE_PEAK_RATIO,
                    harmonics=TTO_COORD_FRE_HARMONICS,
                    include_axes=TTO_COORD_FRE_INCLUDE_AXES,
                    include_diagonals=TTO_COORD_FRE_INCLUDE_DIAGONALS,
                    min_frequency=TTO_COORD_FRE_MIN_FREQUENCY,
                )
                artifact_rows.append({
                    "name": f"{metric_counts[label]:05d}",
                    "label": label,
                    "cg_score": float(cg_score.cpu()),
                })
                metric_counts[label] += 1
    print()
    _write_test_results(rows, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(
        output_dir / "artifact_metrics.csv", "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=["name", "label", "cg_score"])
        writer.writeheader()
        writer.writerows(artifact_rows)
    cg_values = np.asarray(
        [row["cg_score"] for row in artifact_rows], dtype=np.float64
    )
    artifact_lines = [f"num_samples: {len(artifact_rows)}"]
    if cg_values.size:
        artifact_lines.extend((
            f"cg_score_mean: {cg_values.mean():.9f}",
            f"cg_score_std: {cg_values.std():.9f}",
        ))
    (output_dir / "artifact_summary.txt").write_text(
        "\n".join(artifact_lines) + "\n", encoding="utf-8"
    )
    for label in sorted(set(metric_counts) | set(save_counts)):
        label_rows = [row for row in rows if row["label"] == label]
        if label_rows:
            _write_test_results(
                label_rows,
                output_dir / re.sub(r"[^A-Za-z0-9_.-]+", "_", label),
            )
    metrics = {}
    for name in ("psnr", "ssim", "nrmse"):
        values = [row[name] for row in rows]
        metrics[name] = float(np.mean(values)) if values else float("nan")
    metrics["cg_score"] = (
        float(cg_values.mean()) if cg_values.size else float("nan")
    )
    return metrics


def TRAIN_TTO(args=None):
    local_rank = args.gid
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    scaler = torch.amp.GradScaler("cuda")
    psf = _build_psf(args, device)

    output_root = Path(Configs.sample_results_path)
    checkpoint_dir = Path(Configs.log_dir)
    mkdir_or_exsist([str(output_root), str(checkpoint_dir)], verbose=True)
    save_settings(str(output_root), args)

    trainer = MultiModel_Trainner.from_args(args).to(device)
    trainer.config["prompt_enable"] = False
    optimizer = _optimizer(trainer, args)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[args.n_steps], gamma=args.gamma)

    start_epoch = 0
    if args.resume:
        rolling = checkpoint_dir / "MultiModel_CKPT.tar"
        files = sorted(checkpoint_dir.glob("MultiModel_CKPT_epoch_*.tar"))
        resume_path = rolling if rolling.is_file() else (files[-1] if files else None)
        start_epoch, _ = load_checkpoint(
            resume_path, trainer, device, optimizer, scheduler,
            mode="resume")
    elif args.loading_MT_ckpt_path:
        start_epoch, _ = load_checkpoint(
            args.loading_MT_ckpt_path, trainer, device, mode="pretrained"
        )
        start_epoch = 0  # pretrained weights start a fresh SSL run

    raw_loss = args.refine_loss[0] if isinstance(args.refine_loss, list) else args.refine_loss
    loss_config = ast.literal_eval(raw_loss) if isinstance(raw_loss, str) else dict(raw_loss)
    all_iter_times = []
    bl_weight, cg_weight = _tto_constraint_weights(args)
    print(
        f"[TTO constraints] BL={bl_weight:g} | CG={cg_weight:g} | "
        f"epochs={args.n_epochs}"
    )

    if bool(getattr(args, "tto_eval_initial", False)):
        initial_loaded = get_multi_data(
            task_idx=args.task_idx,
            data_config=args.MT_data_config,
            testonly=True,
            num_work=0,
            seed=args.seed or 42,
            local_rank=local_rank,
            require_text_embedding=(trainer.family == "fluoresfm"),
        )
        _, initial_test_loader, initial_norm, initial_task_name = initial_loaded[:4]
        initial_dir = (
            output_root / "ssl_test" / initial_task_name /
            f"{trainer.family}_epoch_00000"
        )
        initial_metrics = _run_tto_test(
            trainer, initial_test_loader, initial_norm, psf,
            args, device, initial_dir,
        )
        print(f"[TTO direct baseline] {initial_metrics}")
        cleanup_dataloader(initial_test_loader)

    for epoch in range(start_epoch, args.n_epochs):
        training_mode = epoch % 2
        is_denoising = training_mode == 0
        mode_name = "denoising" if is_denoising else "deconvolution"
        loaded = get_multi_data(
            task_idx=args.task_idx,
            data_config=args.MT_data_config,
            num_work=args.num_workers,
            seed=args.seed or 42,
            local_rank=local_rank,
            require_text_embedding=(trainer.family == "fluoresfm"),
        )
        train_loader, test_loader, data_norm, task_name = loaded[:4]
        inp_sub, inp_div, _, _ = _normalizer(data_norm, device)
        trainer.train()
        losses = []
        for iteration, batch in enumerate(train_loader, start=1):
            torch.cuda.synchronize(device)
            iter_start = time.perf_counter()
            lr = batch["p"].to(device)
            if is_denoising:
                p_raw, n_raw = _make_noisy_pair(lr)
            else:
                p_raw = lr
                n_raw = batch["n"].to(device)
            #tifffile.imwrite(f'./t/p.{iteration}.tif', np.array(p_raw.cpu().detach()))
            p = (p_raw - inp_sub) / inp_div
            n = (n_raw - inp_sub) / inp_div
            if "txt_emb" in batch:
                batch["txt_emb"] = batch["txt_emb"].to(device)
            output_size = (
                p.shape[-2:] if is_denoising
                else int(round(p.shape[-1] * float(batch["s"][0])))
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                reconstruction = _forward_tto(trainer, batch, p, output_size, args)
                if not is_denoising:
                    if _is_magnet_family(trainer.family):
                        # Keep MAGNET's reconstruction at its native SR size.
                        # The observation is enlarged for data consistency so
                        # high-frequency artifacts cannot hide in the output
                        # downsampling operation used by the former strategy.
                        loss_prediction = _degrade(
                            reconstruction, psf
                        )
                        loss_observation = F.interpolate(
                            n,
                            size=loss_prediction.shape[-2:],
                            mode="bicubic",
                            align_corners=False,
                        )
                    else:
                        # Preserve the original TTO behavior for UNet and all
                        # other model families.
                        loss_prediction = _degrade(
                            F.interpolate(
                                reconstruction,
                                size=n.shape[-2:],
                                mode="bicubic",
                                align_corners=False,
                            ),
                            psf,
                        )
                        loss_observation = n
                else:
                    loss_prediction = reconstruction
                    loss_observation = n
                loss, details = _reconstruction_loss(
                    loss_config, loss_prediction, loss_observation,
                    reconstruction, trainer, device)
                if bl_weight > 0:
                    bg_mask = _background_mask(reconstruction, reconstruction.shape[-2:])
                    bg_lap = _masked_laplacian_loss(reconstruction, bg_mask)
                    weighted_bg_lap = bl_weight * bg_lap
                    loss = loss + weighted_bg_lap
                    details["bg_lap_loss"] = float(weighted_bg_lap.detach().cpu())
                if (
                    cg_weight > 0
                    and _is_magnet_family(trainer.family)
                    and not is_denoising
                ):
                    scale_y = reconstruction.shape[-2] / p.shape[-2]
                    scale_x = reconstruction.shape[-1] / p.shape[-1]
                    coordinate_scale = float((scale_y * scale_x) ** 0.5)
                    coord_fre = coordinate_frequency_loss(
                        reconstruction,
                        scale_factor=coordinate_scale,
                        band_width=TTO_COORD_FRE_BAND_WIDTH,
                        ring_inner_ratio=TTO_COORD_FRE_RING_INNER_RATIO,
                        ring_outer_ratio=TTO_COORD_FRE_RING_OUTER_RATIO,
                        peak_ratio=TTO_COORD_FRE_PEAK_RATIO,
                        harmonics=TTO_COORD_FRE_HARMONICS,
                        include_axes=TTO_COORD_FRE_INCLUDE_AXES,
                        include_diagonals=TTO_COORD_FRE_INCLUDE_DIAGONALS,
                        min_frequency=TTO_COORD_FRE_MIN_FREQUENCY,
                    )
                    weighted_coord_fre = cg_weight * coord_fre
                    loss = loss + weighted_coord_fre
                    details["coord_fre_loss"] = float(
                        weighted_coord_fre.detach().cpu()
                    )
            scaler.scale(loss).backward()
            if args.gclip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(trainer.parameters(), args.gclip)
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize(device)
            iter_seconds = time.perf_counter() - iter_start
            all_iter_times.append(iter_seconds)
            losses.append(float(loss.detach().cpu()))
            detail_text = "{" + ", ".join(
                f"{name}: {value:.3f}" for name, value in details.items()
            ) + "}"
            print(f"\rEpoch {epoch + 1}/{args.n_epochs} | mode={mode_name} "
                  f"| iter={iteration}/{len(train_loader)} "
                  f"| loss={losses[-1]:.3f} | iter_time={iter_seconds:.3f}s "
                  f"| {detail_text}", end="")

        # Keep training and evaluation progress on separate terminal lines.
        print()
        train_l1 = float(np.mean(losses))
        eval_metrics = None
        is_sample_epoch = epoch % args.sample_interval == 0
        if is_sample_epoch:
            test_dir = (output_root / "ssl_test" / task_name /
                        f"{trainer.family}_epoch_{epoch + 1:05d}")
            eval_metrics = _run_tto_test(
                trainer, test_loader, data_norm, psf, args, device, test_dir)
        checkpoint = make_checkpoint(
            trainer, optimizer, scheduler, epoch + 1,
            train_l1, eval_metrics, task_name)
        # Keep one always-current resume state and periodic archival snapshots.
        torch.save(checkpoint, checkpoint_dir / "MultiModel_CKPT.tar")
        if is_sample_epoch:
            torch.save(
                checkpoint,
                checkpoint_dir / f"MultiModel_CKPT_epoch_{epoch + 1:05d}.tar",
            )
        print(f"train={train_l1:.6f} | eval={eval_metrics}")
        scheduler.step()
        cleanup_dataloader(train_loader)
        cleanup_dataloader(test_loader)
    average = float(np.mean(all_iter_times)) if all_iter_times else float("nan")
    print(f"Average training iter time: {average:.4f}s")
