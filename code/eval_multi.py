"""Large-TIFF tiled inference through the unified MultiModel trainer."""
from __future__ import annotations

import glob
import os
import random
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models import MultiModel_Trainner
from utils.checkpoint import load_checkpoint
from utils.dataloader_chunk import get_tiff_prompt_loader, make_coord
from utils.rec import TiffBlockStitcher
from utils.utility import PromptPredictor
from utils.utility import parse_int_sequence


MAGNET_FAMILIES = {"magnet", "magnet_old"}


def _is_magnet_family(family):
    return str(family).lower() in MAGNET_FAMILIES


def Normalize_data(x, is_clip=False, cast_bitdepth=16):
    """Preserve the output convention used by the original eval.py."""
    x = np.asarray(x)
    if is_clip:
        x = np.maximum(x, 0)
    x = (x - x.min()) / (x.max() - x.min() + 1e-20)
    if cast_bitdepth == 16:
        return np.asarray(x * 65535, dtype=np.uint16)
    if cast_bitdepth == 8:
        return np.asarray(x * 255, dtype=np.uint8)
    return x


def _input_files(path):
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, "**", "*.tif"), recursive=True))
    if os.path.isfile(path):
        return [path]
    raise FileNotFoundError(f"Inference input not found: {path}")


def _load_prompt_predictor(path, device, task_idx):
    if not path or not os.path.isfile(path):
        print(f"[PDOR] No prompt predictor found: {path or '<empty>'}")
        return None
    predictor = PromptPredictor(
        z_channels=16, task_dim=12, embed_dim=176
    ).to(device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    state = OrderedDict(
        (key.removeprefix("module."), value) for key, value in state.items()
    )
    predictor.load_state_dict(state, strict=True)
    predictor.task_id = int(task_idx)
    predictor.eval()
    print(f"[PDOR] Loaded prompt predictor: {Path(path).resolve()}")
    return predictor


def _random_paired_crop(prompt, crop_size):
    first, second = prompt
    if first.shape != second.shape:
        raise ValueError(f"Prompt pair shape mismatch: {first.shape} vs {second.shape}")
    height, width = first.shape[-2:]
    if height < crop_size or width < crop_size:
        raise ValueError(
            f"Prompt size {height}x{width} is smaller than crop {crop_size}"
        )
    top = random.randint(0, height - crop_size)
    left = random.randint(0, width - crop_size)
    return [
        item[..., top:top + crop_size, left:left + crop_size]
        for item in prompt
    ]


def _magnet_prompt(prompts, blocks, args, device, predictor):
    if predictor is not None:
        return [blocks, blocks]
    pairs = prompts.get("pairs")
    has_pairs = bool(pairs.item()) if torch.is_tensor(pairs) else bool(pairs)
    if not has_pairs:
        raise RuntimeError("MAGNET inference needs prompt pairs or --pdor")
    prompt = [
        (prompts["input"].unsqueeze(0).to(device) - 0.5) / 0.5,
        (prompts["target"].unsqueeze(0).to(device) - 0.5) / 0.5,
    ]
    if args.sr != 1:
        prompt[0] = F.interpolate(
            prompt[0], size=prompt[1].shape[-2:],
            mode="bicubic", align_corners=False,
        )
        prompt = _random_paired_crop(prompt, args.block_size[-1])
    return prompt


def _fluoresfm_embedding(prompt_path, device):
    root = Path(prompt_path)
    embedding_path = root if root.is_file() else root / "txt_embedding.pt"
    if not embedding_path.is_file():
        raise FileNotFoundError(
            "FluoResFM tiled inference requires txt_embedding.pt at "
            f"{embedding_path}"
        )
    embedding = torch.load(embedding_path, map_location="cpu", weights_only=True)
    if tuple(embedding.shape) == (1, 160, 768):
        embedding = embedding[0]
    if tuple(embedding.shape) != (160, 768):
        raise ValueError(f"Invalid text embedding shape: {tuple(embedding.shape)}")
    return embedding.float().unsqueeze(0).to(device)


def _magnet_runtime(args, device, allow_pdor=True):
    if not _is_magnet_family(args.model_family):
        raise ValueError(
            f"This inference mode only supports MAGNET families, got "
            f"{args.model_family!r}"
        )
    trainer = MultiModel_Trainner.from_args(args).to(device)
    _, checkpoint = load_checkpoint(
        args.loading_MT_ckpt_path, trainer, device, mode="inference"
    )
    if checkpoint is None:
        raise FileNotFoundError(
            f"A valid inference checkpoint is required: {args.loading_MT_ckpt_path}"
        )
    trainer.eval()
    predictor = None
    if args.use_prompt and allow_pdor:
        predictor = _load_prompt_predictor(args.pdor, device, args.task_idx)
    return trainer, predictor


def _crop_single_rgb_pair(single, rgb, crop_size):
    if single.ndim != 4 or rgb.ndim != 5 or rgb.shape[-1] != 3:
        raise ValueError(
            f"Expected prompt shapes [B,Z,H,W] and [B,Z,H,W,3], got "
            f"{tuple(single.shape)} and {tuple(rgb.shape)}"
        )
    if single.shape[:2] != rgb.shape[:2] or single.shape[-2:] != rgb.shape[-3:-1]:
        raise ValueError("VST prompt pair dimensions do not match")
    height, width = single.shape[-2:]
    if height < crop_size or width < crop_size:
        raise ValueError(f"VST prompt is smaller than crop size {crop_size}")
    top = random.randint(0, height - crop_size)
    left = random.randint(0, width - crop_size)
    return [
        single[..., top:top + crop_size, left:left + crop_size],
        rgb[..., top:top + crop_size, left:left + crop_size, :],
    ]


def _predict_block(trainer, blocks, prompts, dataset, args, device,
                   prompt_predictor, text_embedding):
    family = trainer.family
    output_shape = tuple(int(value) for value in dataset.const_block_size_sr)
    if _is_magnet_family(family):
        normalized = (blocks.to(device) - 0.5) / 0.5
        prompt_enabled = bool(trainer.config.get("prompt_enable", False))
        prompt = (
            _magnet_prompt(prompts, normalized, args, device, prompt_predictor)
            if prompt_enabled else ["None", "None"]
        )
        result = trainer.forward_batch({
            "inp": normalized,
            "coord": dataset.coord.unsqueeze(0).to(device),
            "cell": dataset.cell.unsqueeze(0).to(device),
            "prompt": prompt,
            "promptPD": prompt_predictor if prompt_enabled else None,
            "s_factor": torch.full(
                (normalized.shape[0],), float(args.sr), device=device
            ),
        })["prediction"]
        return (result * 0.5 + 0.5).permute(0, 2, 1).reshape(
            result.shape[0], *output_shape
        )[0]

    blocks = blocks.to(device)
    if blocks.ndim != 4:
        raise NotImplementedError(
            f"{family} currently supports 2D tiled inference only; "
            f"received block shape {tuple(blocks.shape)}"
        )
    task_id = parse_int_sequence(args.model_task, "model_task")[0]
    # Temporary compatibility for the official UNiFMIR baseline: its denoising
    # head (task 2, conv_firstdT) was trained with five input channels, while
    # the TIFF inference loader provides one channel.
    if family == "unifmir" and task_id == 2 and blocks.shape[1] == 1:
        blocks = blocks.repeat(1, 5, 1, 1)
    model_batch = {
        "task_id": task_id,
        "s": torch.full((blocks.shape[0],), float(args.sr), device=device),
    }
    if text_embedding is not None:
        model_batch["txt_emb"] = text_embedding.expand(blocks.shape[0], -1, -1)
    prediction = trainer.forward_image(
        blocks,
        batch=model_batch,
        output_size=output_shape[-2:],
    )
    return prediction[0]


def Inference(args):
    if args.VST:
        return Test_Multi_VST(args)
    if args.iso_3d:
        return wsi_iso(args)
    return Test_Multi_wsi_T(args)


def Test_Multi_wsi_T(args=None):
    """Infer every TIFF in a file/folder using overlapped block stitching."""
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda", args.gid)
    torch.cuda.set_device(args.gid)

    trainer = MultiModel_Trainner.from_args(args).to(device)
    _, checkpoint = load_checkpoint(
        args.loading_MT_ckpt_path, trainer, device, mode="inference"
    )
    if checkpoint is None:
        raise FileNotFoundError(
            f"A valid inference checkpoint is required: {args.loading_MT_ckpt_path}"
        )
    trainer.eval()
    trainer.config["prompt_enable"] = (
        _is_magnet_family(trainer.family) and bool(args.use_prompt)
    )
    trainer.config["post_SR"] = False  # prompt is explicitly pre-scaled below

    prompt_predictor = None
    if _is_magnet_family(trainer.family) and args.use_prompt:
        prompt_predictor = _load_prompt_predictor(
            args.pdor, device, args.task_idx
        )
    elif _is_magnet_family(trainer.family):
        print("[Prompt] Disabled by use_prompt=false")
    text_embedding = (
        _fluoresfm_embedding(args.prompt_path, device)
        if trainer.family == "fluoresfm" else None
    )

    files = _input_files(args.inp_path)
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    print(f"[EvalMulti] family={trainer.family} | files={len(files)}")

    for file_index, file_path in enumerate(files):
        print(f"[EvalMulti] {file_index + 1}/{len(files)}: {file_path}")
        image_start = time.perf_counter()
        patch_times = []
        loader, dataset = get_tiff_prompt_loader(
            tiff_path=file_path,
            prompt_root=args.prompt_path,
            block_size=tuple(args.block_size),
            overlap=tuple(args.overlap),
            sr=args.sr,
            batch_size=1,
            num_workers=args.num_workers,
            is_iso=False,
            prenorm=args.prenorm,
            use_rate=args.use_rate,
        )
        stitcher = TiffBlockStitcher(dataset, args.sr, device="cpu")

        with torch.inference_mode():
            for indices, blocks, prompts, rate in tqdm(
                loader, leave=True, desc="WSI predicting"
            ):
                torch.cuda.synchronize(device)
                patch_start = time.perf_counter()
                prediction = _predict_block(
                    trainer, blocks, prompts, dataset, args, device,
                    prompt_predictor, text_embedding,
                )
                expected = stitcher.block_size
                crop_top = max(0, (prediction.shape[-2] - expected[-2]) // 2)
                crop_left = max(0, (prediction.shape[-1] - expected[-1]) // 2)
                prediction = prediction[
                    ..., crop_top:crop_top + expected[-2],
                    crop_left:crop_left + expected[-1],
                ]
                stitcher.add_block(
                    indices,
                    torch.clamp(prediction * rate.to(device), min=0.0),
                )
                torch.cuda.synchronize(device)
                patch_times.append(time.perf_counter() - patch_start)

        final_result = stitcher.get_final_result().cpu().numpy()
        output_file = save_path / f"{file_index:05d}.tif"
        tifffile.imwrite(output_file, Normalize_data(final_result, cast_bitdepth=16))
        image_seconds = time.perf_counter() - image_start
        average_patch = (
            float(np.mean(patch_times)) if patch_times else float("nan")
        )
        print(f"[EvalMulti] saved: {output_file}")
        print(
            f"[Timing] patches={len(patch_times)} | "
            f"avg_patch={average_patch:.4f}s | "
            f"total_image={image_seconds:.4f}s"
        )


def Test_Multi_VST(args=None):
    """MAGNET-only tiled virtual-staining inference with RGB output."""
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda", args.gid)
    torch.cuda.set_device(args.gid)
    trainer, _ = _magnet_runtime(args, device, allow_pdor=False)
    if args.MT_model_name != "MultiModel_X_light_3c":
        raise ValueError("VST requires MT_model_name=MultiModel_X_light_3c")
    if args.use_prompt is False:
        print("[Prompt] VST prompt disabled by use_prompt=false")

    files = _input_files(args.inp_path)
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    for file_index, file_path in enumerate(files):
        image_start = time.perf_counter()
        patch_times = []
        loader, dataset = get_tiff_prompt_loader(
            tiff_path=file_path,
            prompt_root=args.prompt_path,
            block_size=tuple(args.block_size),
            overlap=tuple(args.overlap),
            sr=args.sr,
            batch_size=1,
            num_workers=args.num_workers,
            is_iso=False,
            VST=True,
            use_rate=args.use_rate,
        )
        stitchers = [
            TiffBlockStitcher(dataset, args.sr, device="cpu")
            for _ in range(3)
        ]
        with torch.inference_mode():
            for indices, blocks, input_prompt, target_prompt, _ in tqdm(
                loader, leave=True, desc="VST predicting"
            ):
                torch.cuda.synchronize(device)
                patch_start = time.perf_counter()
                blocks = blocks.to(device)
                
                prompt = ["None", "None"]
                if args.use_prompt:
                    prompt = [
                        input_prompt.unsqueeze(0).to(device),
                        target_prompt.unsqueeze(0).to(device),
                    ]
                    if args.sr != 1:
                        prompt[0] = F.interpolate(
                            prompt[0], size=prompt[1].shape[-3:-1],
                            mode="bicubic", align_corners=False,
                        )
                        prompt = _crop_single_rgb_pair(
                            prompt[0], prompt[1], args.block_size[-1]
                        )
                    prompt[1] = prompt[1].squeeze(0).permute(0, 3, 1, 2)

                prediction = trainer.model(
                    x=blocks,
                    coord=dataset.coord.unsqueeze(0).to(device),
                    cell=dataset.cell.unsqueeze(0).to(device),
                    prompt_tensor=prompt,
                    prompt_enable=bool(args.use_prompt),
                    s_factor=args.sr,
                    outmode="tri",
                )
                height, width = dataset.const_block_size_sr[-2:]
                prediction = prediction.reshape(height, width, 3).permute(2, 0, 1)
                for channel, stitcher in enumerate(stitchers):
                    stitcher.add_block(indices, prediction[channel].clamp_min(0.0))
                torch.cuda.synchronize(device)
                patch_times.append(time.perf_counter() - patch_start)

        channels = [stitcher.get_final_result().cpu() for stitcher in stitchers]
        final_result = torch.cat(channels, dim=0).numpy()
        output_file = save_path / f"{file_index:05d}.tif"
        tifffile.imwrite(output_file, Normalize_data(final_result, cast_bitdepth=8))
        total = time.perf_counter() - image_start
        average = float(np.mean(patch_times)) if patch_times else float("nan")
        print(f"[EvalMulti:VST] saved: {output_file}")
        print(
            f"[Timing] patches={len(patch_times)} | avg_patch={average:.4f}s "
            f"| total_image={total:.4f}s"
        )


def wsi_iso(args=None):
    """MAGNET-only 3D isotropic reconstruction from XY and XZ views."""
    if tuple(args.block_size) != (128, 128, 128):
        raise ValueError("3D ISO requires block_size=[128,128,128]")
    if float(args.sr) != 1.0:
        raise ValueError("3D ISO requires sr=1")
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda", args.gid)
    torch.cuda.set_device(args.gid)
    trainer, predictor = _magnet_runtime(args, device, allow_pdor=True)

    coord = make_coord((128, 128)).unsqueeze(0).to(device)
    cell = torch.ones_like(coord)
    cell[..., 0] *= 2 / 128
    cell[..., 1] *= 2 / 128
    files = _input_files(args.inp_path)
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    for file_index, file_path in enumerate(files):
        image_start = time.perf_counter()
        patch_times = []
        loader, dataset = get_tiff_prompt_loader(
            tiff_path=file_path,
            prompt_root=args.prompt_path,
            block_size=tuple(args.block_size),
            overlap=tuple(args.overlap),
            sr=args.sr,
            batch_size=1,
            num_workers=args.num_workers,
            is_iso=True,
            use_rate=args.use_rate,
        )
        stitch_xy = TiffBlockStitcher(dataset, args.sr, device="cpu", is_iso=True)
        stitch_xz = TiffBlockStitcher(dataset, args.sr, device="cpu", is_iso=True)

        with torch.inference_mode():
            for indices, blocks, prompts, rate in tqdm(
                loader, leave=True, desc="ISO predicting"
            ):
                torch.cuda.synchronize(device)
                patch_start = time.perf_counter()
                blocks = (blocks.to(device) - 0.5) / 0.5
                views = (blocks, blocks.permute(0, 1, 4, 3, 2))
                restored_views = []
                for view in views:
                    restored = torch.zeros_like(view)
                    for slice_index in range(128):
                        image_slice = view[:, :, slice_index]
                        prompt = ["None", "None"]
                        if args.use_prompt:
                            prompt = _magnet_prompt(
                                prompts, image_slice, args, device, predictor
                            )
                        output = trainer.model(
                            x=image_slice,
                            coord=coord,
                            cell=cell,
                            promptPD=predictor if args.use_prompt else None,
                            prompt_tensor=prompt,
                            prompt_enable=bool(args.use_prompt),
                            s_factor=args.sr,
                        )
                        restored[:, :, slice_index] = output.reshape(1, 1, 128, 128)
                    restored_views.append(restored * 0.5 + 0.5)

                block_xy = restored_views[0][0, 0]
                block_xz = restored_views[1].permute(0, 1, 4, 3, 2)[0, 0]
                stitch_xy.add_block(indices, block_xy * rate.to(device))
                stitch_xz.add_block(indices, block_xz * rate.to(device))
                torch.cuda.synchronize(device)
                patch_times.append(time.perf_counter() - patch_start)

        result_xy = stitch_xy.get_final_result().cpu()
        result_xz = stitch_xz.get_final_result().cpu()
        final_result = torch.sqrt(F.relu(result_xy) * F.relu(result_xz)).numpy()
        output_file = save_path / f"iso_result{file_index:04d}.tif"
        tifffile.imwrite(output_file, final_result.astype(np.float32))
        total = time.perf_counter() - image_start
        average = float(np.mean(patch_times)) if patch_times else float("nan")
        print(f"[EvalMulti:ISO] saved: {output_file} | shape={final_result.shape}")
        print(
            f"[Timing] patches={len(patch_times)} | avg_patch={average:.4f}s "
            f"| total_image={total:.4f}s"
        )
