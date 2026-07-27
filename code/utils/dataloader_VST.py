"""Isolated virtual-staining dataloader.

Fluorescence inputs use FluoResFM percentile normalization. Bright-field RGB
targets preserve colour ratios and are converted only by uint8 / 255.
"""
from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


P_LOW = 0.03
P_HIGH = 0.995
CLIP = False


def _fluorescence_normalize(image: np.ndarray) -> np.ndarray:
    image = np.clip(np.asarray(image, dtype=np.float32), 0.0, None)
    low = float(np.percentile(image, P_LOW * 100.0))
    high = float(np.percentile(image, P_HIGH * 100.0))
    image = (image - low) / max(high - low, 1e-8)
    if CLIP:
        image = np.clip(image, 0.0, 1.0)
    return image.astype(np.float32, copy=False)


def _rgb_target(image: np.ndarray, path: Path) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected RGB target [H,W,3], got {image.shape}: {path}")
    # Bright-field RGB uses the image convention requested for VST: /255,
    # never percentile or per-channel min-max normalization.
    maximum = float(np.max(image))
    if maximum > 255.0:
        raise ValueError(f"RGB target exceeds 8-bit range ({maximum}): {path}")
    scale = 255.0 if maximum > 1.0 or np.issubdtype(image.dtype, np.integer) else 1.0
    return (image.astype(np.float32) / scale).transpose(2, 0, 1)


def _paired_files(root: Path, split: str) -> list[tuple[Path, Path]]:
    input_dir, target_dir = root / split / "input", root / split / "target"
    if not input_dir.is_dir() or not target_dir.is_dir():
        raise NotADirectoryError(f"Expected {split}/input and {split}/target under {root}")
    suffixes = {".tif", ".tiff"}
    inputs = {p.stem: p for p in input_dir.iterdir() if p.suffix.lower() in suffixes}
    targets = {p.stem: p for p in target_dir.iterdir() if p.suffix.lower() in suffixes}
    names = sorted(inputs.keys() & targets.keys())
    if not names:
        raise FileNotFoundError(f"No paired TIFF files under {root / split}")
    return [(inputs[name], targets[name]) for name in names]


class VSTDataset(Dataset):
    def __init__(self, root, split="train", patch_size=64, embedding=None):
        self.root = Path(root)
        self.split = split
        self.patch_size = int(patch_size)
        self.pairs = _paired_files(self.root, split)
        self.embedding = embedding

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        input_path, target_path = self.pairs[index]
        inp = np.squeeze(tifffile.imread(input_path))
        if inp.ndim != 2:
            raise ValueError(f"Expected fluorescence input [H,W], got {inp.shape}: {input_path}")
        inp = torch.from_numpy(_fluorescence_normalize(inp)).unsqueeze(0)
        target = torch.from_numpy(_rgb_target(tifffile.imread(target_path), target_path))
        if inp.shape[-2:] != target.shape[-2:]:
            inp = F.interpolate(inp[None], size=target.shape[-2:], mode="bilinear",
                                align_corners=False)[0]

        if self.split == "train":
            height, width = target.shape[-2:]
            if height < self.patch_size or width < self.patch_size:
                raise ValueError(f"Patch {self.patch_size} exceeds image {height}x{width}")
            top = random.randint(0, height - self.patch_size)
            left = random.randint(0, width - self.patch_size)
            slices = (..., slice(top, top + self.patch_size),
                      slice(left, left + self.patch_size))
            inp, target = inp[slices], target[slices]
            if random.random() < 0.5:
                inp, target = inp.flip(-1), target.flip(-1)
            if random.random() < 0.5:
                inp, target = inp.flip(-2), target.flip(-2)

        result = {"inp": inp, "target": target, "name": input_path.stem}
        if self.embedding is not None:
            result["txt_emb"] = self.embedding[0]
        return result


def _settings(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    labels = config.get("Task_labels") or []
    if len(labels) != 1:
        raise ValueError("VST_DDP expects exactly one Task_labels entry")
    return labels[0], config["data_info"][labels[0]]


def get_vst_loaders(yaml_path, num_workers, rank, world_size):
    label, settings = _settings(yaml_path)
    root = Path(settings["data_path"])
    embedding_path = root / "txt_embedding.pt"
    if not embedding_path.is_file():
        raise FileNotFoundError(f"VST text embedding not found: {embedding_path}")
    embedding = torch.load(embedding_path, map_location="cpu", weights_only=True).float()
    if tuple(embedding.shape) != (1, 160, 768):
        raise ValueError(f"Expected embedding [1,160,768], got {tuple(embedding.shape)}")
    patch_size = int(settings.get("patch_size", 64))
    eval_patch_size = int(settings.get("eval_patch_size", 64))
    train_set = VSTDataset(root, "train", patch_size, embedding)
    test_set = VSTDataset(root, "test", eval_patch_size, embedding)
    sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
    loader_args = dict(num_workers=num_workers, pin_memory=True,
                       persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_set, batch_size=int(settings.get("batch_size", 1)),
                              sampler=sampler, **loader_args)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, **loader_args)
    return train_loader, sampler, test_loader, label, eval_patch_size
