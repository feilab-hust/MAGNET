"""Image-form dataloader for PSF-guided self-supervised training."""
import os
import random

import numpy as np
import tifffile
import torch
from omegaconf import ListConfig
from torch.utils.data import DataLoader, Dataset

from .dataloader_MultiTask import _load_text_embedding
from .utility import load_config, load_file_list, normalize_percentile, str2value

__all__ = ["get_multi_data"]
global_seed = 42


def get_multi_data(task_idx, data_config=None, testonly=False, num_work=0,
                   seed=global_seed, local_rank=0, require_text_embedding=False):
    if not data_config or not os.path.isfile(data_config):
        raise FileNotFoundError(f"SSL data config not found: {data_config}")
    config = load_config(data_config)
    task_name = config.Task_labels[task_idx]
    info = getattr(config.data_info, task_name)
    scale = info["sr_ratio"]
    scale_min, scale_max = scale if isinstance(scale, ListConfig) else (scale, scale)
    common = dict(
        root=info["data_path"],
        scale_min=scale_min,
        scale_max=scale_max,
        repeat=info.get("repeat", 1),
        read_folder_mode=info["data_loading_mode"],
        discard_folder=info["discard_train_folder"],
        pre_normalize=info["pre_normalize"],
        load_target=bool(info.get("target", False)),
        require_text_embedding=require_text_embedding,
    )
    generator = torch.Generator().manual_seed(seed or global_seed)
    train_loader = None
    if not testonly:
        train_loader = DataLoader(
            SSLImageDataset(split="train", augment=True, **common),
            batch_size=info["batch_size"], shuffle=True, generator=generator,
            pin_memory=True, num_workers=num_work,
            persistent_workers=num_work > 0,
        )
    test_loader = DataLoader(
        SSLImageDataset(split="test", augment=False, **common),
        batch_size=1, shuffle=False, pin_memory=True, num_workers=0,
    )
    return (train_loader, test_loader, config.data_norm, task_name,
            info["Dims"], info["batch_size"])


class SSLImageDataset(Dataset):
    def __init__(self, root, split, scale_min=1, scale_max=1, repeat=1,
                 augment=False, read_folder_mode="SINGLE", discard_folder=None,
                 pre_normalize=False, load_target=False,
                 require_text_embedding=False):
        self.root = root
        self.split = split
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.repeat = repeat
        self.augment = augment
        self.pre_normalize = pre_normalize
        # Targets are optional visual references and are never loaded into
        # the self-supervised training split or used by its loss.
        self.load_target = bool(load_target) and split == "test"
        self.txt_emb = {}
        discarded = set(discard_folder or []) if isinstance(discard_folder, (list, tuple)) else {discard_folder}

        mode = read_folder_mode.upper()
        if mode == "SINGLE":
            roots = [root]
        elif mode in ("MULTI", "PICK"):
            roots = [os.path.join(root, name) for name in sorted(os.listdir(root))
                     if os.path.isdir(os.path.join(root, name)) and name not in discarded]
            if mode == "PICK" and roots:
                roots = [random.choice(roots)]
        else:
            raise ValueError("data_loading_mode must be SINGLE, MULTI, or PICK")

        self.files, self.targets, self.labels = [], [], []
        for dataset_root in roots:
            label = os.path.basename(os.path.normpath(dataset_root))
            input_dir = os.path.join(dataset_root, split, "input")
            files = load_file_list(input_dir, regx=".*.tif", is_fullpath=True)
            if not files:
                continue
            targets = []
            if self.load_target:
                target_dir = os.path.join(dataset_root, split, "target")
                targets = load_file_list(
                    target_dir, regx=".*.tif", is_fullpath=True
                )
                if len(files) != len(targets):
                    raise RuntimeError(
                        "SSL input/target count mismatch under "
                        f"{dataset_root}/{split}: {len(files)} vs {len(targets)}"
                    )
            embedding = _load_text_embedding(dataset_root, require_text_embedding)
            if embedding is not None:
                self.txt_emb[label] = embedding
            self.files.extend(files)
            self.targets.extend(targets)
            self.labels.extend([label] * len(files))
        if not self.files:
            raise RuntimeError(f"No SSL tif images found under {root}/{split}")

    def __len__(self):
        return len(self.files) * self.repeat

    def _read(self, path):
        image = np.squeeze(np.asarray(tifffile.imread(path), dtype=np.float32))
        if not self.pre_normalize:
            image = normalize_percentile(image, 0, 100, clip=True)
        return torch.from_numpy(np.asarray(image)).unsqueeze(0)

    def __getitem__(self, index):
        source_index = index % len(self.files)
        image = self._read(self.files[source_index])
        p = image.clone()
        n = image.clone()
        if self.augment:
            if random.random() < 0.5:
                p, n = p.flip(-2), n.flip(-2)
            if random.random() < 0.5:
                p, n = p.flip(-1), n.flip(-1)
            if random.random() < 0.5:
                p, n = p.transpose(-2, -1), n.transpose(-2, -1)
        label = self.labels[source_index]
        result = {
            "s": random.uniform(self.scale_min, self.scale_max),
            "p": p,
            "n": n,
            "lr": p.clone(),
            "sample_shape": n.shape,
            "sample_label": label,
        }
        if self.load_target:
            result["target"] = self._read(self.targets[source_index])
        if label in self.txt_emb:
            result["txt_emb"] = self.txt_emb[label]
        return result
