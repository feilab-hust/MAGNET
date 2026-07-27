import os
import random
import tifffile
import torch
from torch.utils.data import Dataset, DataLoader
from math import ceil
from typing import Tuple, List

NORMAX=1.0
SOFTMORM=False
# Inference percentile normalization. Values use numpy.percentile units
# (0--100), matching dataloader_MultiTask.py and FluoResFM's 3%/99.5% setup.
P_LOW = 0 #3.0
P_HIGH = 100 # 99.5
CLIP = True #False

import numpy as np
from scipy.ndimage import convolve

def average_convolution(input_array, kernel_size=3):
    if input_array.ndim != 3:
        raise ValueError("Input shape must be (C, H, W)")

    if isinstance(kernel_size, int):
        if input_array.shape[0] == 1:
            kernel_size = (1, kernel_size, kernel_size)  # 2D卷积
        else:
            kernel_size = (kernel_size, kernel_size, kernel_size)  # 3D卷积
    else:
        if len(kernel_size) == 2:
            kernel_size = (1,) + kernel_size if input_array.shape[0] == 1 else kernel_size + (kernel_size[-1],)

    kernel = np.ones(kernel_size) / np.prod(kernel_size)

    if input_array.shape[0] == 1:
        result = convolve(input_array[0], kernel[0], mode='mirror')
        return result[np.newaxis, ...]
    else:
        return convolve(input_array, kernel, mode='mirror')
def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs,indexing='ij'), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

def normalize_percentile(
        im, low=P_LOW, high=P_HIGH, clip=CLIP,
        is_random=False, dtype=np.float32):
    if is_random:
        _p_low = np.random.uniform(0.1, 0.5)
        p_low = np.percentile(im, _p_low)

        _p_high = np.random.uniform(99.5, 99.9)
        p_high = np.percentile(im, _p_high)
    else:
        p_low = np.percentile(im, low)
        p_high = np.percentile(im, high)
    eps = 1e-8
    if dtype is not None:
        x = ((im - p_low) / (p_high - p_low + eps)).astype(np.float32)
    else:
        x = ((im - p_low) / (p_high - p_low + eps))
    if clip:
        x[x > 1.0] = 1.0
        x[x < .0] = .0

    if dtype is not None:
        x = x.astype(dtype)

    return x


class TiffBlockPromptDataset(Dataset):
    def __init__(self,
                 tiff_path: str,
                 prompt_root: str,
                 block_size: Tuple[int, int, int] = (64, 128, 128),
                 overlap: Tuple[int, int, int] = (16, 32, 32),
                 sr:float = 1,
                 pad_mode: str = 'reflect',
                 is_iso:bool=False,
                 VST:bool=False,
                 prenorm:bool=False,
                 use_rate:bool=True,
                 ):

        self.is_iso = is_iso
        self.VST = VST
        self.prenorm = prenorm
        self.use_rate = use_rate
        self.img = np.asarray(tifffile.imread(tiff_path), dtype=np.float32)
        if not VST and not self.prenorm:
            self.img = normalize_percentile(
                self.img, low=P_LOW, high=P_HIGH, clip=CLIP
            )
        self.sr = sr
        assert len(block_size)==2 or len(block_size)==3 ,f'invalid shape of block size {len(block_size)}!'
        assert len(overlap) == 2 or len(overlap) == 3, f'invalid shape of overlap size {len(overlap)}!'
        block_size = tuple(int(v) for v in block_size)
        overlap = tuple(int(v) for v in overlap)

        if len(block_size) == 2:
            block_size = (1,) + block_size
            self.mode = '2d'
        elif block_size[0] == 1:
            self.mode = '2d'
        else:
            self.mode = '3d'

        if len(overlap) == 2:
            overlap = (0,) + overlap
        elif self.mode == '2d':
            overlap = (0,) + overlap[1:]

        self.block_size = block_size
        self.overlap = overlap
        self.const_block_size = list(self.block_size)
        self.const_block_size_sr = [
            self.const_block_size[0],
            int(self.const_block_size[1] * self.sr),
            int(self.const_block_size[2] * self.sr),
        ]

        #self.tiff_path = tiff_path
        self.prompt_root = prompt_root
        self.pad_mode = pad_mode
        #self.pad_per_block = 128-self.block_size[-1]

        self._load_and_pad_tiff()

        try:
            self.prompt_pairs = self._load_prompt_pairs()
        except:
            self.prompt_pairs = None
            print(f'No Prompt Pairs at {self.prompt_root}! Try to use prompt predictor')


        self.block_indices = self._precompute_block_indices()

        self.coord = make_coord(self.const_block_size_sr if self.mode=='3d' else self.const_block_size_sr[1:])
        self.cell = torch.ones_like(self.coord)

        if self.mode=='3d':
            for _i in range(self.cell.shape[-1]):
                self.cell[:, _i] *= 2 / self.const_block_size_sr[_i] # block_size[_i]
        else:
            for _i in range(self.cell.shape[-1]):
                self.cell[:, _i] *= 2 / self.const_block_size_sr[_i+1]

    def _load_and_pad_tiff(self):
        if len(self.img.shape)==2:
            self.img = np.expand_dims(self.img, axis=0)
        self.original_shape = self.img.shape
        #self.norm_max = np.percentile(self.img, high)

        self.padding = []
        self.padded_shape = []
        self.pad_for_block_crop = []
        for i in range(3):

            pad_for_block_crop = self.const_block_size[i]-self.block_size[i]
            self.pad_for_block_crop.append(pad_for_block_crop)
            stride = self.block_size[i] - self.overlap[i]
            num_blocks = ceil((self.original_shape[i] - self.overlap[i]) / stride)
            padded_dim = num_blocks * stride + self.overlap[i]
            pad_total = max(0, padded_dim - self.original_shape[i])
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
            self.padding.append((pad_before+pad_for_block_crop//2, pad_after+pad_for_block_crop//2))

        print(self.padding)
        self.padded_img = np.pad(
            self.img,
            pad_width=self.padding,
            mode=self.pad_mode
        )
        self.padded_shape = self.padded_img.shape
        print(self.padded_shape)
        if self.mode=='2d':
            self.SSM = average_convolution(self.padded_img)
            self.mask_norm_max = self.SSM.max()
        elif self.mode=='3d':
            self.SSM = average_convolution(self.padded_img)
            self.mask_norm_max = self.SSM.max()
        else:
            raise AssertionError
    def _load_prompt_pairs(self) -> List[Tuple[str, str]]:
        input_dir = os.path.join(self.prompt_root, 'input')
        target_dir = os.path.join(self.prompt_root, 'target')


        input_files = set(os.listdir(input_dir))
        target_files = set(os.listdir(target_dir))
        #common_files = input_files & target_files
        assert len(input_files)==len(target_files)

        pairs = []
        for inp_name,tar_name in zip(input_files,target_files):
            input_path = os.path.join(input_dir, inp_name)
            target_path = os.path.join(target_dir, tar_name)
            pairs.append((input_path, target_path))
        return pairs

    def _read_prompt_file(self, path: str) -> str:

        img = np.asarray(tifffile.imread(path),np.float32)
        return img

    def _precompute_block_indices(self):

        indices = []
        stride = (
            self.block_size[0] - self.overlap[0],
            self.block_size[1] - self.overlap[1],
            self.block_size[2] - self.overlap[2]
        )

        num_blocks = (
            ceil((self.padded_shape[0] - self.pad_for_block_crop[0] - self.overlap[0]) / stride[0]) if self.is_iso else ceil((self.padded_shape[0] - - self.overlap[0]) / stride[0]),#Z轴方向暂时不做推大取小处理
            #ceil((self.padded_shape[0] - self.pad_for_block_crop[0] - self.overlap[0]) / stride[0]),
            ceil((self.padded_shape[1] - self.pad_for_block_crop[1] - self.overlap[1]) / stride[1]),
            ceil((self.padded_shape[2] - self.pad_for_block_crop[2] - self.overlap[2]) / stride[2])
        )

        for a in range(num_blocks[0]):
            d_start = a * stride[0]
            d_end = d_start + self.const_block_size[0]

            for b in range(num_blocks[1]):
                h_start = b * stride[1]
                h_end = h_start + self.const_block_size[1]

                for c in range(num_blocks[2]):
                    w_start = c * stride[2]
                    w_end = w_start + self.const_block_size[2]

                    indices.append((
                        (a, b, c),  # 块索引
                        (d_start, d_end, h_start, h_end, w_start, w_end)  # 坐标
                    ))

        return indices

    def __len__(self):
        return len(self.block_indices)

    def __getitem__(self, idx):
        if not self.VST:
            (a, b, c), (d_s, d_e, h_s, h_e, w_s, w_e) = self.block_indices[idx]

            block = self.padded_img[d_s:d_e, h_s:h_e, w_s:w_e]
            mask = self.SSM[d_s:d_e, h_s:h_e, w_s:w_e]

            if self.use_rate:
                rate = (mask.max()-mask.mean()) / self.mask_norm_max
            else:
                rate = 1.0
            #rate = mask.max() / self.mask_norm_max

            if self.prenorm:
                # The complete 3-D volume was normalized before inference;
                # preserve its shared intensity scale across every block/slice.
                block = np.asarray(block, dtype=np.float32)
            elif SOFTMORM:
                block = normalize_percentile(
                    block, low=P_LOW, high=P_HIGH, clip=CLIP
                )
            else:
                block = normalize_percentile(block)
            if self.mode=='3d':
                block_tensor = torch.from_numpy(block).unsqueeze(0)  # (1, D, H, W)
            else:
                block_tensor = block
            if self.prompt_pairs:
                input_path, target_path = random.choice(self.prompt_pairs)
                input_prompt = self._read_prompt_file(input_path)
                target_prompt = self._read_prompt_file(target_path)
                if SOFTMORM:
                    input_prompt = normalize_percentile(
                        input_prompt, low=P_LOW, high=P_HIGH, clip=CLIP
                    )
                    target_prompt = normalize_percentile(
                        target_prompt, low=P_LOW, high=P_HIGH, clip=CLIP
                    )
                else:
                    input_prompt = normalize_percentile(input_prompt)
                    target_prompt = normalize_percentile(target_prompt)
                prompts = {
                    'pairs': True,
                    'input': input_prompt,
                    'target': target_prompt,
                }
            else:
                prompts = {
                    'pairs':False,
                }
            return (a, b, c), block_tensor, prompts, rate
        else: #VST
            (a, b, c), (d_s, d_e, h_s, h_e, w_s, w_e) = self.block_indices[idx]

            block = self.padded_img[d_s:d_e, h_s:h_e, w_s:w_e]
            rate = 1
            block = block/15000
            if self.mode == '3d':
                block_tensor = torch.from_numpy(block).unsqueeze(0)
            else:
                block_tensor = block
            input_path, target_path = random.choice(self.prompt_pairs)
            input_prompt = self._read_prompt_file(input_path)
            target_prompt = self._read_prompt_file(target_path)
            if SOFTMORM:
                input_prompt = normalize_percentile(
                    input_prompt, low=P_LOW, high=P_HIGH, clip=CLIP
                )
                target_prompt = normalize_percentile(
                    target_prompt, low=P_LOW, high=P_HIGH, clip=CLIP
                )
            else:
                input_prompt = normalize_percentile(input_prompt)
                target_prompt = normalize_percentile(target_prompt)
            return (a, b, c), block_tensor, input_prompt, target_prompt, rate


def get_tiff_prompt_loader(
        tiff_path: str,
        prompt_root: str,
        block_size: Tuple[int, int, int] = (64, 128, 128),
        overlap: Tuple[int, int, int] = (16, 32, 32),
        sr:int=1,
        pad_mode: str = 'reflect',
        batch_size: int = 1,
        num_workers: int = 0,
        is_iso:bool=False,
        VST:bool=False,
        prenorm:bool=False,
        use_rate:bool=True,
):
    dataset = TiffBlockPromptDataset(
        tiff_path=tiff_path,
        prompt_root=prompt_root,
        block_size=block_size,
        overlap=overlap,
        sr=sr,
        pad_mode=pad_mode,
        is_iso=is_iso,
        VST=VST,
        prenorm=prenorm,
        use_rate=use_rate,
    )

    def collate_fn(batch):
        indices = [item[0] for item in batch]
        blocks = torch.stack([item[1] for item in batch])
        input_prompts = [item[2] for item in batch]
        target_prompts = [item[3] for item in batch]
        return indices, blocks, input_prompts, target_prompts

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        #collate_fn=collate_fn
    ), dataset

