import math

import torch


class TiffBlockStitcher:
    def __init__(self, dataset, sr, device='cpu', is_iso=False):
        """
        Stitch network outputs from TiffBlockPromptDataset.

        For non-integer SR, the network returns floor(input_size * sr) for
        each patch. The stitched canvas therefore follows the actual patch
        grid, not floor(full_image_size * sr).
        """
        self.device = device
        self.is_iso = is_iso
        self.sr = sr

        self.raw_block_size = tuple(dataset.block_size)
        self.raw_overlap = tuple(dataset.overlap)
        self.raw_stride = tuple(
            b - o for b, o in zip(self.raw_block_size, self.raw_overlap)
        )
        self.raw_const_block_size = tuple(
            dataset.const_block_size if hasattr(dataset, 'const_block_size') else dataset.block_size
        )

        self.num_blocks = self._get_num_blocks(dataset)
        self.block_size = tuple(
            self._scale_len(v, dim) for dim, v in enumerate(self.raw_block_size)
        )
        self.stride = tuple(
            self._scale_len(v, dim) for dim, v in enumerate(self.raw_stride)
        )
        self.const_block_size = tuple(
            self._scale_len(v, dim) for dim, v in enumerate(self.raw_const_block_size)
        )
        self.crop_before = tuple(
            max(0, (const - block) // 2)
            for const, block in zip(self.const_block_size, self.block_size)
        )
        self.overlap = tuple(
            max(0, block - stride)
            for block, stride in zip(self.block_size, self.stride)
        )
        self.padding = [
            (
                self._scale_len(pad_before, dim),
                self._scale_len(pad_after, dim),
            )
            for dim, (pad_before, pad_after) in enumerate(dataset.padding)
        ]

        self.padded_shape = tuple(
            (n - 1) * stride + const
            for n, stride, const in zip(self.num_blocks, self.stride, self.const_block_size)
        )

        self.result = torch.zeros(self.padded_shape, dtype=torch.float32, device=self.device)
        self.weights = torch.zeros(self.padded_shape, dtype=torch.float32, device=self.device)
        self._precompute_weight_windows()

    def _scale_len(self, value, dim):
        if dim == 0:
            return int(value)
        return int(math.floor(float(value) * float(self.sr) + 1e-6))

    @staticmethod
    def _get_num_blocks(dataset):
        max_idx = [0, 0, 0]
        for block_idx, _ in dataset.block_indices:
            for dim in range(3):
                max_idx[dim] = max(max_idx[dim], int(block_idx[dim]))
        return tuple(v + 1 for v in max_idx)

    def _precompute_weight_windows(self):
        weights = []

        for dim in range(3):
            size = self.block_size[dim]
            overlap = min(self.overlap[dim], max(0, size // 2))

            w = torch.ones(size, device=self.device)
            if overlap > 0:
                ramp_left = torch.linspace(0, 1, overlap + 2, device=self.device)[1:-1]
                w[:overlap] = ramp_left

                ramp_right = torch.linspace(1, 0, overlap + 2, device=self.device)[1:-1]
                w[-overlap:] = ramp_right

            weights.append(w)

        self.weight_window = (
            weights[0][:, None, None]
            * weights[1][None, :, None]
            * weights[2][None, None, :]
        )

    def add_block(self, block_idx: tuple[int, int, int], block_data: torch.Tensor):
        """
        Add one model output block.

        Args:
            block_idx: block index (a, b, c)
            block_data: model output tensor with shape (D, H, W)
        """
        if block_data.dim() == 2:
            block_data = block_data.unsqueeze(0)
        elif block_data.dim() != 3:
            raise ValueError(f'block_data must be 2D or 3D, got shape {tuple(block_data.shape)}')

        a, b, c = (int(v) for v in block_idx)
        starts = (
            a * self.stride[0] + (self.crop_before[0] if self.is_iso else 0),
            b * self.stride[1] + self.crop_before[1],
            c * self.stride[2] + self.crop_before[2],
        )
        ends = tuple(start + size for start, size in zip(starts, block_data.shape))

        block_data = block_data.to(self.device)
        weight_window = self._weight_for_block(block_data.shape)
        weighted_block = block_data * weight_window

        d_start, h_start, w_start = starts
        d_end, h_end, w_end = ends

        self.result[d_start:d_end, h_start:h_end, w_start:w_end] += weighted_block
        self.weights[d_start:d_end, h_start:h_end, w_start:w_end] += weight_window

    def _weight_for_block(self, shape):
        if tuple(shape) == self.block_size:
            return self.weight_window

        if any(actual > expected for actual, expected in zip(shape, self.block_size)):
            raise ValueError(
                f'block_data shape {tuple(shape)} is larger than expected block size {self.block_size}'
            )

        return self.weight_window[:shape[0], :shape[1], :shape[2]]

    def get_final_result(self, clip: bool = True) -> torch.Tensor:
        weights = torch.where(
            self.weights == 0,
            torch.ones_like(self.weights),
            self.weights,
        )
        final_result = self.result / weights

        if clip:
            d_pad, h_pad, w_pad = self.padding

            d_start = d_pad[0]
            d_end = final_result.shape[0] - d_pad[1]

            h_start = h_pad[0]
            h_end = final_result.shape[1] - h_pad[1]

            w_start = w_pad[0]
            w_end = final_result.shape[2] - w_pad[1]

            final_result = final_result[d_start:d_end, h_start:h_end, w_start:w_end]

        return final_result
