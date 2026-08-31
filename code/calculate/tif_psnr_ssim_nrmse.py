import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def list_tiffs(folder):
    if not folder.is_dir():
        raise NotADirectoryError(f"Folder not found: {folder}")
    files = {
        path.name: path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    }
    if not files:
        raise FileNotFoundError(f"No TIFF files found: {folder}")
    return files


def read_image(path):
    image = np.squeeze(tifffile.imread(path))
    if image.ndim not in {2, 3}:
        raise ValueError(f"Expected a 2D image or 3D volume: {path}, shape={image.shape}")
    if not np.all(np.isfinite(image)):
        raise ValueError(f"Image contains NaN or infinity: {path}")
    return image


def normalize_pair(gt, output):
    if gt.dtype != output.dtype:
        raise TypeError(f"Dtype mismatch: GT={gt.dtype}, output={output.dtype}")
    if np.issubdtype(gt.dtype, np.integer):
        info = np.iinfo(gt.dtype)
        scale = float(info.max - info.min)
        gt = (gt.astype(np.float64) - info.min) / scale
        output = (output.astype(np.float64) - info.min) / scale
        return gt, output, 1.0
    if np.issubdtype(gt.dtype, np.floating):
        gt = gt.astype(np.float64)
        output = output.astype(np.float64)
        data_range = float(gt.max() - gt.min())
        if data_range <= 0:
            raise ValueError("GT has zero dynamic range")
        return gt, output, data_range
    raise TypeError(f"Unsupported dtype: {gt.dtype}")


def ssim(gt, output, data_range):
    min_size = min(gt.shape)
    if min_size < 3:
        raise ValueError(f"Image is too small for SSIM: shape={gt.shape}")
    window = 7 if min_size >= 7 else min_size if min_size % 2 else min_size - 1
    return structural_similarity(gt, output, data_range=data_range, win_size=window)


def calculate(gt_path, output_path):
    gt = read_image(gt_path)
    output = read_image(output_path)
    if gt.shape != output.shape:
        raise ValueError(
            f"Shape mismatch for {gt_path.name}: GT={gt.shape}, output={output.shape}"
        )
    gt, output, data_range = normalize_pair(gt, output)
    rmse = float(np.sqrt(np.mean((output - gt) ** 2)))
    return {
        "file": gt_path.name,
        "psnr": float(peak_signal_noise_ratio(gt, output, data_range=data_range)),
        "ssim": float(ssim(gt, output, data_range)),
        "nrmse": rmse / data_range,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gt_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--save", default="metrics.xlsx")
    args = parser.parse_args()

    gt_files = list_tiffs(Path(args.gt_dir).resolve())
    output_files = list_tiffs(Path(args.output_dir).resolve())
    if set(gt_files) != set(output_files):
        missing = sorted(set(gt_files) - set(output_files))
        extra = sorted(set(output_files) - set(gt_files))
        raise ValueError(f"File mismatch: missing={missing}, extra={extra}")

    rows = [calculate(gt_files[name], output_files[name]) for name in sorted(gt_files)]
    dataframe = pd.DataFrame(rows)
    save_path = Path(args.save).resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_excel(save_path, index=False)
    print(dataframe.to_string(index=False))
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
