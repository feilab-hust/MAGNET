import argparse
import os
import numpy as np
import pandas as pd
import tifffile
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects



parser = argparse.ArgumentParser()
parser.add_argument("root_dir")
parser.add_argument("--input-name", default="Input.tif")
parser.add_argument("--output")
args = parser.parse_args()
root_dir = os.path.abspath(args.root_dir)

input_name = args.input_name
output_excel = os.path.abspath(args.output or os.path.join(root_dir, "snr_results.xlsx"))

mask_dir = os.path.join(root_dir, "mask")
os.makedirs(mask_dir, exist_ok=True)


use_existing_mask_for_input_bg = True
remove_small = True
min_size = 20
bg_value_mode = "max"
bg_percentile = 99.5

metric_mode = "snr"
norm_mode = "minmax"

p_low = 0.1
p_high = 99.9


def read_tif(path):
    img = tifffile.imread(path)
    img = np.squeeze(img)
    return img


def normalize_image(img):
    img = img.astype(np.float32)

    finite = np.isfinite(img)
    if not np.any(finite):
        return np.zeros_like(img, dtype=np.float32)

    if norm_mode == "minmax":
        low = np.min(img[finite])
        high = np.max(img[finite])
    elif norm_mode == "percentile":
        low, high = np.percentile(img[finite], [p_low, p_high])
    else:
        raise ValueError("norm_mode must be 'minmax' or 'percentile'")

    if high <= low:
        return np.zeros_like(img, dtype=np.float32)

    img_norm = (img - low) / (high - low)
    img_norm = np.clip(img_norm, 0, 1)
    return img_norm


def find_existing_mask(root_dir, mask_dir):
    search_dirs = [root_dir, mask_dir]

    for cur_dir in search_dirs:
        if not os.path.exists(cur_dir):
            continue

        mask_files = sorted([
            f for f in os.listdir(cur_dir)
            if f.lower().endswith((".tif", ".tiff")) and "mask" in f.lower()
        ])

        if len(mask_files) > 0:
            return os.path.join(cur_dir, mask_files[0])

    return None


def save_mask(mask, save_path):
    tifffile.imwrite(save_path, (mask.astype(np.uint8) * 255))


input_path = os.path.join(root_dir, input_name)



input_img = read_tif(input_path)
input_norm = normalize_image(input_img)

input_bg_mask = None
used_mask_path = None

if use_existing_mask_for_input_bg:
    used_mask_path = find_existing_mask(root_dir, mask_dir)

if used_mask_path is not None:
    existing_mask = read_tif(used_mask_path)
    input_signal_mask = existing_mask > 0
    input_bg_mask = ~input_signal_mask


else:
    otsu_threshold = threshold_otsu(input_norm)
    input_signal_mask = input_norm > otsu_threshold

    if remove_small:
        input_signal_mask = remove_small_objects(input_signal_mask.astype(bool), min_size=min_size)

    input_bg_mask = ~input_signal_mask

    otsu_mask_path = os.path.join(mask_dir, "Input_initial_otsu_mask.tif")
    save_mask(input_signal_mask, otsu_mask_path)
    print(f"Otsu threshold = {otsu_threshold:.6f}")


input_bg_pixels = input_norm[input_bg_mask]

if bg_value_mode == "max":
    input_bg_value = np.max(input_bg_pixels)
elif bg_value_mode == "percentile":
    input_bg_value = np.percentile(input_bg_pixels, bg_percentile)


input_max_value = np.max(input_norm)
threshold_ratio = input_bg_value / input_max_value

tif_files = sorted([
    f for f in os.listdir(root_dir)
    if f.lower().endswith((".tif", ".tiff")) and "mask" not in f.lower()
])

results = []
eps = 1e-8

for file_name in tif_files:
    img_path = os.path.join(root_dir, file_name)
    img = read_tif(img_path)
    img_norm = normalize_image(img)

    img_max_value = np.max(img_norm)
    threshold = threshold_ratio * img_max_value

    signal_mask = img_norm > threshold
    nonsignal_mask = ~signal_mask

    mask_save_name = os.path.splitext(file_name)[0] + "_mask.tif"
    mask_save_path = os.path.join(mask_dir, mask_save_name)
    save_mask(signal_mask, mask_save_path)

    signal_pixels = img_norm[signal_mask]
    nonsignal_pixels = img_norm[nonsignal_mask]

    signal_mean = np.mean(signal_pixels)
    nonsignal_mean = np.mean(nonsignal_pixels)
    nonsignal_std = np.std(nonsignal_pixels)

    snr = signal_mean / (nonsignal_std + eps)
    sbr = signal_mean / (nonsignal_mean + eps)

    row = {
        "file": file_name,
        "threshold_ratio": threshold_ratio,
        "threshold": threshold,
        "signal_pixel_count": int(np.sum(signal_mask)),
        "nonsignal_pixel_count": int(np.sum(nonsignal_mask)),
    }

    if metric_mode == "sbr":
        row["SBR"] = sbr

    elif metric_mode == "snr":
        row["SNR"] = snr

    elif metric_mode == "both":
        row["SNR"] = snr
        row["SBR"] = sbr

    else:
        raise ValueError("metric_mode must be 'sbr', 'snr', or 'both'")

    results.append(row)

df = pd.DataFrame(results)
df.to_excel(output_excel, index=False)
