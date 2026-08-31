import argparse
import cv2
import numpy as np
import pandas as pd
import tifffile as tiff
from pathlib import Path
import re


root_dir = None
gt_folder_name = "GT"

output_excel = None

pixel_size_nm = 65
pixel_size_um = pixel_size_nm / 1000
num_circles_per_image = 5
min_radius = 0
max_radius = 5
hough_param1 = 40
hough_param2 = 6
min_dist = 8
min_ring_contrast = 8
max_center_peak_ratio = 0.88

edge_margin = 25

save_check_images = True
check_dir_name = "_ring_check_same_position"


def natural_key(path):
    text = str(path)
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", text)]


def safe_sheet_name(name, used_names):
    name = re.sub(r'[\[\]\:\*\?\/\\]', "_", name)
    name = name[:31]

    if name == "":
        name = "Sheet"

    base = name
    idx = 1

    while name in used_names:
        suffix = f"_{idx}"
        name = base[:31 - len(suffix)] + suffix
        idx += 1

    used_names.add(name)
    return name


def read_tif_as_2d(img_path):
    img = tiff.imread(str(img_path))
    img = np.asarray(img)
    img = np.squeeze(img)

    if img.ndim == 2:
        return img

    if img.ndim == 3 and img.shape[-1] in [3, 4]:
        img = img[..., :3]
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        return img

    if img.ndim == 3:
        img = np.max(img, axis=0)
        return img

    raise ValueError(f"Unsupported image shape: {img.shape}")


def normalize_to_uint8(img):
    img = img.astype(np.float32)

    low = np.percentile(img, 1)
    high = np.percentile(img, 99.8)

    if high <= low:
        low = img.min()
        high = img.max()

    if high <= low:
        return np.zeros_like(img, dtype=np.uint8)

    img = (img - low) / (high - low)
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)

    return img


def get_radial_profile(image, center_x, center_y, radius_max):
    h, w = image.shape

    x0 = max(0, center_x - radius_max - 2)
    x1 = min(w, center_x + radius_max + 3)
    y0 = max(0, center_y - radius_max - 2)
    y1 = min(h, center_y + radius_max + 3)

    patch = image[y0:y1, x0:x1]

    yy, xx = np.indices(patch.shape)

    rr = np.sqrt(
        (xx + x0 - center_x) ** 2 +
        (yy + y0 - center_y) ** 2
    )

    rr_int = rr.astype(np.int32)
    valid = rr_int <= radius_max

    sums = np.bincount(
        rr_int[valid].ravel(),
        weights=patch[valid].ravel(),
        minlength=radius_max + 1
    )

    counts = np.bincount(
        rr_int[valid].ravel(),
        minlength=radius_max + 1
    )

    profile = sums / np.maximum(counts, 1)

    return profile


def smooth_profile(profile):
    kernel = np.array([1, 2, 3, 2, 1], dtype=np.float32)
    kernel = kernel / kernel.sum()
    return np.convolve(profile, kernel, mode="same")


def get_candidate_centers_by_components(gray):

    blur = cv2.GaussianBlur(gray, (3, 3), 0.8)

    nonzero = blur[blur > 0]

    if nonzero.size == 0:
        return []

    threshold = max(15, np.percentile(nonzero, 45))

    mask = (blur >= threshold).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)

    candidates = []

    for label_idx in range(1, num_labels):
        x, y, w, h, area = stats[label_idx]

        if area < 6:
            continue

        if area > 900:
            continue

        if w < 3 or h < 3:
            continue

        if w > 2 * max_radius + 10:
            continue

        if h > 2 * max_radius + 10:
            continue

        elongation = max(w, h) / max(1, min(w, h))

        if elongation > 4.0:
            continue

        center_x, center_y = centroids[label_idx]

        candidates.append({
            "center_x": float(center_x),
            "center_y": float(center_y),
            "seed_score": float(area)
        })

    return candidates


def merge_candidate_centers(candidates, merge_dist=5):

    candidates = sorted(
        candidates,
        key=lambda x: x.get("seed_score", 0),
        reverse=True
    )

    merged = []

    for candidate in candidates:
        center_x = candidate["center_x"]
        center_y = candidate["center_y"]

        keep = True

        for old in merged:
            old_x = old["center_x"]
            old_y = old["center_y"]

            dist2 = (center_x - old_x) ** 2 + (center_y - old_y) ** 2

            if dist2 < merge_dist ** 2:
                keep = False
                break

        if keep:
            merged.append(candidate)

    return merged


def measure_inner_diameter_at_position(img, center_x, center_y):

    gray = normalize_to_uint8(img)

    h, w = gray.shape

    if center_x < edge_margin:
        return np.nan, None

    if center_x > w - edge_margin:
        return np.nan, None

    if center_y < edge_margin:
        return np.nan, None

    if center_y > h - edge_margin:
        return np.nan, None

    radius_max = max_radius + 8

    profile = get_radial_profile(
        gray,
        int(round(center_x)),
        int(round(center_y)),
        radius_max
    )

    profile_smooth = smooth_profile(profile)

    search_start = min_radius
    search_end = min(len(profile_smooth), max_radius + 1)

    if search_end <= search_start + 2:
        return np.nan, None

    ring_radius = np.argmax(profile_smooth[search_start:search_end]) + search_start

    center_region_end = max(2, ring_radius // 3)
    center_level = np.mean(profile_smooth[:center_region_end])
    peak_level = profile_smooth[ring_radius]

    ring_contrast = peak_level - center_level
    center_peak_ratio = center_level / (peak_level + 1e-6)

    if ring_contrast < min_ring_contrast:
        return np.nan, None

    if center_peak_ratio > max_center_peak_ratio:
        return np.nan, None

    threshold = center_level + 0.5 * (peak_level - center_level)

    candidates = np.where(profile_smooth[:ring_radius + 1] >= threshold)[0]

    if len(candidates) == 0:
        return np.nan, None

    inner_radius_px = int(candidates[0])

    if inner_radius_px < 1:
        return np.nan, None

    if inner_radius_px > 0.90 * ring_radius:
        return np.nan, None

    inner_diameter_px = 2 * inner_radius_px
    inner_diameter_um = inner_diameter_px * pixel_size_um

    info = {
        "inner_radius_px": inner_radius_px,
        "ring_radius_px": ring_radius,
        "inner_diameter_um": inner_diameter_um,
        "ring_contrast": ring_contrast,
        "center_peak_ratio": center_peak_ratio
    }

    return inner_diameter_um, info


def detect_five_ring_positions_from_gt(gt_img):

    gray = normalize_to_uint8(gt_img)
    blur = cv2.GaussianBlur(gray, (5, 5), 1.2)

    all_candidates = []

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_dist,
        param1=hough_param1,
        param2=hough_param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    if circles is not None:
        circles = np.round(circles[0]).astype(int)

        for center_x, center_y, radius in circles:
            all_candidates.append({
                "center_x": float(center_x),
                "center_y": float(center_y),
                "seed_score": float(radius) * 10
            })

    component_candidates = get_candidate_centers_by_components(gray)
    all_candidates.extend(component_candidates)

    all_candidates = merge_candidate_centers(
        all_candidates,
        merge_dist=5
    )

    valid_candidates = []

    for candidate in all_candidates:
        center_x = candidate["center_x"]
        center_y = candidate["center_y"]

        diameter_um, info = measure_inner_diameter_at_position(
            gt_img,
            center_x,
            center_y
        )

        if np.isnan(diameter_um):
            continue

        ring_score = info["ring_contrast"] * (1 - info["center_peak_ratio"])

        valid_candidates.append({
            "center_x": center_x,
            "center_y": center_y,
            "gt_inner_diameter_um": diameter_um,
            "score": ring_score,
            "ring_radius_px": info["ring_radius_px"],
            "inner_radius_px": info["inner_radius_px"]
        })

    valid_candidates = sorted(
        valid_candidates,
        key=lambda x: x["score"],
        reverse=True
    )

    valid_candidates = valid_candidates[:num_circles_per_image]

    valid_candidates = sorted(
        valid_candidates,
        key=lambda x: (x["center_y"], x["center_x"])
    )

    return valid_candidates, gray


def draw_check_image(img, positions, measured_infos=None):
    gray = normalize_to_uint8(img)
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    for idx, pos in enumerate(positions, start=1):
        center_x = int(round(pos["center_x"]))
        center_y = int(round(pos["center_y"]))

        cv2.circle(vis, (center_x, center_y), 2, (255, 0, 0), -1)

        if measured_infos is not None:
            info = measured_infos[idx - 1]

            if info is not None:
                inner_radius = int(info["inner_radius_px"])
                ring_radius = int(info["ring_radius_px"])

                cv2.circle(vis, (center_x, center_y), inner_radius, (0, 0, 255), 1)

                cv2.circle(vis, (center_x, center_y), ring_radius, (0, 255, 0), 1)

        cv2.putText(
            vis,
            str(idx),
            (center_x + 4, center_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )

    return vis


def get_tif_list(folder):

    tif_paths = list(folder.glob("*.tif")) + list(folder.glob("*.tiff"))
    tif_paths = sorted(tif_paths, key=natural_key)
    return tif_paths



parser = argparse.ArgumentParser()
parser.add_argument("root_dir")
parser.add_argument("--gt-folder", default="GT")
parser.add_argument("--output")
args = parser.parse_args()
root_dir = Path(args.root_dir).resolve()
gt_folder_name = args.gt_folder
output_excel = Path(args.output).resolve() if args.output else root_dir / "ring_diameter_same_position.xlsx"

gt_folder = root_dir / gt_folder_name

method_folders = [
    p for p in root_dir.iterdir()
    if p.is_dir() and p.name != check_dir_name
]

method_folders = sorted(method_folders, key=natural_key)
method_folders = [gt_folder] + [p for p in method_folders if p != gt_folder]

gt_tif_list = get_tif_list(gt_folder)


method_tif_lists = {
    folder.name: get_tif_list(folder)
    for folder in method_folders
}

gt_names = {path.name for path in gt_tif_list}
for method_name, paths in method_tif_lists.items():
    method_names = {path.name for path in paths}
    if method_names != gt_names:
        missing = sorted(gt_names - method_names)
        extra = sorted(method_names - gt_names)
        raise ValueError(f"File mismatch for {method_name}: missing={missing}, extra={extra}")


used_sheet_names = set()
all_sheet_tables = {}
summary_data = {}

if save_check_images:
    check_root = root_dir / check_dir_name
    check_root.mkdir(exist_ok=True)

for method_folder in method_folders:
    method_name = method_folder.name
    sheet_name = safe_sheet_name(method_name, used_sheet_names)

    rows = []
    summary_series = {}

    method_tif_list = method_tif_lists[method_name]

    if save_check_images:
        method_check_dir = check_root / method_name
        method_check_dir.mkdir(exist_ok=True)

    for image_idx, gt_path in enumerate(gt_tif_list):
        gt_image_name = gt_path.name

        method_path = next(path for path in method_tif_list if path.name == gt_image_name)
        method_image_name = method_path.name

        gt_img = read_tif_as_2d(gt_path)
        method_img = read_tif_as_2d(method_path)

        gt_positions, _ = detect_five_ring_positions_from_gt(gt_img)

        diameters_um = []
        measured_infos = []

        for pos in gt_positions:
            center_x = pos["center_x"]
            center_y = pos["center_y"]

            if method_img.shape != gt_img.shape:
                scale_y = method_img.shape[0] / gt_img.shape[0]
                scale_x = method_img.shape[1] / gt_img.shape[1]
                center_x = center_x * scale_x
                center_y = center_y * scale_y

            diameter_um, info = measure_inner_diameter_at_position(
                method_img,
                center_x,
                center_y
            )

            diameters_um.append(diameter_um)
            measured_infos.append(info)

        while len(diameters_um) < num_circles_per_image:
            diameters_um.append(np.nan)
            measured_infos.append(None)

        diameters_um = diameters_um[:num_circles_per_image]
        measured_infos = measured_infos[:num_circles_per_image]

        valid_count = np.sum(~np.isnan(diameters_um))
        mean_um = np.nanmean(diameters_um) if valid_count > 0 else np.nan

        row = {
            "Index": image_idx,
            "GT_Image": gt_image_name,
            "Method_Image": method_image_name
        }

        for i in range(num_circles_per_image):
            row[f"Circle_{i + 1}_um"] = diameters_um[i]

        row["Mean_um"] = mean_um
        row["Valid_count"] = valid_count

        rows.append(row)
        summary_series[gt_image_name] = mean_um

        if save_check_images:
            draw_positions = []

            for pos in gt_positions:
                draw_pos = dict(pos)

                if method_img.shape != gt_img.shape:
                    scale_y = method_img.shape[0] / gt_img.shape[0]
                    scale_x = method_img.shape[1] / gt_img.shape[1]
                    draw_pos["center_x"] = pos["center_x"] * scale_x
                    draw_pos["center_y"] = pos["center_y"] * scale_y

                draw_positions.append(draw_pos)

            vis = draw_check_image(method_img, draw_positions, measured_infos)

            save_name = (
                f"{image_idx:06d}_"
                f"{Path(gt_image_name).stem}_to_"
                f"{Path(method_image_name).stem}_check.png"
            )

            save_path = method_check_dir / save_name
            cv2.imwrite(str(save_path), vis)



    method_df = pd.DataFrame(rows)
    all_sheet_tables[sheet_name] = method_df
    summary_data[method_name] = pd.Series(summary_series)



summary_df = pd.DataFrame(summary_data)
summary_df.index.name = "GT_Image"

sorted_index = sorted(summary_df.index, key=natural_key)
summary_df = summary_df.loc[sorted_index]



with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
    summary_df.to_excel(writer, sheet_name="Summary")

    for sheet_name, df in all_sheet_tables.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)

