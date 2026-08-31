import argparse
import os
import re

import numpy as np
import pandas as pd
import tifffile as tiff
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.draw import line as draw_line
from skimage.transform import resize
from skimage.morphology import (
    binary_closing,
    binary_opening,
    disk,
    remove_small_objects,
    skeletonize,
)


parser = argparse.ArgumentParser()
parser.add_argument("root_dir")
parser.add_argument("--output-dir")
args = parser.parse_args()
root_dir = os.path.abspath(args.root_dir)

folder_names = {
    "target": "target",
    "input": "input",
    "output": "output",
}

output_dir = os.path.abspath(args.output_dir or os.path.join(root_dir, "line_count_results"))
overlay_root = os.path.join(output_dir, "line_count_overlays")
os.makedirs(overlay_root, exist_ok=True)

output_excel = os.path.join(output_dir, "line_counts.xlsx")
output_csv = os.path.join(output_dir, "line_counts.csv")

match_mode = "same_name"
threshold_scale = 0.62
min_area = 12
branch_cut_radius = 2
min_line_length_px = 12
min_line_skeleton_pixels = 7
max_local_turn_degrees = 50
turn_window_pixels = 5
min_split_gap_pixels = 8
merge_endpoint_distance_px = 18
merge_angle_degrees = 32
merge_max_iterations = 4
folder_params = {
    "target": {
        "threshold_scale": 0.62,
        "merge_endpoint_distance_px": 18,
        "merge_angle_degrees": 32,
        "max_cumulative_turn_degrees": 120,
        "min_line_length_px": 12,
        "min_line_skeleton_pixels": 7,
        "min_ridge_alignment": 0.0,
        "alignment_filter_max_length_px": 0,
    },
    "input": {
        "threshold_scale": 0.5,
        "merge_endpoint_distance_px": 12,
        "merge_angle_degrees": 32,
        "max_cumulative_turn_degrees": 120,
        "min_line_length_px": 10,
        "min_line_skeleton_pixels": 6,
        "min_ridge_alignment": 0.25,
        "alignment_filter_max_length_px": 12,
    },
    "output": {
        "threshold_scale": 0.62,
        "merge_endpoint_distance_px": 18,
        "merge_angle_degrees": 32,
        "max_cumulative_turn_degrees": 120,
        "min_line_length_px": 12,
        "min_line_skeleton_pixels": 7,
        "min_ridge_alignment": 0.0,
        "alignment_filter_max_length_px": 0,
    },
}

overlay_output_size = 256
label_font_size = 8


def natural_key(name):
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", str(name))]


def list_images(folder):
    names = [
        f
        for f in os.listdir(folder)
        if f.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg"))
    ]
    names.sort(key=natural_key)
    return names


def read_image(path):
    if path.lower().endswith((".tif", ".tiff")):
        img = tiff.imread(path)
    else:
        img = np.asarray(Image.open(path))

    img = np.asarray(img)
    img = np.squeeze(img)

    if img.ndim == 3:
        if img.shape[-1] in [3, 4]:
            img = img[..., :3]
            img = (
                0.299 * img[..., 0].astype(np.float32)
                + 0.587 * img[..., 1].astype(np.float32)
                + 0.114 * img[..., 2].astype(np.float32)
            )
        else:
            img = np.max(img, axis=0)


    return img


def normalize_image(img):
    img = img.astype(np.float32)
    low = np.percentile(img, 1)
    high = np.percentile(img, 99.8)

    if high <= low:
        low = img.min()
        high = img.max()

    if high <= low:
        return np.zeros_like(img, dtype=np.float32)

    img = (img - low) / (high - low)
    return np.clip(img, 0, 1)


def make_line_mask(img_norm, threshold_value):
    blur = ndi.gaussian_filter(img_norm.astype(np.float32), sigma=0.8)

    if np.max(blur) <= np.min(blur):
        return np.zeros_like(img_norm, dtype=bool)

    th = threshold_otsu(blur)
    mask = blur > th * threshold_value
    mask = binary_opening(mask, disk(1))
    mask = binary_closing(mask, disk(1))
    mask = remove_small_objects(mask.astype(bool), min_size=min_area)
    return mask


def count_skeleton_neighbors(skel):
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_count = ndi.convolve(skel.astype(np.uint8), kernel, mode="constant", cval=0)
    neighbor_count = neighbor_count - skel.astype(np.uint8)
    endpoints = skel & (neighbor_count == 1)
    branch_points = skel & (neighbor_count >= 3)
    return endpoints, branch_points


def skeleton_length(skel):
    coords = np.column_stack(np.where(skel))
    if coords.shape[0] < 2:
        return 0.0

    coord_set = set((int(y), int(x)) for y, x in coords)
    edges = set()

    for y, x in coord_set:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                yy = y + dy
                xx = x + dx
                if (yy, xx) in coord_set:
                    edges.add(tuple(sorted([(y, x), (yy, xx)])))

    length = 0.0
    for p1, p2 in edges:
        y1, x1 = p1
        y2, x2 = p2
        length += float(np.hypot(y2 - y1, x2 - x1))
    return length


def neighbors_of(point, coord_set):
    y, x = point
    out = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            nb = (y + dy, x + dx)
            if nb in coord_set:
                out.append(nb)
    return out


def order_skeleton_path(skel):
    coords = [tuple(map(int, p)) for p in np.column_stack(np.where(skel))]
    if len(coords) == 0:
        return []
    if len(coords) == 1:
        return coords

    coord_set = set(coords)
    endpoints, _ = count_skeleton_neighbors(skel)
    endpoint_coords = [tuple(map(int, p)) for p in np.column_stack(np.where(endpoints))]

    if endpoint_coords:
        start = endpoint_coords[0]
    else:
        center = np.asarray(coords, dtype=np.float32).mean(axis=0)
        start = coords[int(np.argmax(np.sum((np.asarray(coords) - center) ** 2, axis=1)))]

    path = [start]
    visited = {start}
    current = start
    previous = None

    while True:
        candidates = [p for p in neighbors_of(current, coord_set) if p not in visited]
        if not candidates:
            break

        if previous is None:
            nxt = candidates[0]
        else:
            direction = np.asarray(current, dtype=np.float32) - np.asarray(previous, dtype=np.float32)

            def turn_cost(p):
                step = np.asarray(p, dtype=np.float32) - np.asarray(current, dtype=np.float32)
                denom = (np.linalg.norm(direction) * np.linalg.norm(step)) + 1e-8
                return -float(np.dot(direction, step) / denom)

            nxt = sorted(candidates, key=turn_cost)[0]

        path.append(nxt)
        visited.add(nxt)
        previous, current = current, nxt

    return path


def path_to_mask(path, shape):
    mask = np.zeros(shape, dtype=bool)
    if not path:
        return mask
    yy = [p[0] for p in path]
    xx = [p[1] for p in path]
    mask[yy, xx] = True
    return mask


def bridge_between_points(p0, p1):
    rr, cc = draw_line(int(p0[0]), int(p0[1]), int(p1[0]), int(p1[1]))
    return list(zip(rr.tolist(), cc.tolist()))


def local_turn_angle(path, i, window):
    if i - window < 0 or i + window >= len(path):
        return 0.0

    p0 = np.asarray(path[i - window], dtype=np.float32)
    p1 = np.asarray(path[i], dtype=np.float32)
    p2 = np.asarray(path[i + window], dtype=np.float32)
    v1 = p1 - p0
    v2 = p2 - p1
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-8
    cosang = float(np.clip(np.dot(v1, v2) / denom, -1, 1))
    return float(np.degrees(np.arccos(cosang)))


def step_turn_angle(path, i):
    if i <= 0 or i >= len(path) - 1:
        return 0.0
    p0 = np.asarray(path[i - 1], dtype=np.float32)
    p1 = np.asarray(path[i], dtype=np.float32)
    p2 = np.asarray(path[i + 1], dtype=np.float32)
    v1 = p1 - p0
    v2 = p2 - p1
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-8
    cosang = float(np.clip(np.dot(v1, v2) / denom, -1, 1))
    return float(np.degrees(np.arccos(cosang)))


def split_path_at_hard_turns(path, max_cumulative_turn):
    if len(path) < min_line_skeleton_pixels:
        return []

    split_indices = []
    last_split = -min_split_gap_pixels
    cumulative_turn = 0.0
    for i in range(turn_window_pixels, len(path) - turn_window_pixels):
        angle = local_turn_angle(path, i, turn_window_pixels)
        cumulative_turn += step_turn_angle(path, i)
        if (
            (angle > max_local_turn_degrees or cumulative_turn > max_cumulative_turn)
            and i - last_split >= min_split_gap_pixels
        ):
            split_indices.append(i)
            last_split = i
            cumulative_turn = 0.0

    pieces = []
    start = 0
    for idx in split_indices:
        pieces.append(path[start:idx])
        start = idx
    pieces.append(path[start:])
    return [p for p in pieces if len(p) >= min_line_skeleton_pixels]


def endpoint_tangent(path, at_start):
    if len(path) < 3:
        return None
    if at_start:
        p0 = np.asarray(path[0], dtype=np.float32)
        p1 = np.asarray(path[min(turn_window_pixels, len(path) - 1)], dtype=np.float32)
    else:
        p0 = np.asarray(path[-1], dtype=np.float32)
        p1 = np.asarray(path[-1 - min(turn_window_pixels, len(path) - 1)], dtype=np.float32)
    vec = p1 - p0
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8:
        return None
    return vec / norm


def ridge_alignment_score(path, grad_y, grad_x):
    if len(path) < 3:
        return 0.0

    scores = []
    h, w = grad_y.shape
    for i in range(1, len(path) - 1):
        p0 = np.asarray(path[i - 1], dtype=np.float32)
        p2 = np.asarray(path[i + 1], dtype=np.float32)
        tangent = p2 - p0
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm <= 1e-8:
            continue
        tangent = tangent / tangent_norm

        y, x = path[i]
        if y < 0 or y >= h or x < 0 or x >= w:
            continue
        grad = np.asarray([grad_y[y, x], grad_x[y, x]], dtype=np.float32)
        grad_norm = float(np.linalg.norm(grad))
        if grad_norm <= 1e-5:
            continue
        grad = grad / grad_norm

        scores.append(1.0 - abs(float(np.dot(tangent, grad))))

    if not scores:
        return 1.0
    return float(np.median(scores))


def can_merge_paths(path_a, path_b, merge_distance, merge_angle):
    endpoints = [
        (path_a[0], True, path_b[0], True),
        (path_a[0], True, path_b[-1], False),
        (path_a[-1], False, path_b[0], True),
        (path_a[-1], False, path_b[-1], False),
    ]

    best = None
    best_distance = np.inf
    for pa, a_start, pb, b_start in endpoints:
        dist = float(np.linalg.norm(np.asarray(pa, dtype=np.float32) - np.asarray(pb, dtype=np.float32)))
        if dist < best_distance:
            best_distance = dist
            best = (pa, a_start, pb, b_start)

    if best is None or best_distance > merge_distance:
        return None

    pa, a_start, pb, b_start = best
    ta = endpoint_tangent(path_a, a_start)
    tb = endpoint_tangent(path_b, b_start)
    if ta is None or tb is None:
        return None

    collinearity = abs(float(np.dot(ta, tb)))
    if collinearity < np.cos(np.deg2rad(merge_angle)):
        return None

    if a_start:
        a = list(reversed(path_a))
    else:
        a = list(path_a)

    if b_start:
        b = list(path_b)
    else:
        b = list(reversed(path_b))

    bridge = bridge_between_points(a[-1], b[0])
    merged = a + bridge[1:-1] + b
    if any(local_turn_angle(merged, i, turn_window_pixels) > max_local_turn_degrees for i in range(turn_window_pixels, len(merged) - turn_window_pixels)):
        return None
    return merged


def merge_collinear_paths(paths, merge_distance, merge_angle):
    paths = [list(p) for p in paths]
    for _ in range(merge_max_iterations):
        used = [False] * len(paths)
        merged_any = False
        new_paths = []

        for i, path in enumerate(paths):
            if used[i]:
                continue
            current = path
            used[i] = True

            changed = True
            while changed:
                changed = False
                for j, other in enumerate(paths):
                    if used[j]:
                        continue
                    merged = can_merge_paths(current, other, merge_distance, merge_angle)
                    if merged is None:
                        continue
                    current = merged
                    used[j] = True
                    changed = True
                    merged_any = True
                    break

            new_paths.append(current)

        paths = new_paths
        if not merged_any:
            break
    return paths


def segment_lines(img_norm, params):
    mask = make_line_mask(img_norm, params["threshold_scale"])
    skel = skeletonize(mask)
    grad_y = ndi.sobel(img_norm.astype(np.float32), axis=0)
    grad_x = ndi.sobel(img_norm.astype(np.float32), axis=1)
    _, branch_points = count_skeleton_neighbors(skel)
    cut_points = ndi.binary_dilation(branch_points, iterations=branch_cut_radius)
    segment_seed = skel & ~cut_points

    raw_paths = []
    labeled = label(segment_seed, connectivity=2)
    for prop in regionprops(labeled):
        comp = labeled == prop.label
        path = order_skeleton_path(comp)
        raw_paths.extend(split_path_at_hard_turns(path, params["max_cumulative_turn_degrees"]))

    merged_paths = merge_collinear_paths(
        raw_paths,
        params["merge_endpoint_distance_px"],
        params["merge_angle_degrees"],
    )
    line_masks = []
    for path in merged_paths:
        line_mask = path_to_mask(path, img_norm.shape)
        if int(line_mask.sum()) < params["min_line_skeleton_pixels"]:
            continue
        length_px = skeleton_length(line_mask)
        if length_px < params["min_line_length_px"]:
            continue
        if (
            params.get("alignment_filter_max_length_px", 0) > 0
            and length_px <= params["alignment_filter_max_length_px"]
            and ridge_alignment_score(path, grad_y, grad_x) < params["min_ridge_alignment"]
        ):
            continue
        line_masks.append(line_mask)

    line_masks = sorted(line_masks, key=skeleton_length, reverse=True)
    return line_masks, mask, skel


def line_colors():
    return [
        (0, 0, 255),
        (0, 255, 255),
        (255, 0, 255),
        (0, 255, 0),
        (255, 128, 0),
        (255, 0, 0),
        (0, 128, 255),
        (128, 0, 255),
        (128, 255, 0),
        (255, 255, 0),
    ]


def save_line_overlay(img_norm, line_masks, save_path):
    if overlay_output_size is not None:
        display_shape = (overlay_output_size, overlay_output_size)
        img_display = resize(
            img_norm,
            display_shape,
            order=1,
            preserve_range=True,
            anti_aliasing=True,
        )
    else:
        display_shape = img_norm.shape
        img_display = img_norm

    img8 = (img_display * 255).astype(np.uint8)
    rgb = np.stack([img8, img8, img8], axis=-1)
    colors = line_colors()
    label_positions = []

    for idx, line_mask in enumerate(line_masks, start=1):
        color = np.asarray(colors[(idx - 1) % len(colors)][::-1], dtype=np.uint8)
        if line_mask.shape != display_shape:
            line_mask_display = resize(
                line_mask.astype(np.uint8),
                display_shape,
                order=0,
                preserve_range=True,
                anti_aliasing=False,
            ).astype(bool)
        else:
            line_mask_display = line_mask

        thick = ndi.binary_dilation(line_mask_display, iterations=1)
        rgb[thick] = (0.65 * color + 0.35 * rgb[thick]).astype(np.uint8)

        coords = np.column_stack(np.where(line_mask_display))
        if coords.size == 0:
            continue
        cy, cx = coords[len(coords) // 2]
        label_positions.append((str(idx), int(cx) + 2, int(cy) - 2))

    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("arial.ttf", label_font_size)
    except OSError:
        font = ImageFont.load_default()
    for text, x, y in label_positions:
        draw.text((x, y), text, fill=(255, 255, 255), font=font)

    pil_img.save(save_path)


def build_image_sets(folder_to_files):
    reference_key = next(iter(folder_to_files))
    reference = set(folder_to_files[reference_key])
    for key, files in folder_to_files.items():
        current = set(files)
        if current != reference:
            missing = sorted(reference - current, key=natural_key)
            extra = sorted(current - reference, key=natural_key)
            raise ValueError(f"File mismatch for {key}: missing={missing}, extra={extra}")
    names = sorted(reference, key=natural_key)
    return [{key: name for key in folder_to_files} for name in names]


folder_dirs = {key: os.path.join(root_dir, folder) for key, folder in folder_names.items()}
for key, folder in folder_dirs.items():
    if not os.path.isdir(folder):
        raise RuntimeError(f"Folder not found for {key}: {folder}")
    os.makedirs(os.path.join(overlay_root, key), exist_ok=True)

folder_to_files = {key: list_images(folder) for key, folder in folder_dirs.items()}
image_sets = build_image_sets(folder_to_files)
max_image_groups = os.environ.get("LINE_COUNT_MAX_IMAGES")
if max_image_groups:
    image_sets = image_sets[: int(max_image_groups)]

if len(image_sets) == 0:
    raise RuntimeError("No matched images found")

print(f"Found {len(image_sets)} matched image groups")

rows = []
for idx, file_map in enumerate(image_sets, start=1):
    row = {"index": idx}

    for key, filename in file_map.items():
        image_id = os.path.splitext(filename)[0]
        row[f"{key}_image"] = image_id

        try:
            img_path = os.path.join(folder_dirs[key], filename)
            img_norm = normalize_image(read_image(img_path))
            params = folder_params.get(
                key,
                {
                    "threshold_scale": threshold_scale,
                    "merge_endpoint_distance_px": merge_endpoint_distance_px,
                    "merge_angle_degrees": merge_angle_degrees,
                    "max_cumulative_turn_degrees": 100,
                    "min_line_length_px": min_line_length_px,
                    "min_line_skeleton_pixels": min_line_skeleton_pixels,
                    "min_ridge_alignment": 0.25,
                    "alignment_filter_max_length_px": 0,
                },
            )
            line_masks, mask, skel = segment_lines(img_norm, params)
            count = len(line_masks)
            row[f"{key}_line_count"] = count
            row[f"{key}_threshold_scale"] = params["threshold_scale"]
            row[f"{key}_merge_endpoint_distance_px"] = params["merge_endpoint_distance_px"]

            save_path = os.path.join(overlay_root, key, image_id + "_line_count.png")
            save_line_overlay(img_norm, line_masks, save_path)
            row[f"{key}_overlay"] = save_path

        except Exception as e:
            row[f"{key}_line_count"] = np.nan
            row[f"{key}_overlay"] = ""
            row[f"{key}_error"] = str(e)

    rows.append(row)

    count_msg = ", ".join(
        f"{key}={row.get(f'{key}_line_count', np.nan)}"
        for key in folder_names
    )
    first_name = next(iter(file_map.values()))
    print(f"[{idx}/{len(image_sets)}] {first_name}: {count_msg}")

df = pd.DataFrame(rows)
df.to_csv(output_csv, index=False, encoding="utf-8-sig")
with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
    df.to_excel(writer, index=False, sheet_name="line_counts")

print("Done")
print("Excel saved to:", output_excel)
print("CSV saved to:", output_csv)
print("One-image overlays saved to:", overlay_root)
