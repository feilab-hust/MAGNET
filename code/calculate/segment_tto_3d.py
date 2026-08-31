import argparse
import csv
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skimage.filters import apply_hysteresis_threshold
from skimage.morphology import closing, disk, remove_small_objects
from skimage.segmentation import clear_border, find_boundaries
import tifffile


def safe_name(folder):
    name = str(folder).strip("/").replace("\\", "__").replace("/", "__")
    return re.sub(r'[<>:"/\\|?*()]', "_", name)


def natural_key(path):
    text = str(path)
    return [
        int(s) if s.isdigit() else s.lower()
        for s in re.split(r"(\d+)", text)
    ]


def read_tif_2d(path):
    image = tifffile.imread(str(path))
    image = np.asarray(image)
    image = np.squeeze(image)

    if image.ndim == 2:
        return image

    if image.ndim == 3 and image.shape[-1] in [3, 4]:
        image = image[..., 0]
        return image

    if image.ndim == 3:
        image = np.max(image, axis=0)
        return image



def normalize_u8(array, low=0.5, high=99.8):
    lo, hi = np.percentile(array, (low, high))
    return np.clip(
        (array.astype(np.float32) - lo) * 255 / max(hi - lo, 1e-9),
        0,
        255,
    ).astype(np.uint8)


def segment_slice(
    image,
    low_percentile=96.5,
    high_percentile=99.2,
    background_radius=28.0,
    border=48,
    min_object_size=32,
    body_closing_radius=3,
    keep_hollow=True,
):

    image = image.astype(np.float32)

    fine = ndi.gaussian_filter(image, 1.2)
    background = ndi.gaussian_filter(image, background_radius)
    response = np.maximum(fine - background, 0)

    positive = response[response > 0]

    if positive.size == 0:
        return np.zeros(image.shape, bool), np.zeros(image.shape, bool)

    low, high = np.percentile(
        positive,
        (low_percentile, high_percentile)
    )

    raw = apply_hysteresis_threshold(response, low, high)
    raw = closing(raw, disk(1))

    raw = remove_small_objects(
        raw.astype(bool),
        min_size=min_object_size
    )

    if keep_hollow:
        mask = closing(raw, disk(body_closing_radius))
    else:
        body_shell = closing(raw, disk(body_closing_radius))
        filled_body = ndi.binary_fill_holes(body_shell)
        mask = raw | filled_body

    if border > 0:
        mask = clear_border(mask, buffer_size=border)

        mask[:border] = False
        mask[-border:] = False
        mask[:, :border] = False
        mask[:, -border:] = False

    strong = response >= high

    if border > 0:
        strong[:border] = False
        strong[-border:] = False
        strong[:, :border] = False
        strong[:, -border:] = False

    return mask, strong
def remove_rectangular_frame_artifacts_2d(
    mask,
    min_bbox_area=2500,
    min_side=35,
    edge_band=3,
    side_contact_ratio=0.30,
    min_contact_sides=2,
):

    mask = mask.astype(bool)

    labels, count = ndi.label(
        mask,
        structure=np.ones((3, 3), dtype=bool)
    )

    if count == 0:
        return mask, np.zeros_like(mask, dtype=bool)

    cleaned = mask.copy()
    removed = np.zeros_like(mask, dtype=bool)

    objects = ndi.find_objects(labels)

    for label_id, slc in enumerate(objects, start=1):
        if slc is None:
            continue

        rr, cc = slc
        minr, maxr = rr.start, rr.stop
        minc, maxc = cc.start, cc.stop

        h = maxr - minr
        w = maxc - minc
        bbox_area = h * w

        if bbox_area < min_bbox_area:
            continue

        if h < min_side or w < min_side:
            continue

        comp = labels[slc] == label_id

        band = max(1, min(edge_band, h // 2, w // 2))

        top_ratio = np.any(comp[:band, :], axis=0).sum() / max(w, 1)
        bottom_ratio = np.any(comp[-band:, :], axis=0).sum() / max(w, 1)
        left_ratio = np.any(comp[:, :band], axis=1).sum() / max(h, 1)
        right_ratio = np.any(comp[:, -band:], axis=1).sum() / max(h, 1)

        side_ratios = [
            top_ratio,
            bottom_ratio,
            left_ratio,
            right_ratio,
        ]

        contact_sides = sum(r >= side_contact_ratio for r in side_ratios)

        if contact_sides >= min_contact_sides:
            cleaned[slc][comp] = False
            removed[slc][comp] = True

    return cleaned, removed


def remove_rectangular_frame_artifacts_3d(
    volume,
    min_bbox_area=2500,
    min_side=35,
    edge_band=3,
    side_contact_ratio=0.30,
    min_contact_sides=2,
):
    volume = volume.astype(bool)
    cleaned = np.zeros_like(volume, dtype=np.uint8)
    removed = np.zeros_like(volume, dtype=np.uint8)

    for z in range(volume.shape[0]):
        clean_z, removed_z = remove_rectangular_frame_artifacts_2d(
            volume[z],
            min_bbox_area=min_bbox_area,
            min_side=min_side,
            edge_band=edge_band,
            side_contact_ratio=side_contact_ratio,
            min_contact_sides=min_contact_sides,
        )

        cleaned[z] = clean_z.astype(np.uint8)
        removed[z] = removed_z.astype(np.uint8)

    return cleaned, removed


def refine_z(raw, strong, min_slice_keep_size=128, min_strong_keep_size=32):

    depth = raw.shape[0]
    output = np.zeros_like(raw, dtype=np.uint8)

    xy_structure = np.ones((3, 3), bool)

    for z in range(depth):
        current = raw[z].astype(bool)

        if z > 0:
            previous = ndi.binary_dilation(
                raw[z - 1].astype(bool),
                structure=xy_structure
            )
        else:
            previous = np.zeros_like(current)

        if z + 1 < depth:
            following = ndi.binary_dilation(
                raw[z + 1].astype(bool),
                structure=xy_structure
            )
        else:
            following = np.zeros_like(current)

        supported = previous | following

        labels, count = ndi.label(
            current,
            structure=np.ones((3, 3), bool)
        )

        if count:
            sizes = np.bincount(labels.ravel())

            touches_strong = np.bincount(
                labels.ravel(),
                weights=strong[z].ravel(),
                minlength=count + 1
            ) > 0

            touches_z = np.bincount(
                labels.ravel(),
                weights=supported.ravel(),
                minlength=count + 1
            ) > 0

            keep = (
                (sizes >= min_slice_keep_size)
                | touches_z
                | (touches_strong & (sizes >= min_strong_keep_size))
            )

            keep[0] = False
            current = keep[labels]

        output[z] = current.astype(np.uint8)

    return output

def enforce_z_continuity(volume, z_radius=1, xy_radius=2, iterations=1):

    out = volume.astype(bool).copy()

    if z_radius <= 0:
        return out.astype(np.uint8)

    xy_structure = disk(xy_radius).astype(bool)

    depth = out.shape[0]

    for _ in range(iterations):
        updated = out.copy()

        for z in range(z_radius, depth - z_radius):
            before = out[z - z_radius]
            after = out[z + z_radius]

            before_dil = ndi.binary_dilation(
                before,
                structure=xy_structure
            )

            after_dil = ndi.binary_dilation(
                after,
                structure=xy_structure
            )

            support = before_dil & after_dil

            updated[z] = updated[z] | support

        out = updated

    return out.astype(np.uint8)

def remove_small_3d_components(volume, min_3d_component_size=800, min_z_slices=2):

    volume = volume.astype(bool)

    labels, count = ndi.label(
        volume,
        structure=np.ones((3, 3, 3), bool)
    )

    if count == 0:
        return volume.astype(np.uint8)

    sizes = np.bincount(labels.ravel())

    z_presence = np.zeros((count + 1, volume.shape[0]), dtype=bool)

    for z in range(volume.shape[0]):
        ids = np.unique(labels[z])
        z_presence[ids, z] = True

    z_counts = z_presence.sum(axis=1)

    keep = (
        (sizes >= min_3d_component_size)
        & (z_counts >= min_z_slices)
    )

    keep[0] = False

    cleaned = keep[labels]

    return cleaned.astype(np.uint8)


def safe_ascii_text(text):
    text = str(text)
    text = re.sub(r"[^\x00-\x7F]+", "_", text)
    text = re.sub(r"[^a-zA-Z0-9_\-./\\ ]+", "_", text)
    text = text.strip()

    if text == "":
        text = "projection"

    return text


def projection_panel(mask, destination, title):
    xy = (mask.max(axis=0) * 255).astype(np.uint8)
    xz = (mask.max(axis=1) * 255).astype(np.uint8)
    yz = (mask.max(axis=2) * 255).astype(np.uint8)

    xy_im = Image.fromarray(xy)

    xz_im = Image.fromarray(xz).resize(
        (xy.shape[1], max(80, xz.shape[0] * 8)),
        Image.Resampling.NEAREST
    )

    yz_im = Image.fromarray(yz).resize(
        (max(80, yz.shape[1] * 8), xy.shape[0]),
        Image.Resampling.NEAREST
    )

    canvas = Image.new(
        "RGB",
        (
            xy_im.width + yz_im.width,
            32 + xy_im.height + xz_im.height
        ),
        "black",
    )

    canvas.paste(xy_im.convert("RGB"), (0, 32))
    canvas.paste(yz_im.convert("RGB"), (xy_im.width, 32))
    canvas.paste(xz_im.convert("RGB"), (0, 32 + xy_im.height))

    draw = ImageDraw.Draw(canvas)

    safe_title = safe_ascii_text(title)

    try:
        draw.text(
            (8, 8),
            f"{safe_title} | XY large, YZ right, XZ bottom",
            fill="white"
        )
    except Exception:
        pass

    canvas.thumbnail((1800, 1800))
    canvas.save(destination)


def write_slice_folder(volume, destination, suffix):
    destination.mkdir(parents=True, exist_ok=True)

    for z in range(volume.shape[0]):
        Image.fromarray((volume[z] * 255).astype(np.uint8)).save(
            destination / f"{z:05d}_{suffix}.tif",
            compression="tiff_deflate",
        )


def collect_tif_groups(input_root):
    tif_files = sorted(
        [
            p for p in input_root.rglob("*")
            if p.is_file() and p.suffix.lower() in [".tif", ".tiff"]
        ],
        key=natural_key
    )

    groups = {}

    for tif_path in tif_files:
        parent = tif_path.parent

        if "results" in tif_path.parts:
            continue

        groups.setdefault(parent, []).append(tif_path)

    return groups


def process_group(folder, entries, input_root, output_root, args):
    entries = sorted(entries, key=natural_key)

    first = read_tif_2d(entries[0])
    shape = (len(entries), *first.shape)

    relative_folder = folder.relative_to(input_root)
    stem = safe_name(relative_folder)

    print(f"Processing {relative_folder} ({len(entries)} slices), shape={shape}")

    raw = np.zeros(shape, dtype=np.uint8)
    strong = np.zeros(shape, dtype=np.uint8)

    for z, path in enumerate(entries):
        image = read_tif_2d(path)

        if image.shape != first.shape:
            raise ValueError(
                f"Shape mismatch in {relative_folder}: "
                f"{path.name} shape={image.shape}, expected={first.shape}"
            )

        mask, seed = segment_slice(
            image=image,
            low_percentile=args.low_percentile,
            high_percentile=args.high_percentile,
            background_radius=args.background_radius,
            border=args.border,
            min_object_size=args.min_object_size,
            body_closing_radius=args.body_closing_radius,
            keep_hollow=args.keep_hollow,
        )

        if args.remove_rect_frames:
            mask, rect_removed = remove_rectangular_frame_artifacts_2d(
                mask,
                min_bbox_area=args.rect_min_bbox_area,
                min_side=args.rect_min_side,
                edge_band=args.rect_edge_band,
                side_contact_ratio=args.rect_side_contact_ratio,
                min_contact_sides=args.rect_min_contact_sides,
            )

            seed[rect_removed] = False

        raw[z] = mask.astype(np.uint8)
        strong[z] = seed.astype(np.uint8)

        if (z + 1) % 10 == 0 or z + 1 == len(entries):
            print(f"  {stem}: {z + 1}/{len(entries)} slices")

    final = refine_z(
        raw,
        strong,
        min_slice_keep_size=args.min_slice_keep_size,
        min_strong_keep_size=args.min_strong_keep_size
    )

    final = enforce_z_continuity(
        final,
        z_radius=args.z_continuity_radius,
        xy_radius=args.xy_continuity_radius,
        iterations=args.z_continuity_iterations
    )

    final = remove_small_3d_components(
        final,
        min_3d_component_size=args.min_3d_component_size,
        min_z_slices=args.min_z_slices
    )

    if args.remove_rect_frames:
        final, rect_removed_final = remove_rectangular_frame_artifacts_3d(
            final,
            min_bbox_area=args.rect_min_bbox_area,
            min_side=args.rect_min_side,
            edge_band=args.rect_edge_band,
            side_contact_ratio=args.rect_side_contact_ratio,
            min_contact_sides=args.rect_min_contact_sides,
        )

    group_out = output_root / stem
    group_out.mkdir(parents=True, exist_ok=True)

    mask_tif = group_out / f"{stem}_mask_3d.tif"
    edge_tif = group_out / f"{stem}_outer_edge_3d.tif"

    tifffile.imwrite(
        mask_tif,
        final,
        photometric="minisblack",
        compression="zlib",
        metadata={"axes": "ZYX"},
    )

    edge = np.zeros_like(final, dtype=np.uint8)

    for z in range(shape[0]):
        edge[z] = find_boundaries(
            final[z].astype(bool),
            mode="outer"
        ).astype(np.uint8)

    tifffile.imwrite(
        edge_tif,
        edge,
        photometric="minisblack",
        compression="zlib",
        metadata={"axes": "ZYX"},
    )

    if args.write_slices:
        write_slice_folder(final, group_out / "mask_slices", "mask")
        write_slice_folder(edge, group_out / "outer_edge_slices", "edge")

    projection_panel(
        final,
        group_out / f"{stem}_projections.png",
        str(relative_folder),
    )

    segmented_voxels = int(np.count_nonzero(final))
    total_voxels = int(final.size)

    result = {
        "folder": str(relative_folder),
        "z_slices": shape[0],
        "height_px": shape[1],
        "width_px": shape[2],
        "segmented_voxels": segmented_voxels,
        "segmented_fraction": float(segmented_voxels / total_voxels),
        "mask_3d_tif": str(mask_tif),
        "outer_edge_3d_tif": str(edge_tif),
    }

    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="3D segmentation from TIFF slices inside folders."
    )

    parser.add_argument(
        "input_root",
        type=Path,
        help="Input root folder containing TIFF slice folders."
    )

    parser.add_argument(
        "--low-percentile",
        type=float,
        default=96.5,
        help="Low hysteresis percentile."
    )

    parser.add_argument(
        "--high-percentile",
        type=float,
        default=99.2,
        help="High hysteresis percentile."
    )

    parser.add_argument(
        "--background-radius",
        type=float,
        default=28,
        help="Gaussian background radius."
    )

    parser.add_argument(
        "--body-closing-radius",
        type=int,
        default=3,
        help="Closing radius for filling object body loops."
    )

    parser.add_argument(
        "--min-object-size",
        type=int,
        default=32,
        help="Minimum object size in each 2D slice."
    )

    parser.add_argument(
        "--border",
        type=int,
        default=48,
        help="Remove components connected to image border."
    )

    parser.add_argument(
        "--write-slices",
        action="store_true",
        help="Also export every mask and edge plane as individual TIFF files."
    )
    parser.add_argument(
        "--min-slice-keep-size",
        type=int,
        default=128,
        help="Minimum 2D component size kept during Z refinement."
    )

    parser.add_argument(
        "--min-strong-keep-size",
        type=int,
        default=32,
        help="Strong 2D components smaller than this are removed."
    )

    parser.add_argument(
        "--min-3d-component-size",
        type=int,
        default=800,
        help="Minimum 3D connected component voxel count."
    )

    parser.add_argument(
        "--min-z-slices",
        type=int,
        default=2,
        help="Minimum number of Z slices a 3D component must appear in."
    )
    parser.add_argument(
        "--keep-hollow",
        action="store_true",
        default=True,
        help="Keep hollow shell-like structures instead of filling holes."
    )

    parser.add_argument(
        "--filled-body",
        action="store_false",
        dest="keep_hollow",
        help="Fill holes and output solid body mask."
    )

    parser.add_argument(
        "--z-continuity-radius",
        type=int,
        default=1,
        help="Use z-radius neighbours to fill missing slices."
    )

    parser.add_argument(
        "--xy-continuity-radius",
        type=int,
        default=2,
        help="XY dilation radius when enforcing Z continuity."
    )

    parser.add_argument(
        "--z-continuity-iterations",
        type=int,
        default=1,
        help="Number of Z continuity filling iterations."
    )
    parser.add_argument(
        "--remove-rect-frames",
        action="store_true",
        default=True,
        help="Remove obvious rectangular frame artifacts from each 2D mask."
    )

    parser.add_argument(
        "--keep-rect-frames",
        action="store_false",
        dest="remove_rect_frames",
        help="Do not remove rectangular frame artifacts."
    )

    parser.add_argument(
        "--rect-min-bbox-area",
        type=int,
        default=2500,
        help="Minimum bounding box area for detecting rectangular frame artifacts."
    )

    parser.add_argument(
        "--rect-min-side",
        type=int,
        default=35,
        help="Minimum bbox width/height for rectangular frame artifacts."
    )

    parser.add_argument(
        "--rect-edge-band",
        type=int,
        default=3,
        help="How many pixels near bbox edges are used to detect straight frames."
    )

    parser.add_argument(
        "--rect-side-contact-ratio",
        type=float,
        default=0.30,
        help="A bbox side is considered frame-like if this fraction is occupied."
    )

    parser.add_argument(
        "--rect-min-contact-sides",
        type=int,
        default=2,
        help="Remove component if at least this many bbox sides look like straight frame edges."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    input_root = args.input_root

    if not input_root.exists():
        raise FileNotFoundError(f"Input folder not found: {input_root}")

    if not input_root.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {input_root}")

    output_root = input_root / "results"
    output_root.mkdir(parents=True, exist_ok=True)

    groups = collect_tif_groups(input_root)

    if len(groups) == 0:
        raise RuntimeError(f"No TIFF files found under: {input_root}")

    rows = []

    for folder, entries in sorted(groups.items(), key=lambda x: natural_key(x[0])):
        rows.append(
            process_group(
                folder=folder,
                entries=entries,
                input_root=input_root,
                output_root=output_root,
                args=args
            )
        )

    summary = output_root / "segmentation_summary.csv"

    with summary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Finished. Summary: {summary.resolve()}")


if __name__ == "__main__":
    main()
