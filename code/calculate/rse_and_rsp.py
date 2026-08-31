import argparse
from pathlib import Path

import numpy as np
import tifffile as tiff
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment
from openpyxl.utils import get_column_letter



save_xlsx_name = "squirrel_metrics.xlsx"


save_error_map = False
error_map_folder_name = "squirrel_error_map"




def list_tifs(folder: Path):
    files = sorted([
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in [".tif", ".tiff"]
    ])
    return files


def is_lr_file(path: Path):
    name = path.stem.lower()
    return ("input" in name) or ("lr" in name)


def normalize_float(img):
    img = img.astype(np.float32)

    p1, p99 = np.percentile(img, (1, 99.9))
    img = np.clip(img, p1, p99)

    img_min = float(img.min())
    img_max = float(img.max())

    if img_max == img_min:
        raise ValueError("Image has zero dynamic range, cannot normalize.")

    img = (img - img_min) / (img_max - img_min)
    return img


def load_for_squirrel(path: Path):
    img = tiff.imread(str(path))

    if img.ndim == 3:
        img = np.mean(img, axis=0)

    img = normalize_float(img)
    return img


def calc_squirrel(ref_img, sr_img):
    from nanopyx.core.transform import ErrorMap
    error_map = ErrorMap()
    error_map.optimise(ref_img, sr_img)

    rse = float(error_map.getRSE())
    rsp = float(error_map.getRSP())
    err = np.array(error_map.imRSE).astype(np.float32)

    return rse, rsp, err


def safe_sheet_name(name: str):
    invalid_chars = ["\\", "/", "*", "?", ":", "[", "]"]
    for ch in invalid_chars:
        name = name.replace(ch, "_")
    return name[:31]


def write_sheet(wb, sheet_name, results):
    ws = wb.create_sheet(title=safe_sheet_name(sheet_name))

    image_names = sorted(results.keys())

    ws.cell(row=1, column=1, value="Metric")

    for col_idx, name in enumerate(image_names, start=2):
        ws.cell(row=1, column=col_idx, value=name)

    metrics = ["RSE", "RSP"]

    for row_idx, metric in enumerate(metrics, start=2):
        ws.cell(row=row_idx, column=1, value=metric)

        for col_idx, name in enumerate(image_names, start=2):
            value = results[name].get(metric, "")
            ws.cell(row=row_idx, column=col_idx, value=value)

    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal="center")

    for row_idx in range(1, len(metrics) + 2):
        ws.cell(row=row_idx, column=1).font = Font(bold=True)

    ws.freeze_panes = "B2"
    ws.column_dimensions["A"].width = 12

    for col_idx in range(2, len(image_names) + 2):
        col_letter = get_column_letter(col_idx)
        ws.column_dimensions[col_letter].width = 24

    for row in ws.iter_rows(min_row=2, max_row=3, min_col=2, max_col=len(image_names) + 1):
        for cell in row:
            if isinstance(cell.value, float):
                cell.number_format = "0.000000"


def process_one_folder(folder: Path):
    tif_files = list_tifs(folder)

    lr_files = [p for p in tif_files if is_lr_file(p)]
    sr_files = [p for p in tif_files if not is_lr_file(p)]
    if len(lr_files) != 1:
        raise ValueError(f"Expected exactly one reference image in {folder}, found {len(lr_files)}")
    if not sr_files:
        raise ValueError(f"No comparison images found in {folder}")
    ref_path = lr_files[0]

    ref_img = load_for_squirrel(ref_path)

    results = {}

    if save_error_map:
        err_dir = folder / error_map_folder_name
        err_dir.mkdir(parents=True, exist_ok=True)

    for sr_path in sr_files:
        sr_img = load_for_squirrel(sr_path)
        if sr_img.shape != ref_img.shape:
            raise ValueError(f"Shape mismatch: reference={ref_img.shape}, comparison={sr_img.shape}, file={sr_path}")

        rse, rsp, err = calc_squirrel(ref_img, sr_img)

        results[sr_path.name] = {
            "RSE": rse,
            "RSP": rsp
        }

        print(f"{sr_path.name}: RSE={rse:.6f}, RSP={rsp:.6f}")

        if save_error_map:
            err_save_path = err_dir / f"{sr_path.stem}_error_map.tif"
            tiff.imwrite(str(err_save_path), err)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir")
    parser.add_argument("--output-name", default="squirrel_metrics.xlsx")
    args = parser.parse_args()
    root = Path(args.root_dir).resolve()

    save_path = root / args.output_name

    wb = Workbook()
    default_sheet = wb.active
    wb.remove(default_sheet)

    subfolders = sorted([
        p for p in root.iterdir()
        if p.is_dir()
    ])

    valid_sheet_count = 0

    for folder in subfolders:
        results = process_one_folder(folder)

        if results is None or len(results) == 0:
            continue

        write_sheet(wb, folder.name, results)
        valid_sheet_count += 1


    wb.save(str(save_path))


if __name__ == "__main__":
    main()
