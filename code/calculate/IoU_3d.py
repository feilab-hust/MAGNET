import argparse
import os
import re
import numpy as np
import tifffile as tiff
import pandas as pd
from PIL import Image
from scipy.stats import ttest_rel, wilcoxon


root_dir = None
gt_method_folder_name = None
output_dir = None
excel_path = None
vis_mip_root = None
vis_stack_root = None



only_use_prefix = False
method_prefix = "TTO_"

ignore_folders = {
    "iou_analysis",
    "iou_analysis_3d",
    "vis",
    "vis_mip",
    "vis_stack",
}


mask_threshold = 0
match_sample_by_name = True
match_slices_by_name = True
save_mip_vis = True
save_stack_vis = True
background_gray = 180
color_tp = (45, 160, 90)
color_fp = (45, 150, 210)
color_fn = (225, 95, 80)



def natural_key(name):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", str(name))
    ]


def safe_sheet_name(name):
    name = re.sub(r"[\[\]\:\*\?\/\\]", "_", name)

    if len(name) > 31:
        name = name[:31]

    if name == "":
        name = "Sheet"

    return name


def list_tif_files(folder):
    if not os.path.isdir(folder):
        return []

    names = [
        name for name in os.listdir(folder)
        if name.lower().endswith((".tif", ".tiff"))
    ]

    names.sort(key=natural_key)
    return names


def list_subfolders(folder):
    if not os.path.isdir(folder):
        return []

    names = [
        name for name in os.listdir(folder)
        if os.path.isdir(os.path.join(folder, name))
    ]

    names.sort(key=natural_key)
    return names


def get_method_folders(root_dir):
    folders = []

    for name in os.listdir(root_dir):
        path = os.path.join(root_dir, name)

        if not os.path.isdir(path):
            continue

        if name in ignore_folders:
            continue

        if only_use_prefix and not name.startswith(method_prefix):
            continue

        folders.append(name)

    folders.sort(key=natural_key)
    return folders


def read_tif_mask(path):
    img = tiff.imread(path)
    img = np.asarray(img)
    img = np.squeeze(img)

    if img.ndim == 2:
        img = img[np.newaxis, :, :]

    elif img.ndim == 3:
        if img.shape[-1] in [3, 4]:
            img = img[..., 0]
            img = img[np.newaxis, :, :]

    else:
        raise ValueError(f"Unsupported tif shape: {img.shape}")

    mask = img > mask_threshold
    return mask.astype(bool)


def read_sample_as_3d(sample_dir):
    tif_names = list_tif_files(sample_dir)

    if len(tif_names) == 0:
        raise RuntimeError(f"No TIFF files found: {sample_dir}")

    if len(tif_names) == 1:
        path = os.path.join(sample_dir, tif_names[0])
        mask = read_tif_mask(path)
        return mask, tif_names

    slice_masks = []

    for name in tif_names:
        path = os.path.join(sample_dir, name)
        mask = read_tif_mask(path)

        if mask.shape[0] != 1:
            raise ValueError(
                f"Expected a single 2D slice in multi-file mode: {path}, shape={mask.shape}"
            )

        slice_masks.append(mask[0])

    volume = np.stack(slice_masks, axis=0)
    return volume.astype(bool), tif_names


def read_sample_as_3d_by_pairs(gt_sample_dir, pred_sample_dir):
    gt_tif_names = list_tif_files(gt_sample_dir)
    pred_tif_names = list_tif_files(pred_sample_dir)

    if len(gt_tif_names) == 0:
        raise RuntimeError(f"No GT TIFF files found: {gt_sample_dir}")

    if len(pred_tif_names) == 0:
        raise RuntimeError(f"No prediction TIFF files found: {pred_sample_dir}")

    if len(gt_tif_names) == 1 and len(pred_tif_names) == 1:
        if gt_tif_names[0] != pred_tif_names[0]:
            raise ValueError(
                f"Slice file mismatch: GT={gt_tif_names[0]}, prediction={pred_tif_names[0]}"
            )
        gt_mask, gt_used = read_sample_as_3d(gt_sample_dir)
        pred_mask, pred_used = read_sample_as_3d(pred_sample_dir)
        return gt_mask, pred_mask, gt_used, pred_used

    if set(gt_tif_names) != set(pred_tif_names):
        missing = sorted(set(gt_tif_names) - set(pred_tif_names), key=natural_key)
        extra = sorted(set(pred_tif_names) - set(gt_tif_names), key=natural_key)
        raise ValueError(f"Slice file mismatch: missing={missing}, extra={extra}")

    common_pairs = [(name, name) for name in gt_tif_names]

    gt_slices = []
    pred_slices = []
    gt_used = []
    pred_used = []

    for gt_name, pred_name in common_pairs:
        gt_path = os.path.join(gt_sample_dir, gt_name)
        pred_path = os.path.join(pred_sample_dir, pred_name)

        gt_mask = read_tif_mask(gt_path)
        pred_mask = read_tif_mask(pred_path)

        if gt_mask.shape[0] != 1 or pred_mask.shape[0] != 1:
            raise ValueError("Each TIFF must contain one 2D slice in multi-file mode")

        gt_slices.append(gt_mask[0])
        pred_slices.append(pred_mask[0])

        gt_used.append(gt_name)
        pred_used.append(pred_name)

    gt_volume = np.stack(gt_slices, axis=0)
    pred_volume = np.stack(pred_slices, axis=0)

    return gt_volume, pred_volume, gt_used, pred_used


def calculate_iou_3d(gt_mask, pred_mask):
    if gt_mask.shape != pred_mask.shape:
        raise ValueError(f"Shape mismatch: gt={gt_mask.shape}, pred={pred_mask.shape}")

    gt_mask = gt_mask.astype(bool)
    pred_mask = pred_mask.astype(bool)

    tp = gt_mask & pred_mask
    fp = (~gt_mask) & pred_mask
    fn = gt_mask & (~pred_mask)

    tp_count = int(tp.sum())
    fp_count = int(fp.sum())
    fn_count = int(fn.sum())

    denom = tp_count + fp_count + fn_count

    if denom == 0:
        iou = 1.0
    else:
        iou = tp_count / denom

    return iou, tp_count, fp_count, fn_count


def make_vis_stack(gt_mask, pred_mask):
    if gt_mask.shape != pred_mask.shape:
        raise ValueError(
            f"Shape mismatch for visualization: gt={gt_mask.shape}, pred={pred_mask.shape}"
        )

    tp = gt_mask & pred_mask
    fp = (~gt_mask) & pred_mask
    fn = gt_mask & (~pred_mask)

    z, h, w = gt_mask.shape

    rgb = np.zeros((z, h, w, 3), dtype=np.uint8)
    rgb[:, :, :, :] = background_gray

    rgb[tp] = color_tp
    rgb[fp] = color_fp
    rgb[fn] = color_fn

    return rgb


def make_vis_mip(gt_mask, pred_mask):
    vis_stack = make_vis_stack(gt_mask, pred_mask)

    gt_mip = np.max(gt_mask, axis=0)
    pred_mip = np.max(pred_mask, axis=0)

    tp_mip = gt_mip & pred_mip
    fp_mip = (~gt_mip) & pred_mip
    fn_mip = gt_mip & (~pred_mip)

    h, w = gt_mip.shape

    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :, :] = background_gray

    rgb[tp_mip] = color_tp
    rgb[fp_mip] = color_fp
    rgb[fn_mip] = color_fn

    return Image.fromarray(rgb)


def p_to_star(p_value):
    if pd.isna(p_value):
        return "NA"

    if p_value < 0.0001:
        return "****"

    if p_value < 0.001:
        return "***"

    if p_value < 0.01:
        return "**"

    if p_value < 0.05:
        return "*"

    return "ns"


def get_compare_key(method_name):
    lower_name = method_name.lower()

    if "ours" in lower_name:
        key = lower_name.replace("ours", "")
        key = key.replace("__", "_").strip("_")
        return "ours", key

    if "unet" in lower_name:
        key = lower_name.replace("unet", "")
        key = key.replace("__", "_").strip("_")
        return "unet", key

    return None, None


def calculate_significance(all_method_results):
    ours_methods = {}
    unet_methods = {}

    for method_name in all_method_results.keys():
        method_type, compare_key = get_compare_key(method_name)

        if method_type == "ours":
            ours_methods[compare_key] = method_name

        elif method_type == "unet":
            unet_methods[compare_key] = method_name

    common_keys = sorted(
        set(ours_methods.keys()) & set(unet_methods.keys()),
        key=natural_key
    )

    rows = []

    for compare_key in common_keys:
        ours_name = ours_methods[compare_key]
        unet_name = unet_methods[compare_key]

        df_ours = all_method_results[ours_name][["sample", "iou_3d"]].copy()
        df_unet = all_method_results[unet_name][["sample", "iou_3d"]].copy()

        df_ours = df_ours.rename(columns={"iou_3d": "iou_ours"})
        df_unet = df_unet.rename(columns={"iou_3d": "iou_unet"})

        merged = pd.merge(
            df_ours,
            df_unet,
            on="sample",
            how="inner"
        )

        merged = merged.dropna(subset=["iou_ours", "iou_unet"])

        n = len(merged)

        if n == 0:
            rows.append({
                "compare_key": compare_key,
                "ours_method": ours_name,
                "unet_method": unet_name,
                "n": 0,
                "mean_iou_ours": np.nan,
                "mean_iou_unet": np.nan,
                "mean_diff_ours_minus_unet": np.nan,
                "paired_t_p": np.nan,
                "paired_t_star": "NA",
                "wilcoxon_p": np.nan,
                "wilcoxon_star": "NA",
            })
            continue

        ours_values = merged["iou_ours"].to_numpy(dtype=np.float64)
        unet_values = merged["iou_unet"].to_numpy(dtype=np.float64)

        diff_values = ours_values - unet_values

        mean_ours = float(np.mean(ours_values))
        mean_unet = float(np.mean(unet_values))
        mean_diff = float(np.mean(diff_values))

        if n >= 2:
            try:
                t_result = ttest_rel(
                    ours_values,
                    unet_values,
                    nan_policy="omit"
                )
                paired_t_p = float(t_result.pvalue)
            except Exception:
                paired_t_p = np.nan
        else:
            paired_t_p = np.nan

        if n >= 2 and np.any(diff_values != 0):
            try:
                w_result = wilcoxon(
                    ours_values,
                    unet_values,
                    zero_method="wilcox"
                )
                wilcoxon_p = float(w_result.pvalue)
            except Exception:
                wilcoxon_p = np.nan
        else:
            wilcoxon_p = np.nan

        rows.append({
            "compare_key": compare_key,
            "ours_method": ours_name,
            "unet_method": unet_name,
            "n": n,
            "mean_iou_ours": mean_ours,
            "mean_iou_unet": mean_unet,
            "mean_diff_ours_minus_unet": mean_diff,
            "paired_t_p": paired_t_p,
            "paired_t_star": p_to_star(paired_t_p),
            "wilcoxon_p": wilcoxon_p,
            "wilcoxon_star": p_to_star(wilcoxon_p),
        })

    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir")
    parser.add_argument("gt_method")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main():
    global root_dir, gt_method_folder_name, output_dir, excel_path
    global vis_mip_root, vis_stack_root
    args = parse_args()
    root_dir = os.path.abspath(args.root_dir)
    gt_method_folder_name = args.gt_method
    output_dir = os.path.abspath(args.output_dir or os.path.join(root_dir, "iou_analysis_3d"))
    excel_path = os.path.join(output_dir, "iou_3d_result.xlsx")
    vis_mip_root = os.path.join(output_dir, "vis_mip")
    vis_stack_root = os.path.join(output_dir, "vis_stack")
    gt_method_dir = os.path.join(root_dir, gt_method_folder_name)

    if not os.path.isdir(gt_method_dir):
        raise RuntimeError(f"GT method folder not found: {gt_method_dir}")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(vis_mip_root, exist_ok=True)
    os.makedirs(vis_stack_root, exist_ok=True)

    method_names = get_method_folders(root_dir)

    if gt_method_folder_name not in method_names:
        method_names.insert(0, gt_method_folder_name)

    method_names = [gt_method_folder_name] + [
        name for name in method_names
        if name != gt_method_folder_name
    ]

    print("GT method folder:", gt_method_folder_name)
    print("Methods:")

    for method in method_names:
        print("  ", method)

    gt_sample_names = list_subfolders(gt_method_dir)

    if len(gt_sample_names) == 0:
        raise RuntimeError(f"No sample folders found in GT directory: {gt_method_dir}")

    all_method_results = {}
    summary_rows = []

    for method_name in method_names:
        method_dir = os.path.join(root_dir, method_name)

        if not os.path.isdir(method_dir):
            print(f"Skipping missing method directory: {method_dir}")
            continue

        vis_mip_dir = os.path.join(vis_mip_root, method_name)
        vis_stack_dir = os.path.join(vis_stack_root, method_name)

        os.makedirs(vis_mip_dir, exist_ok=True)
        os.makedirs(vis_stack_dir, exist_ok=True)

        method_sample_names = list_subfolders(method_dir)
        if set(method_sample_names) != set(gt_sample_names):
            missing = sorted(set(gt_sample_names) - set(method_sample_names), key=natural_key)
            extra = sorted(set(method_sample_names) - set(gt_sample_names), key=natural_key)
            raise ValueError(
                f"Sample folder mismatch for {method_name}: missing={missing}, extra={extra}"
            )

        rows = []

        print(f"\nProcessing method: {method_name}")

        for sample_idx, gt_sample_name in enumerate(gt_sample_names):
            gt_sample_dir = os.path.join(gt_method_dir, gt_sample_name)

            if match_sample_by_name:
                method_sample_name = gt_sample_name
            else:
                if sample_idx >= len(method_sample_names):
                    rows.append({
                        "sample": gt_sample_name,
                        "method_sample": "",
                        "iou_3d": np.nan,
                        "tp_voxels": np.nan,
                        "fp_voxels": np.nan,
                        "fn_voxels": np.nan,
                        "gt_voxels": np.nan,
                        "pred_voxels": np.nan,
                        "volume_shape": "",
                        "z_slices": np.nan,
                        "status": "missing_sample",
                        "error": "",
                    })
                    continue

                method_sample_name = method_sample_names[sample_idx]

            method_sample_dir = os.path.join(method_dir, method_sample_name)

            if not os.path.isdir(method_sample_dir):
                rows.append({
                    "sample": gt_sample_name,
                    "method_sample": method_sample_name,
                    "iou_3d": np.nan,
                    "tp_voxels": np.nan,
                    "fp_voxels": np.nan,
                    "fn_voxels": np.nan,
                    "gt_voxels": np.nan,
                    "pred_voxels": np.nan,
                    "volume_shape": "",
                    "z_slices": np.nan,
                    "status": "missing_sample",
                    "error": f"sample folder not found: {method_sample_dir}",
                })
                print(f"  Missing sample: {method_name} / {method_sample_name}")
                continue

            try:
                gt_mask, pred_mask, gt_used, pred_used = read_sample_as_3d_by_pairs(
                    gt_sample_dir=gt_sample_dir,
                    pred_sample_dir=method_sample_dir
                )

                iou, tp_count, fp_count, fn_count = calculate_iou_3d(
                    gt_mask=gt_mask,
                    pred_mask=pred_mask
                )

                gt_voxels = int(gt_mask.sum())
                pred_voxels = int(pred_mask.sum())

                row = {
                    "sample": gt_sample_name,
                    "method_sample": method_sample_name,
                    "iou_3d": iou,
                    "tp_voxels": tp_count,
                    "fp_voxels": fp_count,
                    "fn_voxels": fn_count,
                    "gt_voxels": gt_voxels,
                    "pred_voxels": pred_voxels,
                    "volume_shape": str(gt_mask.shape),
                    "z_slices": int(gt_mask.shape[0]),
                    "gt_tif_count": len(gt_used),
                    "pred_tif_count": len(pred_used),
                    "status": "ok",
                    "error": "",
                }

                rows.append(row)

                base_name = gt_sample_name

                if save_mip_vis:
                    vis_img = make_vis_mip(gt_mask, pred_mask)
                    save_path = os.path.join(vis_mip_dir, base_name + "_mip.png")
                    vis_img.save(save_path)

                if save_stack_vis:
                    vis_stack = make_vis_stack(gt_mask, pred_mask)
                    save_path = os.path.join(vis_stack_dir, base_name + "_tp_fp_fn_stack.tif")
                    tiff.imwrite(save_path, vis_stack)

                print(
                    f"  {gt_sample_name} vs {method_sample_name} | "
                    f"shape={gt_mask.shape} | IoU3D={iou:.4f}"
                )

            except Exception as e:
                rows.append({
                    "sample": gt_sample_name,
                    "method_sample": method_sample_name,
                    "iou_3d": np.nan,
                    "tp_voxels": np.nan,
                    "fp_voxels": np.nan,
                    "fn_voxels": np.nan,
                    "gt_voxels": np.nan,
                    "pred_voxels": np.nan,
                    "volume_shape": "",
                    "z_slices": np.nan,
                    "gt_tif_count": np.nan,
                    "pred_tif_count": np.nan,
                    "status": "error",
                    "error": str(e),
                })

                print(f"  Error: {gt_sample_name} vs {method_sample_name} | {e}")

        df_method = pd.DataFrame(rows)
        all_method_results[method_name] = df_method

        df_ok = df_method[df_method["status"] == "ok"].copy()

        if len(df_ok) > 0:
            summary_rows.append({
                "method": method_name,
                "mean_iou_3d": df_ok["iou_3d"].mean(),
                "std_iou_3d": df_ok["iou_3d"].std(),
                "mean_tp_voxels": df_ok["tp_voxels"].mean(),
                "mean_fp_voxels": df_ok["fp_voxels"].mean(),
                "mean_fn_voxels": df_ok["fn_voxels"].mean(),
                "mean_gt_voxels": df_ok["gt_voxels"].mean(),
                "mean_pred_voxels": df_ok["pred_voxels"].mean(),
                "count": len(df_ok),
            })
        else:
            summary_rows.append({
                "method": method_name,
                "mean_iou_3d": np.nan,
                "std_iou_3d": np.nan,
                "mean_tp_voxels": np.nan,
                "mean_fp_voxels": np.nan,
                "mean_fn_voxels": np.nan,
                "mean_gt_voxels": np.nan,
                "mean_pred_voxels": np.nan,
                "count": 0,
            })

    summary_df = pd.DataFrame(summary_rows)
    significance_df = calculate_significance(all_method_results)

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for method_name, df_method in all_method_results.items():
            sheet_name = safe_sheet_name(method_name)
            df_method.to_excel(writer, index=False, sheet_name=sheet_name)

        summary_df.to_excel(writer, index=False, sheet_name="Summary")
        significance_df.to_excel(writer, index=False, sheet_name="Significance")

    print("\nAll finished.")
    print("Excel saved to:", excel_path)
    print("MIP visualization saved to:", vis_mip_root)
    print("3D stack visualization saved to:", vis_stack_root)


if __name__ == "__main__":
    main()
