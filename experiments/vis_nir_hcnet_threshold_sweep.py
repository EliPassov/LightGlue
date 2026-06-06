#!/usr/bin/env python3
"""Sweep HC-Net descriptor-distance thresholds for VIS-NIR matching.

Example:
python experiments/vis_nir_hcnet_threshold_sweep.py --data_root /media/eli/storage2/Datasets/visnir_raw --exclude_categories country --limit_per_category 20 --max_num_keypoints 200 --nms_radius 45 --filter_border 32
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

from lightglue import SuperPoint  # noqa: E402
from vis_nir_common import discover_pairs, extract_superpoint_keypoints, mean_or_nan, parse_thresholds, print_pairs  # noqa: E402
from vis_nir_superpoint_hcnet import (  # noqa: E402
    NIR_MEAN,
    VIS_MEAN,
    crop_patches,
    describe_patches,
    load_hcnet_model,
    load_nir_grayscale_uint8,
    load_vis_grayscale_uint8,
    match_descriptors,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep HC-Net squared-L2 descriptor-distance thresholds.")
    parser.add_argument("--data_root", default="/media/eli/storage2/Datasets/visnir_raw", help="Dataset root directory.")
    parser.add_argument("--category", default=None, help="Optional category subfolder, e.g. country.")
    parser.add_argument("--pairs_file", default=None, help="Optional text file with one 'rgb_path nir_path' pair per line.")
    parser.add_argument("--rgb_suffix", default="_rgb.tiff")
    parser.add_argument("--nir_suffix", default="_nir.tiff")
    parser.add_argument("--max_num_keypoints", type=int, default=2048)
    parser.add_argument("--nms_radius", type=int, default=4, help="SuperPoint NMS radius in pixels.")
    parser.add_argument("--resize", type=int, default=None, help="Optional resize value passed to SuperPoint extractor.extract.")
    parser.add_argument("--thresholds", default="1,3,5,10,16,32,48,64", help="Pixel MMA thresholds.")
    parser.add_argument("--fixed_distance_thresholds", default="0.1,0.2,0.4,1", help="Comma-separated HC squared-L2 distance thresholds.")
    parser.add_argument("--train_thresholds_csv", default="/media/eli/storage2/Projects/tensorboard_files/vis_nir_result_backup/weights_and_config/visnir_train_country_thresholds.csv")
    parser.add_argument("--test_thresholds_csv", default="/media/eli/storage2/Projects/tensorboard_files/vis_nir_result_backup/weights_and_config/visnir_test_thresholds.csv")
    parser.add_argument("--max_threshold_tpr_percent", type=float, default=95.0, help="Use CSV thresholds up to this target_tpr_percent.")
    parser.add_argument("--output_csv", default="outputs/vis_nir_hcnet_threshold_sweep.csv", help="Path to write per-threshold aggregate CSV.")
    parser.add_argument("--device", default=None, help="Device to use. Defaults to cuda when available, otherwise cpu.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--limit_per_category", type=int, default=None, help="When scanning all categories, keep this many pairs per first-level category.")
    parser.add_argument("--exclude_categories", default=None, help="Comma-separated category names to skip, e.g. country.")
    parser.add_argument("--filter_border", type=int, default=32, help="Drop SuperPoint keypoints within this many pixels of the image border before HC-Net crops.")
    parser.add_argument("--hcnet_project", default="/media/eli/storage2/Projects/multisensor_yossi")
    parser.add_argument("--hcnet_config", default="/media/eli/storage2/Projects/tensorboard_files/vis_nir_result_backup/weights_and_config/norm_old_cond_in_gelu_hyp_embedding_24x48_patience_3_config.yaml")
    parser.add_argument("--hcnet_weights", default="/media/eli/storage2/Projects/tensorboard_files/vis_nir_result_backup/weights_and_config/weights.pth")
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--descriptor_batch_size", type=int, default=512)
    parser.add_argument("--hcnet_grayscale", choices=["cv2", "pil"], default="cv2", help="Grayscale conversion for HC-Net patches.")
    parser.add_argument("--match_backend", choices=["mnn", "lg_filter", "compare"], default="lg_filter", help="HC descriptor matcher before distance thresholding.")
    return parser.parse_args()


def read_threshold_rows(path: Path, max_tpr_percent: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            tpr_percent = float(row["target_tpr_percent"])
            if tpr_percent <= max_tpr_percent:
                rows.append(
                    {
                        "dataset": row["dataset"],
                        "target_tpr_percent": tpr_percent,
                        "distance_threshold": float(row["distance_threshold"]),
                    }
                )
    return rows


def build_threshold_specs(args: argparse.Namespace) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for value in parse_thresholds(args.fixed_distance_thresholds):
        specs.append({"source": "fixed", "label": f"fixed_{value:g}", "distance_threshold": value, "category_thresholds": None})

    train_rows = read_threshold_rows(Path(args.train_thresholds_csv), args.max_threshold_tpr_percent)
    for row in sorted(train_rows, key=lambda r: float(r["target_tpr_percent"])):
        value = float(row["distance_threshold"])
        specs.append({"source": "train_country", "label": f"train_{value:g}", "distance_threshold": value, "category_thresholds": None})

    test_rows = read_threshold_rows(Path(args.test_thresholds_csv), args.max_threshold_tpr_percent)
    by_tpr: dict[float, dict[str, float]] = {}
    for row in test_rows:
        by_tpr.setdefault(float(row["target_tpr_percent"]), {})[str(row["dataset"])] = float(row["distance_threshold"])
    for tpr_percent in sorted(by_tpr):
        values = list(by_tpr[tpr_percent].values())
        specs.append(
            {
                "source": "test_category",
                "label": f"test_category_tpr{tpr_percent:g}",
                "distance_threshold": float(np.mean(values)),
                "distance_threshold_min": float(np.min(values)),
                "distance_threshold_max": float(np.max(values)),
                "category_thresholds": by_tpr[tpr_percent],
            }
        )
    return specs


def compute_pair_matches(
    rgb_path: Path,
    nir_path: Path,
    extractor: SuperPoint,
    hcnet: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, object]:
    sp_kp_rgb, raw_rgb = extract_superpoint_keypoints(rgb_path, extractor, device, args.resize, args.filter_border)
    sp_kp_nir, raw_nir = extract_superpoint_keypoints(nir_path, extractor, device, args.resize, args.filter_border)

    rgb_image = load_vis_grayscale_uint8(rgb_path, args.hcnet_grayscale)
    nir_image = load_nir_grayscale_uint8(nir_path, args.hcnet_grayscale)
    rgb_patches, rgb_valid = crop_patches(rgb_image, sp_kp_rgb, args.patch_size, VIS_MEAN)
    nir_patches, nir_valid = crop_patches(nir_image, sp_kp_nir, args.patch_size, NIR_MEAN)
    kp_rgb = sp_kp_rgb[rgb_valid.to(sp_kp_rgb.device)].to(device)
    kp_nir = sp_kp_nir[nir_valid.to(sp_kp_nir.device)].to(device)

    desc_rgb = describe_patches(hcnet, rgb_patches, "vis", device, args.descriptor_batch_size)
    desc_nir = describe_patches(hcnet, nir_patches, "nir", device, args.descriptor_batch_size)
    matches = match_descriptors(desc_rgb, desc_nir, args.match_backend)
    if len(matches):
        desc_distances = (desc_rgb[matches[:, 0]] - desc_nir[matches[:, 1]]).pow(2).sum(dim=1)
        pixel_errors = torch.linalg.norm(kp_rgb[matches[:, 0]] - kp_nir[matches[:, 1]], dim=1)
    else:
        desc_distances = torch.empty((0,), dtype=torch.float32, device=device)
        pixel_errors = torch.empty((0,), dtype=torch.float32, device=device)

    return {
        "category": rgb_path.parent.name,
        "raw_rgb": raw_rgb,
        "raw_nir": raw_nir,
        "num_keypoints_rgb": int(kp_rgb.shape[0]),
        "num_keypoints_nir": int(kp_nir.shape[0]),
        "num_unfiltered_matches": int(matches.shape[0]),
        "desc_distances": desc_distances.detach().cpu().numpy(),
        "pixel_errors": pixel_errors.detach().cpu().numpy(),
    }


def summarize_spec(pair_data: list[dict[str, object]], spec: dict[str, object], mma_thresholds: list[float]) -> dict[str, object]:
    per_pair: list[dict[str, float]] = []
    total_matches = 0
    total_correct = {threshold: 0 for threshold in mma_thresholds}
    medians = []

    category_thresholds = spec.get("category_thresholds")
    for item in pair_data:
        threshold = category_thresholds[item["category"]] if category_thresholds is not None else spec["distance_threshold"]
        distances = item["desc_distances"]
        errors = item["pixel_errors"][distances <= threshold]
        total_matches += len(errors)
        if len(errors):
            medians.append(float(np.median(errors)))
        pair_row: dict[str, float] = {"num_matches": float(len(errors))}
        for mma_threshold in mma_thresholds:
            correct = int(np.sum(errors <= mma_threshold))
            total_correct[mma_threshold] += correct
            pair_row[f"mma@{mma_threshold:g}"] = correct / len(errors) if len(errors) else float("nan")
        per_pair.append(pair_row)

    row: dict[str, object] = {
        "source": spec["source"],
        "label": spec["label"],
        "distance_threshold": spec["distance_threshold"],
        "distance_threshold_min": spec.get("distance_threshold_min", spec["distance_threshold"]),
        "distance_threshold_max": spec.get("distance_threshold_max", spec["distance_threshold"]),
        "average_matches": mean_or_nan([r["num_matches"] for r in per_pair]),
        "total_matches": total_matches,
        "median_of_median_pixel_error": float(np.median(medians)) if medians else float("nan"),
    }
    for mma_threshold in mma_thresholds:
        row[f"micro_mma@{mma_threshold:g}"] = total_correct[mma_threshold] / total_matches if total_matches else float("nan")
        row[f"macro_mma@{mma_threshold:g}"] = mean_or_nan([r[f"mma@{mma_threshold:g}"] for r in per_pair])
    return row


def print_summary(rows: list[dict[str, object]], mma_thresholds: list[float]) -> None:
    columns = ["label", "average_matches"] + [f"macro_mma@{t:g}" for t in mma_thresholds]
    print("\nThreshold sweep summary")
    print(" ".join(f"{c:>14}" for c in columns))
    print("-" * (15 * len(columns)))
    for row in rows:
        values = [str(row["label"]), f"{float(row['average_matches']):.3f}"]
        values += [f"{float(row[f'macro_mma@{t:g}']):.6f}" for t in mma_thresholds]
        print(" ".join(f"{v:>14}" for v in values))


def main() -> None:
    args = parse_args()
    mma_thresholds = parse_thresholds(args.thresholds)
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    pairs = discover_pairs(args)
    specs = build_threshold_specs(args)
    print_pairs(pairs)
    print(f"Selected device: {device}")
    print(f"SuperPoint NMS radius: {args.nms_radius}")
    print(f"Pixel MMA thresholds: {args.thresholds}")
    print(f"HC threshold specs: {len(specs)}")
    if not pairs:
        raise RuntimeError("No RGB/NIR pairs found.")

    extractor = SuperPoint(max_num_keypoints=args.max_num_keypoints, nms_radius=args.nms_radius).eval().to(device)
    hcnet = load_hcnet_model(Path(args.hcnet_project), Path(args.hcnet_config), Path(args.hcnet_weights), device)

    pair_data = []
    for i, (rgb_path, nir_path) in enumerate(pairs, 1):
        print(f"[{i}/{len(pairs)}] {rgb_path.name} <-> {nir_path.name}")
        pair_data.append(compute_pair_matches(rgb_path, nir_path, extractor, hcnet, device, args))

    rows = [summarize_spec(pair_data, spec, mma_thresholds) for spec in specs]
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote CSV: {output_csv}")
    print_summary(rows, mma_thresholds)


if __name__ == "__main__":
    main()
