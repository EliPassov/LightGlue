#!/usr/bin/env python3
"""Measure how many SuperPoint+LightGlue matches are also selected by HC-Net.

Example:
python experiments/vis_nir_lg_hcnet_overlap.py --data_root /media/eli/storage2/Datasets/visnir_raw --exclude_categories country --limit_per_category 20 --max_num_keypoints 200 --filter_border 32 --output_csv outputs/vis_nir_lg_hcnet_overlap.csv
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

from lightglue import LightGlue, SuperPoint  # noqa: E402
from lightglue.utils import rbd  # noqa: E402
from vis_nir_common import discover_pairs, extract_superpoint_features, mean_or_nan, median_or_nan, parse_thresholds, print_pairs  # noqa: E402
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
    parser = argparse.ArgumentParser(description="Check exact overlap between SP+LG matches and SP+HCNet matches.")
    parser.add_argument("--data_root", default="/media/eli/storage2/Datasets/visnir_raw", help="Dataset root directory.")
    parser.add_argument("--category", default=None, help="Optional category subfolder, e.g. country.")
    parser.add_argument("--pairs_file", default=None, help="Optional text file with one 'rgb_path nir_path' pair per line.")
    parser.add_argument("--rgb_suffix", default="_rgb.tiff")
    parser.add_argument("--nir_suffix", default="_nir.tiff")
    parser.add_argument("--max_num_keypoints", type=int, default=2048)
    parser.add_argument("--nms_radius", type=int, default=4, help="SuperPoint NMS radius in pixels.")
    parser.add_argument("--resize", type=int, default=None, help="Optional resize value passed to SuperPoint extractor.extract.")
    parser.add_argument("--thresholds", default="1,3,5,10,16,32,48,64")
    parser.add_argument("--output_csv", default="outputs/vis_nir_lg_hcnet_overlap.csv", help="Path to write per-pair CSV results.")
    parser.add_argument("--device", default=None, help="Device to use. Defaults to cuda when available, otherwise cpu.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--limit_per_category", type=int, default=None, help="When scanning all categories, keep this many pairs per first-level category.")
    parser.add_argument("--exclude_categories", default=None, help="Comma-separated category names to skip, e.g. country.")
    parser.add_argument("--filter_border", type=int, default=32, help="Drop SuperPoint keypoints within this many pixels of the image border before matching.")
    parser.add_argument("--hcnet_project", default="/media/eli/storage2/Projects/multisensor_yossi")
    parser.add_argument("--hcnet_config", default="/media/eli/storage2/Projects/tensorboard_files/vis_nir_result_backup/weights_and_config/norm_old_cond_in_gelu_hyp_embedding_24x48_patience_3_config.yaml")
    parser.add_argument("--hcnet_weights", default="/media/eli/storage2/Projects/tensorboard_files/vis_nir_result_backup/weights_and_config/weights.pth")
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--descriptor_batch_size", type=int, default=512)
    parser.add_argument("--hcnet_grayscale", choices=["cv2", "pil"], default="cv2", help="Grayscale conversion for HC-Net patches. cv2 matches the HC-Net dataset creation path.")
    parser.add_argument("--match_backend", choices=["mnn", "lg_filter", "compare"], default="lg_filter", help="HC descriptor matcher.")
    return parser.parse_args()


def as_pair_set(matches: torch.Tensor) -> set[tuple[int, int]]:
    return {(int(i), int(j)) for i, j in matches.detach().cpu().tolist()}


def add_subset_metrics(row: dict[str, object], name: str, pairs: set[tuple[int, int]], kp_rgb: torch.Tensor, kp_nir: torch.Tensor, thresholds: list[float], device: torch.device) -> None:
    row[f"{name}_matches"] = len(pairs)
    if pairs:
        pair_tensor = torch.tensor(sorted(pairs), dtype=torch.long, device=device)
        errors = torch.linalg.norm(kp_rgb[pair_tensor[:, 0]] - kp_nir[pair_tensor[:, 1]], dim=1).detach().cpu().numpy()
        row[f"{name}_median_pixel_error"] = float(np.median(errors))
        row[f"{name}_mean_pixel_error"] = float(np.mean(errors))
    else:
        errors = np.empty((0,), dtype=np.float32)
        row[f"{name}_median_pixel_error"] = float("nan")
        row[f"{name}_mean_pixel_error"] = float("nan")

    for threshold in thresholds:
        suffix = f"{threshold:g}"
        correct = int(np.sum(errors <= threshold)) if len(errors) else 0
        row[f"{name}_correct@{suffix}"] = correct
        row[f"{name}_mma@{suffix}"] = correct / len(errors) if len(errors) else float("nan")


def evaluate_pair(
    rgb_path: Path,
    nir_path: Path,
    extractor: SuperPoint,
    matcher: LightGlue,
    hcnet: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    thresholds: list[float],
) -> dict[str, object]:
    with torch.inference_mode():
        feats0, raw_num_keypoints_rgb = extract_superpoint_features(rgb_path, extractor, device, args.resize, args.filter_border)
        feats1, raw_num_keypoints_nir = extract_superpoint_features(nir_path, extractor, device, args.resize, args.filter_border)
        lg_out = matcher({"image0": feats0, "image1": feats1})

    feats0, feats1, lg_out = [rbd(x) for x in [feats0, feats1, lg_out]]
    kp_rgb = feats0["keypoints"]
    kp_nir = feats1["keypoints"]
    lg_matches = lg_out["matches"]

    rgb_image = load_vis_grayscale_uint8(rgb_path, args.hcnet_grayscale)
    nir_image = load_nir_grayscale_uint8(nir_path, args.hcnet_grayscale)
    rgb_patches, rgb_valid = crop_patches(rgb_image, kp_rgb, args.patch_size, VIS_MEAN)
    nir_patches, nir_valid = crop_patches(nir_image, kp_nir, args.patch_size, NIR_MEAN)

    desc_rgb = describe_patches(hcnet, rgb_patches, "vis", device, args.descriptor_batch_size)
    desc_nir = describe_patches(hcnet, nir_patches, "nir", device, args.descriptor_batch_size)
    hc_matches_local = match_descriptors(desc_rgb, desc_nir, args.match_backend)

    rgb_valid = rgb_valid.to(hc_matches_local.device)
    nir_valid = nir_valid.to(hc_matches_local.device)
    if len(hc_matches_local):
        hc_matches = torch.stack([rgb_valid[hc_matches_local[:, 0]], nir_valid[hc_matches_local[:, 1]]], dim=1)
    else:
        hc_matches = torch.empty((0, 2), dtype=torch.long, device=device)

    lg_set = as_pair_set(lg_matches)
    hc_set = as_pair_set(hc_matches)
    overlap = lg_set & hc_set
    lg_only = lg_set - hc_set
    hc_only = hc_set - lg_set

    num_lg_matches = len(lg_set)
    num_hc_matches = len(hc_set)
    num_overlap = len(overlap)
    lg_covered_by_hc = num_overlap / num_lg_matches if num_lg_matches else float("nan")
    hc_covered_by_lg = num_overlap / num_hc_matches if num_hc_matches else float("nan")

    row: dict[str, object] = {
        "category": rgb_path.parent.name,
        "rgb_path": str(rgb_path),
        "nir_path": str(nir_path),
        "num_superpoint_keypoints_rgb": raw_num_keypoints_rgb,
        "num_superpoint_keypoints_nir": raw_num_keypoints_nir,
        "num_keypoints_rgb": int(kp_rgb.shape[0]),
        "num_keypoints_nir": int(kp_nir.shape[0]),
        "num_lg_matches": num_lg_matches,
        "num_hc_matches": num_hc_matches,
        "num_overlap_matches": num_overlap,
        "lg_covered_by_hc": lg_covered_by_hc,
        "hc_covered_by_lg": hc_covered_by_lg,
        "hcnet_grayscale": args.hcnet_grayscale,
        "hc_match_backend": args.match_backend,
    }
    add_subset_metrics(row, "lg_all", lg_set, kp_rgb, kp_nir, thresholds, device)
    add_subset_metrics(row, "hc_all", hc_set, kp_rgb, kp_nir, thresholds, device)
    add_subset_metrics(row, "overlap", overlap, kp_rgb, kp_nir, thresholds, device)
    add_subset_metrics(row, "lg_only", lg_only, kp_rgb, kp_nir, thresholds, device)
    add_subset_metrics(row, "hc_only", hc_only, kp_rgb, kp_nir, thresholds, device)
    return row


def print_subset_aggregate(rows: list[dict[str, object]], name: str, thresholds: list[float]) -> None:
    total_matches = sum(int(r[f"{name}_matches"]) for r in rows)
    print(f"\n{name}")
    print(f"average_matches: {mean_or_nan([float(r[f'{name}_matches']) for r in rows]):.3f}")
    print(f"median_of_median_pixel_error: {median_or_nan([float(r[f'{name}_median_pixel_error']) for r in rows]):.6f}")
    for threshold in thresholds:
        suffix = f"{threshold:g}"
        total_correct = sum(int(r[f"{name}_correct@{suffix}"]) for r in rows)
        micro_mma = total_correct / total_matches if total_matches else float("nan")
        macro_mma = mean_or_nan([float(r[f"{name}_mma@{suffix}"]) for r in rows])
        print(f"micro_mma@{suffix}: {micro_mma:.6f}  macro_mma@{suffix}: {macro_mma:.6f}")


def print_aggregate(rows: list[dict[str, object]], thresholds: list[float]) -> None:
    total_lg = sum(int(r["num_lg_matches"]) for r in rows)
    total_hc = sum(int(r["num_hc_matches"]) for r in rows)
    total_overlap = sum(int(r["num_overlap_matches"]) for r in rows)
    print("\nAggregate overlap")
    print(f"total_pairs: {len(rows)}")
    print(f"average_rgb_keypoints: {mean_or_nan([float(r['num_keypoints_rgb']) for r in rows]):.3f}")
    print(f"average_nir_keypoints: {mean_or_nan([float(r['num_keypoints_nir']) for r in rows]):.3f}")
    print(f"average_lg_matches: {mean_or_nan([float(r['num_lg_matches']) for r in rows]):.3f}")
    print(f"average_hc_matches: {mean_or_nan([float(r['num_hc_matches']) for r in rows]):.3f}")
    print(f"average_overlap_matches: {mean_or_nan([float(r['num_overlap_matches']) for r in rows]):.3f}")
    print(f"micro_lg_covered_by_hc: {total_overlap / total_lg if total_lg else float('nan'):.6f}")
    print(f"micro_hc_covered_by_lg: {total_overlap / total_hc if total_hc else float('nan'):.6f}")
    print(f"macro_lg_covered_by_hc: {mean_or_nan([float(r['lg_covered_by_hc']) for r in rows]):.6f}")
    print(f"macro_hc_covered_by_lg: {mean_or_nan([float(r['hc_covered_by_lg']) for r in rows]):.6f}")
    for name in ["lg_all", "hc_all", "overlap", "lg_only", "hc_only"]:
        print_subset_aggregate(rows, name, thresholds)


def main() -> None:
    args = parse_args()
    thresholds = parse_thresholds(args.thresholds)
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    pairs = discover_pairs(args)
    print_pairs(pairs)
    print(f"Selected device: {device}")
    print(f"Resize: {args.resize}")
    print(f"Filter border: {args.filter_border}")
    print(f"SuperPoint NMS radius: {args.nms_radius}")
    print(f"Thresholds: {args.thresholds}")
    print(f"HC-Net grayscale: {args.hcnet_grayscale}")
    print(f"HC match backend: {args.match_backend}")
    if not pairs:
        raise RuntimeError("No RGB/NIR pairs found.")

    extractor = SuperPoint(max_num_keypoints=args.max_num_keypoints, nms_radius=args.nms_radius).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)
    hcnet = load_hcnet_model(Path(args.hcnet_project), Path(args.hcnet_config), Path(args.hcnet_weights), device)

    rows = []
    for i, (rgb_path, nir_path) in enumerate(pairs, 1):
        print(f"[{i}/{len(pairs)}] {rgb_path.name} <-> {nir_path.name}")
        rows.append(evaluate_pair(rgb_path, nir_path, extractor, matcher, hcnet, device, args, thresholds))

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote CSV: {output_csv}")
    print_aggregate(rows, thresholds)


if __name__ == "__main__":
    main()
