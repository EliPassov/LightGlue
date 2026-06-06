#!/usr/bin/env python3
"""VIS-NIR matching with the official SuperPoint + LightGlue pipeline.

Examples:
python experiments/vis_nir_superpoint_lightglue.py --data_root /media/eli/storage2/Datasets/visnir_raw --category country --limit 20 --output_csv outputs/vis_nir_country_sp_lightglue.csv
python experiments/vis_nir_superpoint_lightglue.py --data_root /media/eli/storage2/Datasets/visnir_raw --limit 20
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

from lightglue import LightGlue, SuperPoint  # noqa: E402
from lightglue.utils import rbd  # noqa: E402
from vis_nir_common import (  # noqa: E402
    discover_pairs,
    extract_superpoint_features,
    mean_or_nan,
    median_or_nan,
    parse_thresholds,
    print_pairs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VIS-NIR image-pair matches with SuperPoint + LightGlue.")
    parser.add_argument("--data_root", default="/media/eli/storage2/Datasets/visnir_raw", help="Dataset root directory.")
    parser.add_argument("--category", default=None, help="Optional category subfolder, e.g. country.")
    parser.add_argument("--pairs_file", default=None, help="Optional text file with one 'rgb_path nir_path' pair per line.")
    parser.add_argument("--rgb_suffix", default="_rgb.tiff")
    parser.add_argument("--nir_suffix", default="_nir.tiff")
    parser.add_argument("--max_num_keypoints", type=int, default=2048)
    parser.add_argument("--nms_radius", type=int, default=4, help="SuperPoint NMS radius in pixels.")
    parser.add_argument("--resize", type=int, default=None, help="Optional resize value passed to extractor.extract; default disables resizing.")
    parser.add_argument("--thresholds", default="1,3,5,10")
    parser.add_argument("--output_csv", default="outputs/vis_nir_superpoint_lightglue.csv", help="Path to write per-pair CSV results.")
    parser.add_argument("--device", default=None, help="Device to use. Defaults to cuda when available, otherwise cpu.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--limit_per_category", type=int, default=None, help="When scanning all categories, keep this many pairs per first-level category.")
    parser.add_argument("--exclude_categories", default=None, help="Comma-separated category names to skip, e.g. country.")
    parser.add_argument("--filter_border", type=int, default=0, help="Drop keypoints within this many pixels of the image border before matching.")
    return parser.parse_args()


def evaluate_pair(
    rgb_path: Path,
    nir_path: Path,
    extractor: SuperPoint,
    matcher: LightGlue,
    device: torch.device,
    resize: int | None,
    thresholds: list[float],
    filter_border: int,
) -> dict[str, object]:
    with torch.inference_mode():
        feats0, raw_num_keypoints_rgb = extract_superpoint_features(rgb_path, extractor, device, resize, filter_border)
        feats1, raw_num_keypoints_nir = extract_superpoint_features(nir_path, extractor, device, resize, filter_border)
        matches01 = matcher({"image0": feats0, "image1": feats1})

    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
    matches = matches01["matches"]
    num_matches = int(matches.shape[0])
    num_keypoints_rgb = int(feats0["keypoints"].shape[0])
    num_keypoints_nir = int(feats1["keypoints"].shape[0])

    row: dict[str, object] = {
        "category": rgb_path.parent.name,
        "rgb_path": str(rgb_path),
        "nir_path": str(nir_path),
        "num_superpoint_keypoints_rgb": raw_num_keypoints_rgb,
        "num_superpoint_keypoints_nir": raw_num_keypoints_nir,
        "num_keypoints_rgb": num_keypoints_rgb,
        "num_keypoints_nir": num_keypoints_nir,
        "num_matches": num_matches,
    }

    if num_matches:
        points0 = feats0["keypoints"][matches[:, 0]]
        points1 = feats1["keypoints"][matches[:, 1]]
        errors = torch.linalg.norm(points0 - points1, dim=1).detach().cpu().numpy()
        median_error = float(np.median(errors))
        mean_error = float(np.mean(errors))
    else:
        errors = np.empty((0,), dtype=np.float32)
        median_error = float("nan")
        mean_error = float("nan")

    for threshold in thresholds:
        suffix = f"{threshold:g}"
        correct = int(np.sum(errors <= threshold)) if num_matches else 0
        row[f"correct@{suffix}"] = correct
        row[f"mma@{suffix}"] = correct / num_matches if num_matches else float("nan")

    row["median_pixel_error"] = median_error
    row["mean_pixel_error"] = mean_error
    return row
def print_aggregate(rows: list[dict[str, object]], thresholds: list[float]) -> None:
    print("\nAggregate results")
    print(f"total_pairs: {len(rows)}")
    print(f"average_rgb_keypoints: {mean_or_nan([float(r['num_keypoints_rgb']) for r in rows]):.3f}")
    print(f"average_nir_keypoints: {mean_or_nan([float(r['num_keypoints_nir']) for r in rows]):.3f}")
    if "num_superpoint_keypoints_rgb" in rows[0]:
        print(f"average_raw_superpoint_rgb_keypoints: {mean_or_nan([float(r['num_superpoint_keypoints_rgb']) for r in rows]):.3f}")
        print(f"average_raw_superpoint_nir_keypoints: {mean_or_nan([float(r['num_superpoint_keypoints_nir']) for r in rows]):.3f}")
    print(f"average_matches: {mean_or_nan([float(r['num_matches']) for r in rows]):.3f}")
    for threshold in thresholds:
        suffix = f"{threshold:g}"
        print(f"average_mma@{suffix}: {mean_or_nan([float(r[f'mma@{suffix}']) for r in rows]):.6f}")
    print(
        "median_of_median_pixel_error: "
        f"{median_or_nan([float(r['median_pixel_error']) for r in rows]):.6f}"
    )


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
    if not pairs:
        raise RuntimeError("No RGB/NIR pairs found.")

    extractor = SuperPoint(max_num_keypoints=args.max_num_keypoints, nms_radius=args.nms_radius).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)

    rows = []
    for i, (rgb_path, nir_path) in enumerate(pairs, 1):
        print(f"[{i}/{len(pairs)}] {rgb_path.name} <-> {nir_path.name}")
        rows.append(evaluate_pair(rgb_path, nir_path, extractor, matcher, device, args.resize, thresholds, args.filter_border))

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote CSV: {output_csv}")
    print_aggregate(rows, thresholds)


if __name__ == "__main__":
    main()
