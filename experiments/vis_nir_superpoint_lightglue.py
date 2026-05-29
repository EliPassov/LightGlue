#!/usr/bin/env python3
"""VIS-NIR matching with the official SuperPoint + LightGlue pipeline.

Examples:
python experiments/vis_nir_superpoint_lightglue.py --data_root /media/eli/storage2/Datasets/visnir_raw --category country --limit 20 --output_csv outputs/vis_nir_country_sp_lightglue.csv
python experiments/vis_nir_superpoint_lightglue.py --data_root /media/eli/storage2/Datasets/visnir_raw --limit 20
"""

from __future__ import annotations

import argparse
import csv
import math
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
from lightglue.utils import load_image as lg_load_image  # noqa: E402
from lightglue.utils import rbd  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VIS-NIR image-pair matches with SuperPoint + LightGlue.")
    parser.add_argument("--data_root", default="/media/eli/storage2/Datasets/visnir_raw", help="Dataset root directory.")
    parser.add_argument("--category", default=None, help="Optional category subfolder, e.g. country.")
    parser.add_argument("--pairs_file", default=None, help="Optional text file with one 'rgb_path nir_path' pair per line.")
    parser.add_argument("--rgb_suffix", default="_rgb.tiff")
    parser.add_argument("--nir_suffix", default="_nir.tiff")
    parser.add_argument("--max_num_keypoints", type=int, default=2048)
    parser.add_argument("--resize", type=int, default=None, help="Optional resize value passed to extractor.extract; default disables resizing.")
    parser.add_argument("--thresholds", default="1,3,5,10")
    parser.add_argument("--output_csv", default="outputs/vis_nir_superpoint_lightglue.csv", help="Path to write per-pair CSV results.")
    parser.add_argument("--device", default=None, help="Device to use. Defaults to cuda when available, otherwise cpu.")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def selected_root(data_root: str, category: str | None) -> Path:
    root = Path(data_root)
    return root / category if category else root


def discover_pairs(args: argparse.Namespace) -> list[tuple[Path, Path]]:
    if args.pairs_file:
        pairs_path = Path(args.pairs_file)
        base = pairs_path.parent
        pairs = []
        with pairs_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) != 2:
                    raise ValueError(
                        f"{pairs_path}:{line_no} must contain: rgb_path nir_path"
                    )
                rgb, nir = (Path(parts[0]), Path(parts[1]))
                pairs.append(
                    (
                        rgb if rgb.is_absolute() else base / rgb,
                        nir if nir.is_absolute() else base / nir,
                    )
                )
    else:
        root = selected_root(args.data_root, args.category)
        if not root.exists():
            raise FileNotFoundError(f"Selected data directory does not exist: {root}")

        pairs = []
        for rgb_path in sorted(root.rglob(f"*{args.rgb_suffix}")):
            stem = rgb_path.name[: -len(args.rgb_suffix)]
            nir_path = rgb_path.with_name(f"{stem}{args.nir_suffix}")
            if nir_path.exists():
                pairs.append((rgb_path, nir_path))

    if args.limit is not None:
        pairs = pairs[: args.limit]
    return pairs


def pil_load_image(path: Path) -> torch.Tensor:
    from PIL import Image

    image = np.asarray(Image.open(path))
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    elif image.ndim == 3 and image.shape[2] > 3:
        image = image[..., :3]
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.float32)
        max_value = float(np.nanmax(image)) if image.size else 0.0
        image = image / max_value if max_value > 0 else image
    image = np.clip(image, 0.0, 1.0)
    return torch.from_numpy(image.transpose(2, 0, 1)).float()


def load_visnir_image(path: Path, device: torch.device) -> torch.Tensor:
    try:
        image = lg_load_image(path)
    except Exception:
        image = pil_load_image(path)
    return image.to(device)


def parse_thresholds(thresholds: str) -> list[float]:
    values = [float(t.strip()) for t in thresholds.split(",") if t.strip()]
    if not values:
        raise ValueError("--thresholds must contain at least one numeric value")
    return values


def mean_or_nan(values: list[float]) -> float:
    values = [v for v in values if not math.isnan(v)]
    return float(np.mean(values)) if values else float("nan")


def median_or_nan(values: list[float]) -> float:
    values = [v for v in values if not math.isnan(v)]
    return float(np.median(values)) if values else float("nan")


def evaluate_pair(
    rgb_path: Path,
    nir_path: Path,
    extractor: SuperPoint,
    matcher: LightGlue,
    device: torch.device,
    resize: int | None,
    thresholds: list[float],
) -> dict[str, object]:
    image0 = load_visnir_image(rgb_path, device)
    image1 = load_visnir_image(nir_path, device)

    with torch.inference_mode():
        feats0 = extractor.extract(image0, resize=resize)
        feats1 = extractor.extract(image1, resize=resize)
        matches01 = matcher({"image0": feats0, "image1": feats1})

    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
    matches = matches01["matches"]
    num_matches = int(matches.shape[0])
    num_keypoints_rgb = int(feats0["keypoints"].shape[0])
    num_keypoints_nir = int(feats1["keypoints"].shape[0])

    row: dict[str, object] = {
        "rgb_path": str(rgb_path),
        "nir_path": str(nir_path),
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


def print_pairs(pairs: list[tuple[Path, Path]]) -> None:
    print(f"Found {len(pairs)} RGB/NIR pairs")
    for rgb, nir in pairs[:5]:
        print(f"  {rgb}  {nir}")


def print_aggregate(rows: list[dict[str, object]], thresholds: list[float]) -> None:
    print("\nAggregate results")
    print(f"total_pairs: {len(rows)}")
    print(f"average_rgb_keypoints: {mean_or_nan([float(r['num_keypoints_rgb']) for r in rows]):.3f}")
    print(f"average_nir_keypoints: {mean_or_nan([float(r['num_keypoints_nir']) for r in rows]):.3f}")
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
    if not pairs:
        raise RuntimeError("No RGB/NIR pairs found.")

    extractor = SuperPoint(max_num_keypoints=args.max_num_keypoints).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)

    rows = []
    for i, (rgb_path, nir_path) in enumerate(pairs, 1):
        print(f"[{i}/{len(pairs)}] {rgb_path.name} <-> {nir_path.name}")
        rows.append(
            evaluate_pair(
                rgb_path,
                nir_path,
                extractor,
                matcher,
                device,
                args.resize,
                thresholds,
            )
        )

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
