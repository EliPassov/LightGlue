from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from lightglue.utils import load_image as lg_load_image
from lightglue.utils import rbd


def split_csv(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def selected_root(data_root: str, category: str | None) -> Path:
    root = Path(data_root)
    return root / category if category else root


def discover_pairs(args) -> list[tuple[Path, Path]]:
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
                    raise ValueError(f"{pairs_path}:{line_no} must contain: rgb_path nir_path")
                rgb, nir = (Path(parts[0]), Path(parts[1]))
                pairs.append((rgb if rgb.is_absolute() else base / rgb, nir if nir.is_absolute() else base / nir))
        return pairs[: args.limit] if args.limit is not None else pairs

    root = selected_root(args.data_root, args.category)
    if not root.exists():
        raise FileNotFoundError(f"Selected data directory does not exist: {root}")

    exclude_categories = split_csv(getattr(args, "exclude_categories", None))
    grouped: dict[str, list[tuple[Path, Path]]] = {}
    for rgb_path in sorted(root.rglob(f"*{args.rgb_suffix}")):
        stem = rgb_path.name[: -len(args.rgb_suffix)]
        nir_path = rgb_path.with_name(f"{stem}{args.nir_suffix}")
        if not nir_path.exists():
            continue
        rel = rgb_path.relative_to(root)
        group = args.category or (rel.parts[0] if len(rel.parts) > 1 else root.name)
        if group in exclude_categories:
            continue
        grouped.setdefault(group, []).append((rgb_path, nir_path))

    pairs = []
    for group in sorted(grouped):
        group_pairs = grouped[group]
        if args.limit_per_category is not None:
            group_pairs = group_pairs[: args.limit_per_category]
        pairs.extend(group_pairs)
    return pairs[: args.limit] if args.limit is not None else pairs


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


def filter_features_by_border(feats: dict, border: int) -> dict:
    if border <= 0:
        return feats
    keypoints = feats["keypoints"]
    width, height = feats["image_size"][0]
    centers = torch.round(keypoints[0])
    keep = (
        (centers[:, 0] >= border)
        & (centers[:, 0] <= width - border)
        & (centers[:, 1] >= border)
        & (centers[:, 1] <= height - border)
    )
    filtered = dict(feats)
    filtered["keypoints"] = feats["keypoints"][:, keep]
    filtered["keypoint_scores"] = feats["keypoint_scores"][:, keep]
    filtered["descriptors"] = feats["descriptors"][:, keep]
    return filtered


def extract_superpoint_features(path: Path, extractor, device: torch.device, resize: int | None, filter_border: int = 0) -> tuple[dict, int]:
    image = load_visnir_image(path, device)
    with torch.inference_mode():
        feats = extractor.extract(image, resize=resize)
    raw_num_keypoints = int(feats["keypoints"].shape[1])
    feats = filter_features_by_border(feats, filter_border)
    return feats, raw_num_keypoints


def extract_superpoint_keypoints(path: Path, extractor, device: torch.device, resize: int | None, filter_border: int = 0) -> tuple[torch.Tensor, int]:
    feats, raw_num_keypoints = extract_superpoint_features(path, extractor, device, resize, filter_border)
    return rbd(feats)["keypoints"], raw_num_keypoints


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


def print_pairs(pairs: list[tuple[Path, Path]]) -> None:
    print(f"Found {len(pairs)} RGB/NIR pairs")
    for rgb, nir in pairs[:5]:
        print(f"  {rgb}  {nir}")
