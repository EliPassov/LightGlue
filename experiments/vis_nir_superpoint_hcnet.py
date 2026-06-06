#!/usr/bin/env python3
"""VIS-NIR matching with SuperPoint keypoints and HC-Net descriptors.

Examples:
python experiments/vis_nir_superpoint_hcnet.py --data_root /media/eli/storage2/Datasets/visnir_raw --category country --limit 20 --output_csv outputs/vis_nir_country_sp_hcnet.csv
python experiments/vis_nir_superpoint_hcnet.py --data_root /media/eli/storage2/Datasets/visnir_raw --limit 20
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import cv2
from PIL import Image
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

from lightglue import SuperPoint  # noqa: E402
from lightglue.lightglue import filter_matches  # noqa: E402
from vis_nir_common import (  # noqa: E402
    discover_pairs,
    extract_superpoint_keypoints,
    mean_or_nan,
    median_or_nan,
    parse_thresholds,
    print_pairs,
)

VIS_MEAN = 91.96518768436994
NIR_MEAN = 162.9802835004391


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VIS-NIR matches with SuperPoint keypoints and HC-Net descriptors.")
    parser.add_argument("--data_root", default="/media/eli/storage2/Datasets/visnir_raw", help="Dataset root directory.")
    parser.add_argument("--category", default=None, help="Optional category subfolder, e.g. country.")
    parser.add_argument("--pairs_file", default=None, help="Optional text file with one 'rgb_path nir_path' pair per line.")
    parser.add_argument("--rgb_suffix", default="_rgb.tiff")
    parser.add_argument("--nir_suffix", default="_nir.tiff")
    parser.add_argument("--max_num_keypoints", type=int, default=2048)
    parser.add_argument("--nms_radius", type=int, default=4, help="SuperPoint NMS radius in pixels.")
    parser.add_argument("--resize", type=int, default=None, help="Optional resize value passed to SuperPoint extractor.extract.")
    parser.add_argument("--thresholds", default="1,3,5,10")
    parser.add_argument("--output_csv", default="outputs/vis_nir_superpoint_hcnet.csv", help="Path to write per-pair CSV results.")
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
    parser.add_argument("--hcnet_grayscale", choices=["cv2", "pil"], default="cv2", help="Grayscale conversion for HC-Net patches. cv2 matches the HC-Net dataset creation path.")
    parser.add_argument("--match_backend", choices=["mnn", "lg_filter", "compare"], default="lg_filter", help="HC descriptor matcher. compare runs both MNN and LightGlue filter logic and checks equality.")
    parser.add_argument("--hc_distance_threshold", type=float, default=None, help="Optional HC-Net squared-L2 descriptor distance threshold. Matches above it are rejected.")
    return parser.parse_args()


def install_hcnet_import_stubs() -> None:
    transformer_backbone = types.ModuleType("network.transformer_backbone")
    transformer_backbone.CUSTOM_VIT = None
    transformer_backbone.CUSTOM_SWIN = None
    sys.modules.setdefault("network.transformer_backbone", transformer_backbone)

    termcolor = types.ModuleType("termcolor")
    termcolor.colored = lambda text, *args, **kwargs: text
    sys.modules.setdefault("termcolor", termcolor)

    pytorch_msssim = types.ModuleType("pytorch_msssim")
    pytorch_msssim.ssim = lambda *args, **kwargs: None
    sys.modules.setdefault("pytorch_msssim", pytorch_msssim)


def load_hcnet_model(project: Path, config_path: Path, weights_path: Path, device: torch.device) -> nn.Module:
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))
    install_hcnet_import_stubs()

    from network.my_classes import MetricLearningCnn

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    cfg_model = config["model"]
    cfg_aux = config.get("aux_loss", {})
    cfg_data = config["data"]
    hyper_block_type = cfg_model.get("hyper_block_type", None)
    if isinstance(hyper_block_type, str) and hyper_block_type.lower() == "none":
        hyper_block_type = None

    model = MetricLearningCnn(
        cfg_model.get("cnn_mode", "PairwiseSymmetric"),
        cfg_model.get("dropout", 0.5),
        hyper_block_type,
        getattr(nn, cfg_model.get("non_linearity", "GELU")),
        cfg_model.get("ll_normalization", "cond_in"),
        return_layers=cfg_model.get("return_layers", [7]),
        store_layer=None if cfg_aux.get("type", None) is None else cfg_aux.get("aux_layer", None),
        external_embeddings_dim=512 if cfg_data.get("use_embeddings", False) else None,
        backbone=cfg_model.get("backbone", "CNN"),
        skip_connections=cfg_model.get("skip_connections", False),
        hypernet_on_conv=cfg_model.get("hypernet_on_conv", False),
        hypernet_bias=cfg_model.get("hypernet_bias", True),
        shallow_hypernet_config=cfg_model.get("shallow_hypernet", None),
        film_mode=cfg_model.get("film_mode", False),
    )

    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval().to(device)


def load_vis_grayscale_uint8(path: Path, mode: str) -> np.ndarray:
    if mode == "cv2":
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise IOError(f"Could not read image at {path}.")
        return image
    if mode == "pil":
        img = Image.open(path)
        if img.mode == "L":
            return np.array(img, dtype=np.uint8)
        rgb = np.array(img.convert("RGB"), dtype=np.uint8).astype(np.float32)
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        return np.floor(gray + 0.5).astype(np.uint8)
    raise ValueError(f"Unsupported grayscale mode: {mode}")


def load_nir_grayscale_uint8(path: Path, mode: str) -> np.ndarray:
    if mode == "cv2":
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise IOError(f"Could not read image at {path}.")
        return image
    if mode == "pil":
        return np.array(Image.open(path).convert("L"), dtype=np.uint8)
    raise ValueError(f"Unsupported grayscale mode: {mode}")


def crop_patches(image: np.ndarray, keypoints: torch.Tensor, patch_size: int, mean: float) -> tuple[torch.Tensor, torch.Tensor]:
    half = patch_size // 2
    h, w = image.shape[:2]
    points = keypoints.detach().cpu().numpy()
    patches, valid_indices = [], []
    for i, (x, y) in enumerate(points):
        cx, cy = int(round(float(x))), int(round(float(y)))
        x0, x1 = cx - half, cx + half
        y0, y1 = cy - half, cy + half
        if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
            continue
        patch = image[y0:y1, x0:x1].astype(np.float32)
        patch = (patch - mean) * (2.0 / 255.0)
        patches.append(patch[None])
        valid_indices.append(i)

    if not patches:
        return torch.empty((0, 1, patch_size, patch_size), dtype=torch.float32), torch.empty((0,), dtype=torch.long)
    return torch.from_numpy(np.stack(patches)), torch.tensor(valid_indices, dtype=torch.long)


def describe_patches(model: nn.Module, patches: torch.Tensor, modality: str, device: torch.device, batch_size: int) -> torch.Tensor:
    if len(patches) == 0:
        return torch.empty((0, 128), dtype=torch.float32, device=device)

    descs = []
    with torch.inference_mode():
        for start in range(0, len(patches), batch_size):
            batch = patches[start : start + batch_size].to(device)
            dummy = torch.zeros_like(batch)
            if modality == "vis":
                emb = model(batch, dummy)["Emb1"]
            elif modality == "nir":
                emb = model(dummy, batch)["Emb2"]
            else:
                raise ValueError(f"Unsupported modality: {modality}")
            descs.append(F.normalize(emb, dim=1, p=2))
    return torch.cat(descs, dim=0)


def mutual_nearest_matches(desc0: torch.Tensor, desc1: torch.Tensor) -> torch.Tensor:
    if len(desc0) == 0 or len(desc1) == 0:
        return torch.empty((0, 2), dtype=torch.long, device=desc0.device)
    similarity = desc0 @ desc1.T
    nn01 = similarity.argmax(dim=1)
    nn10 = similarity.argmax(dim=0)
    idx0 = torch.arange(desc0.shape[0], device=desc0.device)
    mutual = nn10[nn01] == idx0
    return torch.stack([idx0[mutual], nn01[mutual]], dim=1)


def lg_filter_matches(desc0: torch.Tensor, desc1: torch.Tensor) -> torch.Tensor:
    if len(desc0) == 0 or len(desc1) == 0:
        return torch.empty((0, 2), dtype=torch.long, device=desc0.device)
    similarity = desc0 @ desc1.T
    scores = similarity.new_full((1, desc0.shape[0] + 1, desc1.shape[0] + 1), -1e9)
    scores[0, :-1, :-1] = similarity
    m0, _, _, _ = filter_matches(scores, th=0.0)
    valid = m0[0] > -1
    return torch.stack([torch.where(valid)[0], m0[0][valid]], dim=1)


def match_descriptors(desc0: torch.Tensor, desc1: torch.Tensor, backend: str) -> torch.Tensor:
    if backend == "mnn":
        return mutual_nearest_matches(desc0, desc1)
    if backend == "lg_filter":
        return lg_filter_matches(desc0, desc1)
    if backend == "compare":
        matches_mnn = mutual_nearest_matches(desc0, desc1)
        matches_lg = lg_filter_matches(desc0, desc1)
        if matches_mnn.shape != matches_lg.shape or not torch.equal(matches_mnn, matches_lg):
            raise RuntimeError(f"MNN and LightGlue-filter matches differ: {matches_mnn.shape} vs {matches_lg.shape}")
        return matches_lg
    raise ValueError(f"Unsupported match backend: {backend}")


def filter_matches_by_distance(desc0: torch.Tensor, desc1: torch.Tensor, matches: torch.Tensor, distance_threshold: float | None) -> torch.Tensor:
    if distance_threshold is None or len(matches) == 0:
        return matches
    distances = (desc0[matches[:, 0]] - desc1[matches[:, 1]]).pow(2).sum(dim=1)
    return matches[distances <= distance_threshold]


def evaluate_pair(
    rgb_path: Path,
    nir_path: Path,
    extractor: SuperPoint,
    hcnet: nn.Module,
    device: torch.device,
    resize: int | None,
    thresholds: list[float],
    patch_size: int,
    descriptor_batch_size: int,
    filter_border: int,
    grayscale_mode: str,
    match_backend: str,
    distance_threshold: float | None,
) -> dict[str, object]:
    sp_kp_rgb, raw_num_keypoints_rgb = extract_superpoint_keypoints(rgb_path, extractor, device, resize, filter_border)
    sp_kp_nir, raw_num_keypoints_nir = extract_superpoint_keypoints(nir_path, extractor, device, resize, filter_border)

    rgb_image = load_vis_grayscale_uint8(rgb_path, grayscale_mode)
    nir_image = load_nir_grayscale_uint8(nir_path, grayscale_mode)
    rgb_patches, rgb_valid = crop_patches(rgb_image, sp_kp_rgb, patch_size, VIS_MEAN)
    nir_patches, nir_valid = crop_patches(nir_image, sp_kp_nir, patch_size, NIR_MEAN)
    kp_rgb = sp_kp_rgb[rgb_valid.to(sp_kp_rgb.device)].to(device)
    kp_nir = sp_kp_nir[nir_valid.to(sp_kp_nir.device)].to(device)

    desc_rgb = describe_patches(hcnet, rgb_patches, "vis", device, descriptor_batch_size)
    desc_nir = describe_patches(hcnet, nir_patches, "nir", device, descriptor_batch_size)
    matches = match_descriptors(desc_rgb, desc_nir, match_backend)
    matches = filter_matches_by_distance(desc_rgb, desc_nir, matches, distance_threshold)
    num_matches = int(matches.shape[0])

    row: dict[str, object] = {
        "category": rgb_path.parent.name,
        "rgb_path": str(rgb_path),
        "nir_path": str(nir_path),
        "num_superpoint_keypoints_rgb": raw_num_keypoints_rgb,
        "num_superpoint_keypoints_nir": raw_num_keypoints_nir,
        "num_keypoints_rgb": int(kp_rgb.shape[0]),
        "num_keypoints_nir": int(kp_nir.shape[0]),
        "num_matches": num_matches,
        "hcnet_grayscale": grayscale_mode,
        "match_backend": match_backend,
        "hc_distance_threshold": distance_threshold if distance_threshold is not None else "",
    }

    if num_matches:
        points0 = kp_rgb[matches[:, 0]]
        points1 = kp_nir[matches[:, 1]]
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
    print(f"average_raw_superpoint_rgb_keypoints: {mean_or_nan([float(r['num_superpoint_keypoints_rgb']) for r in rows]):.3f}")
    print(f"average_raw_superpoint_nir_keypoints: {mean_or_nan([float(r['num_superpoint_keypoints_nir']) for r in rows]):.3f}")
    print(f"average_matches: {mean_or_nan([float(r['num_matches']) for r in rows]):.3f}")
    for threshold in thresholds:
        suffix = f"{threshold:g}"
        print(f"average_mma@{suffix}: {mean_or_nan([float(r[f'mma@{suffix}']) for r in rows]):.6f}")
    print(f"median_of_median_pixel_error: {median_or_nan([float(r['median_pixel_error']) for r in rows]):.6f}")


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
    print(f"HC-Net grayscale: {args.hcnet_grayscale}")
    print(f"Match backend: {args.match_backend}")
    print(f"HC distance threshold: {args.hc_distance_threshold}")
    print(f"HC-Net weights: {args.hcnet_weights}")
    if not pairs:
        raise RuntimeError("No RGB/NIR pairs found.")

    extractor = SuperPoint(max_num_keypoints=args.max_num_keypoints, nms_radius=args.nms_radius).eval().to(device)
    hcnet = load_hcnet_model(Path(args.hcnet_project), Path(args.hcnet_config), Path(args.hcnet_weights), device)

    rows = []
    for i, (rgb_path, nir_path) in enumerate(pairs, 1):
        print(f"[{i}/{len(pairs)}] {rgb_path.name} <-> {nir_path.name}")
        rows.append(evaluate_pair(rgb_path, nir_path, extractor, hcnet, device, args.resize, thresholds, args.patch_size, args.descriptor_batch_size, args.filter_border, args.hcnet_grayscale, args.match_backend, args.hc_distance_threshold))

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
