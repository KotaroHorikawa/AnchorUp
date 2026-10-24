"""Gate, PCA and Fourier diagnostics for an AnchorUp checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ANCHORS = ("nearest", "bilinear", "bicubic")
# Radial frequencies (cycles/pixel) where the mean anchor and correction spectra
# of the bilinear 25k model cross; they bound the low, mid and high bands (Figure S3).
BAND_CROSSINGS = (0.06510913314680558, 0.28941297527957593)
FIELDS = ("anchor", "learned", "correction", "pred")


def interpolate_anchor(features: torch.Tensor, size: tuple[int, int], mode: str) -> torch.Tensor:
    options = {} if mode == "nearest" else {"align_corners": False}
    return F.interpolate(features, size=size, mode=mode, **options)


def error_inner_products(anchor: torch.Tensor, learned: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Raw-feature error sums; do not average over channels or spatial elements."""
    # Subtract in the feature dtype, then accumulate in float64.
    anchor_error = (anchor - target).double().flatten()
    learned_error = (learned - target).double().flatten()
    return {
        "bb": float(anchor_error @ anchor_error),
        "bn": float(anchor_error @ learned_error),
        "nn": float(learned_error @ learned_error),
        "num_images": int(anchor.shape[0]),
    }


def alpha_summary(sums: dict[str, float], learned_alpha: float) -> dict[str, Any]:
    """Closed form alpha* and mean per-image SSE for alpha from 0 to 1 in steps of 0.05."""
    bb, bn, nn, count = sums["bb"], sums["bn"], sums["nn"], sums["num_images"]
    denominator = bb - 2 * bn + nn
    alpha_raw = (bb - bn) / denominator if denominator > 0 else None
    alpha_clipped = min(1.0, max(0.0, alpha_raw)) if alpha_raw is not None else None

    def mean_sse(alpha: float | None) -> float | None:
        if alpha is None:
            return None
        return ((1 - alpha) ** 2 * bb + 2 * alpha * (1 - alpha) * bn + alpha**2 * nn) / count

    grid = [index / 20 for index in range(21)]
    return {
        "inner_products": sums,
        "alpha_star_unconstrained": alpha_raw,
        "alpha_star_clipped": alpha_clipped,
        "error_correlation": bn / np.sqrt(bb * nn) if bb > 0 and nn > 0 else None,
        "learned_alpha": learned_alpha,
        "anchor_mean_image_sse": bb / count,
        "branch_mean_image_sse": nn / count,
        "learned_mean_image_sse": mean_sse(learned_alpha),
        "optimal_mean_image_sse": mean_sse(alpha_clipped),
        "alpha_grid": grid,
        "mean_image_sse": [mean_sse(alpha) for alpha in grid],
    }


def _radius(height: int, width: int, device: torch.device) -> torch.Tensor:
    fy = torch.fft.fftfreq(height, device=device)
    fx = torch.fft.fftfreq(width, device=device)
    return torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)


def radial_power_spectrum(field: torch.Tensor, num_bins: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Raw channel/image-averaged FFT power, then annular mean on [0, .5)."""
    radius = _radius(*field.shape[-2:], field.device)
    edges = torch.linspace(0, 0.5, num_bins + 1, device=field.device)
    # No mean removal, window, FFT norm, feature normalization or curve scaling.
    power = torch.zeros(radius.shape, dtype=torch.float64, device=field.device)
    for chunk in field.float().split(64, dim=1):
        power += torch.fft.fft2(chunk).abs().square().sum(dim=(0, 1)).double()
    power /= field.shape[0] * field.shape[1]
    radial = torch.zeros(num_bins, dtype=torch.float64, device=field.device)
    for index in range(num_bins):
        mask = (radius >= edges[index]) & (radius < edges[index + 1])
        if mask.any():
            radial[index] = power[mask].mean()
    centers = (edges[:-1] + edges[1:]) / 2
    return centers.cpu().numpy(), radial.cpu().numpy()


def frequency_bands(
    correction: torch.Tensor, crossings: tuple[float, float] = BAND_CROSSINGS, taper_half_width: float = 0.01,
) -> dict[str, torch.Tensor]:
    """Partition the correction with complementary radial raised-cosine masks."""
    radius = _radius(*correction.shape[-2:], correction.device)

    def low_pass(cutoff: float) -> torch.Tensor:
        transition = ((radius - cutoff + taper_half_width) / (2 * taper_half_width)).clamp(0, 1)
        return (1 + torch.cos(torch.pi * transition)) / 2

    below_low, below_high = (low_pass(edge) for edge in crossings)
    masks = {"low": below_low, "mid": below_high - below_low, "high": 1 - below_high}
    spectrum = torch.fft.fft2(correction.float())
    return {name: torch.fft.ifft2(spectrum * mask).real for name, mask in masks.items()}


def fit_shared_pca(fields: list[np.ndarray], *, seed: int = 0, max_samples: int = 20000) -> dict[str, np.ndarray]:
    """Fit one centered, whitened 3-component basis to absolute feature columns."""
    rng = np.random.default_rng(seed)
    selected = []
    for field in fields:
        flat = field.reshape(field.shape[0], -1).T
        count = min(len(flat), max_samples // len(fields))
        selected.append(flat[rng.choice(len(flat), size=count, replace=False)].astype(np.float64))
    samples = np.concatenate(selected)
    mean = samples.mean(axis=0)
    centered = samples - mean
    values, vectors = np.linalg.eigh(centered.T @ centered / (len(samples) - 1))
    values, axes = values[-3:][::-1], vectors[:, -3:][:, ::-1]
    # Fix eigenvector signs so repeated runs keep the same RGB axes.
    signs = np.sign(axes[np.abs(axes).argmax(axis=0), np.arange(3)])
    axes *= np.where(signs == 0, 1, signs)
    return {"mean": mean, "axes": axes, "variance": values.clip(1e-12)}


def project_pca(field: np.ndarray, basis: dict[str, np.ndarray], *, residual: bool = False) -> np.ndarray:
    """Shared whitening and sigmoid mapping; zero-origin residuals map to gray."""
    channels, height, width = field.shape
    flat = field.reshape(channels, -1).T.astype(np.float64)
    if not residual:
        flat = flat - basis["mean"]
    projected = (flat @ basis["axes"]) / np.sqrt(basis["variance"])
    rgb = 1 / (1 + np.exp(-np.clip(2 * projected, -80, 80)))
    return rgb.reshape(height, width, 3)


def pca_columns(lr_features: torch.Tensor, fields: dict[str, torch.Tensor], seed: int) -> dict[str, np.ndarray]:
    """RGB columns of one image: absolute features, the correction and its frequency bands."""
    size = tuple(fields["pred"].shape[-2:])
    absolute = {"lr": interpolate_anchor(lr_features, size, "nearest")}
    absolute.update({name: fields[name] for name in ("anchor", "learned", "pred")})
    absolute = {name: value[0].cpu().numpy() for name, value in absolute.items()}
    basis = fit_shared_pca(list(absolute.values()), seed=seed)
    columns = {name: project_pca(value, basis) for name, value in absolute.items()}
    residuals = {"correction": fields["correction"], **frequency_bands(fields["correction"])}
    for name, value in residuals.items():
        columns[name] = project_pca(value[0].cpu().numpy(), basis, residual=True)
    return columns


def _read_list(path: Path) -> list[Path]:
    values = json.loads(path.read_text())
    if not isinstance(values, list) or not values or not all(isinstance(value, str) for value in values):
        raise ValueError("Input list must be a nonempty JSON array of file paths")
    return [path.parent / value for value in values]


def _save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _load_pair(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    with np.load(path, allow_pickle=False) as archive:
        pair = {name: torch.from_numpy(archive[name].copy()).to(device) for name in archive.files}
    for name in ("guidance", "lr_features"):
        if name not in pair:
            raise ValueError(f"{path.name}: missing {name}")
    for name, tensor in pair.items():
        if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
            raise ValueError(f"{path.name}/{name}: expected finite float32 [1,C,H,W]")
    return pair


def _draw_pca(path: Path, columns: dict[str, np.ndarray]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, len(columns), figsize=(2 * len(columns), 2.4), squeeze=False)
    for axis, (name, rgb) in zip(axes[0], columns.items()):
        axis.imshow(rgb, interpolation="nearest")
        axis.set_title(name, fontsize=8)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


@torch.no_grad()
def analyze(args: argparse.Namespace) -> None:
    from utils.checkpoint import instantiate_model

    samples = _read_list(args.samples)
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    model = instantiate_model(device=device, checkpoint_path=args.checkpoint).eval()
    if model.name != "anchorup":
        raise ValueError("Diagnostics require an AnchorUp checkpoint")
    if args.alpha_mode == "fixed-bilinear-branch" and model.base_mode != "bilinear":
        raise ValueError("The fixed-bilinear-branch sweep requires a bilinear-trained checkpoint")
    anchor_modes = ANCHORS if args.alpha_mode == "fixed-bilinear-branch" else (model.base_mode,)
    sums = {mode: {"bb": 0.0, "bn": 0.0, "nn": 0.0, "num_images": 0} for mode in anchor_modes}
    mean_spectra = {name: 0.0 for name in FIELDS}

    for index, path in enumerate(samples):
        pair = _load_pair(path, device)
        target = pair.get("hr_features")
        if args.output_size is None and target is None:
            raise ValueError("Provide hr_features or --output-size HEIGHT WIDTH")
        size = tuple(args.output_size or target.shape[-2:])
        if "alpha" in args.analyses and (target is None or tuple(target.shape[-2:]) != size):
            raise ValueError("Alpha requires the native hr_features grid; do not set --output-size")
        aux = model(pair["guidance"], pair["lr_features"], size, return_aux=True)
        alpha = float(aux["gate"])
        # pred - anchor includes the FP32 rounding of the fusion.
        fields = {"anchor": aux["base"], "learned": aux["attn_feat"], "pred": aux["pred"]}
        fields["correction"] = fields["pred"] - fields["anchor"]
        outputs = {}

        if "alpha" in args.analyses:
            for mode in anchor_modes:
                anchor = interpolate_anchor(pair["lr_features"], size, mode)
                for key, value in error_inner_products(anchor, fields["learned"], target).items():
                    sums[mode][key] += value
        if "spectrum" in args.analyses:
            for name in FIELDS:
                frequencies, power = radial_power_spectrum(fields[name])
                mean_spectra[name] = mean_spectra[name] + power / len(samples)
                outputs[f"spectrum_{name}"] = power
            outputs["frequencies"] = frequencies
        if "pca" in args.analyses:
            columns = pca_columns(pair["lr_features"], fields, seed=args.seed + index)
            outputs.update({f"rgb_{name}": rgb for name, rgb in columns.items()})
            _draw_pca(args.output / f"sample_{index:04d}.png", columns)
        if outputs:
            np.savez_compressed(args.output / f"sample_{index:04d}.npz", **outputs)

    if "alpha" in args.analyses:
        _save_json(args.output / "alpha.json", {mode: alpha_summary(value, alpha) for mode, value in sums.items()})
    if "spectrum" in args.analyses:
        np.savez_compressed(args.output / "spectrum.npz", frequencies=frequencies, **mean_spectra)


@torch.no_grad()
def extract(args: argparse.Namespace) -> None:
    """Write 2x feature pairs for an image list with a frozen backbone."""
    from PIL import Image
    from torchvision.transforms import InterpolationMode, functional as TF
    from src.backbone.vit_wrapper import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, PretrainedViTWrapper

    paths = _read_list(args.images)
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    backbone = PretrainedViTWrapper(args.backbone).to(device).eval()
    if args.image_size % (2 * backbone.patch_size):
        raise ValueError("image-size must be divisible by twice the backbone patch size")
    mean = torch.as_tensor(backbone.config["mean"], device=device).reshape(1, 3, 1, 1)
    std = torch.as_tensor(backbone.config["std"], device=device).reshape(1, 3, 1, 1)
    guidance_mean = torch.tensor(IMAGENET_DEFAULT_MEAN, device=device).reshape(1, 3, 1, 1)
    guidance_std = torch.tensor(IMAGENET_DEFAULT_STD, device=device).reshape(1, 3, 1, 1)
    pairs = []
    for index, path in enumerate(paths):
        with Image.open(path) as source:
            rgb = TF.resize(source.convert("RGB"), args.image_size, interpolation=InterpolationMode.BILINEAR)
            rgb = TF.center_crop(rgb, [args.image_size, args.image_size])
        image = TF.to_tensor(rgb).unsqueeze(0).to(device)
        normalized = (image - mean) / std
        hr_features = backbone(normalized)
        lr_features = backbone(F.interpolate(normalized, scale_factor=0.5, mode="bilinear", align_corners=False))
        guidance = (image - guidance_mean) / guidance_std
        filename = f"pair_{index:04d}.npz"
        np.savez_compressed(args.output / filename, guidance=guidance.cpu().numpy(),
                            lr_features=lr_features.cpu().numpy(), hr_features=hr_features.cpu().numpy())
        pairs.append(filename)
    _save_json(args.output / "samples.json", pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("extract", help="Extract 2x feature pairs from an image list")
    prepare.add_argument("--images", type=Path, required=True, help="JSON array of paths relative to this list")
    prepare.add_argument("--backbone", default="vit_base_patch16_dinov3.lvd1689m")
    prepare.add_argument("--image-size", type=int, default=512)
    prepare.set_defaults(run=extract)
    diagnostic = commands.add_parser("analyze", help="Run the diagnostics on feature pairs with a checkpoint")
    diagnostic.add_argument("--samples", type=Path, required=True, help="JSON array of NPZ paths relative to this list")
    diagnostic.add_argument("--checkpoint", type=Path, required=True)
    diagnostic.add_argument("--analyses", nargs="+", choices=("alpha", "spectrum", "pca"), default=["alpha", "spectrum", "pca"])
    diagnostic.add_argument("--alpha-mode", choices=("own-branch", "fixed-bilinear-branch"), default="own-branch")
    diagnostic.add_argument("--output-size", nargs=2, type=int, metavar=("HEIGHT", "WIDTH"))
    diagnostic.add_argument("--seed", type=int, default=0)
    diagnostic.set_defaults(run=analyze)
    for command in (prepare, diagnostic):
        command.add_argument("--output", type=Path, required=True, help="Output directory (must not exist)")
        command.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
