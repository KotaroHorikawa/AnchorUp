#!/usr/bin/env python3
"""Benchmark upsampler-only forward latency.

All methods receive the same inputs: batch 1, FP32, a 448x448 RGB guidance image
and a 384-channel 28x28 feature map upsampled to 448x448 (FeatUp runs its 16x
cascade). Model construction, checkpoint loading and warm-up are not timed.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Final

import torch

from src.model.anyup import AnyUpsampler
from src.model.bicubic import Bicubic
from src.model.bilinear import Bilinear
from src.model.featup import FeatUp
from src.model.jafar import JAFAR
from utils.checkpoint import instantiate_model
from utils.model_config import load_model_config

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]


METHOD_ORDER: Final = (
    "bilinear",
    "bicubic",
    "featup",
    "jafar",
    "anyup_natten",
    "naf",
    "anchorup",
    "anchorup_bicubic",
)


@dataclass(frozen=True)
class BenchmarkResult:
    method: str
    display_name: str
    implementation: str
    latency_ms_mean: float
    latency_ms_std: float
    latency_ms_median: float
    latency_ms_min: float
    latency_ms_max: float
    parameter_count: int
    peak_memory_mib: float
    batch_size: int
    dtype: str
    image_height: int
    image_width: int
    feature_channels: int
    feature_height: int
    feature_width: int
    output_height: int
    output_width: int
    warmup_forwards: int
    timed_forwards: int
    trials: int
    iterations_per_trial: int
    gpu: str
    pytorch_version: str
    cuda_version: str


def _load_anchorup(checkpoint_path: Path, *, base_mode: str, device: str = "cuda") -> torch.nn.Module:
    model = instantiate_model(device=device, checkpoint_path=str(checkpoint_path))
    if model.base_mode != base_mode:
        raise ValueError(
            f"Checkpoint anchor is {model.base_mode!r}; the requested benchmark "
            f"requires {base_mode!r}"
        )
    return model


def _build_model(
    method: str,
    *,
    feature_channels: int,
    upsample_ratio: int,
    anchorup_checkpoint: Path,
    anchorup_bicubic_checkpoint: Path,
) -> tuple[torch.nn.Module, str, str]:
    if method == "bilinear":
        return Bilinear().cuda(), "Bilinear", "torch.interpolate"
    if method == "bicubic":
        return Bicubic().cuda(), "Bicubic", "torch.interpolate"
    if method == "featup":
        model = FeatUp(feature_dim=feature_channels, ratio=upsample_ratio)
        return model.cuda(), "FeatUp", "CUDA AdaptiveConv"
    if method == "jafar":
        model = JAFAR(
            dim=128,
            v_dim=feature_channels,
            feature_dim=feature_channels,
            kernel_size=1,
            num_heads=4,
        )
        return model.cuda(), "JAFAR", "PyTorch MHA"
    if method == "anyup_natten":
        model = AnyUpsampler(use_natten=True, pretrained=False)
        return model.cuda(), "AnyUp (NATTEN)", "NATTEN"
    if method == "naf":
        model = instantiate_model(load_model_config("naf"), device="cuda")
        return model, "NAF", "NATTEN"
    if method == "anchorup":
        model = _load_anchorup(anchorup_checkpoint, base_mode="bilinear")
        return model, "AnchorUp (ours)", "NATTEN + bilinear anchor"
    if method == "anchorup_bicubic":
        model = _load_anchorup(anchorup_bicubic_checkpoint, base_mode="bicubic")
        return model, "AnchorUp (bicubic anchor)", "NATTEN + bicubic anchor"
    raise ValueError(f"Unknown method: {method}")


def _make_inputs(
    *,
    batch_size: int,
    image_size: int,
    feature_channels: int,
    feature_size: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(0)
    image = torch.randn(
        batch_size,
        3,
        image_size,
        image_size,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    features = torch.randn(
        batch_size,
        feature_channels,
        feature_size,
        feature_size,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    return image, features


def _forward(
    model: torch.nn.Module,
    image: torch.Tensor,
    features: torch.Tensor,
    output_size: tuple[int, int],
) -> torch.Tensor:
    output = model(image, features, output_size)
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Expected tensor output, got {type(output)!r}")
    expected = (
        int(features.shape[0]),
        int(features.shape[1]),
        int(output_size[0]),
        int(output_size[1]),
    )
    if tuple(output.shape) != expected:
        raise ValueError(
            f"Output shape mismatch for {type(model).__name__}: "
            f"observed={tuple(output.shape)}, expected={expected}"
        )
    return output


def _measure(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    trials: int,
    iterations: int,
) -> tuple[list[float], float]:
    with torch.inference_mode():
        for _ in range(warmup):
            output = fn()
            del output
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples: list[float] = []
    with torch.inference_mode():
        for _ in range(trials):
            for _ in range(iterations):
                start.record()
                output = fn()
                end.record()
                end.synchronize()
                samples.append(float(start.elapsed_time(end)))
                del output

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        output = fn()
    torch.cuda.synchronize()
    peak_memory_mib = float(torch.cuda.max_memory_allocated()) / (1024**2)
    del output
    return samples, peak_memory_mib


def _write_csv(path: Path, results: list[BenchmarkResult]) -> None:
    rows = [asdict(result) for result in results]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHOD_ORDER,
        default=list(METHOD_ORDER),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--feature-channels", type=int, default=384)
    parser.add_argument("--feature-size", type=int, default=28)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--anchorup-checkpoint",
        type=Path,
        default=None,
        help="Required when anchorup is benchmarked.",
    )
    parser.add_argument(
        "--anchorup-bicubic-checkpoint",
        type=Path,
        default=None,
        help="Required when anchorup_bicubic is benchmarked.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output/latency",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if "featup" in args.methods and args.image_size % args.feature_size != 0:
        raise ValueError(
            "The common FeatUp protocol requires an integer spatial ratio: "
            f"image_size={args.image_size}, feature_size={args.feature_size}"
        )
    ratio = args.image_size // args.feature_size
    if "featup" in args.methods and ratio not in (2, 4, 8, 16):
        raise ValueError(
            "FeatUp supports only ratios 2, 4, 8, and 16; "
            f"the requested common ratio is {ratio}."
        )
    if min(args.warmup, args.trials, args.iterations, args.batch_size) <= 0:
        raise ValueError("warmup, trials, iterations, and batch-size must be positive")
    if "anchorup" in args.methods and args.anchorup_checkpoint is None:
        raise ValueError("--anchorup-checkpoint is required when methods include anchorup")
    if "anchorup_bicubic" in args.methods and args.anchorup_bicubic_checkpoint is None:
        raise ValueError("--anchorup-bicubic-checkpoint is required when methods include anchorup_bicubic")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    dtype = torch.float32
    output_size = (args.image_size, args.image_size)

    results: list[BenchmarkResult] = []
    raw_methods: dict[str, dict[str, object]] = {}
    for method in args.methods:
        gc.collect()
        torch.cuda.empty_cache()
        model, display_name, implementation = _build_model(
            method,
            feature_channels=args.feature_channels,
            upsample_ratio=ratio,
            anchorup_checkpoint=args.anchorup_checkpoint.resolve() if args.anchorup_checkpoint else Path(),
            anchorup_bicubic_checkpoint=(
                args.anchorup_bicubic_checkpoint.resolve() if args.anchorup_bicubic_checkpoint else Path()
            ),
        )
        model.eval()
        image, features = _make_inputs(
            batch_size=args.batch_size,
            image_size=args.image_size,
            feature_channels=args.feature_channels,
            feature_size=args.feature_size,
            dtype=dtype,
        )
        fn = lambda: _forward(model, image, features, output_size)
        samples, peak_memory_mib = _measure(
            fn,
            warmup=args.warmup,
            trials=args.trials,
            iterations=args.iterations,
        )
        result = BenchmarkResult(
            method=method,
            display_name=display_name,
            implementation=implementation,
            latency_ms_mean=float(statistics.fmean(samples)),
            latency_ms_std=float(statistics.pstdev(samples)),
            latency_ms_median=float(statistics.median(samples)),
            latency_ms_min=float(min(samples)),
            latency_ms_max=float(max(samples)),
            parameter_count=sum(parameter.numel() for parameter in model.parameters()),
            peak_memory_mib=peak_memory_mib,
            batch_size=args.batch_size,
            dtype="float32",
            image_height=args.image_size,
            image_width=args.image_size,
            feature_channels=args.feature_channels,
            feature_height=args.feature_size,
            feature_width=args.feature_size,
            output_height=args.image_size,
            output_width=args.image_size,
            warmup_forwards=args.warmup,
            timed_forwards=len(samples),
            trials=args.trials,
            iterations_per_trial=args.iterations,
            gpu=torch.cuda.get_device_name(),
            pytorch_version=torch.__version__,
            cuda_version=str(torch.version.cuda),
        )
        results.append(result)
        raw_methods[method] = {
            "summary": asdict(result),
            "latency_ms_samples": samples,
        }
        print(
            f"{display_name:18s} "
            f"{result.latency_ms_mean:9.3f} +/- {result.latency_ms_std:7.3f} ms "
            f"(median {result.latency_ms_median:9.3f} ms)"
        )
        del fn, image, features, model

    csv_path = output_dir / "upsampler_latency.csv"
    json_path = output_dir / "upsampler_latency_raw.json"
    _write_csv(csv_path, results)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "batch_size": args.batch_size,
            "dtype": "float32",
            "image_shape": [args.batch_size, 3, args.image_size, args.image_size],
            "feature_shape": [
                args.batch_size,
                args.feature_channels,
                args.feature_size,
                args.feature_size,
            ],
            "output_size": list(output_size),
            "warmup_forwards": args.warmup,
            "trials": args.trials,
            "iterations_per_trial": args.iterations,
            "timing": "CUDA events with synchronization after each forward",
        },
        "methods": raw_methods,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {csv_path}")
    print(f"Saved {json_path}")


if __name__ == "__main__":
    main()
