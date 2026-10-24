"""Upsample one image's frozen VFM features with a local checkpoint."""

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as transforms

from src.backbone.vit_wrapper import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, PretrainedViTWrapper
from utils.checkpoint import instantiate_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backbone", default="vit_base_patch16_dinov3.lvd1689m")
    parser.add_argument("--size", type=int, default=448)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    backbone = PretrainedViTWrapper(name=args.backbone).to(args.device).eval()
    upsampler = instantiate_model(device=args.device, checkpoint_path=args.checkpoint).eval()
    with Image.open(args.image) as image:
        rgb = transforms.to_tensor(image.convert("RGB"))
    rgb = transforms.resize(rgb, [args.size, args.size], antialias=True)
    rgb = rgb.unsqueeze(0).to(args.device)
    backbone_rgb = transforms.normalize(rgb, backbone.config["mean"], backbone.config["std"])
    guidance_rgb = transforms.normalize(rgb, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    with torch.inference_mode():
        features = backbone(backbone_rgb)
        output = upsampler(guidance_rgb, features, (args.size, args.size))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output.cpu(), args.output)
    print(f"Saved features {tuple(output.shape)} to {args.output}")


if __name__ == "__main__":
    main()
