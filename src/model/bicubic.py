import torch.nn.functional as F

from src.model.base import BaseUpsampler


class Bicubic(BaseUpsampler):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, image, features, output_size, *args, **kwargs):
        return F.interpolate(features, size=output_size, mode="bicubic", align_corners=False)
