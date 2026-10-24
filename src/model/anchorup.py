import torch
import torch.nn.functional as F
from torch import nn

from src.model.naf import NAF


_SUPPORTED_BASE_MODES = {"bilinear", "bicubic"}


def _normalize_hw(size):
    if len(size) != 2:
        raise ValueError(f"output_size must contain exactly 2 values, got: {size}")
    return tuple(int(value) for value in size)


class AnchorUp(NAF):
    """Classical interpolation plus a learned residual, Eq. 1.

    ``pred = base + sigmoid(gamma) * (learned - base)``. ``gamma`` starts at 0,
    so the gate starts at 1/2. The anchor is bilinear or bicubic.
    """

    def __init__(
        self,
        name="anchorup",
        base_mode="bilinear",
        gate_params=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.name = name
        self.base_mode = str(base_mode).strip().lower()
        if self.base_mode not in _SUPPORTED_BASE_MODES:
            raise ValueError(
                f"base_mode must be one of {sorted(_SUPPORTED_BASE_MODES)}, got: {base_mode}"
            )
        self.gate_param = nn.Parameter(torch.tensor(float(gate_params), dtype=torch.float32))

    def _alpha(self, *, device, dtype):
        return torch.sigmoid(self.gate_param).to(device=device, dtype=dtype)

    def forward(
        self,
        image,
        features,
        output_size,
        return_weights=False,
        return_aux=False,
        *args,
        **kwargs,
    ):
        output_size = _normalize_hw(output_size)
        base = F.interpolate(features, size=output_size, mode=self.base_mode, align_corners=False)
        naf_result = super().forward(
            image,
            features,
            output_size,
            return_weights=return_weights,
            *args,
            **kwargs,
        )
        if return_weights:
            learned, weights = naf_result
        else:
            learned = naf_result
            weights = None
        alpha = self._alpha(device=learned.device, dtype=learned.dtype)
        pred = base + alpha * (learned - base)

        if return_aux:
            aux = {
                "pred": pred,
                "base": base,
                "attn_feat": learned,
                "correction": learned - base,
                "gate": alpha,
            }
            if return_weights:
                aux["weights"] = weights
            return aux
        if return_weights:
            return pred, weights
        return pred
