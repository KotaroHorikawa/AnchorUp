dependencies = ["torch", "natten", "einops", "numpy", "omegaconf", "hydra"]

import torch

RELEASE_URL = "https://github.com/KotaroHorikawa/AnchorUp/releases/download/weights"
ANCHORUP_WEIGHTS = (
    *(f"anchorup-{anchor}-d{dim}-25k" for anchor in ("bilinear", "bicubic") for dim in (32, 64, 128, 256)),
    *(f"anchorup-bilinear-d128-{steps}" for steps in ("12p5k", "50k", "100k", "200k", "300k")),
)


def anchorup(
    pretrained: bool = True,
    anchor: str = "bilinear",
    dim: int = 128,
    steps: str = "25k",
    device="cpu",
    checkpoint_path=None,
    **model_overrides,
):
    """
    AnchorUp: a NAF-style correction added to a fixed bilinear or bicubic anchor.

    Released weights: anchor in {"bilinear", "bicubic"} with dim in {32, 64, 128, 256}
    at steps="25k", plus bilinear dim=128 at steps in {"12p5k", "50k", "100k", "200k", "300k"}.
    A local checkpoint_path is loaded with its own saved anchor and width instead.
    """
    from utils.checkpoint import instantiate_model
    from utils.model_config import load_model_config

    if checkpoint_path is not None:
        model = instantiate_model(device=device, checkpoint_path=checkpoint_path, model_overrides=model_overrides)
    elif pretrained:
        name = f"anchorup-{anchor}-d{dim}-{steps}"
        if name not in ANCHORUP_WEIGHTS:
            raise ValueError(f"No released AnchorUp weights {name!r}; available: {', '.join(ANCHORUP_WEIGHTS)}")
        checkpoint = torch.hub.load_state_dict_from_url(
            f"{RELEASE_URL}/{name}.pth", progress=True, map_location="cpu", weights_only=True,
        )
        model = instantiate_model(device=device, checkpoint=checkpoint, model_overrides=model_overrides)
    else:
        config = load_model_config("anchorup", overrides={"base_mode": anchor, "dim": dim})
        model = instantiate_model(config, device=device, model_overrides=model_overrides)
    if model.name != "anchorup":
        raise ValueError("anchorup() requires an AnchorUp checkpoint")
    return model
