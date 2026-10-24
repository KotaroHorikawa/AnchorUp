"""Model checkpoints that store weights with the config needed for inference."""

from collections.abc import Mapping

import torch
from hydra.utils import instantiate

from utils.model_config import canonical_inference_config, model_constructor_config, resolved_model_config

CHECKPOINT_FORMAT_VERSION = 2
_CORE_MODELS = {"anchorup", "naf"}


def _inference_config(config):
    """NAF and AnchorUp checkpoints keep only the settings needed for inference."""
    config = resolved_model_config(config)
    return canonical_inference_config(config) if config.get("name") in _CORE_MODELS else config


def build_model_checkpoint(model, model_config):
    return {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state_dict": model.state_dict(),
        "model_config": _inference_config(model_config),
    }


def save_model_checkpoint(path, model, model_config):
    torch.save(build_model_checkpoint(model, model_config), path)


def _is_state_dict(value):
    return isinstance(value, Mapping) and bool(value) and all(torch.is_tensor(v) for v in value.values())


def checkpoint_model_config(checkpoint):
    """Return the saved model config of a checkpoint written by build_model_checkpoint."""
    version = checkpoint.get("checkpoint_format_version") if isinstance(checkpoint, Mapping) else None
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"Unsupported model checkpoint (format version {version!r})")
    return resolved_model_config(checkpoint["model_config"])


def extract_model_state_dict(checkpoint, model_config):
    """Check that the saved config matches the requested one and return the state dict.

    A bare state dict has no saved config and is returned as is.
    """
    if _is_state_dict(checkpoint):
        return checkpoint
    saved = _inference_config(checkpoint_model_config(checkpoint))
    requested = _inference_config(model_config)
    differences = [
        f"{key}: checkpoint={saved.get(key)!r}, requested={requested.get(key)!r}"
        for key in sorted(set(saved) | set(requested))
        if saved.get(key) != requested.get(key)
    ]
    if differences:
        raise ValueError("Checkpoint model_config does not match the requested model: " + "; ".join(differences))
    return checkpoint["model_state_dict"]


def _load_external_featup(model, checkpoint):
    if "state_dict" not in checkpoint:
        return False
    converted = {
        key.replace("model.1.", "norm."): value
        for key, value in checkpoint["state_dict"].items()
        if "upsampler" in key or "model.1.norm" in key
    }
    model.load_state_dict(converted, strict=True)
    return True


def _load_external_jafar(model, checkpoint):
    if "jafar" not in checkpoint:
        return False
    model.load_state_dict(checkpoint["jafar"], strict=True)
    return True


def _load_payload(model, checkpoint, model_config):
    model_name = str(model_config.get("name", "")).lower()
    if model_name == "featup" and _load_external_featup(model, checkpoint):
        return model
    if model_name == "jafar" and _load_external_jafar(model, checkpoint):
        return model
    model.load_state_dict(extract_model_state_dict(checkpoint, model_config), strict=True)
    return model


def load_probe_state_dict(path, *, map_location=None):
    """Load a linear probe, which is saved as a plain state dict."""
    return torch.load(path, map_location=map_location, weights_only=False)


def load_model_checkpoint(model, path, model_config, *, device):
    """Load a checkpoint written by build_model_checkpoint, or an official FeatUp/JAFAR file."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    return _load_payload(model, checkpoint, model_config)


def instantiate_model(model_config=None, *, device, checkpoint_path=None, checkpoint=None, model_overrides=None):
    """Build a model from model_config or from the config saved in the checkpoint, then load its weights."""
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if model_config is None:
        if checkpoint is None:
            raise ValueError("Provide model_config or checkpoint_path")
        if _is_state_dict(checkpoint):
            raise ValueError("Bare checkpoint: provide its explicit model_config")
        model_config = checkpoint_model_config(checkpoint)
    config = resolved_model_config(model_config)
    if model_overrides:
        config.update(model_overrides)
    model = instantiate(model_constructor_config(config)).to(device)
    if checkpoint is not None:
        _load_payload(model, checkpoint, config)
    return model
