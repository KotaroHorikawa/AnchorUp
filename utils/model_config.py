"""Helpers for loading model configs and reducing them to inference settings."""

from pathlib import Path

from omegaconf import OmegaConf

_CORE_MODEL_TARGETS = {
    "anchorup": "src.model.anchorup.AnchorUp",
    "naf": "src.model.naf.NAF",
}
_INFERENCE_FIELDS = ("name", "dim", "heads_attn", "heads_rope", "kernel_size", "use_encoder", "img_layers", "rope_base")
# Settings that only affect training or construction, not a trained model.
_TRAINING_FIELDS = {"_target_", "rope_rescale", "gate_params"}


def resolved_model_config(model_config):
    """Plain dict of a model config with interpolations resolved."""
    return OmegaConf.to_container(OmegaConf.create(dict(model_config)), resolve=True)


def load_model_config(model_name, *, overrides=None):
    """Load ``config/model/<model_name>.yaml`` outside a Hydra application."""
    model_config = OmegaConf.load(Path(__file__).resolve().parents[1] / "config" / "model" / f"{model_name}.yaml")
    if overrides:
        model_config = OmegaConf.merge(model_config, dict(overrides))
    return model_config


def canonical_inference_config(model_config):
    """Keep the settings needed to reconstruct NAF or AnchorUp for inference."""
    config = resolved_model_config(model_config)
    if config.get("name") not in _CORE_MODEL_TARGETS:
        raise ValueError(f"Minimal checkpoints support anchorup or naf, got: {config.get('name')!r}")
    fields = _INFERENCE_FIELDS + (("base_mode",) if config["name"] == "anchorup" else ())
    unknown = set(config) - set(fields) - _TRAINING_FIELDS
    if unknown:
        raise ValueError(f"Unknown model configuration fields: {sorted(unknown)}")
    minimal = {key: config[key] for key in fields}
    minimal["rope_base"] = float(minimal["rope_base"])
    return minimal


def model_constructor_config(model_config):
    """Constructor config for Hydra; NAF and AnchorUp are rebuilt from their inference settings."""
    config = resolved_model_config(model_config)
    if config.get("name") not in _CORE_MODEL_TARGETS:
        return config
    constructor = {"_target_": _CORE_MODEL_TARGETS[config["name"]], **canonical_inference_config(config)}
    # Explicit training settings remain effective when initializing a new run.
    for key in ("rope_rescale", "gate_params"):
        if key in config:
            constructor[key] = config[key]
    return constructor
