"""MAGNET construction defaults used by every entry point."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


MODEL_DEFAULTS = {
    "magnet": {
        "PFE_type": "CNN",
        "encoder_type": "Transformer",
        "decoder_type": "LIIF",
        "dim_token": 3,
        "embedder_type_2D": "FourierEncoding",
        "embedder_type_3D": "FourierEncoding",
        "hidden_feat": 256,
        "hidden_lens": 3,
    },
}

# Kept as a named alias because Configs.py exposes these values as legacy args.
MAGNET_DEFAULTS = MODEL_DEFAULTS["magnet"]


def _merge(base: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            base[key] = _merge(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def model_parameters(family: str, overrides=None) -> dict[str, Any]:
    """Return an independent defaults copy with optional recursive overrides."""
    family = family.lower()
    if family not in MODEL_DEFAULTS:
        raise ValueError(f"Unknown model family: {family!r}")
    return _merge(deepcopy(MODEL_DEFAULTS[family]), overrides or {})
