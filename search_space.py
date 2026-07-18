# search_space.py
"""
Utilities for the "Configurable Space" NAS mode in app.py.

A single candidate architecture is described by a small config dict:

    {
        "depth": 2 | 3 | 4,
        "base_channels": 16 | 32 | 64,
        "block_type": "plain" | "residual" | "bottleneck",
        "dropout": 0.0 | 0.2 | 0.5,
    }

This module only deals with naming / estimating that config space.
The actual nn.Module is built by models.get_model_from_config.
"""

from typing import Dict


def config_to_name(config: Dict) -> str:
    """
    Turn a config dict into a short, stable, human-readable identifier.
    e.g. {"depth":3,"base_channels":32,"block_type":"residual","dropout":0.2}
         -> "d3_c32_residual_dr0.2"
    """
    return (
        f"d{config['depth']}_"
        f"c{config['base_channels']}_"
        f"{config['block_type']}_"
        f"dr{config['dropout']}"
    )


def estimate_params(config: Dict, num_classes: int) -> int:
    """
    Estimate the trainable parameter count for a given config, without
    needing a real input tensor. Used to preview/filter candidates before
    training (the "Max parameter budget" slider in app.py).

    We build the actual model on CPU with meta tensors would be nicer, but
    to keep this dependency-free and always accurate we just instantiate
    the real module briefly and count parameters. Construction of these
    small CNNs is cheap (a few hundred thousand params at most), so doing
    this for preview purposes (even ~80 times) is fast.
    """
    # Local import to avoid a circular import at module load time
    # (models.py does not import search_space, but this keeps things safe
    # if that ever changes).
    from models import get_model_from_config

    model = get_model_from_config(config, num_classes)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)