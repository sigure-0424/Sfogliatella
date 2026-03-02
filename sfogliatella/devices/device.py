"""JAX device selection and precision management."""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------------

def setup_device(device: str = "auto", show_tpu_warnings: bool = False) -> None:
    """Configure JAX device and suppress optional warnings."""
    if not show_tpu_warnings:
        # Suppress TPU init warnings that are not actionable
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        os.environ.setdefault("JAX_PLATFORMS", "")  # let JAX auto-select silently

    import jax

    if device == "auto":
        _device = _auto_detect_device()
    else:
        _device = device

    if _device == "cpu":
        jax.config.update("jax_platform_name", "cpu")
    elif _device in ("cuda", "gpu"):
        jax.config.update("jax_platform_name", "gpu")
    elif _device == "tpu":
        jax.config.update("jax_platform_name", "tpu")

    # Verify
    try:
        devs = jax.devices()
        logger.info("JAX device: %s (count=%d)", _device, len(devs))
    except Exception as e:
        logger.warning("JAX device check failed: %s", e)


def _auto_detect_device() -> str:
    try:
        import jax
        devs = jax.devices("gpu")
        if devs:
            return "cuda"
    except Exception:
        pass
    try:
        import jax
        devs = jax.devices("tpu")
        if devs:
            return "tpu"
    except Exception:
        pass
    return "cpu"


def get_device_count(device: str = "auto") -> int:
    import jax
    try:
        if device in ("cuda", "gpu"):
            return len(jax.devices("gpu"))
        elif device == "tpu":
            return len(jax.devices("tpu"))
        elif device == "cpu":
            return len(jax.devices("cpu"))
        else:
            return len(jax.devices())
    except Exception:
        return 1


# ---------------------------------------------------------------------------
# Precision
# ---------------------------------------------------------------------------

def configure_precision(precision: str = "auto") -> None:
    """Set JAX precision policy.

    auto  -> fp32 (safe default; bf16 on TPU if supported)
    fp32  -> force fp32
    bf16  -> use bfloat16 (TPU / Ampere+ GPU)
    fp16  -> use float16
    """
    import jax

    if precision == "bf16":
        from jax import numpy as jnp
        jax.config.update("jax_default_matmul_precision", "bfloat16")
        logger.info("Precision: bfloat16")
    elif precision == "fp16":
        logger.info("Precision: float16 (cast layers manually)")
    elif precision in ("fp32", "auto"):
        jax.config.update("jax_default_matmul_precision", "float32")
        logger.info("Precision: float32")
    else:
        logger.warning("Unknown precision %r, using fp32", precision)
        jax.config.update("jax_default_matmul_precision", "float32")


# ---------------------------------------------------------------------------
# Array helpers
# ---------------------------------------------------------------------------

def to_jnp(arr, dtype=None):
    """Convert numpy array (or any array-like) to JAX array."""
    import jax.numpy as jnp
    import numpy as np
    if hasattr(arr, "__jax_array__"):
        return arr
    a = jnp.asarray(arr)
    if dtype is not None:
        a = a.astype(dtype)
    return a
