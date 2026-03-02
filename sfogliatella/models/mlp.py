"""MLP model (windowed, JAX/equinox)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import equinox as eqx
import jax
import jax.numpy as jnp

from sfogliatella.models.base import output_dim_for_task
from sfogliatella.registry.registry import register


# ---------------------------------------------------------------------------
# Module definition
# ---------------------------------------------------------------------------

class MLP(eqx.Module):
    """Multi-Layer Perceptron for time-series windows.

    Input: (lookback, input_dim) -> flattened -> MLP -> (out_dim,)
    """
    layers: List[eqx.nn.Linear]
    dropout: eqx.nn.Dropout
    in_dim: int
    out_dim: int

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        width: int = 256,
        depth: int = 4,
        dropout_rate: float = 0.0,
        *,
        key: jax.Array,
    ):
        self.in_dim = in_dim
        self.out_dim = out_dim
        keys = jax.random.split(key, depth + 1)
        dims = [in_dim] + [width] * max(1, depth - 1) + [out_dim]
        self.layers = [eqx.nn.Linear(dims[i], dims[i + 1], key=keys[i]) for i in range(len(dims) - 1)]
        self.dropout = eqx.nn.Dropout(p=dropout_rate)

    def __call__(self, x: jnp.ndarray, *, key: Optional[jax.Array] = None) -> jnp.ndarray:
        # x: (lookback, input_dim) or (flat_dim,)
        x = x.reshape(-1)
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
            if key is not None:
                key, subkey = jax.random.split(key)
                x = self.dropout(x, key=subkey)
        return self.layers[-1](x)


# ---------------------------------------------------------------------------
# Baseline config computation
# ---------------------------------------------------------------------------

def _baseline_config(config: Dict[str, Any]) -> Dict[str, Any]:
    lookback  = int(config.get("lookback", 32))
    input_dim = int(config.get("input_dim", 1))
    width = config.get("width", 256)
    depth = config.get("depth", config.get("layers", 4))
    return {"width": int(width), "depth": int(depth)}


# ---------------------------------------------------------------------------
# Registry builder
# ---------------------------------------------------------------------------

@register("mlp")
def build_mlp(config: Dict[str, Any], rng_key=None, sample_x=None):
    import jax
    key = rng_key if rng_key is not None else jax.random.PRNGKey(int(config.get("seed", 42)))

    lookback  = int(config.get("lookback", 32))
    input_dim = int(config.get("input_dim", 1))
    task      = config.get("task", "regression")
    num_cls   = int(config.get("num_classes", 2))
    horizon   = int(config.get("horizon", 1))
    out_dim   = output_dim_for_task(task, num_cls, horizon)

    bc = _baseline_config(config)
    width = int(config.get("width", bc["width"]))
    depth = int(config.get("depth", config.get("layers", bc["depth"])))
    dropout = float(config.get("dropout", 0.0))

    flat_in = lookback * input_dim
    return MLP(flat_in, out_dim, width=width, depth=depth, dropout_rate=dropout, key=key)
