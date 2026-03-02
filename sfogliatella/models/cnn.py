"""1D CNN model for time-series (JAX/equinox)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import equinox as eqx
import jax
import jax.numpy as jnp

from sfogliatella.models.base import output_dim_for_task
from sfogliatella.registry.registry import register


class CNNModel(eqx.Module):
    """Stack of 1D causal convolutions followed by global average pooling and MLP head."""
    convs: List[eqx.nn.Conv1d]
    output_proj: eqx.nn.Linear
    in_dim: int
    out_dim: int
    channels: int
    kernel: int
    n_layers: int

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        channels: int = 64,
        kernel: int = 3,
        n_layers: int = 4,
        *,
        key: jax.Array,
    ):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.channels = channels
        self.kernel = kernel
        self.n_layers = n_layers

        keys = jax.random.split(key, n_layers + 1)
        convs = []
        for i in range(n_layers):
            c_in = in_dim if i == 0 else channels
            convs.append(
                eqx.nn.Conv1d(
                    in_channels=c_in,
                    out_channels=channels,
                    kernel_size=kernel,
                    padding=kernel - 1,  # causal padding applied manually
                    key=keys[i],
                )
            )
        self.convs = convs
        self.output_proj = eqx.nn.Linear(channels, out_dim, key=keys[-1])

    def __call__(self, x: jnp.ndarray, *, key: Optional[jax.Array] = None) -> jnp.ndarray:
        # x: (lookback, in_dim)  -> conv expects (in_dim, seq)
        x = x.T   # (in_dim, lookback)

        for conv in self.convs:
            x = conv(x)
            # Remove future padding (keep only causal part)
            x = x[:, :-(self.kernel - 1)] if self.kernel > 1 else x
            x = jax.nn.relu(x)

        # Global average pool: (channels, seq) -> (channels,)
        x = jnp.mean(x, axis=-1)
        return self.output_proj(x)


@register("cnn")
def build_cnn(config: Dict[str, Any], rng_key=None, sample_x=None):
    import jax
    key = rng_key if rng_key is not None else jax.random.PRNGKey(int(config.get("seed", 42)))

    input_dim = int(config.get("input_dim", 1))
    task      = config.get("task", "regression")
    num_cls   = int(config.get("num_classes", 2))
    horizon   = int(config.get("horizon", 1))
    out_dim   = output_dim_for_task(task, num_cls, horizon)
    channels  = int(config.get("channels", config.get("d_model", 64)))
    kernel    = int(config.get("kernel", 3))
    n_layers  = int(config.get("layers", 4))

    return CNNModel(input_dim, out_dim, channels=channels, kernel=kernel, n_layers=n_layers, key=key)
