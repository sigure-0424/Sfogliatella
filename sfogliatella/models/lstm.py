"""LSTM model (JAX/equinox)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import equinox as eqx
import jax
import jax.numpy as jnp

from sfogliatella.models.base import output_dim_for_task
from sfogliatella.registry.registry import register


class LSTMModel(eqx.Module):
    """Stacked LSTM followed by a linear output head.

    Input: (lookback, input_dim) -> LSTM (last hidden) -> Linear -> (out_dim,)
    """
    cells: List[eqx.nn.LSTMCell]
    output_proj: eqx.nn.Linear
    in_dim: int
    hidden: int
    out_dim: int
    n_layers: int

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: int = 128,
        n_layers: int = 2,
        *,
        key: jax.Array,
    ):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden = hidden
        self.n_layers = n_layers

        keys = jax.random.split(key, n_layers + 1)
        cells = []
        for i in range(n_layers):
            cell_in = in_dim if i == 0 else hidden
            cells.append(eqx.nn.LSTMCell(input_size=cell_in, hidden_size=hidden, key=keys[i]))
        self.cells = cells
        self.output_proj = eqx.nn.Linear(hidden, out_dim, key=keys[-1])

    def __call__(self, x: jnp.ndarray, *, key: Optional[jax.Array] = None) -> jnp.ndarray:
        # x: (lookback, in_dim)
        seq_len = x.shape[0]

        # Initialize hidden/cell states
        states = [(jnp.zeros(self.hidden), jnp.zeros(self.hidden)) for _ in self.cells]

        for t in range(seq_len):
            inp = x[t]  # (in_dim,)
            new_states = []
            for i, cell in enumerate(self.cells):
                h, c = states[i]
                h, c = cell(inp, (h, c))
                inp = h
                new_states.append((h, c))
            states = new_states

        last_h = states[-1][0]  # (hidden,)
        return self.output_proj(last_h)


@register("lstm")
def build_lstm(config: Dict[str, Any], rng_key=None, sample_x=None):
    import jax
    key = rng_key if rng_key is not None else jax.random.PRNGKey(int(config.get("seed", 42)))

    input_dim = int(config.get("input_dim", 1))
    task      = config.get("task", "regression")
    num_cls   = int(config.get("num_classes", 2))
    horizon   = int(config.get("horizon", 1))
    out_dim   = output_dim_for_task(task, num_cls, horizon)
    hidden    = int(config.get("hidden", config.get("d_model", 128)))
    n_layers  = int(config.get("layers", 2))

    return LSTMModel(input_dim, out_dim, hidden=hidden, n_layers=n_layers, key=key)
