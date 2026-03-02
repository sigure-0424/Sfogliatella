"""Transformer model for time-series (JAX/equinox)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import equinox as eqx
import jax
import jax.numpy as jnp

from sfogliatella.models.base import output_dim_for_task
from sfogliatella.registry.registry import register


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

def sinusoidal_pos_enc(seq_len: int, d_model: int) -> jnp.ndarray:
    positions = jnp.arange(seq_len)[:, None]
    dims = jnp.arange(d_model)[None, :]
    angles = positions / jnp.power(10000.0, (2 * (dims // 2)) / d_model)
    pe = jnp.where(dims % 2 == 0, jnp.sin(angles), jnp.cos(angles))
    return pe  # (seq_len, d_model)


# ---------------------------------------------------------------------------
# Transformer components
# ---------------------------------------------------------------------------

class MultiHeadAttention(eqx.Module):
    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear
    n_heads: int
    d_head: int

    def __init__(self, d_model: int, n_heads: int, *, key: jax.Array):
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        keys = jax.random.split(key, 4)
        self.q_proj = eqx.nn.Linear(d_model, d_model, use_bias=False, key=keys[0])
        self.k_proj = eqx.nn.Linear(d_model, d_model, use_bias=False, key=keys[1])
        self.v_proj = eqx.nn.Linear(d_model, d_model, use_bias=False, key=keys[2])
        self.out_proj = eqx.nn.Linear(d_model, d_model, key=keys[3])

    def __call__(self, x: jnp.ndarray, *, key: Optional[jax.Array] = None) -> jnp.ndarray:
        # x: (seq, d_model)
        seq_len, d_model = x.shape
        H, dh = self.n_heads, self.d_head

        Q = jax.vmap(self.q_proj)(x).reshape(seq_len, H, dh).transpose(1, 0, 2)
        K = jax.vmap(self.k_proj)(x).reshape(seq_len, H, dh).transpose(1, 0, 2)
        V = jax.vmap(self.v_proj)(x).reshape(seq_len, H, dh).transpose(1, 0, 2)

        scale = jnp.sqrt(jnp.array(dh, dtype=jnp.float32))
        scores = jnp.matmul(Q, K.transpose(0, 2, 1)) / scale   # (H, seq, seq)
        # Causal mask (for autoregressive use)
        mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
        scores = jnp.where(mask[None], scores, -1e9)
        attn = jax.nn.softmax(scores, axis=-1)                  # (H, seq, seq)

        out = jnp.matmul(attn, V)                               # (H, seq, dh)
        out = out.transpose(1, 0, 2).reshape(seq_len, d_model)  # (seq, d_model)
        return jax.vmap(self.out_proj)(out)


class TransformerBlock(eqx.Module):
    attn: MultiHeadAttention
    ff1: eqx.nn.Linear
    ff2: eqx.nn.Linear
    ln1: eqx.nn.LayerNorm
    ln2: eqx.nn.LayerNorm

    def __init__(self, d_model: int, n_heads: int, d_ff: int, *, key: jax.Array):
        keys = jax.random.split(key, 3)
        self.attn = MultiHeadAttention(d_model, n_heads, key=keys[0])
        self.ff1  = eqx.nn.Linear(d_model, d_ff, key=keys[1])
        self.ff2  = eqx.nn.Linear(d_ff, d_model, key=keys[2])
        self.ln1  = eqx.nn.LayerNorm((d_model,))
        self.ln2  = eqx.nn.LayerNorm((d_model,))

    def __call__(self, x: jnp.ndarray, *, key: Optional[jax.Array] = None) -> jnp.ndarray:
        # Pre-norm + attention
        x = x + self.attn(jax.vmap(self.ln1)(x), key=key)
        # Pre-norm + FFN
        def ffn(z): return self.ff2(jax.nn.gelu(self.ff1(z)))
        x = x + jax.vmap(ffn)(jax.vmap(self.ln2)(x))
        return x


class TransformerModel(eqx.Module):
    input_proj: eqx.nn.Linear
    blocks: List[TransformerBlock]
    output_proj: eqx.nn.Linear
    ln_final: eqx.nn.LayerNorm
    d_model: int
    out_dim: int

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: Optional[int] = None,
        *,
        key: jax.Array,
    ):
        if d_ff is None:
            d_ff = 4 * d_model
        self.d_model = d_model
        self.out_dim = out_dim

        keys = jax.random.split(key, n_layers + 2)
        self.input_proj  = eqx.nn.Linear(in_dim, d_model, key=keys[0])
        self.blocks      = [TransformerBlock(d_model, n_heads, d_ff, key=keys[i + 1]) for i in range(n_layers)]
        self.output_proj = eqx.nn.Linear(d_model, out_dim, key=keys[-1])
        self.ln_final    = eqx.nn.LayerNorm((d_model,))

    def __call__(self, x: jnp.ndarray, *, key: Optional[jax.Array] = None) -> jnp.ndarray:
        # x: (lookback, in_dim)
        seq_len = x.shape[0]
        x = jax.vmap(self.input_proj)(x)           # (seq, d_model)
        pe = sinusoidal_pos_enc(seq_len, self.d_model)
        x = x + pe

        for block in self.blocks:
            x = block(x, key=key)

        x = jax.vmap(self.ln_final)(x)
        last = x[-1]                               # use last token
        return self.output_proj(last)


@register("transformer")
def build_transformer(config: Dict[str, Any], rng_key=None, sample_x=None):
    import jax
    key = rng_key if rng_key is not None else jax.random.PRNGKey(int(config.get("seed", 42)))

    input_dim = int(config.get("input_dim", 1))
    task      = config.get("task", "regression")
    num_cls   = int(config.get("num_classes", 2))
    horizon   = int(config.get("horizon", 1))
    out_dim   = output_dim_for_task(task, num_cls, horizon)
    d_model   = int(config.get("d_model", 128))
    n_heads   = int(config.get("n_heads", config.get("heads", max(1, d_model // 64))))
    n_layers  = int(config.get("layers", 4))
    d_ff      = int(config.get("d_ff", 4 * d_model))

    # Ensure d_model divisible by n_heads
    if d_model % n_heads != 0:
        n_heads = max(1, d_model // 64)

    return TransformerModel(
        input_dim, out_dim,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff,
        key=key,
    )
