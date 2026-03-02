"""Classification loss functions (JAX)."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def cross_entropy_loss(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Multiclass cross-entropy. logits: (B, C), labels: (B,) int."""
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    labels_int = labels.astype(jnp.int32)
    return -jnp.mean(log_probs[jnp.arange(len(labels_int)), labels_int])


def binary_cross_entropy_loss(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Binary cross-entropy with logits. logits and labels: (B,)."""
    return jnp.mean(
        jnp.maximum(logits, 0) - logits * labels + jnp.log1p(jnp.exp(-jnp.abs(logits)))
    )


def focal_loss(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> jnp.ndarray:
    """Binary focal loss."""
    proba = jax.nn.sigmoid(logits)
    ce = binary_cross_entropy_loss(logits, labels)
    p_t = jnp.where(labels == 1, proba, 1.0 - proba)
    focal_weight = alpha * (1.0 - p_t) ** gamma
    return jnp.mean(focal_weight * ce)
