"""Regression loss functions (JAX)."""

from __future__ import annotations

import jax.numpy as jnp


def mse_loss(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((y_pred - y_true) ** 2)


def mae_loss(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.abs(y_pred - y_true))


def huber_loss(y_pred: jnp.ndarray, y_true: jnp.ndarray, delta: float = 1.0) -> jnp.ndarray:
    diff = jnp.abs(y_pred - y_true)
    return jnp.mean(
        jnp.where(diff <= delta, 0.5 * diff ** 2, delta * (diff - 0.5 * delta))
    )


def log_cosh_loss(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
    diff = y_pred - y_true
    return jnp.mean(jnp.log(jnp.cosh(diff + 1e-12)))


def quantile_loss(y_pred: jnp.ndarray, y_true: jnp.ndarray, tau: float = 0.5) -> jnp.ndarray:
    diff = y_true - y_pred
    return jnp.mean(jnp.where(diff >= 0, tau * diff, (tau - 1.0) * diff))


def smape_loss(y_pred: jnp.ndarray, y_true: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    denom = (jnp.abs(y_true) + jnp.abs(y_pred)) / 2.0 + eps
    return jnp.mean(jnp.abs(y_pred - y_true) / denom)


def mape_loss(y_pred: jnp.ndarray, y_true: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    return jnp.mean(jnp.abs((y_pred - y_true) / (jnp.abs(y_true) + eps)))
