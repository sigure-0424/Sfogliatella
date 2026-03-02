"""Loss functions for all tasks."""

from sfogliatella.losses.regression import (
    mse_loss, mae_loss, huber_loss, log_cosh_loss,
    quantile_loss, smape_loss, mape_loss,
)
from sfogliatella.losses.classification import (
    cross_entropy_loss, binary_cross_entropy_loss, focal_loss,
)
from sfogliatella.losses.loader import load_loss_fn, get_default_loss

__all__ = [
    "mse_loss", "mae_loss", "huber_loss", "log_cosh_loss",
    "quantile_loss", "smape_loss", "mape_loss",
    "cross_entropy_loss", "binary_cross_entropy_loss", "focal_loss",
    "load_loss_fn", "get_default_loss",
]
