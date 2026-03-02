"""Data loading, windowing, and preprocessing utilities."""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_data(
    path: Union[str, Path],
    data_format: Optional[str] = None,
) -> np.ndarray:
    """Load CSV or Parquet into a 2-D numpy array (rows x columns).

    Column names are dropped; caller works with column indices only.
    Time order is preserved (no shuffling).
    """
    path = Path(path)
    fmt = data_format or _detect_format(path)

    if fmt == "csv":
        return _load_csv(path)
    elif fmt == "parquet":
        return _load_parquet(path)
    else:
        raise ValueError(f"Unsupported data format: {fmt!r}. Use 'csv' or 'parquet'.")


def _detect_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".csv", ".tsv"):
        return "csv"
    elif suffix in (".parquet", ".pq"):
        return "parquet"
    raise ValueError(f"Cannot auto-detect format from extension: {path.suffix!r}")


def _load_csv(path: Path) -> np.ndarray:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # skip header (column names ignored)
        if header is None:
            raise ValueError(f"Empty CSV: {path}")
        for row in reader:
            if row:
                rows.append([float(v) for v in row])
    if not rows:
        raise ValueError(f"No data rows in CSV: {path}")
    arr = np.array(rows, dtype=np.float32)
    logger.debug("Loaded CSV %s: shape=%s", path, arr.shape)
    return arr


def _load_parquet(path: Path) -> np.ndarray:
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise ImportError("pyarrow is required for Parquet support. Install with: pip install pyarrow") from e
    table = pq.read_table(path)
    arr = table.to_pandas().values.astype(np.float32)
    logger.debug("Loaded Parquet %s: shape=%s", path, arr.shape)
    return arr


# ---------------------------------------------------------------------------
# Column resolution
# ---------------------------------------------------------------------------

def resolve_feature_cols(
    n_cols: int,
    target_col: int,
    feature_cols: Optional[List[int]] = None,
) -> List[int]:
    """Return feature column indices.

    If feature_cols is None, use all columns except target_col.
    """
    if feature_cols is not None:
        for c in feature_cols:
            if c < 0 or c >= n_cols:
                raise IndexError(f"feature_col {c} out of range [0, {n_cols-1}]")
        return list(feature_cols)
    return [c for c in range(n_cols) if c != target_col]


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def make_windows(
    data: np.ndarray,
    lookback: int,
    horizon: int,
    target_col: int,
    feature_cols: Optional[List[int]] = None,
    target_cols: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create (X, y) windows without leakage.

    X shape: (N_windows, lookback, n_features)
    y shape: (N_windows, horizon, n_targets)  or (N_windows, n_targets) if horizon==1

    For each window i:
        X[i] = data[i : i+lookback, feature_cols]
        y[i] = data[i+lookback : i+lookback+horizon, target_col(s)]

    Time order is preserved (no shuffling).
    """
    n_rows, n_cols = data.shape

    # Resolve columns
    if target_cols is None:
        target_cols = [target_col]
    for tc in target_cols:
        if tc < 0 or tc >= n_cols:
            raise IndexError(f"target_col {tc} out of range [0, {n_cols-1}]")

    fcols = resolve_feature_cols(n_cols, target_col, feature_cols)

    min_rows = lookback + horizon
    if n_rows < min_rows:
        raise ValueError(
            f"Not enough rows: need at least lookback+horizon={min_rows}, got {n_rows}"
        )

    n_windows = n_rows - lookback - horizon + 1
    Xs = []
    ys = []

    feat_data = data[:, fcols]
    tgt_data = data[:, target_cols]

    for i in range(n_windows):
        Xs.append(feat_data[i : i + lookback])          # (lookback, n_feat)
        ys.append(tgt_data[i + lookback : i + lookback + horizon])  # (horizon, n_tgt)

    X = np.stack(Xs, axis=0).astype(np.float32)   # (N, lookback, n_feat)
    y = np.stack(ys, axis=0).astype(np.float32)   # (N, horizon, n_tgt)

    # Squeeze horizon and target dims for common cases
    if y.shape[2] == 1:
        y = y[:, :, 0]          # (N, horizon)
    if y.shape[1] == 1:
        y = y[:, 0]             # (N,) or (N, n_tgt) already squeezed

    logger.debug("Windows: X=%s y=%s", X.shape, y.shape)
    return X, y


# ---------------------------------------------------------------------------
# Chronological split
# ---------------------------------------------------------------------------

def chronological_split(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """Split (X, y) chronologically into train/val/test without shuffling."""
    n = len(X)
    n_test = max(1, int(n * test_ratio))
    n_val  = max(1, int(n * val_ratio))
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError(f"Dataset too small for split: n={n}, val={n_val}, test={n_test}")

    X_train, y_train = X[:n_train], y[:n_train]
    X_val,   y_val   = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test,  y_test  = X[n_train+n_val:], y[n_train+n_val:]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def time_series_folds(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int,
    val_ratio: float = 0.1,
) -> List[Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]:
    """Generate n_folds time-series aware splits (expanding window).

    Returns list of (train_split, val_split).
    """
    n = len(X)
    folds = []
    fold_size = n // (n_folds + 1)

    for k in range(n_folds):
        train_end = fold_size * (k + 1)
        val_end   = min(n, train_end + max(1, int(train_end * val_ratio)))
        X_tr, y_tr = X[:train_end], y[:train_end]
        X_vl, y_vl = X[train_end:val_end], y[train_end:val_end]
        if len(X_tr) == 0 or len(X_vl) == 0:
            continue
        folds.append(((X_tr, y_tr), (X_vl, y_vl)))

    return folds


# ---------------------------------------------------------------------------
# Batch iterator
# ---------------------------------------------------------------------------

def make_batches(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return list of (X_batch, y_batch). No shuffle for time-series by default."""
    n = len(X)
    indices = np.arange(n)
    if shuffle and rng is not None:
        rng.shuffle(indices)

    batches = []
    for start in range(0, n, batch_size):
        idx = indices[start : start + batch_size]
        batches.append((X[idx], y[idx]))
    return batches
