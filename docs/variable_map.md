# Variable Map

Key variable / spec definitions for Sfogliatella. Updated whenever variables or specs change.

## CLI / Config Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `lookback` | int | 32 | Number of past time steps used as input window |
| `horizon` | int | 1 | Number of steps ahead to predict |
| `target_col` | int | 0 | Column index (0-based) of the target variable |
| `feature_cols` | list[int] | all except target | Column indices used as input features |
| `task` | str | "regression" | Task type: regression/classification/ranking/anomaly_score/probability/clustering/embedding |
| `num_classes` | int | 2 | Number of classes (classification only) |
| `model` | str | "mlp" | Model: mlp/lstm/rnn/transformer/cnn/xgboost/timesfm/timesfm_ft/custom |
| `run_id` | str | YYYYMMDD_HHMMSS | Run identifier |
| `model_dir` | str | "outputs/models" | Directory for model artifacts |
| `batch_size` | int | 32 | Training batch size |
| `epochs` | int | 10 | Number of training epochs |
| `lr` | float | 1e-3 | Learning rate |
| `seed` | int | 42 | Random seed |
| `val_ratio` | float | 0.1 | Fraction of data for validation |
| `test_ratio` | float | 0.1 | Fraction of data for testing |
| `folds` | int | 1 | Number of time-series folds (1 = no cross-validation) |
| `loss` | str | task-dependent | Loss function name |
| `precision` | str | "auto" | Numeric precision: auto/fp32/bf16/fp16 |

## Model Output Dimensions (output_dim_for_task)

| Task | num_classes | output_dim |
|------|------------|------------|
| regression | — | horizon |
| classification | ≤ 2 (binary) | 1 (sigmoid BCE) |
| classification | > 2 (multiclass) | num_classes (softmax CE) |
| ranking | — | horizon |
| anomaly_score | — | horizon |
| probability | — | horizon |
| embedding | — | 64 (default) |
| clustering | — | 64 (embedding for k-means) |

**Important:** Binary classification uses output_dim=1 (not 2). See ADR-001.

## Default Loss Functions

| Task | num_classes | Default Loss |
|------|------------|-------------|
| regression | — | mse |
| classification | ≤ 2 | binary_cross_entropy |
| classification | > 2 | cross_entropy |
| ranking | — | mse |
| anomaly_score | — | mse |
| probability | — | mse |
| clustering | — | mse |
| embedding | — | mse |

## Archive Format

| Key | Value |
|-----|-------|
| extension | `.sfog` |
| format | tar.gz containing weights.pkl + config.json + version.json |
