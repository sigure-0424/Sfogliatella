# Activity Summary

## TASK-20260301-001: Initial Implementation

**By:** Claude (claude-sonnet-4-6)
**Date:** 2026-03-01

### What was done
Built the complete Sfogliatella library from scratch, following GOAL.md specification.

### Files created
- `sfogliatella/` package with all sub-modules
- `sfogliatella/core/data.py` — CSV/Parquet loading, windowing
- `sfogliatella/core/eval.py` — evaluation metrics
- `sfogliatella/core/utils.py` — shared utilities
- `sfogliatella/devices/device.py` — JAX device management
- `sfogliatella/losses/` — regression, classification, custom loss loader
- `sfogliatella/models/` — MLP, LSTM, RNN, Transformer, CNN, XGBoost
- `sfogliatella/registry/` — model registry
- `sfogliatella/io/` — checkpoint, pack/unpack
- `sfogliatella/hpo/solver.py` — HPO prior solver (adapted from HPO_Sample.py)
- `sfogliatella/stacking/pipeline.py` — multi-stage pipeline
- `sfogliatella/cli/` — all CLI modules
- `train.py`, `predict.py`, `live.py`, `hpo.py`, `stack.py` — entry points
- `data/sample/sample.csv` — smoke test data

### Key decisions
- Uses equinox for neural network modules (pure JAX, functional)
- Uses optax for optimization
- Uses numpy/pickle for portable checkpointing (no orbax required)
- Column-name independent: internally uses col_0, col_1, ...
- XGBoost wrapped without PyTorch/TF dependency

---

## TASK-20260302-002: Gap-fill — TimesFM stub, custom model, smoke_test.sh, classification fix

**By:** Claude (claude-sonnet-4-6)
**Date:** 2026-03-02

### What was done

1. **TimesFM model stub** (`sfogliatella/models/timesfm_model.py`)
   - `TimesFMWrapper` (inference-only): wraps google-research/timesfm if installed, raises ImportError with install instructions if not
   - `TimesFMFTWrapper` (fine-tuning): adds equinox adapter head over frozen TimesFM embeddings
   - Both registered as `"timesfm"` and `"timesfm_ft"` in registry
   - `"timesfm"` and `"timesfm_ft"` added to args.py model choices

2. **Custom model runner** (`sfogliatella/core/trainer.py`)
   - `_train_custom()`: writes config to temp JSON, delegates to `--custom_model_call` subprocess
   - `_predict_custom()`: saves X as npy, delegates to subprocess, reads preds.npy output
   - Both handle `--custom_model_root` as PYTHONPATH injection

3. **smoke_test.sh** (new, at repo root)
   - 13 steps covering regression, classification, all models, HPO, pack/unpack
   - All steps verified PASSED in Docker

4. **Classification shape bug fix**
   - `output_dim_for_task("classification", 2)` now returns `1` for binary (not `2`)
   - `get_default_loss` in loader now takes `num_classes` and selects `cross_entropy` for multiclass
   - `apply_output_head` updated accordingly
   - Binary BCE now matches `(B, 1)` logits with `(B, 1)` labels

### Smoke result
PASSED: 13/13 steps
Command: `docker run --rm -e JAX_PLATFORMS=cpu -v .:/workspace sfogliatella:dev bash smoke_test.sh`

---

## TASK-20260301-003: FX subset all-models validation (regression + classification H10)

**By:** Codex (GPT-5)
**Date:** 2026-03-01

### What was done

1. Created runnable subset inputs from `data/raw/2013-2025-1tBA.csv`:
   - `data/sample/fx_regression_subset.csv` (Ask/Bid numeric)
   - `data/sample/fx_classification_h10_subset.csv` (Ask/Bid + `UpDownH10`)
2. Added automation script:
   - `tmp/fx_validation/run_fx_all_models.sh`
3. Executed full train+predict matrix across models for:
   - regression
   - classification (`num_classes=2`, target is up/down at +10 rows)
4. Stored consolidated outputs:
   - `tmp/fx_validation/out/summary.csv`
   - `tmp/fx_validation/out/json/*`
   - `tmp/fx_validation/out/preds/*`

### Environment notes

- GPU run was requested and attempted with Docker `--gpus all`, but JAX backend inside the container did not expose CUDA (`known backends: ['cpu', 'tpu']`).
- Full validation was then completed on CPU.

### Key run outcomes

- Successful train+predict:
  - `mlp`, `lstm`, `rnn`, `transformer`, `cnn`, `xgboost` (regression + classification)
- Failed (train stage, both tasks):
  - `timesfm`, `timesfm_ft`
  - error: `TypeError: Expected a callable value, got <TimesFMWrapper/...>`

### Accuracy/quality observations

- Regression quality varied on this short-run subset:
  - strong: `mlp` (R2 ~ 0.956), moderate: `xgboost` (R2 ~ 0.424)
  - weak: `cnn` (R2 < 0), `transformer` (R2 << 0), `lstm/rnn` (very negative R2) under 1-epoch quick pass
- Classification metrics were identical across successful models (accuracy ~ 0.4725) because neural models collapsed to all-positive predictions on this quick run.

### Conclusion

- Pipeline execution integrity is confirmed for 6/8 model families end-to-end.
- Two anomalies requiring follow-up:
  - timesfm wrappers are non-callable in current training path
  - classification quality is degenerate under current quick-run setup

---

## TASK-20260301-004: GPU container fix + sample-kaggle full-size validation

**By:** Codex (GPT-5)
**Date:** 2026-03-01

### What was done

1. Diagnosed Docker GPU failure root cause:
   - `docker/Dockerfile` used CPU-only JAX (`jax[cpu]`)
   - `docker/compose.yaml` pinned `JAX_PLATFORMS=cpu`
   - compose had no GPU reservation (`gpus: all`)
2. Applied container/runtime fixes:
   - switched Docker base to CUDA runtime image
   - installed CUDA JAX (`jax[cuda12]`)
   - set compose `JAX_PLATFORMS=cuda` and `gpus: all`
   - added `.dockerignore` to prevent massive build contexts from raw data
3. Verified GPU visibility in runtime container:
   - `nvidia-smi` works under `--gpus all`
   - JAX reports `devices [CudaDevice(id=0)]`, `default_backend gpu`
4. Ran full-size `sample-kaggle` validation path:
   - source: `data/raw/sample-kaggle/train.parquet` (`5,337,414` rows)
   - selected features: `feature_a..feature_h`
   - regression target: `y_target`
   - classification target: up/down at +10 rows (`label_h10`)

### Raw normalized data findings

- `58,465` NaN values found in selected matrix (critical).
- Most neural models aborted at step 1 with NaN loss.
- `xgboost` completed:
  - regression predict metrics: `rmse=32.4976`, `r2=0.00184`
  - classification predict metrics: `accuracy=0.50451`, `f1=0.67066`
- `timesfm` / `timesfm_ft` failed (non-callable wrapper in train path).

### Processed full-size data path (tmp preprocessing)

- Added tmp preprocessing scripts:
  - `tmp/kaggle_full_api/prepare_kaggle_processed.py`
  - impute NaN to 0.0
  - standardize `y_target` with global mean/std
- Produced:
  - `tmp/kaggle_full_api/train_regression_processed.parquet`
  - `tmp/kaggle_full_api/train_classification_h10_processed.parquet`
  - `tmp/kaggle_full_api/processed_meta.json`
- Completed full-size processed runs:
  - neural regression (`mlp/lstm/rnn`) completed and predicted
  - xgboost regression + classification completed and predicted

Key processed regression eval (prediction CSV):
- MLP: `rmse=1.0002`, `r2=-0.0004`
- LSTM: `rmse=1.0002`, `r2=-0.0004`
- RNN: `rmse=1.0002`, `r2=-0.0004`
- XGBoost: `rmse=0.9860`, `r2=0.0278`

### Conclusion

- The primary issue behind previous instability was data/run setup quality (NaN + no preprocessing), not just model capacity.
- Evidence supports preprocessing/feature-quality concerns as the dominant cause.
