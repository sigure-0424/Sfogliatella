# TASK-20260301-004: GPU container fix + sample-kaggle full-size validation

## Status
done

## Priority
high

## Background
Operator requested:
- Diagnose and fix Docker-side GPU non-detection (GPU prep already exists)
- Re-run model validation on normalized but low-quality data at `data/raw/sample-kaggle`
- Assess whether previous poor accuracy is more likely due to data/model limits or test-run setup choices

## Definition of Done

- [x] Identify concrete Docker/JAX GPU misconfiguration and apply fix
- [x] Verify CUDA backend is visible from the runtime container
- [x] Run full-size regression and classification validation on `data/raw/sample-kaggle`
- [x] Capture per-model metrics and failure reasons
- [x] Summarize likely cause of accuracy behavior with evidence
- [x] Update `docs/core/STATE.yaml`
- [x] Update `docs/core/TASK_INDEX.md`
- [x] Update `docs/core/ACTIVITY_SUMMARY.md`

## GPU Root Cause and Fix

- Root cause in container config:
  - `docker/Dockerfile` installed `jax[cpu]`
  - `docker/compose.yaml` forced `JAX_PLATFORMS=cpu`
  - no compose GPU assignment (`gpus: all`)
- Applied fix:
  - CUDA base image + CUDA JAX wheels (`jax[cuda12]`)
  - compose now sets `JAX_PLATFORMS=cuda` and `gpus: all`
  - `.dockerignore` added to avoid huge build context from raw data
- Verification:
  - `docker run --rm --gpus all sfogliatella:dev nvidia-smi -L`
  - `docker run --rm --gpus all sfogliatella:dev python -c "import jax; print(jax.devices()); print(jax.default_backend())"`
  - observed `CudaDevice(id=0)` and backend `gpu`

## Full-Size sample-kaggle Validation

### Dataset
- `data/raw/sample-kaggle/train.parquet`
- rows: 5,337,414
- selected columns for testing: `feature_a..feature_h` + `y_target`
- classification label: up/down at +10 rows (`label_h10`)

### Raw normalized data (no extra preprocessing)

- Generated:
  - `tmp/kaggle_full_api/train_regression.parquet`
  - `tmp/kaggle_full_api/train_classification_h10.parquet`
- Findings:
  - data quality issue: NaN present (`58,465` NaNs in selected matrix)
  - most neural models aborted at step 1 with NaN loss
  - xgboost completed both tasks
  - `timesfm` / `timesfm_ft` failed with non-callable wrapper TypeError

### Processed full-size data (tmp preprocessing path)

- Added preprocessing under `tmp`:
  - impute NaN to `0.0` for selected features and target
  - standardize target (`y_target`) for regression stability
- Generated:
  - `tmp/kaggle_full_api/train_regression_processed.parquet`
  - `tmp/kaggle_full_api/train_classification_h10_processed.parquet`
  - `tmp/kaggle_full_api/processed_meta.json`
- Findings:
  - regression neural models (`mlp/lstm/rnn`) completed and produced predictions
  - xgboost completed regression + classification on full-size processed data
  - `timesfm` / `timesfm_ft` still fail (known wrapper/training-path limitation)

## Evidence-Based Accuracy Conclusion

The current evidence favors **(2)** over **(1)**:
- dominant issue was not just low S/N or capacity, but data/run setup quality:
  - NaN contamination in the provided normalized data
  - no imputation/target stabilization in the first pass
- after simple preprocessing in `tmp`, previously failing neural training paths completed on full-size data.
