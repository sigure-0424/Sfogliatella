# TASK-20260301-003: FX subset all-models validation (regression + classification H10)

## Status
done

## Priority
high

## Background
Operator requested end-to-end validation using FX data prepared in `data/raw`, with:
- Regression train + predict for all models
- Classification train + predict for up/down target 10 rows ahead
- Verification that the pipeline runs correctly and has no abnormal accuracy issues

## Definition of Done

- [x] Create bounded subsets from `data/raw` for runnable validation
- [x] Run regression train + predict on all available built-in models
- [x] Run classification train + predict on all available built-in models using `UpDownH10`
- [x] Capture metrics and run status in a consolidated report
- [x] Evaluate for anomalies (failures, NaN/Inf, pathological metrics)
- [x] Update `docs/core/STATE.yaml`
- [x] Update `docs/core/TASK_INDEX.md`
- [x] Update `docs/core/ACTIVITY_SUMMARY.md`

## Execution Notes

- GPU attempt was requested and executed, but container JAX backend did not expose CUDA (`known backends: ['cpu', 'tpu']`), so runs were retried on CPU.
- Validation subset artifacts:
  - `data/sample/fx_regression_subset.csv`
  - `data/sample/fx_classification_h10_subset.csv`
- Consolidated run report:
  - `tmp/fx_validation/out/summary.csv`
- Per-run outputs and metrics:
  - `tmp/fx_validation/out/json/`
  - `tmp/fx_validation/out/preds/`

## Observed Issues

- `timesfm` and `timesfm_ft` train failed for both regression/classification with:
  - `TypeError: Expected a callable value, got <TimesFMWrapper/...>`
- Classification quick pass showed degenerate behavior on this subset:
  - neural models predicted positive class for all rows (`pred_rate=1.0`), producing identical metrics
  - indicates pipeline execution is functional but classification quality is not healthy under this short-run setting
