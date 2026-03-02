# General-Purpose Time-Series ML Library — Specification (GOAL)

## 1. Purpose
This library provides a **unified interface** for multiple models on time-series style tabular data. The primary use is **regression** (default), and it also supports **classification** (via arguments), plus secondary/optional task categories (ranking, anomaly scoring, probability outputs, clustering, embeddings). The library enables the following with **only import + CLI/API arguments**:

- Training (with checkpoints + resume)
- Hyperparameter prior / HPO helpers (optional)
- Packaging / loading pretrained models (serving)
- **Batch “strict” forecasting** from CSV/Parquet (no future leakage)
- **Live forecasting** that waits for CSV updates and predicts on each update
- Model stacking / multi-stage inference pipelines (optional)

## 2. Assumptions and Constraints

### 2.1 Assumptions
- All preprocessing (e.g., normalization) is completed **outside** this library. The library consumes already-preprocessed data.
- For CSV, the time order is **older at the top, newer at the bottom**.
- Execution environment is **Docker + Jupyter Notebook**. Notebooks must stay “thin” (mostly CLI calls).

### 2.2 Dependencies and Implementation Policy
- Use **JAX** as the primary compute backend.
- Use **JIT (jax.jit / pjit)** where possible for acceleration.
- **Do not introduce PyTorch or TensorFlow** (to avoid dependency conflicts and environment instability).
- If some components cannot be implemented with JAX, exceptions may be allowed, but the above “no PyTorch/TensorFlow” rule must still hold.

## 3. Supported Tasks and Models

### 3.1 Tasks (Primary and Secondary)
- **regression** (default)
- **classification** (enabled via arguments)
- Secondary / optional task categories (best-effort; specs kept lightweight):
  - **ranking** (score-based)
  - **anomaly_score** (anomaly scoring)
  - **probability** (probability outputs; e.g., event probability or distribution parameters)
  - **clustering** (cluster assignment over samples/windows)
  - **embedding** (vector embedding output)

Task selection:
- Live mode (`python -m live`) is intended for forecasting-style tasks (regression/classification/probability). For other tasks, `predict` is the default entry point.
- A run selects one task via `--task` (default: `regression`).
- The same model family must be runnable in **regression** and **classification** by switching the output head and loss.
- Secondary task categories may reuse shared heads (score head / embedding head) and reuse evaluation/export conventions.

#### 3.1.1 Task I/O Contracts (Minimum)
- **regression**:
  - output: `y_pred` (float)
- **classification**:
  - output: `logits` and optional `proba` (`softmax` for multiclass, `sigmoid` for binary)
  - config: `num_classes` (required for multiclass), and optional threshold for binary
- **ranking**:
  - output: `score` (float)
  - config: `group_col` / `query_id_col` (optional; index-based, since column names are ignored)
- **anomaly_score**:
  - output: `anomaly_score` (float; higher = more anomalous)
  - mode: supervised (binary label) or unsupervised (default; e.g., reconstruction error)
- **probability**:
  - output: event probability (`p_event`) or distribution parameters (e.g., `mu`,`sigma`) depending on `prob_mode`
- **clustering**:
  - output: `cluster_id` (int) and optional `distance_to_centroid`
  - config: `n_clusters` (or algorithm-defined)
- **embedding**:
  - output: `embedding` vector (float, length `embedding_dim`)

### 3.2 Planned Models
- Transformer
- LSTM
- RNN
- MLP
- CNN (e.g., 1D conv)
- XGBoost
- TimesFM (inference)
- TimesFM fine-tuned version (trained within this project)

### 3.3 Common Model Requirements
- Input: a past window of length `lookback` (features over time).
- Task outputs must follow the selected `--task` contract (above).
- Forecasting-style tasks (e.g., regression/classification/probability) must support:
  - single-step or multi-step via `horizon`
  - univariate or multivariate targets

### 3.4 Heads and Task-Aware Design (Per Model)
- Each built-in model implementation must provide at least:
  - a **regression head** (float outputs), and
  - a **classification head** (logits; and probabilities at export time).
- For secondary tasks, implementations may map them onto:
  - **score head** (ranking/anomaly/probability-event), and/or
  - **embedding head** (embedding; clustering may be implemented as embedding + clustering algorithm).

### 3.5 Model-Specific Parameters (Default vs Override)
- Model parameters (e.g., Transformer depth / heads / hidden size) may be computed by an internal method that returns **baseline values** believed to be near-optimal.
- If the user **does not specify** parameters, the library uses the baseline computation output.
- If the user **does specify** parameters, argument values **override** the defaults.
- Different models have different tunable parameters, so the interface must support **common args + model-specific args/config**.

### 3.6 User-Defined Models (Custom)
- Users can run their own model implementation (any supported task) by specifying a custom runner.
- The custom model may consist of multiple files (a package or directory), not only a single Python file.

Custom model integration (proposed):
- Use `--model custom` and provide:
  - `--custom_model_root` : path to a directory/package that contains the user model code
  - `--custom_model_call` : a common invocation command (string) to run the user model

Contract (minimum):
- The runner must support both `train` and `predict` modes.
- The runner must accept the library’s standardized config (JSON/YAML path or JSON string) and I/O paths, and write artifacts/predictions into the same output conventions (`model_dir`, `run_id`, `output_path`).
- The runner must follow the same leakage rules (time order) and must be able to load and run the produced artifacts on CPU/CUDA/TPU.

## 4. Data Specification

### 4.1 Supported Input Formats
- Training / batch prediction: **CSV, Parquet**
- Live prediction: **CSV only**

### 4.2 Column-Name Independent Management
- Do not depend on column names.
- After loading, internally assign sequential IDs like `col_0, col_1, ...`.
- Users specify target/features by **column index (0-based)**.

### 4.3 Time Order
- CSV rows: older → newer (top → bottom).
- Parquet: the loaded row order is assumed to already be chronological.

### 4.4 Passing Data
- In addition to file input, Python API accepts **jax.numpy (jnp) arrays** directly.
- If `numpy.ndarray` is provided, convert via `jnp.asarray`.

### 4.5 Task-Specific Column Conventions (Minimum)
- **Regression**: `--target_col` (or target indices in task/model config) indicates the numeric target.
- **Classification**: `--target_col` indicates the label column.
  - Binary: labels are 0/1 (int or float).
  - Multiclass: labels are integer in `[0, num_classes-1]`.
- **Ranking** (optional): group/query id can be provided via task params (index-based).
- **Unsupervised tasks** (clustering/embedding): labels may be absent; the task may ignore `--target_col`.


## 5. Functional Requirements

### 5.1 Training
- Train the selected model and save weights + metadata to `model_dir`.
- Time-series split must be **chronological** (no shuffling):
  - by indices, or
  - by ratios (e.g., train/val/test)

Minimum artifacts to save:
- Model weights / checkpoints (JAX/Flax/Eqx via orbax-checkpoint or equivalent)
- Training configuration (args, hyperparameters, `lookback/horizon`, target settings, etc.)
- Dependency/version info (reproducibility)

#### 5.1.1 Single-File Model Packaging (If Possible)
- Prefer not to scatter artifacts across multiple files.
- If possible, package into a **single archive with a custom extension**, e.g. `*.grml`, `*.rmlpack`, `*.jaxreg` (name to be decided).
- The archive should include at least:
  - weights/checkpoint (portable, host-readable)
  - config (common + model-specific)
  - version info
  - (optional) feature definition (e.g., `feature_cols`)
- Even if single-file packaging is not feasible internally, external usage should still accept **one path** (the library may manage a directory behind the scenes).

#### 5.1.2 Checkpoints and Resume
- Training must create checkpoints periodically (by steps and/or by epochs).
- Training must be resumable from:
  - latest checkpoint in `model_dir`, or
  - an explicitly specified checkpoint path.
- Resume must restore:
  - model weights
  - optimizer state
  - RNG state (as feasible)
  - step/epoch counters

#### 5.1.3 Loss Curves (Image + Numeric)
- Save training/validation loss curves as:
  - an **image** (e.g., PNG), and
  - a **numeric series** (e.g., CSV/JSON/NPY with step/epoch and value).
- Store both under `model_dir` (or inside the single-file package).

#### 5.1.4 Folds (Default = 1)
- Default training runs with **one fold** (no cross-validation split beyond train/val/test).
- If `--folds K` is specified (K ≥ 2), run **time-series aware fold splitting** and aggregate metrics.
- Fold definition and split indices must be saved for reproducibility.

#### 5.1.5 Loss Functions (Built-in + Custom Plugin)
- Major/common loss functions are built-in, selected by `--task` and `--loss`.
  - Regression examples: MSE, MAE, Huber, LogCosh, Quantile/Pinball, SMAPE, MAPE.
  - Classification examples: cross-entropy (multiclass), binary cross-entropy (sigmoid), focal (optional).
- A custom loss function must be supported via an external Python file:
  - `--loss_path path/to/loss.py`
  - The file must expose a known entry point (e.g., `build_loss(config)` or `loss_fn(y_pred, y_true, **kwargs)`).
- The chosen loss (name or file) and its config must be saved in the run artifacts.

#### 5.1.6 RunID and Returnable Results
- Each training/prediction run has a `run_id`.
- If not specified, default format is `YYYYMMDD_HHMMSS` (example: `20260211_062435`).
- If specified, use the provided string verbatim.
- Results are saved to files **and** can be returned as in-memory objects:
  - Python API returns a structured dict/object.
  - CLI can optionally emit the same structured result as JSON to stdout (for pipeline usage).

### 5.2 Pretrained Model Distribution / Loading
- Distribute either `model_dir/` or the single packaged archive.
- Loading must enforce consistency with saved settings (`lookback/horizon/target_dim`, etc.).
- Models trained on TPU must be saved in a format that is runnable on CPU/CUDA as well (portable checkpoints; avoid TPU-only sharding requirements).
- The saved model must be runnable on **any supported hardware** (CPU/CUDA/TPU) regardless of the hardware used for training.
- Avoid device-topology-specific requirements at load time (e.g., TPU-only sharding metadata). If sharded checkpoints are used during training, provide an export/consolidation path to a portable format.

### 5.3 Batch “Strict” Forecasting (No Leakage)
Goal: predict using **only data available up to that point**, preventing future leakage.

#### 5.3.1 Forecasting Method
- For each time `t`, use only the past `lookback` window and predict `t + horizon`.
- Output modes:
  - last N points (production-like)
  - full walk-forward over the whole range (evaluation)

#### 5.3.2 Output
- Save predictions to CSV/Parquet.
- Each prediction row should include at least:
  - prediction time/index (index-based is mandatory because column names are ignored)
  - task outputs (e.g., `y_pred`, `logits`, `proba`, `score`, `anomaly_score`, `cluster_id`, `embedding`)
  - (optional) ground truth (e.g., `y_true` / `y_label`) if available
- If an output file name/path is specified, it must be provided via an argument.

### 5.4 Live Forecasting (CSV Watch + Predict)
Goal: monitor CSV updates (appends) and run prediction on each update, then wait again.

#### 5.4.1 Assumptions
- The CSV grows by appending new rows at the end.
- Live forecasting supports **CSV only**.

#### 5.4.2 Behavior
- Detect updates via file mtime and/or row count increase.
- Ingest appended rows and maintain the required `lookback` buffer.
- Run prediction and append/overwrite output.
- Wait until the next update.

#### 5.4.3 Output
- Prediction CSV (e.g., `predictions_live.csv`)
- Logs (stdout and optionally file)
- If an output file name/path is specified, it must be provided via an argument.

### 5.5 Hyperparameter Prior / HPO Helper (Optional)
Goal: provide an optional “good default / starting point” configuration generator, and make the **reasoning** retrievable.

Minimum requirements:
- Compute baseline model structure / hyperparameters from:
  - dataset scale (samples, lookback, horizon, dims),
  - compute constraints (time budget, device memory), and
  - target loss/accuracy objective (optional).
- The HPO helper must support returning a “why” payload (depending on args), such as:
  - active constraint (target vs data-cap vs hardware)
  - intermediate derived quantities (e.g., ideal N, data cap N, estimated VRAM/time)
  - shrink steps taken (if any), with which parameter changed and why
  - caveats / calibration notes
- HPO output must be savable to file **and** returnable via the Python API.

### 5.6 Model Stacking / Multi-Stage Pipelines
The design must support stacking and more general multi-stage pipelines.

Minimum requirements:
- Specify, via arguments/config:
  - which models are used in each stage,
  - each model’s parameters and artifact paths,
  - which data each stage consumes (raw features, previous stage outputs, etc.),
  - which outputs are emitted at each stage,
  - how stage outputs are post-processed and fed into the next stage.
- Optional per-stage transform plugin:
  - If a stage output needs custom processing before the next stage, allow:
    - `--stage_transform_path path/to/transform.py`
  - The file must expose a known entry point (e.g., `transform(preds, context) -> new_features`).
- Prevent leakage:
  - provide time-series Out-of-Fold prediction generation when training a meta-model.
- Save and reproduce stacking configuration (full pipeline graph + versions + artifact refs).

## 6. API / CLI Design

### 6.1 Keep Notebooks Thin
Notebooks should call CLIs with minimal code, e.g.:

```bash
!python -m train --data_path data/train.csv --model transformer --target_col 0 --lookback 128 --horizon 1 --model_dir outputs/model_x
```

This avoids needing to update notebooks when implementation changes.

### 6.2 CLI Entry Points (Proposed)
Core:
- `python -m train` : training
- `python -m predict` : batch strict forecasting
- `python -m live` : live forecasting

Optional / pipeline-oriented (fine-grained):
- `python -m hpo` : generate baseline/HPO-prior config (and optionally reasoning payload)
- `python -m stack` : run stacking / multi-stage pipeline (train and/or predict)
- `python -m pack` / `python -m unpack` : single-file packaging utilities
- `python -m eval` : evaluation utilities (walk-forward, metrics, plots)

Each entry point should also have a Python-callable function so orchestration tools (e.g., code generators) can call small units directly.

### 6.3 Common CLI Arguments (Proposed)
- `--model` : model name (transformer / lstm / rnn / mlp / cnn / xgboost / timesfm / timesfm_ft)
- `--task` : task name (regression / classification / ranking / anomaly_score / probability / clustering / embedding) (default: regression)
- `--num_classes` : number of classes (required for multiclass classification)
- `--task_params` : task-specific parameters as JSON string (optional)
- `--task_params_path` : task-specific parameters JSON/YAML path (optional)
- `--model_dir` : save/load location (or archive path)
- `--data_path` : input data path
- `--data_format` : csv / parquet (auto-detect is acceptable)
- `--target_col` : target/label column index (0-based). For classification, this is the label column. For unsupervised tasks (e.g., clustering/embedding), it may be omitted or ignored by the task.
- `--feature_cols` : feature column indices (policy if omitted must be explicitly defined in implementation)
- `--lookback` : input window length
- `--horizon` : forecast horizon
- `--batch_size`, `--epochs`, `--lr`, `--seed`, etc.
- `--folds` : number of folds (default 1)
- `--loss` : built-in loss name
- `--loss_path` : path to custom loss python file (overrides `--loss` if provided)
- `--run_id` : run identifier (default `YYYYMMDD_HHMMSS`)
- `--output_path` : prediction output file path
- `--return_json` : emit structured JSON result to stdout (optional)
- `--precision` : numeric precision policy (e.g., `auto` / `fp32` / `bf16` / `fp16`)
- `--num_workers` : CPU workers for data processing (loading/windowing/eval) (default: auto)
- `--log_level` : logging level (error / warn / info / debug)
- `--log_items` : choose which log groups to show (e.g., `run,device,data,train,health,io`)
- `--no_banner` : do not print the startup ASCII-art banner (optional)
- `--show_tpu_warnings` : show TPU initialization warnings if they are suppressible (optional)
- `--custom_model_root` : path to user model code (used when `--model custom`)
- `--custom_model_call` : common invocation command for the user model runner (used when `--model custom`)

Device / parallel:
- `--device` : cpu / cuda / tpu (or auto)
- `--num_devices` : number of devices to use (for pmap/pjit sharding)
- `--parallel_mode` : none / pmap / pjit (exact policy to be defined)

#### 6.3.1 Model-Specific Parameter Passing (Proposed)
- Use baseline auto-computation if not provided.
- Override with user-specified values if provided.
- Passing methods (either):
  - `--model_params` as a JSON string, e.g. `'{"d_model":256,"n_heads":8}'`
  - `--model_params_path` pointing to a JSON/YAML file
- Each model defines its own schema. Handling of unknown keys (error vs ignore) will be decided during implementation.

#### 6.3.2 Stacking / Pipeline Config Passing (Proposed)
- `--pipeline` as JSON/YAML describing stages (models, inputs, outputs, transforms)
- or `--pipeline_path` pointing to JSON/YAML
- Ensure the pipeline config is saved (embedded in `model_dir` / package).

### 6.4 Python API (Proposed)
- `library.load_model(model_dir_or_pack)`
- `library.train(config, data) -> TrainResult`
- `library.predict(config_or_model, data, output_path=None) -> PredictResult`
- `library.live_predict(config_or_model, csv_path, output_path=None, ...)`
- `library.hpo(config, baseline_measurements=None, constraints=...) -> HPOResult`
- `library.stack(pipeline_config, data, ...) -> StackResult`

All result objects must be serializable (for file saving) and also usable directly as return values.

## 7. Internal Architecture

### 7.1 Package Structure (Proposed)
- `core/` : data loading, windowing, evaluation, shared utilities
- `models/` : model implementations
- `registry/` : mapping model name → implementation class
- `io/` : checkpoints, metadata, output formats, pack/unpack (single-file packaging)
- `hpo/` : baseline/HPO-prior solver + reasoning payload export
- `losses/` : built-in losses + custom-loss loader
- `plugins/` : plugin loading (custom loss / stage transforms)
- `stacking/` : stacking and general multi-stage pipelines
- `devices/` : device selection, pmap/pjit wrappers, portability helpers
- `cli/` : train/predict/live/hpo/stack/... entry points

### 7.2 Common Model Interface (Concept)
- `init(rng, sample_batch)`
- `fit(dataset, ...)` (supports checkpoint + resume)
- `predict(batch)`
- `save(path)` / `load(path)` (portable checkpoints)

### 7.3 JIT / Performance
- JIT where possible:
  - forward/predict
  - loss computation
  - gradient update step
- Do not JIT I/O (CSV/Parquet).
- Data processing is parallelized by default (CPU-side), such as:
  - CSV/Parquet reading (when safe),
  - window generation / batching,
  - evaluation metrics and plotting exports.
- Concurrency is controlled by `--num_workers`, while preserving chronological order for time-series.
- Parallelization support:
  - `pmap` for data-parallel training where appropriate
  - `pjit` for sharding where appropriate
  - number of devices controlled by arguments

## 8. Compatibility and Dependencies
- Default precision aims for **FP32-equivalent** behavior. Prefer BF16 when supported (especially on TPU) unless overridden.
- If specified by arguments, allow lower precision such as FP16 for performance.
- All exported artifacts must remain runnable across CPU/CUDA/TPU.
- Python version fixed in Docker image (e.g., 3.10/3.11) and pinned in requirements.
- Core deps: `jax`, `jaxlib`, and optionally `flax` / `equinox` / `optax` / `orbax-checkpoint`.
- Extra deps:
  - Parquet: `pyarrow`
  - XGBoost: `xgboost` (without TF/PyTorch)
  - File watch: `watchdog` (for live prediction)
- Must support CPU, CUDA, and TPU execution through JAX.
- TPU-trained artifacts must be runnable on CPU/CUDA (portable checkpoint export).

## 9. Error Handling
- Minimum length check: `len(data) >= lookback + horizon`
- Live prediction:
  - handle updates without row increases (overwrite)
  - retry on read failures while the file is being written
- Column index range checks
- Model-specific parameter checks:
  - unknown keys
  - type mismatch
  - consistency constraints (e.g., `d_model % n_heads == 0`)
- Plugin checks:
  - validate that `loss_path` / `transform_path` exports expected entry points
  - provide clear errors for import failures

## 10. Artifacts and Operation
- All outputs are organized under `model_dir` (or inside the single-file package) by `run_id`.
- Training outputs (per run):
  - checkpoints (with resume metadata)
  - config + versions
  - metrics (JSON/CSV)
  - loss curves (PNG + numeric series)
- Batch prediction output: `predictions.csv` / `predictions.parquet` (overridable via `--output_path`)
- Live prediction output: `predictions_live.csv` (overridable via `--output_path`)
- Optional: write a single structured `result.json` per run that mirrors the Python return object.


## 11. Logging, Diagnostics, and Startup UX

### 11.1 Startup Banner + Version
- On startup, logs must print a **large ASCII-art banner** for the system name: `Sfogliatella`.
- Immediately after the banner, print the library version (and if available: commit hash / build info).

### 11.2 Warnings (TPU init)
- Some environments emit TPU initialization warnings. If these warnings are suppressible without hiding actionable errors, suppress them by default.
- Provide a flag (e.g., `--show_tpu_warnings`) to re-enable them.

### 11.3 Logging Policy (Not Verbose, But Detects Failure)
- Default logs should be minimal and focused:
  - run_id, device/precision, dataset shapes, key hyperparameters
  - periodic training progress (train/val loss)
  - artifact paths written
- Logs must still make it **obvious** when training is going in a meaningless direction (without being verbose), such as:
  - NaN/Inf detection
  - loss divergence / explosion
  - stagnation beyond a threshold
  - validation loss worsening for sustained periods
- Allow selecting which log groups to print via arguments (e.g., `--log_items`), and coarse verbosity via `--log_level`.


---

## Appendix: Terms
- **lookback**: length of past window used for prediction
- **horizon**: how many steps ahead to predict
- **strict forecasting**: forecasting without using any future information (no leakage)
- **stacking**: combining multiple model outputs via a meta-model for final prediction
- **pipeline**: a multi-stage graph of models + transforms defined by config
- **run_id**: identifier for a training/prediction run (default `YYYYMMDD_HHMMSS`)
