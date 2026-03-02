# Knowledge Base

## Architecture Decisions

### Backend
- **JAX** is the primary compute backend
- **equinox** for neural network modules (pytree-based, pure JAX)
- **optax** for optimization
- **numpy/pickle** for portable checkpointing

### Column Management
- All data columns renamed internally to `col_0, col_1, ...`
- Users specify columns by 0-based index
- Target columns specified via `--target_col`

### Time Order
- CSV: older rows at top, newer rows at bottom
- No shuffling for time-series splits

### Model Heads
- regression head: linear output, float
- classification head: logits → softmax/sigmoid
- score head: single float (ranking/anomaly)
- embedding head: vector output

### Checkpointing
- Saved as `.pkl` (pickle of numpy arrays) for portability
- Can be loaded on CPU/CUDA/TPU regardless of training hardware

### Single-file Packaging
- `.sfog` extension (tar.gz archive containing weights + config + version)

### TimesFM
- TimesFM is an optional dependency — never hard-imported at module level
- `sfogliatella/models/timesfm_model.py` provides `TimesFMWrapper` (inference) and `TimesFMFTWrapper` (adapter fine-tuning)
- Registered as `"timesfm"` and `"timesfm_ft"` in args.py choices
- If not installed, raises `ImportError` with install instructions at build time

### Custom Model
- `--model custom` requires `--custom_model_call` (subprocess command)
- Optional `--custom_model_root` injected into PYTHONPATH
- Train: writes config.json temp file, calls `<cmd> --config <path> --mode train`
- Predict: saves X.npy, calls `<cmd> --config <path> --mode predict`, reads preds.npy

### Classification Output Shape
- Binary classification (num_classes≤2): output_dim=1, loss=binary_cross_entropy
- Multiclass (num_classes>2): output_dim=num_classes, loss=cross_entropy
- `output_dim_for_task()` returns 1 for binary case (critical: do not revert to 2)

## Anti-Regression Notes
- Never shuffle time-series data
- Never depend on column names (always use index)
- No PyTorch or TensorFlow imports anywhere
- XGBoost is CPU-trained but prediction integrated into pipeline
- Binary classification output_dim=1 (not 2) — BCE loss expects scalar logits
- TimesFM is optional-import only — do not add to hard requirements
