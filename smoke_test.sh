#!/usr/bin/env bash
# Sfogliatella smoke test — runs inside Docker container at /workspace
# Usage: bash smoke_test.sh
# All steps must pass; any failure exits with code 1.

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
DATA="${WORKSPACE}/data/sample/sample.csv"
OUT="${WORKSPACE}/tmp/smoke_out"
MODELS_DIR="${OUT}/models"

rm -rf "${OUT}"
mkdir -p "${OUT}" "${MODELS_DIR}"

pass() { echo "[PASS] $1"; }
fail() { echo "[FAIL] $1"; exit 1; }

cd "${WORKSPACE}"

# ── 1. MLP train (regression, 3 epochs) ────────────────────────────────────
echo "==> MLP train (3 epochs)"
python -m train \
  --data_path "${DATA}" \
  --model mlp --task regression \
  --target_col 0 --lookback 10 --horizon 1 \
  --epochs 3 --batch_size 16 \
  --model_dir "${MODELS_DIR}" --run_id smoke_mlp \
  --no_banner --log_level warn \
  && pass "MLP train (3 epochs)" || fail "MLP train (3 epochs)"

# ── 2. MLP resume (→ 5 epochs) ─────────────────────────────────────────────
echo "==> MLP resume (→ 5 epochs)"
python -m train \
  --data_path "${DATA}" \
  --model mlp --task regression \
  --target_col 0 --lookback 10 --horizon 1 \
  --epochs 5 --batch_size 16 \
  --model_dir "${MODELS_DIR}" --run_id smoke_mlp \
  --no_banner --log_level warn \
  && pass "MLP resume (→ 5 epochs)" || fail "MLP resume (→ 5 epochs)"

# ── 3. MLP batch predict ────────────────────────────────────────────────────
echo "==> MLP batch predict"
python -m predict \
  --data_path "${DATA}" \
  --model mlp --task regression \
  --target_col 0 --lookback 10 --horizon 1 \
  --model_dir "${MODELS_DIR}" --run_id smoke_mlp \
  --output_path "${OUT}/preds_mlp.csv" \
  --no_banner --log_level warn \
  && pass "MLP batch predict" || fail "MLP batch predict"

# ── 4. eval metrics ─────────────────────────────────────────────────────────
echo "==> eval metrics"
python -m eval \
  --predictions_path "${OUT}/preds_mlp.csv" \
  --task regression \
  --no_banner --log_level warn \
  && pass "eval metrics" || fail "eval metrics"

# ── 5. HPO all 5 JAX models ─────────────────────────────────────────────────
echo "==> HPO (5 models)"
for MODEL in mlp lstm rnn transformer cnn; do
  python -m hpo \
    --model "${MODEL}" \
    --num_train_samples 1000 --lookback 10 --horizon 1 --input_dim 5 \
    --no_banner --log_level warn \
    && pass "HPO ${MODEL}" || fail "HPO ${MODEL}"
done

# ── 6. HPO with scaling law + JSON output ───────────────────────────────────
echo "==> HPO scaling law + JSON"
python -m hpo \
  --model mlp \
  --num_train_samples 1000 --lookback 10 --horizon 1 --input_dim 5 \
  --return_json \
  --no_banner --log_level warn \
  > "${OUT}/hpo_result.json" \
  && pass "HPO scaling law + JSON" || fail "HPO scaling law + JSON"

# ── 7. Pack model to .sfog ───────────────────────────────────────────────────
echo "==> Pack model"
python -m pack \
  "${MODELS_DIR}" smoke_mlp \
  --output "${OUT}/smoke_mlp.sfog" \
  --no_banner --log_level warn \
  && pass "Pack model" || fail "Pack model"

# ── 8. Unpack .sfog ──────────────────────────────────────────────────────────
echo "==> Unpack model"
python -m unpack \
  "${OUT}/smoke_mlp.sfog" \
  --output_dir "${OUT}/unpacked" \
  --no_banner --log_level warn \
  && pass "Unpack model" || fail "Unpack model"

# ── 9. XGBoost train + predict ───────────────────────────────────────────────
echo "==> XGBoost train"
python -m train \
  --data_path "${DATA}" \
  --model xgboost --task regression \
  --target_col 0 --lookback 10 --horizon 1 \
  --model_dir "${MODELS_DIR}" --run_id smoke_xgb \
  --no_banner --log_level warn \
  && pass "XGBoost train" || fail "XGBoost train"

echo "==> XGBoost predict"
python -m predict \
  --data_path "${DATA}" \
  --model xgboost --task regression \
  --target_col 0 --lookback 10 --horizon 1 \
  --model_dir "${MODELS_DIR}" --run_id smoke_xgb \
  --output_path "${OUT}/preds_xgb.csv" \
  --no_banner --log_level warn \
  && pass "XGBoost predict" || fail "XGBoost predict"

# ── 10. LSTM train ────────────────────────────────────────────────────────────
echo "==> LSTM train"
python -m train \
  --data_path "${DATA}" \
  --model lstm --task regression \
  --target_col 0 --lookback 10 --horizon 1 \
  --epochs 2 --batch_size 16 \
  --model_dir "${MODELS_DIR}" --run_id smoke_lstm \
  --no_banner --log_level warn \
  && pass "LSTM train" || fail "LSTM train"

# ── 11. Transformer train ─────────────────────────────────────────────────────
echo "==> Transformer train"
python -m train \
  --data_path "${DATA}" \
  --model transformer --task regression \
  --target_col 0 --lookback 10 --horizon 1 \
  --epochs 2 --batch_size 16 \
  --model_dir "${MODELS_DIR}" --run_id smoke_transformer \
  --no_banner --log_level warn \
  && pass "Transformer train" || fail "Transformer train"

# ── 12. CNN train ─────────────────────────────────────────────────────────────
echo "==> CNN train"
python -m train \
  --data_path "${DATA}" \
  --model cnn --task regression \
  --target_col 0 --lookback 10 --horizon 1 \
  --epochs 2 --batch_size 16 \
  --model_dir "${MODELS_DIR}" --run_id smoke_cnn \
  --no_banner --log_level warn \
  && pass "CNN train" || fail "CNN train"

# ── 12b. RNN train ────────────────────────────────────────────────────────────
echo "==> RNN train"
python -m train \
  --data_path "${DATA}" \
  --model rnn --task regression \
  --target_col 0 --lookback 10 --horizon 1 \
  --epochs 2 --batch_size 16 \
  --model_dir "${MODELS_DIR}" --run_id smoke_rnn \
  --no_banner --log_level warn \
  && pass "RNN train" || fail "RNN train"

# ── 13. MLP classification train + predict ───────────────────────────────────
echo "==> MLP classification train"
python -m train \
  --data_path "${DATA}" \
  --model mlp --task classification --num_classes 2 \
  --target_col 0 --lookback 10 --horizon 1 \
  --epochs 2 --batch_size 16 \
  --model_dir "${MODELS_DIR}" --run_id smoke_cls \
  --no_banner --log_level warn \
  && pass "MLP classification train" || fail "MLP classification train"

echo "==> MLP classification predict"
python -m predict \
  --data_path "${DATA}" \
  --model mlp --task classification --num_classes 2 \
  --target_col 0 --lookback 10 --horizon 1 \
  --model_dir "${MODELS_DIR}" --run_id smoke_cls \
  --output_path "${OUT}/preds_cls.csv" \
  --no_banner --log_level warn \
  && pass "MLP classification predict" || fail "MLP classification predict"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  ALL SMOKE TESTS PASSED"
echo "========================================"
