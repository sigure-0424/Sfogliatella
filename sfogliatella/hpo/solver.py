"""HPO prior solver — adapted from HPO_Sample.py.

Computes baseline model structure/hyperparameters from:
  - dataset scale (samples, lookback, horizon, dims)
  - compute constraints (time budget, device memory)
  - target loss/accuracy objective (optional)
Returns a "why" payload with active constraint, derived quantities, and shrink steps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Scaling law
# ---------------------------------------------------------------------------

@dataclass
class ScalingLaw:
    """loss(N) = noise_floor + A * N^{-alpha}"""
    alpha: float
    A: float
    noise_floor: float

    @staticmethod
    def fit(
        baseline_measurements: List[Tuple[float, float]],
        noise_floor: float,
    ) -> "ScalingLaw":
        Ns = np.array([m[0] for m in baseline_measurements], dtype=float)
        Ls = np.array([m[1] for m in baseline_measurements], dtype=float)
        if np.any(Ls <= noise_floor):
            raise ValueError("Baseline loss <= noise floor; cannot fit scaling law.")
        x = np.log(Ns)
        y = np.log(Ls - noise_floor)
        slope, intercept = np.polyfit(x, y, 1)
        return ScalingLaw(alpha=-float(slope), A=float(np.exp(intercept)), noise_floor=float(noise_floor))

    def ideal_N(self, target_loss: float) -> float:
        if target_loss <= self.noise_floor:
            return float("inf")
        return ((target_loss - self.noise_floor) / self.A) ** (-1.0 / self.alpha)


# ---------------------------------------------------------------------------
# Hardware config
# ---------------------------------------------------------------------------

@dataclass
class HardwareConfig:
    gpu_vram_gb:   float = 24.0
    gpu_peak_tflops: float = 30.0
    gpu_utilization: float = 0.4
    mixed_precision: bool = True
    use_checkpointing: bool = True
    use_flash_attention: bool = True
    optimizer: str = "adamw"

    @property
    def vram_bytes(self) -> float:
        return self.gpu_vram_gb * 1e9 * 0.95

    @property
    def flops_per_sec(self) -> float:
        return self.gpu_peak_tflops * 1e12 * self.gpu_utilization

    @property
    def bytes_per_element(self) -> int:
        return 2 if self.mixed_precision else 4

    @property
    def bytes_per_param(self) -> int:
        if self.optimizer in ("adam", "adamw"):
            return 18 if self.mixed_precision else 34
        if self.optimizer == "adafactor":
            return 14 if self.mixed_precision else 26
        if self.optimizer == "sgd":
            return 10 if self.mixed_precision else 18
        return 18 if self.mixed_precision else 34


# ---------------------------------------------------------------------------
# Solver inputs
# ---------------------------------------------------------------------------

@dataclass
class SolverInputs:
    # Scaling law baseline (optional)
    baseline_measurements: Optional[List[Tuple[float, float]]] = None
    target_loss: float = 0.1
    noise_floor: float = 0.0
    # Dataset
    num_train_samples: int = 10000
    num_epochs: int = 10
    batch_size: int = 32
    lookback: int = 32
    horizon: int = 1
    input_dim: int = 1
    output_dim: int = 1
    # Compute constraints
    max_training_hours: float = 24.0
    lambda_data_cap: float = 20.0
    # Hardware
    hardware: Optional[HardwareConfig] = None
    # Shrink order
    shrink_order: Optional[List[str]] = None
    # Task
    task: str = "regression"

    def __post_init__(self):
        if self.hardware is None:
            self.hardware = HardwareConfig()
        if self.shrink_order is None:
            self.shrink_order = ["d_model", "layers", "seq_len"]


# ---------------------------------------------------------------------------
# Model specs
# ---------------------------------------------------------------------------

class ModelSpec:
    name: str = "base"

    def init_struct(self, inp: SolverInputs) -> Dict[str, Any]: raise NotImplementedError
    def derive_struct_from_N(self, inp: SolverInputs, N_target: float, struct: Dict) -> Dict: raise NotImplementedError
    def param_count(self, inp: SolverInputs, struct: Dict) -> int: raise NotImplementedError
    def effective_data_units(self, inp: SolverInputs, struct: Dict) -> int:
        return int(inp.num_train_samples * struct.get("seq_len", inp.lookback) * inp.num_epochs)
    def estimate_vram_bytes(self, inp: SolverInputs, N: int, struct: Dict) -> float: raise NotImplementedError
    def estimate_train_time_hours(self, inp: SolverInputs, N: int, data_units: int, struct: Dict) -> float: raise NotImplementedError
    def shrink_once(self, inp: SolverInputs, struct: Dict, key: str) -> bool: raise NotImplementedError
    def hpo_search_space(self, inp: SolverInputs, struct: Dict) -> Dict: raise NotImplementedError


class TransformerSpec(ModelSpec):
    name = "transformer"
    def __init__(self, r_ff=4.0, c_attn=12.0):
        self.r_ff = r_ff; self.c_attn = c_attn

    def init_struct(self, inp):
        S = max(inp.lookback, int(64 * math.sqrt(max(1, inp.horizon))))
        L = int(np.clip(6 + 1.5 * np.log2(max(1, inp.horizon)), 4, 32))
        return {"seq_len": S, "layers": L, "d_model": 128, "heads": 4}

    def derive_struct_from_N(self, inp, N_target, struct):
        L = int(struct["layers"])
        per_layer = 4.0 + 2.0 * self.r_ff
        d = math.sqrt(max(1.0, N_target) / (per_layer * max(1, L)))
        d = max(64, int(round(d / 64) * 64))
        struct["d_model"] = d; struct["heads"] = max(1, d // 64)
        return struct

    def param_count(self, inp, struct):
        L, d = int(struct["layers"]), int(struct["d_model"])
        io = d * (inp.input_dim + inp.output_dim)
        return int((4.0 + 2.0 * self.r_ff) * L * d ** 2 + io)

    def estimate_vram_bytes(self, inp, N, struct):
        hw = inp.hardware; B = inp.batch_size; S = struct["seq_len"]; L = struct["layers"]; d = struct["d_model"]; h = struct["heads"]
        mem_model = N * hw.bytes_per_param
        af = 2 if hw.use_checkpointing else 12
        mem_act = B * S * L * d * hw.bytes_per_element * af
        mem_attn = 0 if hw.use_flash_attention else B * h * S * S * hw.bytes_per_element * (1 if hw.use_checkpointing else 2)
        return float(mem_model + mem_act + mem_attn)

    def estimate_train_time_hours(self, inp, N, data_units, struct):
        hw = inp.hardware; S = struct["seq_len"]; L = struct["layers"]; d = struct["d_model"]
        flops = 6.0 * N * data_units + self.c_attn * L * data_units * S * d
        return float(flops / hw.flops_per_sec / 3600.0)

    def shrink_once(self, inp, struct, key):
        if key == "d_model" and struct["d_model"] > 128:
            struct["d_model"] = max(64, int(struct["d_model"] * 0.8 // 64 * 64)); struct["heads"] = max(1, struct["d_model"] // 64); return True
        if key == "layers" and struct["layers"] > 2:
            struct["layers"] -= 1; return True
        if key == "seq_len" and struct["seq_len"] > 32:
            struct["seq_len"] = max(32, int(struct["seq_len"] * 0.8)); return True
        return False

    def hpo_search_space(self, inp, struct):
        d, L, S = struct["d_model"], struct["layers"], struct["seq_len"]
        return {"d_model": [max(64, d-64), d+128], "layers": [max(2,L-2), L+2], "seq_len": [max(32,int(S*0.8)), int(S*1.2)], "lr": [1e-4, 1e-2]}


class LSTMSpec(ModelSpec):
    name = "lstm"
    def init_struct(self, inp):
        return {"seq_len": inp.lookback, "layers": 2, "hidden": 128}
    def derive_struct_from_N(self, inp, N_target, struct):
        L = int(struct["layers"])
        H = max(32, int(round(math.sqrt(max(1.0, N_target) / (4.0 * max(1, L))) / 32) * 32))
        struct["hidden"] = H; return struct
    def param_count(self, inp, struct):
        L, H, in0 = int(struct["layers"]), int(struct["hidden"]), int(inp.input_dim)
        total = sum(4*(in0 if i==0 else H + H + 1) * H for i in range(L))
        return int(total + (H+1)*int(inp.output_dim))
    def estimate_vram_bytes(self, inp, N, struct):
        hw = inp.hardware; B = inp.batch_size; S = struct["seq_len"]; L = struct["layers"]; H = struct["hidden"]
        af = 4 if hw.use_checkpointing else 16
        return float(N * hw.bytes_per_param + B * S * L * H * hw.bytes_per_element * af)
    def estimate_train_time_hours(self, inp, N, data_units, struct):
        hw = inp.hardware; L = struct["layers"]; H = struct["hidden"]; I = inp.input_dim
        return float(8.0 * L * H * (H + I) * data_units * 2.5 / hw.flops_per_sec / 3600.0)
    def shrink_once(self, inp, struct, key):
        if key == "d_model" and struct["hidden"] > 64:
            struct["hidden"] = max(32, int(struct["hidden"] * 0.8 // 32 * 32)); return True
        if key == "layers" and struct["layers"] > 1:
            struct["layers"] -= 1; return True
        if key == "seq_len" and struct["seq_len"] > 16:
            struct["seq_len"] = max(16, int(struct["seq_len"] * 0.8)); return True
        return False
    def hpo_search_space(self, inp, struct):
        H, L, S = struct["hidden"], struct["layers"], struct["seq_len"]
        return {"hidden": [max(32,H-64), H+128], "layers": [max(1,L-1), L+2], "seq_len": [max(16,int(S*0.8)), int(S*1.2)], "dropout": [0.0,0.3], "lr": [1e-4,5e-3]}


class MLPSpec(ModelSpec):
    name = "mlp"
    def init_struct(self, inp):
        return {"seq_len": inp.lookback, "layers": 4, "width": 256}
    def effective_data_units(self, inp, struct):
        return int(inp.num_train_samples * inp.num_epochs)
    def derive_struct_from_N(self, inp, N_target, struct):
        D = int(struct["layers"])
        W = max(64, int(round(math.sqrt(max(1.0, N_target) / max(1.0, D-2)) / 64) * 64)) if D >= 4 else max(64, int(N_target / max(1.0, struct["seq_len"] * inp.input_dim)))
        struct["width"] = W; return struct
    def param_count(self, inp, struct):
        S, D, W, I, O = struct["seq_len"], int(struct["layers"]), int(struct["width"]), int(inp.input_dim), int(inp.output_dim)
        return int((S*I+1)*W + max(0,D-2)*(W+1)*W + (W+1)*O)
    def estimate_vram_bytes(self, inp, N, struct):
        hw = inp.hardware; B = inp.batch_size; D = struct["layers"]; W = struct["width"]
        af = 2 if hw.use_checkpointing else 6
        return float(N * hw.bytes_per_param + B * D * W * hw.bytes_per_element * af)
    def estimate_train_time_hours(self, inp, N, data_units, struct):
        return float(6.0 * N * data_units / inp.hardware.flops_per_sec / 3600.0)
    def shrink_once(self, inp, struct, key):
        if key == "d_model" and struct["width"] > 128:
            struct["width"] = max(64, int(struct["width"] * 0.8 // 64 * 64)); return True
        if key == "layers" and struct["layers"] > 3:
            struct["layers"] -= 1; return True
        if key == "seq_len" and struct["seq_len"] > 8:
            struct["seq_len"] = max(8, int(struct["seq_len"] * 0.8)); return True
        return False
    def hpo_search_space(self, inp, struct):
        W, D, S = struct["width"], struct["layers"], struct["seq_len"]
        return {"width": [max(64,W-128), W+256], "layers": [max(3,D-2), D+2], "seq_len": [max(8,int(S*0.8)), int(S*1.2)], "lr": [1e-4,1e-2]}


class CNNSpec(ModelSpec):
    name = "cnn"
    def init_struct(self, inp):
        return {"seq_len": inp.lookback, "layers": 4, "channels": 64, "kernel": 3}
    def derive_struct_from_N(self, inp, N_target, struct):
        L, k = int(struct["layers"]), int(struct["kernel"])
        C = max(32, int(round(math.sqrt(max(1.0, N_target) / max(1.0, L*k)) / 32) * 32))
        struct["channels"] = C; return struct
    def param_count(self, inp, struct):
        L, C, k, I, O = int(struct["layers"]), int(struct["channels"]), int(struct["kernel"]), int(inp.input_dim), int(inp.output_dim)
        return int((k*I+1)*C + max(0,L-2)*(k*C+1)*C + (C+1)*O)
    def estimate_vram_bytes(self, inp, N, struct):
        hw = inp.hardware; B = inp.batch_size; S = struct["seq_len"]; L = struct["layers"]; C = struct["channels"]
        af = 2 if hw.use_checkpointing else 8
        return float(N * hw.bytes_per_param + B * S * L * C * hw.bytes_per_element * af)
    def estimate_train_time_hours(self, inp, N, data_units, struct):
        L, C, k = struct["layers"], struct["channels"], struct["kernel"]
        return float(2.0 * k * L * C**2 * data_units * 1.8 / inp.hardware.flops_per_sec / 3600.0)
    def shrink_once(self, inp, struct, key):
        if key == "d_model" and struct["channels"] > 64:
            struct["channels"] = max(32, int(struct["channels"] * 0.8 // 32 * 32)); return True
        if key == "layers" and struct["layers"] > 2:
            struct["layers"] -= 1; return True
        if key == "seq_len" and struct["seq_len"] > 16:
            struct["seq_len"] = max(16, int(struct["seq_len"] * 0.8)); return True
        return False
    def hpo_search_space(self, inp, struct):
        C, L, S, k = struct["channels"], struct["layers"], struct["seq_len"], struct["kernel"]
        return {"channels": [max(32,C-64), C+128], "layers": [max(2,L-2), L+2], "kernel": [k, k+4], "seq_len": [max(16,int(S*0.8)), int(S*1.2)], "lr": [1e-4,1e-2]}


class XGBoostSpec(ModelSpec):
    name = "xgboost"
    def init_struct(self, inp):
        return {"seq_len": 1, "n_estimators": 500, "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8}
    def effective_data_units(self, inp, struct):
        return int(inp.num_train_samples)
    def derive_struct_from_N(self, inp, N_target, struct):
        return struct
    def param_count(self, inp, struct):
        T, D = int(struct["n_estimators"]), int(struct["max_depth"])
        return int(T * ((2 ** (D+1)) - 1))
    def estimate_vram_bytes(self, inp, N, struct):
        return float(N * 64)
    def estimate_train_time_hours(self, inp, N, data_units, struct):
        T, D = struct["n_estimators"], struct["max_depth"]
        return float(T * data_units * D * 200.0 / inp.hardware.flops_per_sec / 3600.0)
    def shrink_once(self, inp, struct, key):
        if key == "layers" and struct["n_estimators"] > 50:
            struct["n_estimators"] = int(struct["n_estimators"] * 0.8); return True
        if key == "d_model" and struct["max_depth"] > 3:
            struct["max_depth"] -= 1; return True
        return False
    def hpo_search_space(self, inp, struct):
        T, D = struct["n_estimators"], struct["max_depth"]
        return {"n_estimators": [max(50,int(T*0.7)), int(T*1.3)], "max_depth": [max(2,D-2), D+2], "learning_rate": [0.01,0.3], "subsample": [0.5,1.0]}


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

_SPEC_MAP: Dict[str, ModelSpec] = {
    "transformer": TransformerSpec(),
    "lstm":        LSTMSpec(),
    "mlp":         MLPSpec(),
    "cnn":         CNNSpec(),
    "rnn":         LSTMSpec(),  # reuse LSTM spec as proxy
    "xgboost":     XGBoostSpec(),
}


def solve_hpo_prior(
    model_name: str,
    inp: SolverInputs,
    return_why: bool = True,
) -> Dict[str, Any]:
    """Solve for the baseline model config given constraints.

    Returns a result dict with status, config, estimates, search_space, and
    optionally a 'why' payload.
    """
    if model_name not in _SPEC_MAP:
        raise ValueError(f"No HPO spec for model: {model_name}. Available: {list(_SPEC_MAP)}")

    spec = _SPEC_MAP[model_name]
    hw = inp.hardware

    # Fit scaling law if measurements provided
    if inp.baseline_measurements and len(inp.baseline_measurements) >= 2:
        sl = ScalingLaw.fit(inp.baseline_measurements, inp.noise_floor)
        N_ideal = sl.ideal_N(inp.target_loss)
        scaling_law_used = True
    else:
        N_ideal = float("inf")
        scaling_law_used = False

    struct = spec.init_struct(inp)
    why_log: List[Dict] = []
    active_constraint = "Target"

    for attempt in range(100):
        data_units = int(spec.effective_data_units(inp, struct))
        N_data_cap = data_units / float(inp.lambda_data_cap)
        N_target = min(N_ideal, N_data_cap) if scaling_law_used else N_data_cap

        struct = spec.derive_struct_from_N(inp, N_target, struct)
        N_actual = int(spec.param_count(inp, struct))
        vram_est = float(spec.estimate_vram_bytes(inp, N_actual, struct))
        time_est = float(spec.estimate_train_time_hours(inp, N_actual, data_units, struct))

        is_vram_ok = vram_est <= hw.vram_bytes
        is_time_ok = time_est <= inp.max_training_hours

        if is_vram_ok and is_time_ok:
            if scaling_law_used and N_actual >= N_ideal * 0.95:
                active_constraint = "Target"
            elif N_actual >= N_data_cap * 0.95:
                active_constraint = "Data"
            else:
                active_constraint = "Hardware"
            break

        violation = "VRAM" if not is_vram_ok else "Time"
        active_constraint = f"Hardware ({violation})"

        shrunk = False
        for shrink_key in inp.shrink_order:
            prev_struct = dict(struct)
            if spec.shrink_once(inp, struct, shrink_key):
                if return_why:
                    why_log.append({
                        "attempt": attempt,
                        "shrink_key": shrink_key,
                        "violation": violation,
                        "vram_gb": round(vram_est / 1e9, 3),
                        "time_h": round(time_est, 3),
                        "before": {k: prev_struct.get(k) for k in struct if prev_struct.get(k) != struct.get(k)},
                        "after":  {k: struct.get(k) for k in struct if prev_struct.get(k) != struct.get(k)},
                    })
                shrunk = True
                break

        if not shrunk:
            return {
                "status": "Failed",
                "model": model_name,
                "msg": f"Cannot satisfy constraints at minimum config for {model_name}.",
                "why": why_log,
            }

    result: Dict[str, Any] = {
        "status": "Solved",
        "model": model_name,
        "active_constraint": active_constraint,
        "config": {**struct, "N": int(N_actual)},
        "estimates": {
            "VRAM_GB":   f"{vram_est/1e9:.2f}",
            "Time_Hours": f"{time_est:.3f}",
            "N_data_cap": f"{N_data_cap:.2e}",
            "data_units": f"{data_units:.2e}",
            **({"N_ideal": f"{N_ideal:.2e}"} if scaling_law_used else {}),
        },
        "hpo_search_space": spec.hpo_search_space(inp, struct),
        "notes": {
            "calibration_required": [
                "gpu_utilization", "bytes_per_param (optimizer/framework)",
                "activation_factor (checkpointing/implementation)",
            ],
            "scaling_law_used": scaling_law_used,
        },
    }
    if return_why:
        result["why"] = {
            "active_constraint": active_constraint,
            "shrink_steps": why_log,
            "final_N": int(N_actual),
            "final_vram_gb": round(vram_est / 1e9, 3),
            "final_time_h": round(time_est, 3),
            "caveats": result["notes"]["calibration_required"],
        }
    return result


# ---------------------------------------------------------------------------
# HPOResult
# ---------------------------------------------------------------------------

from dataclasses import dataclass


@dataclass
class HPOResult:
    model: str
    status: str
    config: Dict[str, Any]
    estimates: Dict[str, str]
    search_space: Dict[str, Any]
    why: Optional[Dict[str, Any]] = None
    raw: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "status": self.status,
            "config": self.config,
            "estimates": self.estimates,
            "search_space": self.search_space,
            "why": self.why,
        }
