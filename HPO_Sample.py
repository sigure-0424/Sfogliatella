import math
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional, Literal

# ============================================================
# v0.2 changes (from v0)
# - Fix Transformer attention FLOPs term to scale as O(L * D * S * d)
# - Add per-model effective_data_units (Data Cap + time modeling)
# - Make bytes_per_param depend on optimizer (configurable)
# ============================================================

# ----------------------------
# 1) Shared: Scaling law fit
# ----------------------------
@dataclass
class ScalingLaw:
    """loss(N) = noise_floor + A * N^{-alpha}"""
    alpha: float
    A: float
    noise_floor: float

    @staticmethod
    def fit(baseline_measurements: List[Tuple[float, float]], noise_floor_at_horizon: float) -> "ScalingLaw":
        Ns = np.array([m[0] for m in baseline_measurements], dtype=float)
        Ls = np.array([m[1] for m in baseline_measurements], dtype=float)

        if np.any(Ls <= noise_floor_at_horizon):
            raise ValueError("Baseline loss <= noise floor (cannot fit scaling law).")

        x = np.log(Ns)
        y = np.log(Ls - noise_floor_at_horizon)
        slope, intercept = np.polyfit(x, y, 1)
        alpha = -float(slope)
        A = float(np.exp(intercept))
        return ScalingLaw(alpha=alpha, A=A, noise_floor=float(noise_floor_at_horizon))

    def ideal_N_for_target_loss(self, target_loss: float) -> float:
        if target_loss <= self.noise_floor:
            return float("inf")
        return ((target_loss - self.noise_floor) / self.A) ** (-1.0 / self.alpha)


# ----------------------------
# 2) Shared: Hardware physics
# ----------------------------
@dataclass
class HardwareConfig:
    gpu_vram_gb: float = 24.0
    gpu_peak_tflops: float = 30.0
    gpu_utilization: float = 0.4

    mixed_precision: bool = True
    use_checkpointing: bool = True
    use_flash_attention: bool = True

    optimizer: Literal["adamw", "adam", "sgd", "adafactor"] = "adamw"
    bytes_per_param_override: Optional[int] = None

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
        """Rough state memory per trainable param.

        This is intentionally configurable because it depends on:
        - optimizer states (Adam has 2 moments; SGD has none)
        - master weights (mixed precision)
        - framework overhead

        Defaults are conservative, tuned for "it won't OOM" rather than exact.
        """
        if self.bytes_per_param_override is not None:
            return int(self.bytes_per_param_override)

        # Conservative heuristics (bytes / param)
        if self.optimizer in ("adam", "adamw"):
            # weight + grad + m + v (+ possible master)
            return 18 if self.mixed_precision else 34
        if self.optimizer == "adafactor":
            return 14 if self.mixed_precision else 26
        if self.optimizer == "sgd":
            # weight + grad (+ momentum maybe)
            return 10 if self.mixed_precision else 18

        return 18 if self.mixed_precision else 34


# ----------------------------
# 3) Shared: Solver inputs
# ----------------------------
@dataclass
class SolverInputs:
    # Scaling-law baseline (same horizon / same protocol)
    baseline_measurements: List[Tuple[float, float]]  # [(N, loss), ...]
    target_loss: float
    target_distance: int
    noise_floor_at_horizon: float

    # Data & schedule
    num_train_samples: int
    num_epochs: int = 3
    batch_size: int = 32

    # Constraints
    max_training_hours: float = 24.0
    lambda_data_cap: float = 20.0

    # Policy
    shrink_order: Optional[List[str]] = None
    hardware: Optional[HardwareConfig] = None

    # Model I/O dims
    input_dim: int = 1
    output_dim: int = 1

    def __post_init__(self):
        if self.shrink_order is None:
            self.shrink_order = ["d_model", "layers", "seq_len"]
        if self.hardware is None:
            self.hardware = HardwareConfig()


# ----------------------------
# 4) ModelSpec interface
# ----------------------------
class ModelSpec:
    name: str = "base"

    def init_struct(self, inp: SolverInputs) -> Dict[str, Any]:
        raise NotImplementedError

    def derive_struct_from_N(self, inp: SolverInputs, N_target: float, struct: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def param_count(self, inp: SolverInputs, struct: Dict[str, Any]) -> int:
        raise NotImplementedError

    def effective_data_units(self, inp: SolverInputs, struct: Dict[str, Any]) -> int:
        """Default: sequential token units (timestep tokens)."""
        return int(inp.num_train_samples * struct["seq_len"] * inp.num_epochs)

    def estimate_vram_bytes(self, inp: SolverInputs, N: int, struct: Dict[str, Any]) -> float:
        raise NotImplementedError

    def estimate_train_time_hours(self, inp: SolverInputs, N: int, data_units: int, struct: Dict[str, Any]) -> float:
        raise NotImplementedError

    def shrink_once(self, inp: SolverInputs, struct: Dict[str, Any], shrink_key: str) -> bool:
        raise NotImplementedError

    def hpo_search_space(self, inp: SolverInputs, struct: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


# ----------------------------
# 5) Shared: Constrained solver loop
# ----------------------------
def solve_hpo_prior_generic(inp: SolverInputs, spec: ModelSpec) -> Dict[str, Any]:
    hw = inp.hardware
    sl = ScalingLaw.fit(inp.baseline_measurements, inp.noise_floor_at_horizon)
    N_ideal = sl.ideal_N_for_target_loss(inp.target_loss)

    struct = spec.init_struct(inp)
    active_constraint = "Target"

    while True:
        data_units = int(spec.effective_data_units(inp, struct))

        # Data cap (model-specific units)
        N_data_cap = data_units / float(inp.lambda_data_cap)
        N_target = min(N_ideal, N_data_cap)

        # Convert N_target -> structural params
        struct = spec.derive_struct_from_N(inp, N_target, struct)
        N_actual = int(spec.param_count(inp, struct))

        vram_est = float(spec.estimate_vram_bytes(inp, N_actual, struct))
        time_est = float(spec.estimate_train_time_hours(inp, N_actual, data_units, struct))

        is_vram_ok = vram_est <= hw.vram_bytes
        is_time_ok = time_est <= inp.max_training_hours

        if is_vram_ok and is_time_ok:
            if N_actual >= N_ideal * 0.95:
                active_constraint = "Target"
            elif N_actual >= N_data_cap * 0.95:
                active_constraint = "Data"
            else:
                active_constraint = "Hardware"
            break

        violation = "VRAM" if not is_vram_ok else "Time"
        active_constraint = f"Hardware ({violation})"

        shrunk = False
        for key in inp.shrink_order:
            if spec.shrink_once(inp, struct, key):
                shrunk = True
                break

        if not shrunk:
            return {"status": "Failed", "msg": f"{spec.name}: cannot satisfy constraints at min config."}

    # Expose both sequential-token count and model-specific units (often same)
    token_units = None
    if "seq_len" in struct:
        token_units = int(inp.num_train_samples * struct["seq_len"] * inp.num_epochs)

    return {
        "status": "Solved",
        "model": spec.name,
        "active_constraint": active_constraint,
        "config": {**struct, "N": int(N_actual)},
        "estimates": {
            "VRAM_GB": f"{vram_est/1e9:.2f}",
            "Time_Hours": f"{time_est:.2f}",
            "N_ideal": f"{N_ideal:.2e}",
            "N_data_cap": f"{N_data_cap:.2e}",
            "data_units": f"{data_units:.2e}",
            **({"token_units": f"{token_units:.2e}"} if token_units is not None else {}),
        },
        "hpo_search_space": spec.hpo_search_space(inp, struct),
        "notes": {
            "calibration_required": [
                "gpu_utilization",
                "bytes_per_param (optimizer / framework)",
                "activation factor (checkpointing / implementation)",
                "attention_flops_coef (transformer-like)",
            ]
        },
    }


# ============================================================
# 6) Model specs (v0.2)
# ============================================================

# -------- Transformer --------
class TransformerSpec(ModelSpec):
    name = "transformer"

    def __init__(self, r_ff: float = 4.0, attention_flops_coef: float = 12.0):
        # r_ff: d_ff = r_ff * d_model
        # attention_flops_coef: training (fwd+bwd) coefficient for L*D*S*d term
        self.r_ff = float(r_ff)
        self.c_attn = float(attention_flops_coef)

    def init_struct(self, inp: SolverInputs) -> Dict[str, Any]:
        S = int(64 * math.sqrt(max(1, inp.target_distance)))
        L = int(np.clip(6 + 1.5 * np.log2(max(1, inp.target_distance)), 4, 32))
        return {"seq_len": max(32, S), "layers": L, "d_model": 256, "heads": 4}

    def derive_struct_from_N(self, inp: SolverInputs, N_target: float, struct: Dict[str, Any]) -> Dict[str, Any]:
        L = int(struct["layers"])
        # Per-layer params ~ (4 + 2*r_ff) * d^2
        per_layer = (4.0 + 2.0 * self.r_ff)
        d = math.sqrt(max(1.0, N_target) / (per_layer * max(1, L)))
        d = int(round(d / 64) * 64)
        d = max(64, d)
        struct["d_model"] = d
        struct["heads"] = max(1, d // 64)
        return struct

    def param_count(self, inp: SolverInputs, struct: Dict[str, Any]) -> int:
        L = int(struct["layers"])
        d = int(struct["d_model"])
        per_layer = (4.0 + 2.0 * self.r_ff)
        # Add small I/O projections
        io = d * (inp.input_dim + inp.output_dim)
        return int(per_layer * L * (d ** 2) + io)

    def estimate_vram_bytes(self, inp: SolverInputs, N: int, struct: Dict[str, Any]) -> float:
        hw = inp.hardware
        B = int(inp.batch_size)
        S = int(struct["seq_len"])
        L = int(struct["layers"])
        d = int(struct["d_model"])
        h = int(struct["heads"])

        mem_model = N * hw.bytes_per_param

        # Activation proxy (absorbs FFN width, QKV buffers, etc. into act_factor)
        act_factor = 2 if hw.use_checkpointing else 12
        mem_act_linear = B * S * L * d * hw.bytes_per_element * act_factor

        # Attention matrix storage (only when not using Flash/MemEfficient)
        mem_act_attn = 0.0
        if not hw.use_flash_attention:
            mem_act_attn = B * h * S * S * hw.bytes_per_element
            if not hw.use_checkpointing:
                mem_act_attn *= 2

        return float(mem_model + mem_act_linear + mem_act_attn)

    def estimate_train_time_hours(self, inp: SolverInputs, N: int, data_units: int, struct: Dict[str, Any]) -> float:
        """Training FLOPs ~ 6*N*D + c_attn*L*D*S*d

        Where:
        - D = data_units (token units)
        - Attention term models the O(S) work per token for QK^T and AV.
        """
        hw = inp.hardware
        S = int(struct["seq_len"])
        L = int(struct["layers"])
        d = int(struct["d_model"])

        D = float(data_units)
        flops_linear = 6.0 * float(N) * D
        flops_attn = self.c_attn * float(L) * D * float(S) * float(d)

        total_flops = flops_linear + flops_attn
        return float(total_flops / hw.flops_per_sec / 3600.0)

    def shrink_once(self, inp: SolverInputs, struct: Dict[str, Any], shrink_key: str) -> bool:
        if shrink_key == "d_model" and struct["d_model"] > 128:
            struct["d_model"] = max(64, int(struct["d_model"] * 0.8 // 64 * 64))
            struct["heads"] = max(1, int(struct["d_model"]) // 64)
            return True
        if shrink_key == "layers" and struct["layers"] > 2:
            struct["layers"] -= 1
            return True
        if shrink_key == "seq_len" and struct["seq_len"] > 32:
            struct["seq_len"] = max(32, int(struct["seq_len"] * 0.8))
            return True
        return False

    def hpo_search_space(self, inp: SolverInputs, struct: Dict[str, Any]) -> Dict[str, Any]:
        d = int(struct["d_model"])
        L = int(struct["layers"])
        S = int(struct["seq_len"])
        return {
            "d_model": [max(64, d - 64), d + 128],
            "layers": [max(2, L - 2), L + 2],
            "seq_len": [max(32, int(S * 0.8)), int(S * 1.2)],
            "lr": [1e-4, 1e-2],
        }


# -------- LSTM --------
class LSTMSpec(ModelSpec):
    name = "lstm"

    def init_struct(self, inp: SolverInputs) -> Dict[str, Any]:
        S = int(64 * math.sqrt(max(1, inp.target_distance)))
        L = int(np.clip(2 + 0.8 * np.log2(max(1, inp.target_distance)), 1, 8))
        return {"seq_len": max(16, S), "layers": L, "hidden": 256}

    def derive_struct_from_N(self, inp: SolverInputs, N_target: float, struct: Dict[str, Any]) -> Dict[str, Any]:
        L = int(struct["layers"])
        # Rough inversion (ignores input term): N ~ 4*L*H^2
        H = math.sqrt(max(1.0, N_target) / (4.0 * max(1, L)))
        H = int(round(H / 32) * 32)
        struct["hidden"] = max(32, H)
        return struct

    def param_count(self, inp: SolverInputs, struct: Dict[str, Any]) -> int:
        L = int(struct["layers"])
        H = int(struct["hidden"])
        in0 = int(inp.input_dim)
        total = 0
        for i in range(L):
            din = in0 if i == 0 else H
            total += 4 * (din + H + 1) * H
        total += (H + 1) * int(inp.output_dim)
        return int(total)

    def estimate_vram_bytes(self, inp: SolverInputs, N: int, struct: Dict[str, Any]) -> float:
        hw = inp.hardware
        B = int(inp.batch_size)
        S = int(struct["seq_len"])
        L = int(struct["layers"])
        H = int(struct["hidden"])

        mem_model = N * hw.bytes_per_param
        act_factor = 4 if hw.use_checkpointing else 16
        mem_act = B * S * L * H * hw.bytes_per_element * act_factor
        return float(mem_model + mem_act)

    def estimate_train_time_hours(self, inp: SolverInputs, N: int, data_units: int, struct: Dict[str, Any]) -> float:
        hw = inp.hardware
        L = int(struct["layers"])
        H = int(struct["hidden"])
        I = int(inp.input_dim)

        # O(L*D*H*(H+I)) (with a conservative multiplier for training)
        flops_per_token = 8.0 * L * H * (H + I)
        total_flops = flops_per_token * float(data_units) * 2.5
        return float(total_flops / hw.flops_per_sec / 3600.0)

    def shrink_once(self, inp: SolverInputs, struct: Dict[str, Any], shrink_key: str) -> bool:
        if shrink_key == "d_model" and struct["hidden"] > 64:
            struct["hidden"] = max(32, int(struct["hidden"] * 0.8 // 32 * 32))
            return True
        if shrink_key == "layers" and struct["layers"] > 1:
            struct["layers"] -= 1
            return True
        if shrink_key == "seq_len" and struct["seq_len"] > 16:
            struct["seq_len"] = max(16, int(struct["seq_len"] * 0.8))
            return True
        return False

    def hpo_search_space(self, inp: SolverInputs, struct: Dict[str, Any]) -> Dict[str, Any]:
        H = int(struct["hidden"])
        L = int(struct["layers"])
        S = int(struct["seq_len"])
        return {
            "hidden": [max(32, H - 64), H + 128],
            "layers": [max(1, L - 1), L + 2],
            "seq_len": [max(16, int(S * 0.8)), int(S * 1.2)],
            "dropout": [0.0, 0.3],
            "lr": [1e-4, 5e-3],
        }


# -------- RNN (vanilla) --------
class RNNSpec(ModelSpec):
    name = "rnn"

    def init_struct(self, inp: SolverInputs) -> Dict[str, Any]:
        S = int(64 * math.sqrt(max(1, inp.target_distance)))
        L = int(np.clip(2 + 0.6 * np.log2(max(1, inp.target_distance)), 1, 8))
        return {"seq_len": max(16, S), "layers": L, "hidden": 256}

    def derive_struct_from_N(self, inp: SolverInputs, N_target: float, struct: Dict[str, Any]) -> Dict[str, Any]:
        L = int(struct["layers"])
        H = math.sqrt(max(1.0, N_target) / max(1.0, L))
        H = int(round(H / 32) * 32)
        struct["hidden"] = max(32, H)
        return struct

    def param_count(self, inp: SolverInputs, struct: Dict[str, Any]) -> int:
        L = int(struct["layers"])
        H = int(struct["hidden"])
        in0 = int(inp.input_dim)
        total = 0
        for i in range(L):
            din = in0 if i == 0 else H
            total += (din + H + 1) * H
        total += (H + 1) * int(inp.output_dim)
        return int(total)

    def estimate_vram_bytes(self, inp: SolverInputs, N: int, struct: Dict[str, Any]) -> float:
        hw = inp.hardware
        B = int(inp.batch_size)
        S = int(struct["seq_len"])
        L = int(struct["layers"])
        H = int(struct["hidden"])

        mem_model = N * hw.bytes_per_param
        act_factor = 3 if hw.use_checkpointing else 10
        mem_act = B * S * L * H * hw.bytes_per_element * act_factor
        return float(mem_model + mem_act)

    def estimate_train_time_hours(self, inp: SolverInputs, N: int, data_units: int, struct: Dict[str, Any]) -> float:
        hw = inp.hardware
        L = int(struct["layers"])
        H = int(struct["hidden"])
        I = int(inp.input_dim)

        flops_per_token = 4.0 * L * H * (H + I)
        total_flops = flops_per_token * float(data_units) * 2.0
        return float(total_flops / hw.flops_per_sec / 3600.0)

    def shrink_once(self, inp: SolverInputs, struct: Dict[str, Any], shrink_key: str) -> bool:
        if shrink_key == "d_model" and struct["hidden"] > 64:
            struct["hidden"] = max(32, int(struct["hidden"] * 0.8 // 32 * 32))
            return True
        if shrink_key == "layers" and struct["layers"] > 1:
            struct["layers"] -= 1
            return True
        if shrink_key == "seq_len" and struct["seq_len"] > 16:
            struct["seq_len"] = max(16, int(struct["seq_len"] * 0.8))
            return True
        return False

    def hpo_search_space(self, inp: SolverInputs, struct: Dict[str, Any]) -> Dict[str, Any]:
        H = int(struct["hidden"])
        L = int(struct["layers"])
        S = int(struct["seq_len"])
        return {
            "hidden": [max(32, H - 64), H + 128],
            "layers": [max(1, L - 1), L + 2],
            "seq_len": [max(16, int(S * 0.8)), int(S * 1.2)],
            "lr": [1e-4, 5e-3],
        }


# -------- MLP (windowed regression) --------
class MLPSpec(ModelSpec):
    name = "mlp"

    def init_struct(self, inp: SolverInputs) -> Dict[str, Any]:
        S = int(32 * math.sqrt(max(1, inp.target_distance)))
        D = int(np.clip(4 + 0.5 * np.log2(max(1, inp.target_distance)), 3, 10))
        return {"seq_len": max(8, S), "layers": D, "width": 512}

    def effective_data_units(self, inp: SolverInputs, struct: Dict[str, Any]) -> int:
        # Windowed MLP: training unit is sample (not timestep token)
        return int(inp.num_train_samples * inp.num_epochs)

    def derive_struct_from_N(self, inp: SolverInputs, N_target: float, struct: Dict[str, Any]) -> Dict[str, Any]:
        D = int(struct["layers"])
        S = int(struct["seq_len"])
        I = int(inp.input_dim)

        # N ≈ (S*I)*W + (D-2)*W^2 + W*out
        if D >= 4:
            W = math.sqrt(max(1.0, N_target) / max(1.0, (D - 2)))
        else:
            W = max(64.0, N_target / max(1.0, S * I))

        W = int(round(W / 64) * 64)
        struct["width"] = max(64, W)
        return struct

    def param_count(self, inp: SolverInputs, struct: Dict[str, Any]) -> int:
        S = int(struct["seq_len"])
        D = int(struct["layers"])
        W = int(struct["width"])
        I = int(inp.input_dim)
        O = int(inp.output_dim)

        in_dim = S * I
        total = (in_dim + 1) * W
        for _ in range(max(0, D - 2)):
            total += (W + 1) * W
        total += (W + 1) * O
        return int(total)

    def estimate_vram_bytes(self, inp: SolverInputs, N: int, struct: Dict[str, Any]) -> float:
        hw = inp.hardware
        B = int(inp.batch_size)
        D = int(struct["layers"])
        W = int(struct["width"])

        mem_model = N * hw.bytes_per_param
        act_factor = 2 if hw.use_checkpointing else 6
        mem_act = B * D * W * hw.bytes_per_element * act_factor
        return float(mem_model + mem_act)

    def estimate_train_time_hours(self, inp: SolverInputs, N: int, data_units: int, struct: Dict[str, Any]) -> float:
        hw = inp.hardware
        # Kaplan-like proxy over sample-steps
        total_flops = 6.0 * float(N) * float(data_units)
        return float(total_flops / hw.flops_per_sec / 3600.0)

    def shrink_once(self, inp: SolverInputs, struct: Dict[str, Any], shrink_key: str) -> bool:
        if shrink_key == "d_model" and struct["width"] > 128:
            struct["width"] = max(64, int(struct["width"] * 0.8 // 64 * 64))
            return True
        if shrink_key == "layers" and struct["layers"] > 3:
            struct["layers"] -= 1
            return True
        if shrink_key == "seq_len" and struct["seq_len"] > 8:
            struct["seq_len"] = max(8, int(struct["seq_len"] * 0.8))
            return True
        return False

    def hpo_search_space(self, inp: SolverInputs, struct: Dict[str, Any]) -> Dict[str, Any]:
        W = int(struct["width"])
        D = int(struct["layers"])
        S = int(struct["seq_len"])
        return {
            "width": [max(64, W - 128), W + 256],
            "layers": [max(3, D - 2), D + 2],
            "seq_len": [max(8, int(S * 0.8)), int(S * 1.2)],
            "lr": [1e-4, 1e-2],
            "weight_decay": [0.0, 1e-2],
        }


# -------- CNN (1D temporal conv stack) --------
class CNNSpec(ModelSpec):
    name = "cnn_1d"

    def init_struct(self, inp: SolverInputs) -> Dict[str, Any]:
        S = int(64 * math.sqrt(max(1, inp.target_distance)))
        L = int(np.clip(4 + 0.7 * np.log2(max(1, inp.target_distance)), 2, 16))
        return {"seq_len": max(16, S), "layers": L, "channels": 128, "kernel": 3}

    def derive_struct_from_N(self, inp: SolverInputs, N_target: float, struct: Dict[str, Any]) -> Dict[str, Any]:
        L = int(struct["layers"])
        k = int(struct["kernel"])
        C = math.sqrt(max(1.0, N_target) / max(1.0, L * k))
        C = int(round(C / 32) * 32)
        struct["channels"] = max(32, C)
        return struct

    def param_count(self, inp: SolverInputs, struct: Dict[str, Any]) -> int:
        L = int(struct["layers"])
        C = int(struct["channels"])
        k = int(struct["kernel"])
        I = int(inp.input_dim)
        O = int(inp.output_dim)

        total = 0
        total += (k * I + 1) * C
        for _ in range(max(0, L - 2)):
            total += (k * C + 1) * C
        total += (C + 1) * O
        return int(total)

    def estimate_vram_bytes(self, inp: SolverInputs, N: int, struct: Dict[str, Any]) -> float:
        hw = inp.hardware
        B = int(inp.batch_size)
        S = int(struct["seq_len"])
        L = int(struct["layers"])
        C = int(struct["channels"])

        mem_model = N * hw.bytes_per_param
        act_factor = 2 if hw.use_checkpointing else 8
        mem_act = B * S * L * C * hw.bytes_per_element * act_factor
        return float(mem_model + mem_act)

    def estimate_train_time_hours(self, inp: SolverInputs, N: int, data_units: int, struct: Dict[str, Any]) -> float:
        hw = inp.hardware
        L = int(struct["layers"])
        C = int(struct["channels"])
        k = int(struct["kernel"])

        flops_per_token = 2.0 * k * L * (C ** 2)
        total_flops = flops_per_token * float(data_units) * 1.8
        return float(total_flops / hw.flops_per_sec / 3600.0)

    def shrink_once(self, inp: SolverInputs, struct: Dict[str, Any], shrink_key: str) -> bool:
        if shrink_key == "d_model" and struct["channels"] > 64:
            struct["channels"] = max(32, int(struct["channels"] * 0.8 // 32 * 32))
            return True
        if shrink_key == "layers" and struct["layers"] > 2:
            struct["layers"] -= 1
            return True
        if shrink_key == "seq_len" and struct["seq_len"] > 16:
            struct["seq_len"] = max(16, int(struct["seq_len"] * 0.8))
            return True
        return False

    def hpo_search_space(self, inp: SolverInputs, struct: Dict[str, Any]) -> Dict[str, Any]:
        C = int(struct["channels"])
        L = int(struct["layers"])
        S = int(struct["seq_len"])
        k = int(struct["kernel"])
        return {
            "channels": [max(32, C - 64), C + 128],
            "layers": [max(2, L - 2), L + 2],
            "kernel": [k, k + 4],
            "seq_len": [max(16, int(S * 0.8)), int(S * 1.2)],
            "lr": [1e-4, 1e-2],
        }


# -------- TimesNet (proxy) --------
class TimesNetSpec(ModelSpec):
    name = "timesnet"

    def __init__(self, K: float = 8.0):
        self.K = float(K)

    def init_struct(self, inp: SolverInputs) -> Dict[str, Any]:
        S = int(96 * math.sqrt(max(1, inp.target_distance)))
        L = int(np.clip(2 + 0.9 * np.log2(max(1, inp.target_distance)), 2, 12))
        return {"seq_len": max(32, S), "layers": L, "d_model": 256}

    def derive_struct_from_N(self, inp: SolverInputs, N_target: float, struct: Dict[str, Any]) -> Dict[str, Any]:
        L = int(struct["layers"])
        d = math.sqrt(max(1.0, N_target) / (self.K * max(1, L)))
        d = int(round(d / 64) * 64)
        struct["d_model"] = max(64, d)
        return struct

    def param_count(self, inp: SolverInputs, struct: Dict[str, Any]) -> int:
        L = int(struct["layers"])
        d = int(struct["d_model"])
        io = d * (inp.input_dim + inp.output_dim)
        return int(self.K * L * (d ** 2) + io)

    def estimate_vram_bytes(self, inp: SolverInputs, N: int, struct: Dict[str, Any]) -> float:
        hw = inp.hardware
        B = int(inp.batch_size)
        S = int(struct["seq_len"])
        L = int(struct["layers"])
        d = int(struct["d_model"])

        mem_model = N * hw.bytes_per_param
        act_factor = 3 if hw.use_checkpointing else 10
        mem_act = B * S * L * d * hw.bytes_per_element * act_factor
        return float(mem_model + mem_act)

    def estimate_train_time_hours(self, inp: SolverInputs, N: int, data_units: int, struct: Dict[str, Any]) -> float:
        hw = inp.hardware
        total_flops = 6.0 * float(N) * float(data_units)
        return float(total_flops / hw.flops_per_sec / 3600.0)

    def shrink_once(self, inp: SolverInputs, struct: Dict[str, Any], shrink_key: str) -> bool:
        if shrink_key == "d_model" and struct["d_model"] > 128:
            struct["d_model"] = max(64, int(struct["d_model"] * 0.8 // 64 * 64))
            return True
        if shrink_key == "layers" and struct["layers"] > 2:
            struct["layers"] -= 1
            return True
        if shrink_key == "seq_len" and struct["seq_len"] > 32:
            struct["seq_len"] = max(32, int(struct["seq_len"] * 0.8))
            return True
        return False

    def hpo_search_space(self, inp: SolverInputs, struct: Dict[str, Any]) -> Dict[str, Any]:
        d = int(struct["d_model"])
        L = int(struct["layers"])
        S = int(struct["seq_len"])
        return {
            "d_model": [max(64, d - 64), d + 128],
            "layers": [max(2, L - 2), L + 2],
            "seq_len": [max(32, int(S * 0.8)), int(S * 1.2)],
            "lr": [1e-4, 1e-2],
        }


# -------- TimesFM (pretrained fine-tune proxy) --------
class TimesFMSpec(ModelSpec):
    name = "timesfm"

    def init_struct(self, inp: SolverInputs) -> Dict[str, Any]:
        S = int(128 * math.sqrt(max(1, inp.target_distance)))
        return {
            "seq_len": max(64, S),
            "variant_params": 200_000_000,
            "finetune_ratio": 0.02,
            "layers": 1,
            "d_model": 1,
        }

    def derive_struct_from_N(self, inp: SolverInputs, N_target: float, struct: Dict[str, Any]) -> Dict[str, Any]:
        return struct

    def effective_data_units(self, inp: SolverInputs, struct: Dict[str, Any]) -> int:
        # Fine-tuning often uses token units; keep sequential tokens
        return int(inp.num_train_samples * struct["seq_len"] * inp.num_epochs)

    def param_count(self, inp: SolverInputs, struct: Dict[str, Any]) -> int:
        N_total = int(struct["variant_params"])
        r = float(struct["finetune_ratio"])
        return int(max(1, N_total * r))

    def estimate_vram_bytes(self, inp: SolverInputs, N: int, struct: Dict[str, Any]) -> float:
        hw = inp.hardware
        B = int(inp.batch_size)
        S = int(struct["seq_len"])
        N_total = int(struct["variant_params"])

        mem_frozen = N_total * hw.bytes_per_element
        mem_trainable = N * hw.bytes_per_param
        act_factor = 2 if hw.use_checkpointing else 8
        mem_act = B * S * 1024 * hw.bytes_per_element * act_factor
        return float(mem_frozen + mem_trainable + mem_act)

    def estimate_train_time_hours(self, inp: SolverInputs, N: int, data_units: int, struct: Dict[str, Any]) -> float:
        hw = inp.hardware
        total_flops = 6.0 * float(N) * float(data_units)
        return float(total_flops / hw.flops_per_sec / 3600.0)

    def shrink_once(self, inp: SolverInputs, struct: Dict[str, Any], shrink_key: str) -> bool:
        if shrink_key == "seq_len" and struct["seq_len"] > 64:
            struct["seq_len"] = max(64, int(struct["seq_len"] * 0.8))
            return True
        if shrink_key == "d_model":
            if struct["finetune_ratio"] > 0.005:
                struct["finetune_ratio"] *= 0.7
                return True
        if shrink_key == "layers":
            if struct["variant_params"] > 50_000_000:
                struct["variant_params"] = int(struct["variant_params"] * 0.7)
                return True
        return False

    def hpo_search_space(self, inp: SolverInputs, struct: Dict[str, Any]) -> Dict[str, Any]:
        S = int(struct["seq_len"])
        r = float(struct["finetune_ratio"])
        vp = int(struct["variant_params"])
        return {
            "seq_len": [max(64, int(S * 0.8)), int(S * 1.2)],
            "finetune_ratio": [max(0.002, r * 0.5), min(0.2, r * 2.0)],
            "lr": [1e-5, 5e-4],
            "variant_params": [int(vp * 0.7), int(vp * 1.0)],
        }


# -------- XGBoost (tree proxy) --------
class XGBoostSpec(ModelSpec):
    name = "xgboost"

    def init_struct(self, inp: SolverInputs) -> Dict[str, Any]:
        return {"seq_len": 1, "n_estimators": 1000, "max_depth": 8, "subsample": 0.8, "colsample_bytree": 0.8}

    def derive_struct_from_N(self, inp: SolverInputs, N_target: float, struct: Dict[str, Any]) -> Dict[str, Any]:
        return struct

    def effective_data_units(self, inp: SolverInputs, struct: Dict[str, Any]) -> int:
        # For trees, use sample units (not timestep tokens)
        return int(inp.num_train_samples)

    def param_count(self, inp: SolverInputs, struct: Dict[str, Any]) -> int:
        T = int(struct["n_estimators"])
        D = int(struct["max_depth"])
        nodes_per_tree = (2 ** (D + 1)) - 1
        return int(T * nodes_per_tree)

    def estimate_vram_bytes(self, inp: SolverInputs, N: int, struct: Dict[str, Any]) -> float:
        bytes_per_node = 64
        return float(N * bytes_per_node)

    def estimate_train_time_hours(self, inp: SolverInputs, N: int, data_units: int, struct: Dict[str, Any]) -> float:
        hw = inp.hardware
        T = int(struct["n_estimators"])
        D = int(struct["max_depth"])
        ops = float(T * data_units * D) * 200.0
        return float(ops / hw.flops_per_sec / 3600.0)

    def shrink_once(self, inp: SolverInputs, struct: Dict[str, Any], shrink_key: str) -> bool:
        if shrink_key == "layers":
            if struct["n_estimators"] > 50:
                struct["n_estimators"] = int(struct["n_estimators"] * 0.8)
                return True
        if shrink_key == "d_model":
            if struct["max_depth"] > 3:
                struct["max_depth"] -= 1
                return True
        if shrink_key == "seq_len":
            if struct["subsample"] > 0.5:
                struct["subsample"] = max(0.5, struct["subsample"] * 0.9)
                struct["colsample_bytree"] = max(0.5, struct["colsample_bytree"] * 0.9)
                return True
        return False

    def hpo_search_space(self, inp: SolverInputs, struct: Dict[str, Any]) -> Dict[str, Any]:
        T = int(struct["n_estimators"])
        D = int(struct["max_depth"])
        ss = float(struct["subsample"])
        cs = float(struct["colsample_bytree"])
        return {
            "n_estimators": [max(50, int(T * 0.7)), int(T * 1.3)],
            "max_depth": [max(2, D - 2), D + 2],
            "learning_rate": [0.01, 0.3],
            "subsample": [max(0.5, ss - 0.2), min(1.0, ss + 0.2)],
            "colsample_bytree": [max(0.5, cs - 0.2), min(1.0, cs + 0.2)],
        }


# ----------------------------
# 7) Registry
# ----------------------------
MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "transformer": TransformerSpec(r_ff=4.0, attention_flops_coef=12.0),
    "lstm": LSTMSpec(),
    "rnn": RNNSpec(),
    "mlp": MLPSpec(),
    "cnn": CNNSpec(),
    "timesnet": TimesNetSpec(K=8.0),
    "timesfm": TimesFMSpec(),
    "xgboost": XGBoostSpec(),
}


def solve_model(model_name: str, inp: SolverInputs) -> Dict[str, Any]:
    spec = MODEL_REGISTRY[model_name]
    return solve_hpo_prior_generic(inp, spec)


# Example usage
# inp = SolverInputs(
#     baseline_measurements=[(1e6, 0.50), (4e6, 0.35), (1.6e7, 0.25)],
#     target_loss=0.22,
#     target_distance=48,
#     noise_floor_at_horizon=0.18,
#     num_train_samples=10000,
#     num_epochs=3,
#     batch_size=32,
#     max_training_hours=24,
#     lambda_data_cap=20,
#     shrink_order=["d_model", "layers", "seq_len"],
#     input_dim=8,
#     output_dim=1,
# )
# print(solve_model("transformer", inp))
# print(solve_model("mlp", inp))
