# ADR-001: Binary Classification Output Shape

**Date:** 2026-03-02
**Task:** TASK-20260302-002
**Status:** Accepted

## Context

During smoke testing of the classification path, training failed with:

```
ValueError: Incompatible shapes for broadcasting: shapes=[(16, 2), (16,)]
```

The root cause: `output_dim_for_task("classification", num_classes=2)` returned `2`,
producing model output of shape `(B, 2)`. The binary cross-entropy loss expected
scalar logits `(B,)` or `(B, 1)`.

## Decision

- **Binary classification** (`num_classes <= 2`): output_dim = **1** (single logit → sigmoid BCE)
- **Multiclass** (`num_classes > 2`): output_dim = **num_classes** (logit vector → softmax CE)

`get_default_loss()` now takes `num_classes` and selects the appropriate loss automatically.

## Rationale

- Consistent with standard practice: binary = single sigmoid, multiclass = softmax
- Avoids shape mismatch between model output and labels from `make_windows` (which produces `(B, horizon)`)
- `apply_output_head` updated to match: binary uses `sigmoid(logits)` without slicing

## Consequences

- Any saved binary classification checkpoints from before this fix will produce
  `(B, 2)` outputs — incompatible with new prediction path. Such checkpoints must be retrained.
- Multiclass (num_classes > 2) is unaffected.
