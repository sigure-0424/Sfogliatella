# TASK-20260302-002: Gap-fill — TimesFM stub, custom model, smoke_test.sh, classification smoke

## Status
in_progress

## Priority
high

## Background
TASK-20260301-001 completed the initial implementation. Smoke passed for regression path.
Gap analysis revealed the following missing pieces vs GOAL.md:

1. TimesFM / TimesFM_FT model — not in registry, not in args.py choices
2. Custom model runner — args.py has --model custom but trainer.py has no handler
3. smoke_test.sh — referenced in STATE.yaml smoke_status but not in repo
4. Classification path never smoke-tested

## Definition of Done

- [ ] `sfogliatella/models/timesfm_model.py` added (stub + optional-import wrapper)
- [ ] `timesfm` and `timesfm_ft` added to args.py model choices
- [ ] `_ensure_models_registered()` imports timesfm_model
- [ ] Custom model runner implemented in trainer.py (subprocess delegation)
- [ ] `smoke_test.sh` exists, runnable in Docker, covers: MLP regression, MLP classification, XGBoost, pack/unpack
- [ ] Docker smoke passes
- [ ] STATE.yaml smoke_status updated
- [ ] ACTIVITY_SUMMARY.md updated
- [ ] TASK_INDEX.md updated

## Notes
- TimesFM (google-research) has heavy deps (torch, etc.) — MUST be optional import, never hard-required
- timesfm_ft (fine-tuned) can be a subclass or variant of the stub
- Custom model: subprocess call, pass config as JSON tempfile, wait for completion
