# FINAL ECG RESULTS — CPSC2018 + Zheng External Validation

Contents of `FINAL_ECG_RESULTS/`:
- `ECG_CPSC2018_FINAL_EXTERNAL_VALIDATION_PUBLICATION_READY.ipynb` — the single
  canonical, fully executed notebook (CPSC experiment + external branch).
- `results/CPSC_golden_numbers.json` — protected canonical CPSC reference
  (Macro-AUC 0.9714, Macro-F1 0.833;
  provenance inside). This session's replica budget is recorded in
  `results/final_config.json`; the golden reference is never overwritten.
- `results/external_zheng/` — all external-branch artifacts (manifest, label mapping,
  metrics, per-class metrics, bootstrap CIs, robustness, morphology, quality report,
  gates).
- `predictions/external_zheng_predictions.npz` — per-record external predictions
  (ids, labels, probabilities, predictions, cohort, status).
- `figures/external_zheng/`, `paper_tables/` (incl. `ext_*.tex`) — publication assets.
- `checkpoints/GOLDEN_CPSC_FINAL/` — immutable golden bundle of this session's frozen
  CPSC system (checkpoints + config + thresholds + defense).
- `README_EXTERNAL_VALIDATION.md` — full external protocol, results and limitations.

Protocol summary: CPSC2018 train/validate/freeze → internal test + Zheng zero-shot
external validation (frozen model, frozen thresholds, frozen AWARD) → robustness,
morphology and statistics on both corpora.
