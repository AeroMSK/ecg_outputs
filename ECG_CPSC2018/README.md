# ECG CPSC2018 + Zheng — High-Performance Final Optimization

Persistent research workspace for the ECG CPSC2018 + Zheng / Chapman-Shaoxing-Ningbo
12-lead ECG classification project.

## Golden CPSC reference (immutable, never overwrite)

| Metric | Value |
|--------|-------|
| Macro-AUC       | 0.9714 |
| Macro-F1        | 0.8330 |
| Macro-Precision | 0.8398 |
| Macro-Recall    | 0.8336 |
| Macro-AP        | 0.8883 |
| ECE             | 0.0316 |
| Brier           | 0.0291 |

These live under `checkpoints/GOLDEN_CPSC/` once recovered. Until recovery is
complete, the directory is reserved and empty.

## Repository layout

```
ECG_CPSC2018/
├── SESSION_01_ENVIRONMENT_AUDIT.md   # reality check (this session)
├── src/                              # implementation modules
├── configs/                          # YAML / JSON configs
├── notebooks/                        # canonical notebook lives here once supplied
├── results/                          # CSV / JSON outputs
├── figures/                          # publication-quality figures
├── checkpoints/
│   ├── GOLDEN_CPSC/                  # immutable canonical baseline
│   ├── BEST_CPSC/
│   ├── BEST_ZERO_SHOT/
│   ├── BEST_UDA/
│   └── BEST_SUPERVISED_ADAPTATION/
├── predictions/
├── manifests/                        # SHA256 hashes, provenance
├── logs/
├── paper_tables/                     # Table 1 .. Table 11
└── FINAL_ECG_RESULTS/                # final delivery bundle (mirrors above)
```

## Selection protocol (frozen)

```
TRAIN  →  VAL_TUNE  →  VAL_SELECT  →  FREEZE  →  TEST  →  EXTERNAL
```

- VAL_TUNE: training monitoring, threshold fit, calibration fit, ensemble weight fit.
- VAL_SELECT: candidate ranking / selection ONLY.
- TEST: never used for any tuning.
- EXTERNAL: evaluated once on the frozen system.

## External tracks (kept separate at all times)

- **Track A — STRICT ZERO-SHOT:** CPSC frozen, no external labels used.
- **Track B — UNSUPERVISED DOMAIN ADAPTATION:** external signals may be used unlabeled.
- **Track C — SUPERVISED DOMAIN ADAPTATION:** labeled Chapman/Ningbo per predeclared split.
- **Track D — JOINT CPSC + EXTERNAL SUPERVISED MODEL.**

Never mix these tracks in one headline number.

## Session log

- **Session 01 (2026-09-23):** environment audit, scaffolding, security setup.
  No training executed (no GPU, no PyTorch, no datasets, no notebook on disk).
  See `SESSION_01_ENVIRONMENT_AUDIT.md`.
