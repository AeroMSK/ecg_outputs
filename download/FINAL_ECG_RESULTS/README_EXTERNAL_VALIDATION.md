# External Validation — Zheng / Chapman-Shaoxing-Ningbo

## Dataset source
- **Official name**: A Large Scale 12-lead Electrocardiogram Database for Arrhythmia
  Study, version 1.0.0 (Zheng, Guo, Chu et al.; Chapman University / Shaoxing
  People's Hospital / Ningbo First Hospital).
- **PhysioNet**: https://physionet.org/content/ecg-arrhythmia/1.0.0/
- **User-provided archive (used here)**: Google Drive folder
  https://drive.google.com/drive/folders/1F8ZB1A5umCS0mYWf48T7OyrnYEx1NNs0 → `a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0.zip`
  (byte-identical to the canonical PhysioNet release;
  SHA256 `2e4c5f2e89153c1acd2e418d4dec9a6b59881167ac07156061ccafadf5fce1d3`).
- **Citation**: Zheng J., Guo H., Chu H. et al. A large-scale 12-lead electrocardiogram database for arrhythmia study. PhysioNet 2020 (ecg-arrhythmia/1.0.0).

## Dataset audit
- Archive: 2,498,709,074 bytes; 45,152 `.mat` + 45,152 `.hea`
  (all paired; sampled members cross-checked against the archive's own
  SHA256SUMS.txt).
- Records: 45,152 × 12-lead, 500 Hz, exactly 10.0 s, gain 1000/mV, WFDB fmt 16+24.
- Cohorts: Chapman-Shaoxing 10,646 + Ningbo 34,506 = 45,152 (combined = PRIMARY).
  Cohort rule: JS number <= 11045 -> Chapman-Shaoxing; else Ningbo (empirical boundary reproducing the published cohort sizes; all 380 JS-numbering gaps lie inside the Chapman block)
- Signal format: int16 raw; physical mV = raw/1000 (WFDB reference-parser verified,
  max |diff| < 1e-9 mV; physiological cross-check vs the CPSC corpus).

## Preprocessing
IDENTICAL to the frozen CPSC pipeline: physical mV → `resample_poly` 500→100 Hz →
record-level 0.5–40 Hz Butterworth-4 zero-phase band-pass → 10 s / 1000-sample centre
window → per-window per-lead z-normalisation. The whole cohort is exactly 10 s, so the
model window is the full record (pre-declared policy; no external tuning).

## Label mapping
SNOMED-CT mapping frozen BEFORE any scoring (`results/external_label_mapping.csv`,
sha256 `c1b73616947bed0fba91a4ec8b1b8c194157d4d509d4b1f2cdfe410bdc703705`). It reuses the CPSC-validated 13-code table; only exact
or clinically-equivalent concepts are mapped. Class coverage:
SR=8,125; AF=1,780; IAVB=1,140; LBBB=240; RBBB=1,745; PAC=1,321; PVC=1,385; STD=1,668; STE=176.
Records with no target-class code: 29538 (excluded with IDs+reason).

## Frozen-model / zero-shot policy
- The CPSC-trained MultiScaleECGNet ensemble (seeds [42, 43]), thresholds
  and AWARD configuration are FROZEN before any external label is read for scoring.
- NO external training, fine-tuning, threshold calibration, model/defense selection,
  or preprocessing selection was performed.
- The primary external result is the COMBINED cohort; cohort-specific numbers are
  secondary diagnostics.

## Results (zero-shot, combined cohort, frozen thresholds)
- Macro-AUC 0.8622 (95% CI [0.8569,
  0.8673])
- Macro-F1 0.4787 (95% CI [0.4709,
  0.4859]) — threshold-dependent, frozen CPSC thresholds
- Macro-AP 0.5230 | Macro-F2 0.5166 |
  Micro-F1 0.5562 | ECE 0.1333 |
  Brier 0.0987
- Robustness (pre-declared stratified subset, frozen defenses): see
  `results/external_robustness.csv`.
- Morphology (clean→AWARD, lead II): PRD 3.81 ± 0.57%, SNR 28.49 ± 1.55 dB,
  correlation 1.00 ± 0.00, R-peak detection 0.99 ± 0.06.

## Statistical protocol
Record-level bootstrap, 2,000 resamples, seed
42, percentile 95% CIs, fixed admissible-class set.

## Limitations
- Dataset shift: different hospitals, devices, patient mix and annotation conventions
  (GE MUSE-exported SNOMED diagnoses) vs the CPSC2018 challenge labels; threshold-
  dependent metrics mix transfer with label-definition shift.
- The external dataset was NOT used to train the primary model.
- External robustness/morphology/defense-clean comparisons use pre-declared stratified
  subsets (192 / 48 / 512 records) for CPU-compute
  reasons; the primary classification metrics use the FULL evaluable cohort.
- Non-evaluable classes: none (zero external
  support; never fabricated).
- Excluded records: 29538 (no target-class SNOMED code).
- No claim of clinical deployment readiness, universal ECG generalization, or
  robustness to all real-world attacks is made.

## Traceability
`configs/external_final_config.json` records archive hashes, mapping hash, frozen
checkpoints + system signature, thresholds, subsets, seeds and the predictions SHA256.
