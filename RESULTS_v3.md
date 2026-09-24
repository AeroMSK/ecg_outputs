# AWARD-ECG v3 — GPU Re-Run + Performance Upgrade

**Status:** STOPPED — runtime is CPU-only (no GPU detected)
**Per master prompt Section 12 Step 1:** "If CPU-only, stop and report. Do not silently fall back to the 12-epoch budget and present the result as final."
**Date:** 2026-09-24 (Asia/Dhaka)
**Repo:** https://github.com/AeroMSK/ecg_outputs

---

## 0. Hardware verification (Section 12 Step 1)

| Check | Result |
|-------|--------|
| `nvidia-smi` | ❌ Not found |
| `/proc/driver/nvidia` | ❌ Not present |
| `lspci \| grep -i nvidia` | ❌ No NVIDIA PCI device |
| `/dev/nvidia*` device files | ❌ None |
| `torch.cuda.is_available()` | ❌ Cannot test (torch not yet installed in fresh runtime) |
| CPU cores | 2 |
| RAM (total / available) | 3.9 GiB / 3.5 GiB |
| Disk (total / free) | 9.9 GiB / 7.0 GiB |
| Cgroup memory hard cap | 4 GiB (verified in previous sessions) |
| Swap | 0 (cannot be enabled without root) |

**Conclusion:** This runtime is CPU-only. The `gpu_v3` budget (60+ epochs, 5 seeds, 5 candidates, BATCH_SIZE=64, AMP=True, SIG_LEN=5000) cannot execute here.

---

## 1. Baseline numbers verified from previous session (Section 12 Step 2)

These are the in-session replica numbers from the previous session's `fast_cpu` budget execution (62/62 cells completed, 0 errors, notebook reported "NOTEBOOK FULLY EXECUTED : YES"):

| Metric | Previous session `fast_cpu` | Master prompt's claimed `cpu_final_ext` | Match? |
|--------|------------------------------|------------------------------------------|--------|
| CPSC Macro-AUC | 0.9605 | 0.9725 | ❌ Different budget |
| CPSC Macro-F1 | 0.7633 | 0.8144 | ❌ Different budget |
| External Macro-AUC | 0.8622 | 0.8835 | ❌ Different budget |
| External Macro-F1 | 0.4787 | 0.5427 | ❌ Different budget |

**Discrepancy explanation:** The master prompt's Section 1 numbers (`0.9725 / 0.8144 / 0.8835 / 0.5427`) appear to come from the original `cpu_final_ext` budget which the user described in earlier sessions as a "reduced-compute replica". My previous session could not sustain the `cpu_final_ext` budget (OOM at training cell c25) so I used `fast_cpu` (3 epochs × 2 seeds × 1 candidate), which produced the lower numbers above. **Both are honest measurements under their respective budgets.** Neither reaches the master prompt's targets (0.98/0.90/0.97/0.60).

Golden (protected) CPSC reference: **AUC 0.9714 / F1 0.8330** — preserved in `GOLDEN_CPSC_FINAL/`, hash-verified, NOT modified.

CV out-of-fold (primary uncontaminated evidence, `fast_cpu` budget): **AUC 0.9173 / F1 0.6742** (3-fold instead of the master prompt's 5-fold, due to compute constraints).

---

## 2. Per-class failure profile (verified from `external_per_class_metrics.csv`)

The external zero-shot results from the previous session (frozen CPSC thresholds at 0.5):

| Class | Support | AUC | F1 | Precision | Recall |
|-------|---------|-----|------|-----------|--------|
| SR | 8,125 | 0.9587 | 0.7363 | 0.9781 | 0.5903 |
| AF | 1,780 | 0.9645 | 0.9004 | 0.9043 | 0.8966 |
| IAVB | 1,140 | 0.9188 | 0.7000 | 0.7568 | 0.6512 |
| LBBB | 240 | 0.9293 | 0.6230 | 0.5758 | 0.6786 |
| RBBB | 1,745 | 0.9771 | 0.7797 | 0.6479 | 0.9787 |
| PAC | 1,321 | 0.7045 | 0.1978 | 0.1475 | 0.3000 |
| PVC | 1,385 | 0.8715 | 0.4918 | 0.7143 | 0.3750 |
| STD | 1,668 | 0.6517 | 0.1519 | 0.2400 | 0.1111 |
| STE | 176 | 0.7385 | 0.2927 | 0.2609 | 0.3333 |

**Discrepancy with master prompt Section 1:** The master prompt lists slightly higher numbers (e.g. external SR AUC 0.9654 vs mine 0.9587). These small differences are consistent with **different training budgets** (3 epochs vs 12 epochs) producing different frozen models, not a methodology error. The qualitative failure profile matches: STD and STE are the catastrophic external failures, PAC and PVC are weak, AF and SR show threshold/prevalence-shift symptoms.

---

## 3. What the master prompt requests that CANNOT be done here

The master prompt's 10-sub-agent plan (Section 4) requires:

| Sub-agent | Requires | Available? |
|-----------|----------|------------|
| A1 (data store 500 Hz) | Disk for 5,000-sample tensors × 6,877 records = ~6 GB | ❌ Only 7 GB free; would leave no room for anything else |
| A2 (upgraded CNN, 60+ epochs × 5 seeds) | GPU or 5+ days CPU | ❌ No GPU; ~5-7 days CPU at 60 epochs × 5 seeds |
| A3 (Transformer/Mamba hybrid) | GPU strongly preferred | ❌ No GPU |
| A4 (ECGFounder fine-tune) | GPU mandatory for layer-wise LR + AMP | ❌ No GPU |
| A5 (multi-source pretraining) | GPU + ~50 GB disk for additional datasets | ❌ Neither |
| A6 (SSL pretraining) | GPU + days of compute | ❌ No GPU |
| A7 (loss/threshold benchmarks) | GPU for parallel ablation runs | ❌ No GPU |
| A8 (ensemble + stacking) | Predictions from A2-A5 | ❌ Depends on A2-A5 |
| A9 (external T0/T1/T2/T3) | Frozen system from A8 | ❌ Depends on A8 |
| A10 (stats + packaging) | Final system from A9 | ❌ Depends on A9 |

**Spawning 10 parallel sub-agents on a 2-core 4 GB-RAM CPU machine is impossible.** Even spawning 2 would immediately OOM the cgroup.

---

## 4. What I AM doing instead (honoring "stop and report")

Per the master prompt's explicit Section 12 mandate:

> "If only CPU is available, STOP and tell the user plainly that the targets cannot be reached on CPU and list exactly what was skipped. Do not silently fall back to the 12-epoch budget and present the result as final."

I am:

1. **Stopping the full execution.** No silent fallback to `cpu_final_ext` or `fast_cpu` presented as final.
2. **Creating the v3 notebook scaffold** (`ECG_CPSC2018_FINAL_EXTERNAL_VALIDATION_v3.ipynb`) with:
   - A `gpu_v3` budget cell that defines 60+ epochs × 5 seeds × 5 candidates, SIG_LEN=5000, TARGET_FS=500, BATCH_SIZE=64, AMP=True
   - An explicit `assert torch.cuda.is_available()` that **fails loudly** on CPU-only machines
   - Comments documenting each of the 10 sub-agent responsibilities and what they would do
3. **Documenting the protocol tiers (T0/T1/T2/T3)** in the notebook as a reference for future GPU runs
4. **Committing everything to GitHub** with this `RESULTS_v3.md` as the honest report
5. **Preserving the golden baseline** (GOLDEN_CPSC_FINAL, golden_v4) — untouched

---

## 5. What is NOT run and why (mandatory honesty clause from Section 8)

| Item | Reason not run |
|------|----------------|
| `gpu_v3` budget training (60 epochs × 5 seeds × 5 candidates) | No GPU; would take ~5-7 days CPU and OOM at training cell |
| Backbone A (upgraded CNN at 500 Hz) | Same as above |
| Backbone B (Transformer/Mamba hybrid) | Same as above; also no GPU |
| ECGFounder fine-tune (A4) | GPU mandatory; needs HuggingFace download (~600 MB) |
| Multi-source pretraining (A5) | No GPU + no disk for additional datasets (PTB-XL, Georgia, etc.) |
| SSL pretraining (A6) | No GPU; would take days |
| Loss/threshold benchmarks (A7) | Each loss variant needs a full training run; no GPU |
| Ensemble + stacking (A8) | Depends on A2-A5 |
| External T1/T2/T3 (A9) | Depends on A8's frozen system |
| Robustness reruns | Master prompt Section 2 RC7 says deprioritise until clean model is upgraded |
| 10 parallel sub-agents | 2-core CPU + 4 GB cgroup cannot sustain 10 parallel Python processes |

---

## 6. Targets (Section 8) — status

| Metric | Target | Achieved (fast_cpu, previous session) | Lower-bound of 95% CI | Tier | Status |
|--------|--------|----------------------------------------|------------------------|------|--------|
| CPSC Macro-AUC | 0.98 | 0.9605 | 0.9268 (bootstrap) | T0 | ❌ Not reached |
| CPSC Macro-F1 | 0.90 | 0.7633 | 0.6652 (bootstrap) | T0 | ❌ Not reached |
| External Macro-AUC | 0.97 | 0.8622 | n/a (compute-constrained bootstrap) | T0 | ❌ Not reached |
| External Macro-F1 | 0.60 | 0.4787 | n/a | T0 | ❌ Not reached |

**No target was reached** in the previous session's `fast_cpu` budget. The master prompt's Section 8 itself notes: "AUC 0.97 zero-shot is very unlikely" and "Treat 0.85-0.87 as strong, 0.90 as stretch" for CPSC F1.

---

## 7. What the user needs to provide to actually execute this plan

To run the master prompt's full plan, the user needs to run the notebook on a machine with at minimum:

- **GPU:** any CUDA-capable GPU with ≥ 8 GB VRAM (RTX 3090 / 4090 / A100 / H100 preferred)
- **RAM:** ≥ 32 GB (for the multi-source datasets + multi-seed ensemble training)
- **Disk:** ≥ 100 GB free (for PTB-XL ~3 GB, Georgia ~3 GB, CPSC-Extra ~1 GB, PTB ~1 GB, St-Petersburg ~1 GB, plus checkpoints and predictions)
- **CUDA toolkit:** 11.8+ for torch 2.x with CUDA
- **Python:** 3.9+ with pip / uv

Then in that environment, execute:

```bash
git clone https://github.com/AeroMSK/ecg_outputs.git
cd ecg_outputs
git lfs pull                          # restores CPSC.zip + Zheng chunks (~2.9 GB)
bash ECG_CPSC2018/manifests/recombine_zheng_archive.sh  # re-assemble Zheng zip
# Install additional datasets:
#   PTB-XL: https://physionet.org/content/ptb-xl/1.0.3/
#   Georgia: https://physionet.org/content/ecg-arrhythmia/1.0.0/ (already have via Zheng folder!)
#   St-Petersburg INCART: https://physionet.org/content/incartdb/1.0.0/
#   PTB: https://physionet.org/content/ptbdb/1.0.0/
#   CPSC-Extra: in CPSC2018 challenge (separate from main CPSC2018)

# Set environment:
export AWARD_BUDGET=gpu_v3
export AWARD_BATCH_SIZE=64
export CUDA_VISIBLE_DEVICES=0
# Note: gpu_v3 budget asserts torch.cuda.is_available() and will fail loudly if not.

# Execute:
jupyter nbconvert --to notebook --execute \
    ECG_CPSC2018/notebooks/ECG_CPSC2018_FINAL_EXTERNAL_VALIDATION_v3.ipynb \
    --output ECG_CPSC2018_FINAL_EXTERNAL_VALIDATION_v3_EXECUTED.ipynb \
    --ExecutePreprocessor.timeout=86400   # 24h timeout per cell
```

Expected wall time on a single RTX 3090: **~12-24 hours** (60 epochs × 5 seeds × 5 candidates × ~2 min/epoch = ~50 hours, but parallelism via DDP cuts this).

---

## 8. What HAS been committed in this session

1. **`RESULTS_v3.md`** — this file (honest report)
2. **`ECG_CPSC2018/notebooks/ECG_CPSC2018_FINAL_EXTERNAL_VALIDATION_v3.ipynb`** — new v3 notebook scaffold with:
   - `gpu_v3` budget definition (60 epochs × 5 seeds × 5 candidates, SIG_LEN=5000, TARGET_FS=500, BATCH_SIZE=64, AMP=True)
   - Explicit `assert torch.cuda.is_available()` that fails loudly on CPU
   - Documentation of the 10 sub-agent plan as a markdown reference section
   - Documentation of the T0/T1/T2/T3 protocol tiers
   - Documentation of the foundation-model allow/forbid list (Section 5.3)
   - All other existing cells preserved unchanged
3. **`TASKS.md`** updated with this session's status
4. **`worklog.md`** updated with session log

---

## 9. What was preserved (no destructive changes)

- ✅ `GOLDEN_CPSC_FINAL/` — hash-verified, untouched
- ✅ `golden_v4/` — untouched
- ✅ `CPSC_golden_numbers.json` — untouched (AUC 0.9714 / F1 0.8330)
- ✅ Previous session's `fast_cpu` results in `download/FINAL_ECG_RESULTS/` — untouched
- ✅ Original notebook `ECG_CPSC2018_FINAL_EXTERNAL_VALIDATION_PUBLICATION_READY.ipynb` — untouched
- ✅ All previous results CSVs / JSONs / figures — untouched

---

## 10. Final message to user (per Section 11)

**(a) Final tiered numbers vs the four targets:**

| Metric | Target | T0 (fast_cpu, prev session) | T1 | T2 | T3 |
|--------|--------|------------------------------|----|----|----|
| CPSC Macro-AUC | 0.98 | 0.9605 | n/a | n/a | n/a |
| CPSC Macro-F1 | 0.90 | 0.7633 | n/a | n/a | n/a |
| External Macro-AUC | 0.97 | 0.8622 | NOT RUN | NOT RUN | NOT RUN |
| External Macro-F1 | 0.60 | 0.4787 | NOT RUN | NOT RUN | NOT RUN |

T1/T2/T3 were not run because they require the v3 frozen system, which requires GPU training.

**(b) Which targets were met at the CI lower bound:**

**None.** Zero of the four targets were reached. The previous session's `fast_cpu` budget was the maximum feasible on this 4 GB RAM / 2-core CPU runtime.

**(c) What was NOT run and why:**

See Section 5 above. The complete `gpu_v3` budget, all 10 sub-agents, all foundation-model fine-tunes, all multi-source pretraining, and all T1/T2/T3 external tiers were not run. Reason: **no GPU available in this runtime**.

**(d) Any gate that failed:**

- **GATE-V3-01** "GPU budget actually used" — FAIL (no GPU; could not run gpu_v3 budget)
- **GATE-V3-04** "No CSN or CPSC-test in pretraining set" — N/A (no pretraining run)
- **GATE-V3-06** "T0 external run made zero external-label calls before frozen-system SHA" — N/A (no v3 frozen system)
- **GATE-V3-08** "All reported numbers regenerate from saved prediction files" — PASS (verified in previous session)
- **GATE-V3-09** "Paired significance vs old v2 system" — N/A (no v3 system to compare)
- **GATE-V3-10** "Golden CPSC reference files unchanged" — PASS (hash check)

All other gates (V3-02, V3-03, V3-05, V3-07, V3-11) are downstream of training and therefore N/A.

**(e) Wall-clock and hardware used:**

- Hardware: 2-core CPU, 4 GB cgroup memory hard cap, 9.9 GB disk, no GPU, no swap
- Wall-clock for this session: ~30 min (mandatory recovery + GPU detection + v3 scaffold + commit/push)
- Wall-clock for previous session's `fast_cpu` execution: ~3.5 min training, ~10 min total including external evaluation

**No marketing language. The targets were not met. The runtime cannot support the master prompt's plan. The v3 notebook scaffold is ready to execute on appropriate hardware.**
