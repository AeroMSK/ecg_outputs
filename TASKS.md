# ECG CPSC2018 + Zheng — Master Task Tracker

**Purpose:** Persistent memory across sessions. Updated at every milestone.
**Repository:** https://github.com/AeroMSK/ecg_outputs
**Last updated:** 2026-09-24 (Asia/Dhaka)

---

## 🎯 GOALS (what we aim to achieve)

### Primary dataset (CPSC2018)
| Metric | Smoke result | Target | Gap |
|--------|--------------|--------|-----|
| Macro-AUC | 0.9457 | **0.9800+** | -0.034 |
| Macro-F1 | 0.7197 | **0.9000+** | -0.180 |
| Macro-Precision | 0.7177 | ≥0.85 | -0.132 |
| Macro-Recall | 0.7303 | ≥0.85 | -0.120 |

### External dataset (Zheng / Chapman-Shaoxing-Ningbo)
| Metric | Smoke result | Target | Gap |
|--------|--------------|--------|-----|
| Macro-AUC | ~0.85 | **0.9700+** | -0.12 |
| Macro-F1 | ~0.40 | **0.6000+** | -0.20 |

### Weak classes (per master prompt)
PAC, PVC, STD, STE, SR — special attention required.

---

## ✅ WHAT WE HAVE DONE (completed tasks)

### Session 01 (2026-09-23)
- [x] GitHub remote `AeroMSK/ecg_outputs` set up with secure PAT auth (env-var askpass)
- [x] All source materials downloaded from Google Drive to local disk
- [x] Notebook (1.06 MB), CPSC.zip (485 MB), Zheng archive (2.5 GB) all stored in Git LFS
- [x] Zheng archive chunked into 2 parts (each ≤2 GB) to fit GitHub LFS per-file limit
- [x] `recombine_zheng_archive.sh` helper script + SHA256 verification
- [x] `redownload_source_files.sh` for full re-download with integrity check
- [x] `SOURCE_FILES_MANIFEST.json` provenance manifest
- [x] Environment audit documented in `ECG_CPSC2018/SESSION_01_ENVIRONMENT_AUDIT.md`
- [x] Token security verified (PAT never in any committed file)
- [x] Template `.env` purged from git history via `git filter-branch`

### Session 02 (2026-09-23)
- [x] TASK.txt pulled from GitHub (master execution prompt)
- [x] Notebook audited: 62 code cells, 140k lines, kernel=python3, 9-class output
- [x] Budget modes documented: smoke / cpu_final_v2 / cpu_final_ext / standard / mini (new)
- [x] Dependencies installed: torch 2.14.0+cpu, wfdb, neurokit2, pywt, papermill
- [x] CPSC dataset extracted (6,877 records) to expected path
- [x] Zheng archive restored at `/home/z/my-project/work/zheng.zip`
- [x] Notebook patched with `gc.collect()` in all 50 code cells (memory hygiene)
- [x] New `mini` budget added (6 epochs, 1 seed, control only)
- [x] **Smoke test FULLY EXECUTED** — 62/62 cells, 0 errors, 2.8 min wall time
- [x] Smoke results: Macro-AUC=0.9457, Macro-F1=0.7197
- [x] All 22 external gates PASSED
- [x] Reproducibility checks: 23/24 passed

### Session 03 (2026-09-24 — current)
- [x] Pulled latest from GitHub
- [x] Created this TASKS.md persistent tracker
- [ ] Analyze notebook for improvement opportunities (weak cells, missing ML techniques)
- [ ] Analyze datasets for class imbalance, signal quality, domain shift
- [ ] Implement ML improvements (see "TODO" below)
- [ ] Run full cpu_final_ext budget (NOT smoke)
- [ ] Push final executed notebook + results to GitHub

---

## 📋 TODO (what we want to do in this session)

### Phase A — Analysis (no execution needed)
1. [ ] **Notebook audit** (Sub-agent 1 mandate): inspect each cell, identify weak points
2. [ ] **Dataset audit** (Sub-agent 2 mandate): CPSC class distribution, Zheng cohort split, signal quality
3. [ ] **Label harmonization audit** (Sub-agent 3 mandate): CPSC labels vs Zheng SNOMED codes
4. [ ] **Identify OOM root cause** for cpu_final_ext budget failures

### Phase B — ML Improvements to Implement
Per master prompt's 19 diagnoses, the highest-ROI changes for our constrained runtime:

1. [ ] **Reduce batch_size to 32** (from 64) — halves training memory, fixes OOM
2. [ ] **Add a "cpu_full_v3" budget** sized for this runtime:
   - 15 epochs × 3 seeds × 2 candidates (control + SE) = 90 epoch-runs
   - Estimated wall time: ~5-7 hours on 2-core CPU
3. [ ] **Improve preprocessing** — add ST-morphology view (Diagnosis 1)
4. [ ] **Improve loss** — test ASL or BCE+ranking for PAC/PVC/STD/STE (Diagnosis 6)
5. [ ] **Better calibration** — per-class F1 thresholds fit on val_tune (Diagnosis 7)
6. [ ] **Multi-window inference** — K=3 with median aggregation (Diagnosis 8)
7. [ ] **Stronger threshold search** — F2, recall-constrained policies
8. [ ] **Test-time augmentation** — average predictions over multiple windows

### Phase C — Execution
1. [ ] Run full cpu_full_v3 budget in background (papermill)
2. [ ] Commit results to GitHub at every cell-completion checkpoint
3. [ ] Generate final paper-ready tables (Tables 1–11)
4. [ ] Generate final figures (publication quality)
5. [ ] Create final ZIP package

### Phase D — Delivery
1. [ ] Push fully-executed notebook to GitHub
2. [ ] Push all results CSVs/JSONs/figures to GitHub
3. [ ] Update TASKS.md with final metrics
4. [ ] Honest comparison vs targets

---

## 🚧 BLOCKERS / RISKS

### Hard constraints of this runtime
- **RAM:** 4.1 GB total (peak ~3.5 GB available after OS)
  - Smoke test peak: ~2.2 GB during c49 attack cell
  - cpu_final_ext OOM at c25 training cell — model + activations + gradients + optimizer state > 1.5 GB
- **Disk:** 9.9 GB total, currently 607 MB free
  - Need to clean up before full run
- **CPU:** 2 cores
  - One training epoch ≈ 100 seconds
  - Full cpu_final_v2 budget (30 epochs × 3 seeds × 5 candidates) ≈ 12+ days
- **No GPU** — CPU-only torch 2.14.0+cpu
- **No swap** — `swapon` requires root, not available in this container

### Mitigation strategy
- Reduce batch_size to 32 (halves peak training memory)
- Use gradient accumulation if needed
- Use `cpu_full_v3` budget sized for 4 GB RAM
- Add `torch.cuda.empty_cache()` and `gc.collect()` aggressively
- Use `del` for large tensors after use

---

## 📊 CURRENT METRICS (smoke budget — to be replaced by full run)

### CPSC primary (smoke, 3 epochs, 1 seed)
```
Macro-AUC       0.9457
Macro-F1        0.7197
Macro-Precision 0.7177
Macro-Recall    0.7303
Macro-AP        0.7689
```

### Per-class AUC (smoke)
```
SR    0.9466    AF    0.9838    IAVB  0.9599
LBBB  0.9972    RBBB  0.9729
PAC   0.8380    PVC   0.9242    STD   0.9527    STE  0.9361
```

### External (Zheng) — smoke subset only
```
Zero-shot AUC: 0.85 (approx)
Zero-shot F1:  0.40 (approx)
```

---

## 🗂️ KEY FILE LOCATIONS

### In repo (committed to GitHub)
- `ECG_CPSC2018/notebooks/ECG_CPSC2018_FINAL_EXTERNAL_VALIDATION_PUBLICATION_READY.ipynb` — canonical notebook (modified: gc.collect patches + mini budget)
- `ECG_CPSC2018/notebooks/ECG_CPSC2018_FINAL_EXTERNAL_VALIDATION_PUBLICATION_READY.original.ipynb` — original preserved
- `ECG_CPSC2018/notebooks/ECG_CPSC2018_FINAL_EXECUTED_SMOKE.ipynb` — smoke-test-executed output
- `ECG_CPSC2018/manifests/SOURCE_FILES_MANIFEST.json` — provenance + recovery
- `ECG_CPSC2018/manifests/RUNTIME_ENVIRONMENT.md` — env setup docs
- `TASK.txt` — master execution prompt from user
- `TASKS.md` — this file
- `worklog.md` — full session log

### In repo LFS (committed, large files)
- `ECG_CPSC2018/datasets/CPSC.zip` (485 MB)
- `ECG_CPSC2018/datasets/zheng_archive_parts/zheng_part_0.zip` (1.4 GB)
- `ECG_CPSC2018/datasets/zheng_archive_parts/zheng_part_1.zip` (953 MB)

### Local only (gitignored — too large or volatile)
- `/home/z/my-project/work/cpsc_extract/CPSC/cpsc_2018/` — extracted CPSC records (1.3 GB)
- `/home/z/my-project/work/zheng.zip` — Zheng archive (2.5 GB)
- `/home/z/my-project/work/store/` — preprocessed memmap cache (503 MB)
- `/home/z/my-project/download/FINAL_ECG_RESULTS/checkpoints/*.pth` — model weights (69 MB)
- `/home/z/my-project/download/FINAL_ECG_RESULTS/predictions/*.npy` — predictions (772 KB)

---

## 🔁 RECOVERY PROCEDURE (for any future session)

```bash
# 1. Clone the repo
git clone https://github.com/AeroMSK/ecg_outputs.git
cd ecg_outputs

# 2. Pull LFS objects (datasets + executed notebook)
git lfs pull

# 3. Set up runtime environment
bash scripts/setup_runtime_env.sh

# 4. Re-download Zheng archive from Google Drive (too large for LFS to keep locally)
bash ECG_CPSC2018/manifests/redownload_source_files.sh

# 5. Recombine Zheng (if it was chunked)
bash ECG_CPSC2018/manifests/recombine_zheng_archive.sh /home/z/my-project/work/zheng.zip

# 6. Extract CPSC dataset
mkdir -p /home/z/my-project/work/cpsc_extract
cd /home/z/my-project/work/cpsc_extract
unzip /home/z/my-project/upload/CPSC.zip
unrar-free x -y CPSC/CPSC.rar

# 7. Run the notebook (smoke for fast verification, cpu_final_ext for full)
export AWARD_WORK_DIR=/home/z/my-project/download/FINAL_ECG_RESULTS
export AWARD_RUN_MODE=final
export AWARD_BUDGET=cpu_final_ext   # or smoke, cpu_final_v2, standard
export CPSC_RAW_ROOT=/home/z/my-project/work/cpsc_extract/CPSC/cpsc_2018
export CPSC_STORE_DIR=/home/z/my-project/work/store
export ZHENG_ZIP=/home/z/my-project/work/zheng.zip

/home/z/.venv/bin/python -m papermill \
  ECG_CPSC2018/notebooks/ECG_CPSC2018_FINAL_EXTERNAL_VALIDATION_PUBLICATION_READY.ipynb \
  ECG_CPSC2018/notebooks/ECG_CPSC2018_FINAL_EXECUTED.ipynb \
  --kernel python3 --request-save-on-cell-execute --log-output
```

---

## 📝 COMMIT HISTORY (recent)

```
bbb2a6c  checkpoint: final delivery - smoke test fully executed
a6e1038  checkpoint: update worklog with session 02 progress
bc0265e  checkpoint: smoke test FULLY EXECUTED - 62/62 cells, 0 errors
7e8beff  checkpoint: smoke test - 46/62 cells executed, kernel OOM in e11
59e5026  checkpoint: smoke test - reached c49 cell, kernel OOM on heavy cells
506d549  checkpoint: smoke test partial - 36/60 cells completed
9c8d3af  checkpoint: runtime env setup (deps installed, CPSC extracted)
f7f50d6  Add files via upload (user - TASK.txt)
c622cc1  checkpoint: update worklog with security cleanup details
5b15738  checkpoint: session 01 worklog
```

---

## ⚠️ HONEST EXPECTATIONS

Given the runtime constraints (4 GB RAM, 2-core CPU, no GPU, no swap):
- **CPSC F1=90 / AUC=98** may not be fully achievable — the master prompt itself notes these are "optimization targets, NOT guaranteed outcomes"
- **External F1=60 / AUC=97** is ambitious — the v2 reference achieved F1=0.81 / AUC=0.97 on CPSC but only F1=0.54 / AUC=0.88 on external
- We will pursue every valid ML improvement within the compute budget, document honestly, and never fabricate metrics

The goal is the strongest genuinely reproducible result, not just bigger numbers.

---

## 📊 SESSION 03 PROGRESS (2026-09-24)

### What I tried
1. **Added `cpu_full_v3` budget** — 12 epochs × 2 seeds × 3 candidates (control+SE+random_window)
   - Reduced from 30 epochs (cpu_final_v2) to fit runtime
   - Added ensemble, 5-fold CV, full robustness suite
2. **Reduced BATCH_SIZE from 64 to 32** — env-overridable via `AWARD_BATCH_SIZE`
3. **Added gc.collect() to all 50 code cells** (carried over from session 02)
4. **Multiple execution attempts**:
   - papermill: died at cell c25 (training cell) repeatedly
   - nbclient with allow_errors=True: died at same cell
   - jupyter nbconvert with --allow-errors: died at same cell
   - Direct Python test outside jupyter: works fine (training succeeds)

### Root cause analysis
The kernel is being killed by something OUTSIDE my control. Possibilities:
- Container cgroup memory limit (not visible via `free`)
- IM gateway killing long-running processes after ~8 minutes
- Bash tool timeout (30 min max per call) truncating monitoring

### Honest path forward
Given the runtime keeps killing the kernel during the training cell, I cannot complete a full-budget training run in this session. The options are:
1. **Persist what we have** (smoke test fully executed, TASKS.md updated with all analysis)
2. **Use the smoke results as the final delivery** — they're scientifically valid, just lower-budget
3. **Document the blocker** so the next session (with different runtime?) can resume

### What is committed (so far) in session 03
- TASKS.md (this file, persistent tracker)
- Updated .gitignore
- Modified notebook (with cpu_full_v3 budget + batch_size=32)
- 5+ commits pushed to GitHub

