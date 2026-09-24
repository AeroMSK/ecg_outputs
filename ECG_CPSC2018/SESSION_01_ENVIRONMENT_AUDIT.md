# ECG CPSC2018 + Zheng — Session 01 Environment Audit

**Date:** 2026-09-23 (Asia/Dhaka)
**Operator:** Senior ML/research implementation engineer (AI assistant)
**GitHub repo:** https://github.com/AeroMSK/ecg_outputs
**Identity:** musakhan5572@gmail.com / AeroMSK

---

## 1. Mandatory Startup Recovery — Result

| Step | Protocol requirement | Actual state |
|------|----------------------|--------------|
| 1 | Locate project workspace | `/home/z/my-project` |
| 2 | Verify whether it is a Git repository | Local `.git` exists but only contains a boilerplate "Initial commit" with `skills/` and `.gitignore`. It is NOT the `AeroMSK/ecg_outputs` repo. |
| 3 | If repo not present locally, clone | `git remote add origin https://github.com/AeroMSK/ecg_outputs.git` added. |
| 4 | `git fetch origin` / `git pull --rebase origin main` | `git fetch origin` succeeded but returned **zero objects** — remote is empty. |
| 5 | `git remote -v` | `origin → https://github.com/AeroMSK/ecg_outputs.git` (fetch+push) ✅ |
| 6 | Inspect repo before modifying | Repo is **empty** (GitHub API: `"size": 0`, `"message": "Git Repository is empty."`). Last push timestamp `2026-09-23T15:37:59Z` — i.e. the repo was created today. |
| 7 | Never assume current local FS is newest | Confirmed — there is no remote state to be newer than. |

**Conclusion of recovery step:** there is **no prior checkpoint, no notebook, no configuration, no golden bundle** on GitHub to recover. The recovery source is empty.

---

## 2. Authentication — Result

- GitHub Personal Access Token (PAT) provided by the user was used to authenticate against the GitHub API and git operations.
- Authentication verified: `GET /user` returned `"login": "AeroMSK"` (id 187884754). ✅
- Token handling (per protocol):
  - Stored in a process env var `GH_TOKEN` (in-memory only).
  - A `git_askpass.sh` helper reads `GH_TOKEN` from the environment; the script file itself contains no secret.
  - `.gh_env` file (chmod 600) holds the export statement for re-sourcing within the session — it is git-ignored and never committed.
  - Token is **never** placed in remote URLs, `.git/config`, Python source, notebooks, README, logs, JSON, CSV, or any committed file.
- ⚠️ **Security recommendation to the user:** the PAT was pasted in plaintext in the chat message. After this session, **rotate / revoke** that token in GitHub → Settings → Developer settings → Personal access tokens, and issue a fresh one for subsequent sessions.

---

## 3. Environment Reality Check

| # | Resource needed by the master prompt | Actual availability in this runtime |
|---|--------------------------------------|-------------------------------------|
| 1 | GPU (CUDA) | ❌ `nvidia-smi` not found. No GPU. |
| 2 | PyTorch | ❌ Not installed in `/home/z/.venv` (`ModuleNotFoundError: No module named 'torch'`). |
| 3 | Mixed precision (AMP) | ❌ N/A without CUDA. |
| 4 | Disk for full CPSC (6,877 records, ~3–5 GB) | ⚠️ 9.3 GB free — feasible for CPSC alone. |
| 5 | Disk for full Zheng archive (45,152 records, ~30+ GB) | ❌ Insufficient. |
| 6 | RAM for batched 12-lead ECG training | ⚠️ 4.1 GB total / 3.6 GB available. Barely enough for tiny batches at 100 Hz. |
| 7 | Google Drive access for source datasets | ❌ `gdown` not installed, no OAuth credentials, no API key. Direct folder URL https://drive.google.com/drive/folders/1F8ZB1A5umCS0mYWf48T7OyrnYEx1NNs0 cannot be programmatically accessed from this sandbox. |
| 8 | Local copy of the canonical notebook `ECG_CPSC2018_FINAL_EXTERNAL_VALIDATION_PUBLICATION_READY (2).ipynb` | ❌ Not present anywhere on the filesystem. |
| 9 | Local copy of `CPSC.zip` | ❌ Not present. |
| 10 | Local copy of `a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0.zip` (Zheng / Chapman-Shaoxing-Ningbo) | ❌ Not present. |
| 11 | Any prior model checkpoint (`.pth`/`.ckpt`/`.pt`) | ❌ None. |
| 12 | Any prior predictions / metrics / manifests | ❌ None. |

---

## 4. What the master prompt requires vs. what this environment can provide

The master prompt requests, in priority order:

1. **Recover the exact canonical 0.9714 / 0.8330 system.** — Impossible: no checkpoint exists in the (empty) GitHub repo, no notebook on disk, no Google Drive access.
2. **Re-run canonical external zero-shot.** — Impossible: no Zheng dataset, no trained model.
3. **Dual-view rhythm + ST morphology representation.** — Implementable as code modules, but cannot be trained or validated without data + GPU.
4. **Temporal attention pooling for PAC/PVC.** — Same: code-only, no validation possible.
5. **200 Hz candidate.** — Same.
6. **Calibration / classwise thresholds.** — Same.
7. **Loss optimization (ASL / BCE+ranking).** — Same.
8. **Long-budget GPU training (40–80 epochs, multiple seeds, multiple architectures).** — ❌ Infeasible: no GPU, no PyTorch, no data.
9. **Heterogeneous ensemble (≥3 seeds × ≥3 architectures).** — ❌ Infeasible in one session.
10. **SSL pretraining on 45,152 unlabeled ECGs.** — ❌ Infeasible: dataset not present, no GPU.
11. **Unsupervised domain adaptation.** — Same.
12. **Supervised Chapman/Ningbo cross-domain adaptation.** — Same.
13. **Final 3–5 seed full-budget execution.** — Same.
14. **Full robustness/morphology/statistics.** — Same.
15. **Fully executed notebook + final ZIP.** — Cannot execute the notebook (no kernel, no data, no torch).

---

## 5. Honest options forward

The protocol explicitly forbids:
- faking successful outputs,
- creating placeholder values,
- fabricating missing metrics,
- using test data for optimization,
- pretending the local filesystem is the newest version.

Therefore, **the only scientifically valid path forward is to first obtain the actual source materials**:

### Option A — User uploads source materials to this runtime (fastest)
1. Upload `ECG_CPSC2018_FINAL_EXTERNAL_VALIDATION_PUBLICATION_READY (2).ipynb` into `/home/z/my-project/upload/` (this directory is shared with the user's session).
2. Upload `CPSC.zip` (≤ 5 GB) — also via the upload mechanism.
3. For the Zheng archive (likely > 9 GB), we cannot fit it on the current 9.3 GB disk. Options:
   - Upload a *subset* sufficient to test the pipeline (e.g. 2,000 records).
   - Provide a public HTTP download URL (e.g. a signed presigned URL or a public mirror) — I can `wget` directly.
   - Or skip Zheng entirely for this session and focus only on CPSC.

### Option B — User pushes source materials to the GitHub repo first, then I pull
- Push the notebook + small config files to `AeroMSK/ecg_outputs` directly (via `git push` from the user's local machine). I will `git pull` them here.
- Datasets are too large for plain Git — use Git LFS or external storage and provide URLs.

### Option C — User runs the heavy training on their own GPU machine
- I produce the **improved notebook + supporting Python modules + configuration files** as deliverables (architecture, loss, calibration, etc.), commit them to GitHub.
- User pulls and executes on their GPU machine.
- Results are pushed back to GitHub; I audit and package.

### Option D — Reality-limited session
- Without source materials, the only valid outputs I can produce in this session are:
  - Project scaffolding (directory tree, `.gitignore`, configs).
  - A standalone audit / planning document.
  - Implementation skeletons for the **dual-view attention ECG network**, **temporal attention pooling**, **lead attention**, **calibration pipeline**, **robustness suite** — written as reviewable Python modules so the user can actually use them later on a GPU machine.
  - A documented recovery plan for the canonical 0.9714 / 0.8330 (which we **cannot** reproduce in this session — the user must supply the original checkpoint, config, and split files).

---

## 6. Recommendation

Given the gap between the prompt's scope and the runtime's reality, I recommend:

1. **In this session (no waiting on the user):** proceed under **Option D**. Produce:
   - Audit (this file).
   - Project scaffolding + git scaffolding (commit + push as the first checkpoint).
   - Implementation skeletons for the high-ROI architecture pieces (dual-view, temporal attention, lead attention, calibration), written so they are drop-in compatible with the canonical notebook once the user supplies it.
   - A documented "recovery checklist" so the next session (with data + GPU) can resume cleanly.
2. **User action required before a real training session can happen:** supply the canonical notebook and the CPSC dataset (at minimum) via the upload directory or via the GitHub repo. Without these, **no scientifically valid execution is possible**, and the protocol explicitly forbids fabricating results.

---

## 7. Golden baseline protection status

| Artifact | Status |
|----------|--------|
| `GOLDEN_CPSC/` directory | Created (empty) under `ECG_CPSC2018/checkpoints/GOLDEN_CPSC/` and `ECG_CPSC2018/FINAL_ECG_RESULTS/checkpoints/GOLDEN_CPSC/`. |
| Canonical checkpoint (0.9714 / 0.8330) | ❌ Not present. Cannot be regenerated without the original training data + GPU. |
| Canonical thresholds | ❌ Not present. |
| Canonical split IDs | ❌ Not present. |
| Canonical config | ❌ Not present. |

The `GOLDEN_CPSC/` directory is reserved and will never be overwritten by experimental outputs. However, **it is currently empty** because there is nothing to fill it with.

---

## 8. Final verification gates — current status

| Gate | Description | Status |
|------|-------------|--------|
| GATE 01 | Correct CPSC dataset | ❌ NOT AVAILABLE |
| GATE 02 | Correct labels | ❌ NOT AVAILABLE |
| GATE 03 | No split overlap | ❌ CANNOT VERIFY — no split present |
| GATE 04 | No val_select leakage | ❌ CANNOT VERIFY |
| GATE 05 | No test leakage | ❌ CANNOT VERIFY |
| GATE 06 | Independent seeds | ❌ N/A |
| GATE 07 | Exact canonical checkpoint recovered OR discrepancy documented | ✅ Discrepancy documented (no checkpoint exists in repo). |
| GATE 08 | All selected checkpoints exist | ❌ None exist |
| GATE 09 | All final metrics regenerated | ❌ None generated |
| GATE 13 | Golden baseline immutable | ✅ Empty directory reserved, write-protected by policy. |
| GATE 16 | External archive verified | ❌ NOT AVAILABLE |
| GATE 17 | External 45,152 census complete | ❌ NOT AVAILABLE |
| GATE 23 | Final ZIP complete | ❌ N/A |

The other gates (10, 11, 12, 14, 15, 18–22) are downstream of training/evaluation and therefore also ❌ pending.

---

## 9. Next concrete action

This audit will be committed to `AeroMSK/ecg_outputs` as the first checkpoint:

```
checkpoint: initial environment audit and project scaffolding
```

After the user supplies source materials (notebook + CPSC dataset minimum), the next session will:

1. Audit the notebook (Sub-agent 1's mandate).
2. Audit the dataset / signals (Sub-agent 2's mandate).
3. Audit label harmonization (Sub-agent 3's mandate).
4. Implement the dual-view + temporal-attention + lead-attention architecture.
5. Run a tiny smoke test on CPU (1 epoch, small subset) just to verify the code path is sound — **not** as a real training run.
6. Document the recovery plan for the canonical 0.9714 / 0.8330.

Sub-agents 4–10 (loss/preprocessing/SSL/robustness/statistics/packaging) will be activated once real training is possible.
