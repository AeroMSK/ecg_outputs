#!/usr/bin/env bash
# One-time setup script for the ECG CPSC2018 runtime environment.
# Idempotent: re-runnable in fresh sessions to restore the working env.
set -euo pipefail

echo "=== Setting up ECG CPSC2018 runtime environment ==="

# 1. Activate the uv-managed venv
export PATH=/home/z/.venv/bin:$PATH

# 2. Install Python dependencies (CPU-only torch to save disk)
cd /home/z
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install wfdb neurokit2 PyWavelets seaborn tqdm gdown

# 3. Set environment variables the notebook reads
export AWARD_WORK_DIR="/home/z/my-project/download/FINAL_ECG_RESULTS"
export AWARD_RUN_MODE="smoke"   # smoke | final
export AWARD_BUDGET="smoke"     # smoke | cpu_final_v2 | cpu_final_ext | standard
export AWARD_RESUME="1"
export CPSC_RAW_ROOT="/home/z/my-project/work/cpsc_extract/CPSC/cpsc_2018"
export CPSC_STORE_DIR="/home/z/my-project/work/store"

# 4. Verify CPSC dataset is present
if [ ! -d "$CPSC_RAW_ROOT" ]; then
  echo "CPSC dataset not found at $CPSC_RAW_ROOT"
  echo "Run: bash ECG_CPSC2018/manifests/redownload_source_files.sh"
  echo "Then: cd /home/z/my-project/work && unzip /home/z/my-project/upload/CPSC.zip && unrar-free x CPSC/CPSC.rar"
  exit 1
fi

N_MAT=$(find "$CPSC_RAW_ROOT" -name "*.mat" | wc -l)
N_HEA=$(find "$CPSC_RAW_ROOT" -name "*.hea" | wc -l)
echo "CPSC dataset: $N_MAT .mat files, $N_HEA .hea files"
if [ "$N_MAT" -ne 6877 ] || [ "$N_HEA" -ne 6877 ]; then
  echo "WARNING: expected 6877 each, got $N_MAT/$N_HEA"
fi

echo ""
echo "=== Environment ready ==="
echo "Python: $(python3 --version)"
echo "Torch:  $(python3 -c 'import torch; print(torch.__version__)')"
echo "Workdir: $AWARD_WORK_DIR"
echo "Budget:  $AWARD_BUDGET (smoke=fast test, cpu_final_v2=production CPU run)"
