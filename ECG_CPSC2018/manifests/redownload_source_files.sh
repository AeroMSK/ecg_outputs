#!/usr/bin/env bash
# Re-download all source materials from the user's Google Drive folder.
# Usage: bash ECG_CPSC2018/manifests/redownload_source_files.sh [destination_dir]
#
# Prerequisites:
#   pip install gdown     (or:  uv pip install gdown)
#
# Source folder:
#   https://drive.google.com/drive/folders/1F8ZB1A5umCS0mYWf48T7OyrnYEx1NNs0

set -euo pipefail

DEST="${1:-/home/z/my-project/upload}"
mkdir -p "$DEST"
cd "$DEST"

verify_sha256() {
  local file="$1"
  local expected="$2"
  if [ -f "$file" ]; then
    actual=$(sha256sum "$file" | awk '{print $1}')
    if [ "$actual" = "$expected" ]; then
      echo "[OK] $file (sha256 verified)"
      return 0
    else
      echo "[WARN] $file exists but sha256 mismatch"
      echo "       expected: $expected"
      echo "       actual:   $actual"
      return 1
    fi
  else
    echo "[MISSING] $file"
    return 1
  fi
}

echo "=== Downloading canonical notebook ==="
if ! verify_sha256 "ECG_CPSC2018_FINAL_EXTERNAL_VALIDATION_PUBLICATION_READY.ipynb" \
   "1cb41fc26d741c626aa64c9954039be8bef0a9a20bf2983b6aad650124d22ac2"; then
  gdown "https://drive.google.com/uc?id=1R8eo8w_MAZdtD_OKuNjHG67rnv1CiBTM" \
        -O "ECG_CPSC2018_FINAL_EXTERNAL_VALIDATION_PUBLICATION_READY.ipynb"
  verify_sha256 "ECG_CPSC2018_FINAL_EXTERNAL_VALIDATION_PUBLICATION_READY.ipynb" \
                "1cb41fc26d741c626aa64c9954039be8bef0a9a20bf2983b6aad650124d22ac2"
fi

echo ""
echo "=== Downloading CPSC.zip (485 MB) ==="
if ! verify_sha256 "CPSC.zip" "b4fc50c7b05cb994e874b1c2a5487d37da63293320ff0a3e77f601ecdbe06ff7"; then
  gdown "https://drive.google.com/uc?id=1BQCG9kdZvKh5nrSyWX0nw0IcLO5qdpRC" -O "CPSC.zip"
  verify_sha256 "CPSC.zip" "b4fc50c7b05cb994e874b1c2a5487d37da63293320ff0a3e77f601ecdbe06ff7"
fi

echo ""
echo "=== Downloading Zheng / Chapman-Shaoxing archive (2.4 GB) ==="
if ! verify_sha256 "a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0.zip" \
   "2e4c5f2e89153c1acd2e418d4dec9a6b59881167ac07156061ccafadf5fce1d3"; then
  gdown "https://drive.google.com/uc?id=1T3AJ3h2TEVSNvNf57NGHOF-tbBuGwDws" \
        -O "a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0.zip"
  verify_sha256 "a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0.zip" \
                "2e4c5f2e89153c1acd2e418d4dec9a6b59881167ac07156061ccafadf5fce1d3"
fi

echo ""
echo "=== All source files present and verified ==="
ls -lah "$DEST"
