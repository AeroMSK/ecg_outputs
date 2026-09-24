#!/usr/bin/env bash
# Recombine the Zheng/Chapman-Shaoxing archive from LFS-stored chunks.
#
# GitHub LFS has a 2 GB per-file hard limit. The original Zheng archive is 2.5 GB,
# so it was split into 2 parts of ~1.4 GB and ~953 MB before being pushed to LFS.
#
# Usage:
#   bash ECG_CPSC2018/manifests/recombine_zheng_archive.sh [output_path]
#
# Default output: /home/z/my-project/upload/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0.zip
#
# After recombination, the script verifies SHA256 against the canonical hash
# 2e4c5f2e89153c1acd2e418d4dec9a6b59881167ac07156061ccafadf5fce1d3
# which matches the original file downloaded from Google Drive
# (https://drive.google.com/uc?id=1T3AJ3h2TEVSNvNf57NGHOF-tbBuGwDws).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PARTS_DIR="$REPO_ROOT/ECG_CPSC2018/datasets/zheng_archive_parts"
OUTPUT="${1:-/home/z/my-project/upload/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0.zip}"
EXPECTED_SHA256="2e4c5f2e89153c1acd2e418d4dec9a6b59881167ac07156061ccafadf5fce1d3"

mkdir -p "$(dirname "$OUTPUT")"

echo "=== Recombining Zheng archive parts ==="
echo "Parts dir: $PARTS_DIR"
ls -lah "$PARTS_DIR"
echo ""

# Concatenate parts in order (zheng_part_0.zip, zheng_part_1.zip)
PART_FILES=$(ls "$PARTS_DIR"/zheng_part_*.zip 2>/dev/null | sort)
if [ -z "$PART_FILES" ]; then
  echo "ERROR: No zheng_part_*.zip files found in $PARTS_DIR" >&2
  exit 1
fi

echo "Concatenating parts into: $OUTPUT"
cat $PART_FILES > "$OUTPUT"

echo ""
echo "=== Verifying SHA256 ==="
ACTUAL_SHA256=$(sha256sum "$OUTPUT" | awk '{print $1}')
echo "Expected: $EXPECTED_SHA256"
echo "Actual:   $ACTUAL_SHA256"

if [ "$ACTUAL_SHA256" = "$EXPECTED_SHA256" ]; then
  echo ""
  echo "[OK] Zheng archive recombined successfully and SHA256 verified."
  ls -lah "$OUTPUT"
else
  echo ""
  echo "[ERROR] SHA256 mismatch! Archive may be corrupt." >&2
  exit 1
fi
