#!/usr/bin/env bash
# Askpass helper: reads token from GH_TOKEN env var (no on-disk storage)
if [ -z "$GH_TOKEN" ]; then
  echo "ERROR: GH_TOKEN not set in environment" >&2
  exit 1
fi
echo "$GH_TOKEN"
