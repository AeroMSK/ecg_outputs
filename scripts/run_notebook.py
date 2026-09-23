#!/usr/bin/env python
"""Custom notebook executor using nbclient.
Continues past errors so partial execution is preserved.
"""
import os
import sys
import time
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')

INPUT_NB = '/home/z/my-project/ECG_CPSC2018/notebooks/ECG_CPSC2018_FINAL_EXTERNAL_VALIDATION_PUBLICATION_READY.ipynb'
OUTPUT_NB = '/home/z/my-project/ECG_CPSC2018/notebooks/ECG_CPSC2018_FINAL_EXECUTED_cpu_full_v3.ipynb'
LOG_PATH = '/home/z/my-project/download/logs/cpu_full_v3_run4.log'

# Load the notebook
print(f"Loading: {INPUT_NB}")
with open(INPUT_NB) as f:
    nb = nbformat.read(f, as_version=4)
print(f"  Total cells: {len(nb['cells'])}")

# Create the client
client = NotebookClient(
    nb,
    timeout=7200,           # 2 hour per-cell timeout
    kernel_name='python3',
    allow_errors=True,      # Continue past errors
    resources={'metadata': {'path': '/home/z/my-project/'}},
    record_timing=True,
)

print("Starting kernel...")
client.execute(cleanup_kc=False)

# Save the output
print(f"Saving to: {OUTPUT_NB}")
with open(OUTPUT_NB, 'w') as f:
    nbformat.write(nb, f)

# Count cells with outputs/errors
code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
executed = sum(1 for c in code_cells if c.get('execution_count') is not None)
errors = sum(1 for c in code_cells if any(o.get('output_type') == 'error' for o in c.get('outputs', [])))
print(f"Done. {executed}/{len(code_cells)} cells executed, {errors} errors")
