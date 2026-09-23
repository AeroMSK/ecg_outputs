#!/usr/bin/env python
"""Direct test of the training cell to see what fails."""
import os, sys
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['AWARD_WORK_DIR'] = '/home/z/my-project/download/FINAL_ECG_RESULTS'
os.environ['AWARD_RUN_MODE'] = 'final'
os.environ['AWARD_BUDGET'] = 'cpu_full_v3'
os.environ['AWARD_RESUME'] = '0'
os.environ['AWARD_BATCH_SIZE'] = '32'
os.environ['CPSC_RAW_ROOT'] = '/home/z/my-project/work/cpsc_extract/CPSC/cpsc_2018'
os.environ['CPSC_STORE_DIR'] = '/home/z/my-project/work/store'
os.environ['ZHENG_ZIP'] = '/home/z/my-project/work/zheng.zip'

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

print(f"PID: {os.getpid()}", flush=True)
print(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS')}", flush=True)

# Try to actually do a tiny training step
import torch
torch.set_num_threads(2)
print(f"torch: {torch.__version__}", flush=True)
print(f"threads: {torch.get_num_threads()}", flush=True)

# Create a simple model
import torch.nn as nn
model = nn.Sequential(
    nn.Conv1d(12, 24, 5, padding=2),
    nn.BatchNorm1d(24),
    nn.ReLU(),
    nn.Conv1d(24, 24, 5, padding=2),
    nn.BatchNorm1d(24),
    nn.ReLU(),
    nn.AdaptiveAvgPool1d(1),
    nn.Flatten(),
    nn.Linear(24, 9)
)
print(f"Model params: {sum(p.numel() for p in model.parameters())}", flush=True)

# Test forward+backward
x = torch.randn(32, 12, 1000)
y = torch.randint(0, 2, (32, 9)).float()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

print("Starting 3 training steps...", flush=True)
for step in range(3):
    optimizer.zero_grad()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    print(f"  step {step}: loss={loss.item():.4f}", flush=True)

print("OK - training works in this environment", flush=True)
