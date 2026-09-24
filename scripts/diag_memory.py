#!/usr/bin/env python
"""Diagnostic: simulate the exact training step from cell c25 to find memory peak."""
import os, gc, time, sys
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['MALLOC_ARENA_MAX'] = '1'
os.environ['AWARD_WORK_DIR'] = '/home/z/my-project/download/FINAL_ECG_RESULTS'
os.environ['AWARD_RUN_MODE'] = 'final'
os.environ['AWARD_BUDGET'] = 'cpu_full_v3'
os.environ['AWARD_RESUME'] = '0'
os.environ['AWARD_BATCH_SIZE'] = '16'
os.environ['CPSC_RAW_ROOT'] = '/home/z/my-project/work/cpsc_extract/CPSC/cpsc_2018'
os.environ['CPSC_STORE_DIR'] = '/home/z/my-project/work/store'
os.environ['ZHENG_ZIP'] = '/home/z/my-project/work/zheng.zip'

import torch
torch.set_num_threads(1)
import numpy as np

def mem_mb():
    return int(open('/proc/self/status').read().split('VmRSS:')[1].split('kB')[0]) // 1024

print(f"After imports: {mem_mb()} MB", flush=True)

# Load the store (memmap)
store_flat = np.load('/home/z/my-project/work/store/recstore_flat.npy', mmap_mode='r')
store_off = np.load('/home/z/my-project/work/store/recstore_off.npy', mmap_mode='r')
print(f"After store mmap: {mem_mb()} MB", flush=True)

# Define a tiny model (similar to MultiScaleECGNet but smaller)
import torch.nn as nn
import torch.nn.functional as F

class TinyECGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv1d(12, 24, 5, padding=2)
        self.bn1 = nn.BatchNorm1d(24)
        self.layer1 = nn.Sequential(
            nn.Conv1d(24, 24, 5, padding=2),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.Conv1d(24, 24, 5, padding=2),
            nn.BatchNorm1d(24),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(24, 9)
    def forward(self, x):
        x = F.relu(self.bn1(self.stem(x)))
        x = self.layer1(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

model = TinyECGNet()
print(f"After model creation: {mem_mb()} MB", flush=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()
print(f"After optimizer+loss: {mem_mb()} MB", flush=True)

# Now do 5 training steps with batch=16
batch_size = 16
print(f"\n=== Simulating training with batch={batch_size} ===", flush=True)
for step in range(5):
    # Generate a fake batch
    x = torch.randn(batch_size, 12, 1000, dtype=torch.float32)
    y = torch.randint(0, 2, (batch_size, 9), dtype=torch.float32)
    
    optimizer.zero_grad()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    
    print(f"  Step {step}: loss={loss.item():.4f}, mem={mem_mb()} MB", flush=True)
    del x, y, out, loss
    gc.collect()

print(f"\nFinal memory: {mem_mb()} MB", flush=True)
print("Diagnostic complete.", flush=True)
