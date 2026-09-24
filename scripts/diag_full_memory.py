#!/usr/bin/env python
"""Run the notebook script and report memory after each cell."""
import os, sys, time, resource
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

def mem_mb():
    return int(open('/proc/self/status').read().split('VmRSS:')[1].split('kB')[0]) // 1024

# Read the script
with open('/home/z/my-project/scripts/run_full_notebook.py') as f:
    script = f.read()

# Print mem before exec
print(f"Before exec: {mem_mb()} MB", flush=True)

# Execute in our process
try:
    exec(compile(script, 'run_full_notebook.py', 'exec'), {'__name__': '__main__'})
    print(f"\nScript complete! Final mem: {mem_mb()} MB", flush=True)
except MemoryError as e:
    print(f"\n*** MEMORY ERROR: {e} ***", flush=True)
    print(f"Mem at crash: {mem_mb()} MB", flush=True)
except Exception as e:
    print(f"\n*** ERROR: {type(e).__name__}: {e} ***", flush=True)
    import traceback
    traceback.print_exc()
