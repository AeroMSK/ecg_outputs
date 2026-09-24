# Runtime Environment

## Python venv
- Path: `/home/z/.venv/` (uv-managed, Python 3.12)
- Activate: `export PATH=/home/z/.venv/bin:$PATH`

## Required packages
| Package    | Version   | Purpose                              |
|------------|-----------|--------------------------------------|
| torch      | 2.14.0+cpu| Deep learning (CPU build)            |
| numpy      | 2.1.3     | Numerics                             |
| pandas     | 2.2.3     | DataFrames                           |
| scipy      | 1.14.1    | Signal processing (butter, find_peaks)|
| sklearn    | 1.5.2     | Metrics, splits                      |
| pywt       | 1.8.0     | Wavelet transforms (AWARD defense)   |
| matplotlib | 3.9.2     | Figures                              |
| seaborn    | 0.13.2    | Statistical plots                    |
| tqdm       | 4.67.1    | Progress bars                        |
| wfdb       | 4.3.1     | ECG format I/O                       |
| neurokit2  | 0.2.13    | ECG-specific signal processing       |
| gdown      | 6.4.0     | Google Drive downloads               |

Install all in one shot:
```bash
cd /home/z
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install wfdb neurokit2 PyWavelets seaborn tqdm gdown
```

## Environment variables the notebook reads
| Variable           | Default                                              | Notes                          |
|--------------------|------------------------------------------------------|--------------------------------|
| `AWARD_WORK_DIR`   | `/home/z/my-project/download/FINAL_ECG_RESULTS`     | Output root                    |
| `AWARD_RUN_MODE`   | `final`                                              | `smoke` for fast smoke test    |
| `AWARD_BUDGET`     | `cpu_final_v2` (CPU) or `standard` (GPU)             | `smoke` for quick validation   |
| `AWARD_RESUME`     | `1`                                                  | Resume from stage caches       |
| `CPSC_RAW_ROOT`    | `/home/z/my-project/work/cpsc_extract/CPSC/cpsc_2018`| Path to CPSC2018 records       |
| `CPSC_STORE_DIR`   | `/home/z/my-project/work/store`                      | Preprocessed memmap cache      |

## Disk budget notes (this runtime)
- Total disk: 9.9 GB
- After dataset extraction + dependencies: ~7 GB used, ~2 GB free
- The full Zheng archive (2.5 GB) is **not** extracted locally — only CPSC.
  External zero-shot is run by re-downloading Zheng chunks from GitHub LFS
  and using `recombine_zheng_archive.sh`.

## Quick smoke test
```bash
export AWARD_RUN_MODE=smoke
export AWARD_BUDGET=smoke
jupyter nbconvert --to notebook --execute \
    ECG_CPSC2018/notebooks/ECG_CPSC2018_FINAL_EXTERNAL_VALIDATION_PUBLICATION_READY.ipynb \
    --output ECG_CPSC2018_FINAL_EXECUTED_SMOKE.ipynb \
    --ExecutePreprocessor.timeout=1200
```
