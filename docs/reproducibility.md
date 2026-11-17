# Reproducibility & Cross-Machine Determinism (F5-06)

This document describes how to reproduce the final competition submission CSV from a clean repository clone with a single command, and records the cross-machine verification on two CPUs of different generations.

## 1. Requirements

- **Operating System:** Linux, macOS, or Windows (WSL2)
- **Python:** Python 3.12
- **Dependencies:** Installed from `requirements-modeling.txt` (`numpy==2.0.2`, `scikit-learn==1.5.2`, and `joblib==1.6.0` / `threadpoolctl==3.8.0`, which load the trained models in `models/**/all.joblib` and keep inference single-threaded):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-modeling.txt
```

## 2. Data Placement

Unpack the official competition match bundle into `./data` at the root of the fresh clone (as specified in `DATA.md`), or set the `MTAH_DATA_DIR` environment variable:

```text
data/
├── tactical/
│   ├── BAR-ATH_tactical.mp4
│   ├── BAR-ATH_tactical_atd.xml
│   ├── BAR-ATH_tactical_atd.txt
│   ├── BAR-ATH_tactical_homography.csv
│   └── BAR-ATH_tactical_smart_tagging.xml
└── broadcast/
    ├── BAR-ATH_broadcast.mp4
    ├── BAR-ATH_broadcast_atd.xml
    ├── BAR-ATH_broadcast_atd.txt
    └── BAR-ATH_broadcast_homography.csv
```

The command reads `MTAH_DATA_DIR` first and falls back to `./data`. When the data is in `./data`, make
sure the variable is not set (`unset MTAH_DATA_DIR`). If the bundle is already unpacked elsewhere,
symlinks avoid copying it; the processed files are still written inside the clone:

```bash
mkdir -p data && ln -s <bundle>/tactical data/tactical && ln -s <bundle>/broadcast data/broadcast
```

## 3. Single-Command Reproduction

From the repository root, run:

```bash
python -m modeling.make_submission --config configs/final.yaml --out outputs/<team_name>.csv
```

Our team name is `brasileirao`, so the file is `outputs/brasileirao.csv` (the file name does not change
the hash). The command runs the whole preprocessing (tactical ball fused with the broadcast ball) before
the model. Do not pass `--skip-preprocessing`: it reuses an existing
`data/processed_fused/master_features.parquet` instead of rebuilding it from the raw files.

- **Expected Config SHA-256:** `a84e812099d2df782dd2b66576ff47cd4892c01ab7d227e0c93f0d394066f1f8`
- **Expected CSV SHA-256:** `25064d90ff2808ffb325f43d51253a5dd58be765d70c2ddee6d608f5fb9f0503`

## 4. Cross-Machine Verification

The single-command pipeline was executed from scratch on two independent machines with different CPUs to verify that SIMD instruction selection in NumPy (`exp`, `log`, `tanh`) produces a bit-for-bit identical CSV.

| Field | Machine 1 (Clean Clone A) | Machine 2 (Clean Clone B) |
| :--- | :--- | :--- |
| **Executed by** | Fernando | Ahmed |
| **CPU Model** | Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz | 12th Gen Intel(R) Core(TM) i5-1235U |
| **SIMD Extensions** | AVX, AVX2, FMA3, SSE, SSE2, SSE3, SSE41, SSE42, SSSE3 | AVX, AVX2, FMA3, SSE, SSE2, SSE3, SSE41, SSE42, SSSE3 |
| **OS / Platform** | Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39 | Linux-7.0.0-43-generic-x86_64-with-glibc2.39 (Ubuntu 24.04.4 LTS) |
| **Python / NumPy** | Python 3.12.16 / NumPy 2.0.2 | Python 3.12.3 / NumPy 2.0.2 |
| **Config Hash (SHA-256)** | `a84e812099d2df782dd2b66576ff47cd4892c01ab7d227e0c93f0d394066f1f8` | `a84e812099d2df782dd2b66576ff47cd4892c01ab7d227e0c93f0d394066f1f8` |
| **CSV SHA-256** | `25064d90ff2808ffb325f43d51253a5dd58be765d70c2ddee6d608f5fb9f0503` | `25064d90ff2808ffb325f43d51253a5dd58be765d70c2ddee6d608f5fb9f0503` |
| **Predicted Shares (Graded)** | H 0.3190 / A 0.2207 / D 0.4603 | H 0.3190 / A 0.2207 / D 0.4603 |
| **Predicted Possessions** | half 1 = 177, half 2 = 193 (total = 370, mean = 9.3 s) | half 1 = 177, half 2 = 193 (total = 370, mean = 9.3 s) |
| **Verification Result** | **IDENTICAL** | **IDENTICAL** |

Both runs used a clean clone of `main` at `9d72bcd` (same method code as the tag `submission-v1`), a
fresh venv with the pinned versions above, and the full pipeline from the raw data. The two CPUs differ in
generation (Comet Lake vs Alder Lake), but neither has AVX-512, so both runs used the AVX2 kernels of
numpy. A run on an AVX-512 or ARM CPU would test a different SIMD path.

## 5. Why the output is deterministic

- **Single thread:** `modeling/make_submission.py` sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and
  `MKL_NUM_THREADS` to 1 before numpy is imported, and the gradient-boosting models run inside
  `threadpool_limits(1)`.
- **Stable order:** sorts use `kind="mergesort"`, and ties go to the lowest index.
- **No BLAS in inference:** linear scores are written as `(X * w).sum(axis=1)` in `float64`, never `@`.
- **Deterministic preprocessing:** `python -m preprocessing.run_all --check-determinism` runs it twice
  and compares the SHA-256 of the master table.

## 6. If the hash differs

1. Check the environment first: Python 3.12, `pip list` against the pins, `MTAH_DATA_DIR` not set,
   and no `--skip-preprocessing`.
2. Find the stage that diverges. Compare `sha256sum data/processed_fused/master_features.parquet`
   across machines. If the master tables match, the difference is in inference. If they differ, it is
   in preprocessing.
3. A fix after the code freeze needs approval from 2 people (G1). Then repeat both runs.
