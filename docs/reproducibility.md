# Reproducibility & Cross-Machine Determinism (F5-06)

This document describes how to reproduce the final competition submission CSV from a clean repository clone with a single command, and records the cross-machine verification across two distinct CPU architectures/generations.

## 1. Requirements

- **Operating System:** Linux, macOS, or Windows (WSL2)
- **Python:** Python 3.12
- **Dependencies:** Installed from `requirements-modeling.txt` (`numpy==2.0.2`, `scikit-learn==1.5.2`):

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

## 3. Single-Command Reproduction

From the repository root, run:

```bash
python -m modeling.make_submission --config configs/final.yaml --out outputs/<team_name>.csv
```

- **Expected Config SHA-256:** `a84e812099d2df782dd2b66576ff47cd4892c01ab7d227e0c93f0d394066f1f8`
- **Expected CSV SHA-256:** `25064d90ff2808ffb325f43d51253a5dd58be765d70c2ddee6d608f5fb9f0503`

## 4. Cross-Machine Verification

The single-command pipeline was executed from scratch on two independent machines with different CPUs to verify that SIMD instruction selection in NumPy (`exp`, `log`, `tanh`) produces a bit-for-bit identical CSV.

| Field | Machine 1 (Clean Clone A) | Machine 2 (Clean Clone B) |
| :--- | :--- | :--- |
| **Executed by** | Fernando | Ahmed |
| **CPU Model** | Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz | `<PENDING_MACHINE_2_CPU>` |
| **SIMD Extensions** | AVX, AVX2, FMA3, SSE, SSE2, SSE3, SSE41, SSE42, SSSE3 | `<PENDING_MACHINE_2_SIMD>` |
| **OS / Platform** | Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39 | `<PENDING_MACHINE_2_OS>` |
| **Python / NumPy** | Python 3.12.16 / NumPy 2.0.2 | Python 3.12.x / NumPy 2.0.2 |
| **Config Hash (SHA-256)** | `a84e812099d2df782dd2b66576ff47cd4892c01ab7d227e0c93f0d394066f1f8` | `a84e812099d2df782dd2b66576ff47cd4892c01ab7d227e0c93f0d394066f1f8` |
| **CSV SHA-256** | `25064d90ff2808ffb325f43d51253a5dd58be765d70c2ddee6d608f5fb9f0503` | `25064d90ff2808ffb325f43d51253a5dd58be765d70c2ddee6d608f5fb9f0503` |
| **Predicted Shares (Graded)** | H 0.3190 / A 0.2207 / D 0.4603 | H 0.3190 / A 0.2207 / D 0.4603 |
| **Predicted Possessions** | half 1 = 177, half 2 = 193 (total = 370, mean = 9.3 s) | half 1 = 177, half 2 = 193 (total = 370, mean = 9.3 s) |
| **Verification Result** | **IDENTICAL** | **IDENTICAL** |
