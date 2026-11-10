# External data: IDSSE

The IDSSE tracking data is provided by the PySport Foundation under the Creative Commons
Attribution 4.0 International license (CC BY 4.0). The converter downloads the original XML files
from the Hugging Face mirror at the immutable revision
[`a715a38dfbaf5f58e431727c2b78d174101a703c`](https://huggingface.co/datasets/pysport/idsse-data/tree/a715a38dfbaf5f58e431727c2b78d174101a703c).

The expected filenames and source file checksums are listed below. Sizes and SHA-256 values are
filled from the downloaded source files.

| Match | COM | Metadata file | Bytes | SHA-256 | Tracking file | Bytes | SHA-256 |
|---|---|---|---:|---|---|---:|---|
| J03WMX | 000001 | `DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-J03WMX.xml` | 11930 | `a884728647a564cf8b20e5223a1d04a5da6f6725e1e506c8651505327c197ba9` | `DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-J03WMX.xml` | 418524833 | `d3dd92e509421a6466acb0a083c009035f1751bc5fd9ef42ffa8f20d01cf7d49` |
| J03WN1 | 000001 | `DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-J03WN1.xml` | 11722 | `5490286436c86e6f80a538d2b1b379d0efd016e0defcaf594bc59d9ef5079f4c` | `DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-J03WN1.xml` | 391722803 | `56ef948961a18e150dba0b7b16869d0008ff66f3050a0c355d31fa5b8ec2fcec` |
| J03WOH | 000002 | `DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WOH.xml` | 11745 | `f23b3ee13b988b1ff9a0bce438c416dac47e71355bbc7ee1dab21aaf3e33c18c` | `DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WOH.xml` | 348945715 | `ec4dcf30200aad694dea263d5289485fcd0ca42badddaeb7c81f5cd5a4229d4f` |
| J03WOY | 000002 | `DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WOY.xml` | 11952 | `4c497fac683eee8292901c18ca02c0be1d9609d0b9d4bd8aa092c61f9e0b19e9` | `DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WOY.xml` | 362567859 | `a228ef0e676d009cefff1580fcb721b53f598454eb120ec347e0a628a3a4646b` |
| J03WPY | 000002 | `DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WPY.xml` | 11922 | `566dd5e4c86e124ab90fcdba51100881f0436e22afa71832945a451a2d95a420` | `DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WPY.xml` | 371688499 | `845202c29ad93d696653436da426a7cdaefb848298e07b7343de6529581671e1` |
| J03WQQ | 000002 | `DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WQQ.xml` | 11932 | `e976aa0c5dba862cfa898ea57ec226efce9d1bc04b23b8d775908db86aa6fa41` | `DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WQQ.xml` | 356219423 | `cbefcb0bd7f39f3b94ee69a7bc97aac0f936425062c1cc1b144733df36804d6f` |
| J03WR9 | 000002 | `DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WR9.xml` | 11823 | `429b584e89df3e3e8b81f4bd0d004b38bd0b4298bb390b8fb43efccb60963e01` | `DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WR9.xml` | 372934140 | `70b63069189019066fb7e3d9db891122edfe787f3a43ba522ebe45ed89ea0269` |

## Array version

Converted arrays (`$MTAH_EXT_DIR/arrays/*.npz`), external masters and any shared package must be
generated with the code on `main` from [#109](https://github.com/AntonioEscobarTorres/more-than-a-hack-2027-brasileirao/pull/109) (2026-11-10) onward. Earlier conversions loaded the
tracking in kloppy's normalised coordinate system and normalised it a second time (every position
collapsed into x_n in [0.500, 0.510]), and stored speeds in km/h instead of m/s. Earlier
degradations stacked duplicate tracks and did not enforce the target tolerances; the order of the
random draws changed too, so the same seed now gives a different degraded match. Delete any older
files and regenerate:

```bash
rm -rf "$MTAH_EXT_DIR/arrays" "$MTAH_EXT_DIR/processed"
python -m modeling.external.idsse --matches all --delete-raw
```

The per-frame labels did not change, so statistics computed from labels only (F2-03) remain valid.

Citation:

> Bassek, M., Rein, R., Weber, H., Memmert, D. (2025). *An integrated dataset of spatiotemporal
> and event data in elite soccer.* Scientific Data 12, 195. https://doi.org/10.1038/s41597-025-04505-y
