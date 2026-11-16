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

## External masters (F4-04)

The standard way to get the external files is to regenerate them locally with the code on `main`:

```bash
export MTAH_EXT_DIR=~/mtah-external
python -m modeling.external.idsse --matches all --delete-raw
python -m modeling.external.build --matches all --degrade none --seed 0
python -m modeling.external.build --matches all --degrade default --seed 0
python -m modeling.external.build --matches all --degrade harder --seed 0
python -m modeling.external.build --matches all --degrade easier --seed 0
python -m modeling.external.build --matches all --degrade hidelo --seed 0
python -m modeling.external.build --matches all --degrade hidehi --seed 0
python -m modeling.external.build --verify
```

`--verify` compares every local file with the SHA-256 values below and prints `IDENTICAL` or `DIFFERENT`
per file. A last-bit difference between CPUs is possible and irrelevant for the calibration, but tell the
team in the group if you see one. The degradation settings and how they were chosen are in
[D-016](decisions.md#d-016--degradação-do-idsse-ajustada-pelo-realismo-f4-04); per-match statistics are in
[`reports/external_masters.md`](../reports/external_masters.md). Masters `__none` are for diagnostics only
and are never used to calibrate or select (D-009).

Each degraded build takes 3-4 minutes per match. **Plan B**, if the download or the build is late for
someone: the F4 owner zips `$MTAH_EXT_DIR/arrays` and `$MTAH_EXT_DIR/processed` together with this
attribution notice and shares the archive in the team folder (CC BY 4.0 allows it). These files never go
into the repository.

### Expected SHA-256

Generated on 2026-11-15 with the F4-04 code (#74), seed 0; `harder`/`easier` regenerated after the review (D-017).

| File | SHA-256 |
| --- | --- |
| `J03WMX.npz` | `c1532f4fa72f94bd1930aa1b8bbf116afdf66d34ea84fbc0f729732d28effb93` |
| `J03WN1.npz` | `8ca279ffc4d5525014e9d174fbc51314fce076cbaa3dd0020cf978924df98d6f` |
| `J03WOH.npz` | `c01596c0d17b9cca02f6a8ac52c2a62f21a664a8083f8dc8a8f7212a9ca0dbc6` |
| `J03WOY.npz` | `a157d7a5722f877c4582920ff6fbeba4455915c8515b02e99f11aed77ce805d4` |
| `J03WPY.npz` | `6e2414f0bb863c6a4d851d0ec3a0993f66551dd4656384d595d76c2ce1249a8d` |
| `J03WQQ.npz` | `12c9e7f1004a375c60acf1306f24ff6e980cf93b23fd05d12d586502b119cf16` |
| `J03WR9.npz` | `f76e02fb6c52e825bc219d9e6000fa135f8b7c928cd0ad488ef1de2780d3fbed` |
| `J03WMX__none.parquet` | `577c83735c17f55a35cda9c3ea3948cf1783411576b68784e12aba9cd6100c82` |
| `J03WMX__none_labels.npy` | `cbbac0011d2b7021600617149ed8907402e8ee14a6a7b5229aa142222cc2de82` |
| `J03WN1__none.parquet` | `44415a2dcc3a304f398190da6eb036ca7c6a789221431b94deeaa343f9449ca3` |
| `J03WN1__none_labels.npy` | `25000b6d2f33b8b4e37275025c3608454b7c3c009586b3c1a0ca253a236f1686` |
| `J03WOH__none.parquet` | `3dc57d6707e70b98cbc982e2bd8dd3a63d69eab6ebd5fa2c9203e6c5e48d64c7` |
| `J03WOH__none_labels.npy` | `b0c547527fbb1d65074a84fe7d26da806e1ad5ad66ea80e5a8dc771265aaf8d2` |
| `J03WOY__none.parquet` | `da929b1e4fda1f1762a425101c9cee8efa779b1b50d37ac2e26bf117b82b314a` |
| `J03WOY__none_labels.npy` | `660199b59c77d50d86aa2f36c645241ef1b2598d0fa9aa6e83b49f0dd10155a0` |
| `J03WPY__none.parquet` | `f32d783639b1eca91c571d863eaedd8919f884c3cdb00e28b49eddb0065cbeb9` |
| `J03WPY__none_labels.npy` | `704c77b3d8de2ac63c5dae4b50b63311b67a3615147203f87e7e49edfdf0edc0` |
| `J03WQQ__none.parquet` | `69cfb3b06b1f7bdc4d12c4cda8435124c47668dc54130e94602cd081144172b3` |
| `J03WQQ__none_labels.npy` | `eea603073d79ce275a40bcc4e173972af098b1eded3d68d1f4311c43bb7aacdc` |
| `J03WR9__none.parquet` | `ab724d3dbfc015f3e8d87d03e8a85b70b175b20b872d99560c9555c05a3906bd` |
| `J03WR9__none_labels.npy` | `c2c39be67571923e9aeba01260b2edc10482579fc3b3bd0ebeafb51f5b2b1fa4` |
| `J03WMX__default.parquet` | `7575b41622eed11b96c6c57e9acd91b7a10791efda45d6ab8f1743911df7f6ec` |
| `J03WMX__default_labels.npy` | `cbbac0011d2b7021600617149ed8907402e8ee14a6a7b5229aa142222cc2de82` |
| `J03WN1__default.parquet` | `02d05a75ed5d71d20ae94fa665bb586e0e77a2a36a795bab1f41a437bed1056b` |
| `J03WN1__default_labels.npy` | `25000b6d2f33b8b4e37275025c3608454b7c3c009586b3c1a0ca253a236f1686` |
| `J03WOH__default.parquet` | `0ba3c8569477da2e3fbac07406d9b08236fb251c4465b43b814899ea004e4c56` |
| `J03WOH__default_labels.npy` | `b0c547527fbb1d65074a84fe7d26da806e1ad5ad66ea80e5a8dc771265aaf8d2` |
| `J03WOY__default.parquet` | `ec0f9df78871f85aa8af00a172beb2f0948ae72b60a30714e25e7ff2f038be83` |
| `J03WOY__default_labels.npy` | `660199b59c77d50d86aa2f36c645241ef1b2598d0fa9aa6e83b49f0dd10155a0` |
| `J03WPY__default.parquet` | `eda08690a4a17e72b7dd270f88b7bcdeb97dfd9d027c3eef4c3ea953eab1289c` |
| `J03WPY__default_labels.npy` | `704c77b3d8de2ac63c5dae4b50b63311b67a3615147203f87e7e49edfdf0edc0` |
| `J03WQQ__default.parquet` | `1d4cf7004f8d8bb3e5cdce5b0d2a8f0eadd861e625274dced04b8f1f6854bbcb` |
| `J03WQQ__default_labels.npy` | `eea603073d79ce275a40bcc4e173972af098b1eded3d68d1f4311c43bb7aacdc` |
| `J03WR9__default.parquet` | `d45cfe4eda1301dc8d1ae96341c0e2033384d0f8463edca3c22194484fd4637e` |
| `J03WR9__default_labels.npy` | `c2c39be67571923e9aeba01260b2edc10482579fc3b3bd0ebeafb51f5b2b1fa4` |
| `J03WMX__harder.parquet` | `a0215b6d0f62cc138c39236eb496c87820e511d8b22e9a591863bacbbae7cce4` |
| `J03WMX__harder_labels.npy` | `cbbac0011d2b7021600617149ed8907402e8ee14a6a7b5229aa142222cc2de82` |
| `J03WN1__harder.parquet` | `6fbc6250ad7c6e345130d9144712ebfd963f34b1af761f23d5a9dbc5da99267b` |
| `J03WN1__harder_labels.npy` | `25000b6d2f33b8b4e37275025c3608454b7c3c009586b3c1a0ca253a236f1686` |
| `J03WOH__harder.parquet` | `337efe4262643a8160fd0c212fecd6c08e67fb20dd5ea6bf5ab81f94337d8fc8` |
| `J03WOH__harder_labels.npy` | `b0c547527fbb1d65074a84fe7d26da806e1ad5ad66ea80e5a8dc771265aaf8d2` |
| `J03WOY__harder.parquet` | `e338e9a9de2b1ba38de8ceedd5327b40fff6c66aab08f46270c6d6f01f65aa91` |
| `J03WOY__harder_labels.npy` | `660199b59c77d50d86aa2f36c645241ef1b2598d0fa9aa6e83b49f0dd10155a0` |
| `J03WPY__harder.parquet` | `30b524c3bfc6f5e741f89a979f573d1159bf724d0d9f7e18160e86ff2efef547` |
| `J03WPY__harder_labels.npy` | `704c77b3d8de2ac63c5dae4b50b63311b67a3615147203f87e7e49edfdf0edc0` |
| `J03WQQ__harder.parquet` | `2708fd2cfd195a44b9335e4869651128a8919b52040f892bfcfa33328e191070` |
| `J03WQQ__harder_labels.npy` | `eea603073d79ce275a40bcc4e173972af098b1eded3d68d1f4311c43bb7aacdc` |
| `J03WR9__harder.parquet` | `91a6153eb01ec3ce90bcdda8f03bfc2f2b7e656a9df047b6453767bd9ecf38f4` |
| `J03WR9__harder_labels.npy` | `c2c39be67571923e9aeba01260b2edc10482579fc3b3bd0ebeafb51f5b2b1fa4` |
| `J03WMX__easier.parquet` | `9f2286a826adc8e2bd1d1da60bc1a8642675c929bac0e5a487c7f1ab7c535431` |
| `J03WMX__easier_labels.npy` | `cbbac0011d2b7021600617149ed8907402e8ee14a6a7b5229aa142222cc2de82` |
| `J03WN1__easier.parquet` | `655f6a6101f8ec5e1675036214bdd560d824713d2ef1abcf2d9a07162933204f` |
| `J03WN1__easier_labels.npy` | `25000b6d2f33b8b4e37275025c3608454b7c3c009586b3c1a0ca253a236f1686` |
| `J03WOH__easier.parquet` | `39ec0b2268da651687f5bc32807fc8d22c538f2c3737982785817f229a68495e` |
| `J03WOH__easier_labels.npy` | `b0c547527fbb1d65074a84fe7d26da806e1ad5ad66ea80e5a8dc771265aaf8d2` |
| `J03WOY__easier.parquet` | `cd4d8cc37ec02128d01034bb3936bf903b6c787a50b30885c0d947aaff3f7f54` |
| `J03WOY__easier_labels.npy` | `660199b59c77d50d86aa2f36c645241ef1b2598d0fa9aa6e83b49f0dd10155a0` |
| `J03WPY__easier.parquet` | `53faa66b916ff579b6068139eabc8ea95a79e753a9e679bde1fb76c90be788ec` |
| `J03WPY__easier_labels.npy` | `704c77b3d8de2ac63c5dae4b50b63311b67a3615147203f87e7e49edfdf0edc0` |
| `J03WQQ__easier.parquet` | `47cac8d7240b447f08d9cd8fc552c86756c2c6d26330169dfd395150bc54812f` |
| `J03WQQ__easier_labels.npy` | `eea603073d79ce275a40bcc4e173972af098b1eded3d68d1f4311c43bb7aacdc` |
| `J03WR9__easier.parquet` | `d237ae730fabaa0a81a70b9e168945382dfcfe7556cc59fb9dde81ac6c78b77e` |
| `J03WR9__easier_labels.npy` | `c2c39be67571923e9aeba01260b2edc10482579fc3b3bd0ebeafb51f5b2b1fa4` |
| `J03WMX__hidelo.parquet` | `1a4f05a39d983cba236d8ddf735c597ee8675e13833858fc2073773ff11df351` |
| `J03WMX__hidelo_labels.npy` | `cbbac0011d2b7021600617149ed8907402e8ee14a6a7b5229aa142222cc2de82` |
| `J03WN1__hidelo.parquet` | `db8de652f74d820a81b218f3365c7bdf5fa25543739146108c0881ff8b11f1b2` |
| `J03WN1__hidelo_labels.npy` | `25000b6d2f33b8b4e37275025c3608454b7c3c009586b3c1a0ca253a236f1686` |
| `J03WOH__hidelo.parquet` | `8c427793912ac4190c9282c72a4b6852841608d21a10895b8debcedc8472e3f7` |
| `J03WOH__hidelo_labels.npy` | `b0c547527fbb1d65074a84fe7d26da806e1ad5ad66ea80e5a8dc771265aaf8d2` |
| `J03WOY__hidelo.parquet` | `93a72e5aba241bab489acf10f4dc868dec2bf35d484bd390e6832ea8a08dae2f` |
| `J03WOY__hidelo_labels.npy` | `660199b59c77d50d86aa2f36c645241ef1b2598d0fa9aa6e83b49f0dd10155a0` |
| `J03WPY__hidelo.parquet` | `05f7a253c42b222eeb7af4104afdd0524476c4209f8ee549902005735ec3ea99` |
| `J03WPY__hidelo_labels.npy` | `704c77b3d8de2ac63c5dae4b50b63311b67a3615147203f87e7e49edfdf0edc0` |
| `J03WQQ__hidelo.parquet` | `5bcb831e9e6e255a9055a2d2dbf643ed223968a26e08e67777007f4f61323b9f` |
| `J03WQQ__hidelo_labels.npy` | `eea603073d79ce275a40bcc4e173972af098b1eded3d68d1f4311c43bb7aacdc` |
| `J03WR9__hidelo.parquet` | `1f199686022c3f9860c3e15b71b29c500209098454834eefb5f5a52012a9a5d8` |
| `J03WR9__hidelo_labels.npy` | `c2c39be67571923e9aeba01260b2edc10482579fc3b3bd0ebeafb51f5b2b1fa4` |
| `J03WMX__hidehi.parquet` | `143e76f7b18a2bf1d8e2acd11f5f59b2923b5e71c2908b48aef47be4245a720b` |
| `J03WMX__hidehi_labels.npy` | `cbbac0011d2b7021600617149ed8907402e8ee14a6a7b5229aa142222cc2de82` |
| `J03WN1__hidehi.parquet` | `9b41a3fd02645025c78d3200d2750d52c2fb4494961c6f6564d5f2ee67ac9811` |
| `J03WN1__hidehi_labels.npy` | `25000b6d2f33b8b4e37275025c3608454b7c3c009586b3c1a0ca253a236f1686` |
| `J03WOH__hidehi.parquet` | `9f3ad547aa6cdca6e63bf97df9bec3e0d00337659122124bda4446f4bfa5a8a6` |
| `J03WOH__hidehi_labels.npy` | `b0c547527fbb1d65074a84fe7d26da806e1ad5ad66ea80e5a8dc771265aaf8d2` |
| `J03WOY__hidehi.parquet` | `bcd186160d4d903d6fd926732ea418024adfdc9f747ef3aa6cf719d831f90a9f` |
| `J03WOY__hidehi_labels.npy` | `660199b59c77d50d86aa2f36c645241ef1b2598d0fa9aa6e83b49f0dd10155a0` |
| `J03WPY__hidehi.parquet` | `b4ec76fdd3997c82b25eae00031cbbb5813f539bde4b8b1deec39e9ae3145fe2` |
| `J03WPY__hidehi_labels.npy` | `704c77b3d8de2ac63c5dae4b50b63311b67a3615147203f87e7e49edfdf0edc0` |
| `J03WQQ__hidehi.parquet` | `7a1036894b0596807351cfdf3597515bb72979eed4e8124d40e98a8a222b6005` |
| `J03WQQ__hidehi_labels.npy` | `eea603073d79ce275a40bcc4e173972af098b1eded3d68d1f4311c43bb7aacdc` |
| `J03WR9__hidehi.parquet` | `cd1f7816a437ba67ecb8ba06d3d86b79dde45dc696e429f8fedcc887e8f96a47` |
| `J03WR9__hidehi_labels.npy` | `c2c39be67571923e9aeba01260b2edc10482579fc3b3bd0ebeafb51f5b2b1fa4` |

Citation:

> Bassek, M., Rein, R., Weber, H., Memmert, D. (2025). *An integrated dataset of spatiotemporal
> and event data in elite soccer.* Scientific Data 12, 195. https://doi.org/10.1038/s41597-025-04505-y
