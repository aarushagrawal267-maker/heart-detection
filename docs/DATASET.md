# Dataset: MIT-BIH Arrhythmia Database

| | |
|---|---|
| Name | MIT-BIH Arrhythmia Database, version 1.0.0 |
| Host | PhysioNet: <https://physionet.org/content/mitdb/1.0.0/> |
| DOI | <https://doi.org/10.13026/C2F305> |
| Licence | Open Data Commons Attribution License v1.0 (ODC-By 1.0) |
| Content | 48 half-hour two-channel ambulatory ECG recordings from 47 subjects |
| Sampling | 360 Hz per channel, 11-bit resolution over a 10 mV range |
| Annotations | ≈110,000 beat labels reviewed by two or more cardiologists (`.atr`) |

## Required citations

Anyone using this data must cite:

- Moody GB, Mark RG. *The impact of the MIT-BIH Arrhythmia Database.* IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001). (PMID: 11446209)
- Goldberger A, Amaral L, Glass L, Hausdorff J, Ivanov PC, Mark R, Mietus JE, Moody GB, Peng CK, Stanley HE. *PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals.* Circulation 101(23):e215–e220 (2000).

## Acquisition procedure (implemented in `backend/app/data/acquire.py`)

1. Download `SHA256SUMS.txt` from `https://physionet.org/files/mitdb/1.0.0/`.
2. Download the official archive `mit-bih-arrhythmia-database-1.0.0.zip` (≈77 MB). Only the 144 top-level record files (`.dat`, `.hea`, `.atr` for the 48 records in `RECORDS`) are extracted; auxiliary folders are ignored. If the archive fails, each file is fetched individually.
3. Verify the SHA-256 of every required file against PhysioNet's published checksums. If any file is missing or mismatched, the step fails loudly.
4. Mark the raw files read-only. The pipeline never writes to `data/raw/`.
5. Write `data/raw/mitdb/1.0.0/provenance.json` (source URL, DOI, version, licence, citations, download time and method, all checksums).

Run it with `setup.ps1`, the Dataset Explorer page, `POST /api/dataset/download`, or:

```bash
cd backend && python -m app.cli download
```

The dataset is **not** committed to this repository.

## File formats

| Extension | Content |
|---|---|
| `.hea` | Header: number of signals, sampling frequency, length, lead names, gain, ADC resolution and zero, comments (age, sex, medications) |
| `.dat` | Signal samples in WFDB format 212 (two 12-bit samples packed in 3 bytes) |
| `.atr` | Reference annotations: sample index, beat or non-beat symbol, rhythm (`aux`) strings |

Files are read with the `wfdb` Python package (`rdheader`, `rdrecord(physical=True)`, `rdann`). Signals are converted to millivolts using the header gain and baseline.

## Leads

The analysis lead is **MLII**, identified by name in the header:

| Leads | Records |
|---|---|
| MLII + V1 | 40 records |
| MLII + V2 / V4 / V5 | 5 records |
| V5 + MLII (swapped) | 114 |
| V5 + V2 (no MLII) | 102, 104 (paced, excluded) |

## Subjects

The MIT-BIH directory states that **records 201 and 202 come from the same subject**. No other shared subjects are documented. All other recordings are treated as separate subjects, as the database documentation describes, but that cannot be verified independently. Subject ids used here: `S201_202` for records 201 and 202, `S<record>` otherwise (47 subjects).

## Verified counts (computed from the downloaded files)

| Quantity | Value |
|---|---|
| Beat annotations (AAMI-mappable symbols) | 109,494 |
| Segments after boundary exclusion | 109,444 (50 beats excluded at record boundaries) |
| Class totals (all 48 records) | N 90,587 · S 2,781 · V 7,235 · F 802 · Q 8,039 |
| Training pool (22 subjects, 23 records) | N 47,903 · S 999 · V 3,807 · F 415 · Q 8 |
| Held-out test (21 subjects) | N 42,178 · S 1,782 · V 3,201 · F 387 · Q 7 |
| Excluded paced records (102, 104, 107, 217) | N 506 · V 227 · Q 8,024 |

The S beats in the test partition are heavily concentrated: record 232 alone contributes 1,382 of the 1,782.

See `configs/aami_mapping.json` for the symbol mapping and `configs/splits.json` for the partitions.
