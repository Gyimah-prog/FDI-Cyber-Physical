# FDI Cyber Physical

Detection of false data injection (FDI) attacks on railway SCADA data, with a focus on track geometry measurements.

## Overview

Rail SCADA systems report track geometry readings such as gage, crosslevel, warp, profile and alignment. An attacker who alters these readings can hide real defects or trigger false maintenance alarms. This repository holds the code used to inject synthetic attacks into track geometry records and to test how well different detectors find them.

The main approach pairs a variational autoencoder (VAE) trained on clean data with statistical scoring and a median absolute deviation (MAD) threshold. Isolation Forest and PCA serve as baselines.

## Repository layout

| Folder | Contents |
|---|---|
| `fdi1/` | First study. Feature selection, attack injection and VAE based detection |
| `fdi2026/` | Attack aware framework. Statistical, temporal and replay attacks, feature susceptibility analysis and bootstrap confidence intervals |
| `utc_report/notebook/` | Notebook that produces the tables and figures for the REPS UTC final report |
| `utc_report/scripts/` | Standalone pipelines for the hybrid VAE search and raw value attack injection |

## Data

The track geometry data is not included in this repository. The notebooks expect a file named `WRM_F.csv` and were written for Google Colab with Google Drive mounted. Update the data path at the top of each notebook or pass it to the scripts as an argument.

```
python utc_report/scripts/fdi_route1_raw_injection.py path/to/WRM_F.csv output_dir
```

## Requirements

Python 3.10 or later. Install the packages with

```
pip install -r requirements.txt
```

## Author

Benjamin Gyimah Boateng, Department of Civil and Environmental Engineering, University of Maryland
