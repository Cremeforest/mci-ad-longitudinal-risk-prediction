# MCI-to-AD Longitudinal Risk Prediction

A leakage-aware longitudinal machine learning project for multi-horizon prediction of conversion from mild cognitive impairment (MCI) to Alzheimer's disease dementia using routine clinical variables.

This repository provides a clean research-demo package: model architecture, preprocessing, inference code, a local Streamlit demo, selected result summaries, and documentation.

## Important notice

This project is a retrospective research demonstration.

It is not a clinically validated calculator and must not be used for diagnosis, treatment, triage, prognosis communication, or patient-level medical decision-making.

## What this project does

- Builds a leakage-aware longitudinal MCI-to-AD prediction pipeline.
- Uses landmark-style prediction with 1-, 2-, 3-, and 5-year horizons.
- Uses ADNI for internal retrospective evaluation.
- Uses NACC only as a preliminary transportability dry-run, not final external validation.
- Provides a compact low-overlap demo model using routine clinical variables.

## Default demo model

The public-facing demo model uses five routine clinical variables:

- age at visit
- sex
- years of education
- MMSE
- FAQ total score

The default demo model does not use ADAS13, CDR, CDR-SB, APOE, PET, CSF, MRI, diagnosis labels, or future outcome information.

## Model package

```text
src/mci_ad/model.py          # model architecture
src/mci_ad/preprocessing.py  # input preprocessing and token construction
src/mci_ad/inference.py      # trained-model inference
demo/streamlit_app.py        # local Streamlit research demo
models/default_low_overlap_demo_mmse_faq.pt  # trained demo checkpoint
models/MODEL_CARD.md         # model card and use boundary
```

## Quick inference example

```bash
python -m src.mci_ad.inference --age 72 --sex female --education 16 --mmse 26 --faq 6
```

Expected reference output:

```text
1y: 5.3%
2y: 32.3%
3y: 58.1%
5y: 92.6%
```

## Run local Streamlit demo

```bash
pip install -r requirements.txt
python -m streamlit run demo/streamlit_app.py
```

## Key results

### Internal ADNI evaluation

| Model | 1-year AUROC | 2-year AUROC | 3-year AUROC | 5-year AUROC |
|---|---:|---:|---:|---:|
| Full dynamic model | 0.911 | 0.902 | 0.921 | 0.939 |
| Reduced-overlap model | 0.891 | 0.880 | 0.891 | 0.922 |
| Demographics-only model | 0.630 | 0.643 | 0.665 | 0.651 |

### Preliminary NACC transportability dry-run

| Horizon | AUROC |
|---|---:|
| 1 year | 0.696 |
| 2 years | 0.701 |
| 3 years | 0.709 |
| 5 years | 0.728 |

The NACC analysis is a preliminary external dry-run / transportability analysis and should not be interpreted as final external validation.

## Repository structure

```text
README.md
DATA.md
requirements.txt
src/mci_ad/
demo/
models/
docs/
results/
```

## Data availability

Raw ADNI and NACC data are not included because they are access-controlled research datasets. See DATA.md for details.

## Reproducibility boundary

This repository supports reproducibility of trained-model inference when the demo checkpoint is included.

Full training reproduction requires authorized access to the original ADNI/NACC data and reconstruction of the cohort, landmark, preprocessing, and training pipeline.

## Limitations

- Retrospective research pipeline.
- ADNI internal evaluation may not represent routine clinical populations.
- NACC analysis is preliminary and requires further mapping verification.
- External calibration and prospective validation have not been completed.
- The demo is not a clinical calculator.

