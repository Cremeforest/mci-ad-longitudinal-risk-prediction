# Model Card: Default Low-Overlap MCI-to-AD Research Demo Model

## Model file

models/default_low_overlap_demo_mmse_faq.pt

## What this file is

This is a PyTorch checkpoint for the trained default research-demo model.
It contains trained model parameters and metadata.
It is not a text file and is not intended to be opened directly.

## Model architecture

The model architecture is defined in src/mci_ad/model.py.
The default architecture is a web-compatible dynamic Transformer model.

## Inference code

The inference entry point is src/mci_ad/inference.py.

Example:

python -m src.mci_ad.inference --age 72 --sex female --education 16 --mmse 26 --faq 6

Expected reference output:

1y: 5.3%
2y: 32.3%
3y: 58.1%
5y: 92.6%

## Inputs used by the default demo model

- age at visit
- sex
- years of education
- MMSE
- FAQ total score

## Inputs not used by the default demo model

- ADAS13
- global CDR
- CDR-SB
- APOE
- PET
- CSF
- MRI
- diagnosis labels
- future outcome information

## Outputs

The model outputs four exploratory scores for 1-, 2-, 3-, and 5-year MCI-to-AD conversion risk.

These are retrospective research model scores, not clinically validated probabilities.

## Validation status

- ADNI: internal retrospective evaluation
- NACC: preliminary transportability dry-run only
- Prospective validation: not performed
- Clinical deployment validation: not performed

## Intended use

This model is intended only for research demonstration and reproducibility of trained-model inference.
It must not be used for diagnosis, treatment, triage, prognosis communication, or patient-level medical decision-making.
