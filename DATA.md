# Data Availability

This repository does not include raw ADNI or NACC data.

The original project used access-controlled research datasets.
Raw participant-level files must not be redistributed through this repository.

## Included

- model architecture code
- preprocessing and inference code
- selected result tables
- documentation and model card
- optional trained demo checkpoint

## Not included

- data/
- raw ADNI CSV files
- raw NACC files
- NACC_investigator.csv
- intermediate .npy or .npz feature tensors
- full intermediate training checkpoints
- invalid or quarantined Step48 outputs

## Reproducibility boundary

This repository supports reproducibility of trained-model inference when the demo checkpoint is included.
Full training reproduction requires authorized access to the original ADNI/NACC data and reconstruction of the cohort, landmark, preprocessing, and training pipeline.
