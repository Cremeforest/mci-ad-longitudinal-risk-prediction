# Minimal Model Package Summary

The project now contains a minimal trained-model inference package.

## Core files

- src/mci_ad/model.py
- src/mci_ad/preprocessing.py
- src/mci_ad/inference.py
- demo/streamlit_app.py
- models/default_low_overlap_demo_mmse_faq.pt
- models/MODEL_CARD.md
- DATA.md
- requirements.txt

## Meaning

- model.py defines the Transformer model architecture.
- default_low_overlap_demo_mmse_faq.pt stores trained model weights.
- preprocessing.py converts clinical inputs into a [1, 8, 36] token.
- inference.py loads the model and produces 1-, 2-, 3-, and 5-year exploratory scores.
- demo/streamlit_app.py provides a local research demo interface.

## Verified reference case

Input: age=72, sex=female, education=16, MMSE=26, FAQ=6

Output: 1y=5.3%, 2y=32.3%, 3y=58.1%, 5y=92.6%

## Boundary

This is a retrospective research demonstration, not a clinically validated calculator.
