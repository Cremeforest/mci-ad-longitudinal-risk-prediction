"""
Preprocessing utilities for the MCI-to-AD research demo.

This file converts simple clinical inputs into the [1, 8, 36] token used by
the web-compatible Transformer model.

Research use only. Not clinically validated.
"""

from __future__ import annotations

import numpy as np
import torch


FULL_FEATURE_ORDER = [
    "age_at_visit",
    "sex_male",
    "PTEDUCAT",
    "MMSE",
    "ADAS13",
    "CDGLOBAL",
    "CDRSB",
    "FAQTOTAL",
]

DEMO_FEATURES = [
    "age_at_visit",
    "sex_male",
    "PTEDUCAT",
    "MMSE",
    "FAQTOTAL",
]

# Train-only preprocessing statistics recovered from the audited preprocessing source.
TRAIN_MEAN = {
    "age_at_visit": 73.99517059326172,
    "sex_male": 0.5853901505470276,
    "PTEDUCAT": 15.97974681854248,
    "MMSE": 28.02777862548828,
    "ADAS13": 13.713915824890137,
    "CDGLOBAL": 0.4172657430171966,
    "CDRSB": 1.256975769996643,
    "FAQTOTAL": 2.460960149765014,
}

TRAIN_STD = {
    "age_at_visit": 7.819310665130615,
    "sex_male": 0.4926545619964599,
    "PTEDUCAT": 2.965334177017212,
    "MMSE": 1.9962093830108645,
    "ADAS13": 6.631774425506592,
    "CDGLOBAL": 0.1974071711301803,
    "CDRSB": 1.0195504426956177,
    "FAQTOTAL": 3.694160461425781,
}


def z_score(value: float, feature: str) -> float:
    std = float(TRAIN_STD[feature])
    if std == 0:
        return 0.0
    return (float(value) - float(TRAIN_MEAN[feature])) / std


def build_single_visit_token(
    age_at_visit: float,
    sex_male: int,
    pteducat: float,
    mmse: float,
    faqtotal: float,
) -> torch.Tensor:
    """
    Build a [1, 8, 36] single-visit approximation token.

    The compact public demo uses five routine variables:
    age_at_visit, sex_male, PTEDUCAT, MMSE, FAQTOTAL.

    The full token still has 8 base feature slots because the audited model
    expects the original web-compatible token layout.

    Token layout:
    - channels 0:8   = scaled feature values
    - channels 8:16  = feature observed masks
    - channels 16:36 = timing/delta/slope auxiliary channels, set to zero here

    The current visit is placed in the final visit slot.
    """
    raw = {
        "age_at_visit": float(age_at_visit),
        "sex_male": float(sex_male),
        "PTEDUCAT": float(pteducat),
        "MMSE": float(mmse),
        "FAQTOTAL": float(faqtotal),
    }

    token = np.zeros((1, 8, 36), dtype=np.float32)
    final_visit = 7

    for feature_index, feature_name in enumerate(FULL_FEATURE_ORDER):
        if feature_name in raw:
            token[0, final_visit, feature_index] = z_score(raw[feature_name], feature_name)
            token[0, final_visit, 8 + feature_index] = 1.0
        else:
            token[0, final_visit, feature_index] = 0.0
            token[0, final_visit, 8 + feature_index] = 0.0

    return torch.tensor(token, dtype=torch.float32)
