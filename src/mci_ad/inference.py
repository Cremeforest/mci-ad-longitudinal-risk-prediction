"""
Inference utilities for the trained MCI-to-AD research demo model.

This file shows how the public demo model is reconstructed:

1. import model architecture from src/mci_ad/model.py
2. load trained weights from default_low_overlap_demo_mmse_faq.pt
3. preprocess clinical inputs into [1, 8, 36]
4. output 1-, 2-, 3-, and 5-year exploratory risk scores

Research use only. Not clinically validated.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.mci_ad.model import WebCompatibleDynamicTransformer
from src.mci_ad.preprocessing import build_single_visit_token


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CHECKPOINT = ROOT / "models" / "default_low_overlap_demo_mmse_faq.pt"


def load_demo_model(checkpoint_path: str | Path = DEFAULT_CHECKPOINT) -> WebCompatibleDynamicTransformer:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise TypeError("Unsupported checkpoint format.")

    model = WebCompatibleDynamicTransformer(
        input_dim=36,
        d_model=64,
        max_visits=8,
        n_layers=2,
        n_heads=4,
        dim_feedforward=128,
        head_hidden_1=128,
        head_hidden_2=96,
        out_dim=4,
        dropout=0.0,
    )

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


@torch.no_grad()
def predict_risk_scores(
    age_at_visit: float,
    sex_male: int,
    pteducat: float,
    mmse: float,
    faqtotal: float,
    monotone_display: bool = True,
) -> dict[str, float]:
    model = load_demo_model()
    token = build_single_visit_token(
        age_at_visit=age_at_visit,
        sex_male=sex_male,
        pteducat=pteducat,
        mmse=mmse,
        faqtotal=faqtotal,
    )

    logits = model(token)
    probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)[:4]

    if monotone_display:
        probs = np.maximum.accumulate(probs)

    return {
        "1y": float(probs[0] * 100.0),
        "2y": float(probs[1] * 100.0),
        "3y": float(probs[2] * 100.0),
        "5y": float(probs[3] * 100.0),
    }


def parse_sex(value: str) -> int:
    value = value.strip().lower()
    if value in {"male", "m", "1"}:
        return 1
    if value in {"female", "f", "0"}:
        return 0
    raise ValueError("sex must be one of: female, male, f, m, 0, 1")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MCI-to-AD research demo inference.")
    parser.add_argument("--age", type=float, default=72.0)
    parser.add_argument("--sex", type=str, default="female")
    parser.add_argument("--education", type=float, default=16.0)
    parser.add_argument("--mmse", type=float, default=26.0)
    parser.add_argument("--faq", type=float, default=6.0)
    args = parser.parse_args()

    sex_male = parse_sex(args.sex)

    scores = predict_risk_scores(
        age_at_visit=args.age,
        sex_male=sex_male,
        pteducat=args.education,
        mmse=args.mmse,
        faqtotal=args.faq,
    )

    print("=" * 80)
    print("MCI-to-AD research demo inference")
    print("=" * 80)
    print(f"Input: age={args.age}, sex={args.sex}, education={args.education}, MMSE={args.mmse}, FAQ={args.faq}")
    print("Output exploratory model scores:")
    for horizon, value in scores.items():
        print(f"  {horizon}: {value:.1f}%")
    print("=" * 80)
    print("Research use only. Not clinically validated.")
    print("=" * 80)


if __name__ == "__main__":
    main()
