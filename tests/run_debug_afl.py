"""Lightweight debug script for upgraded AFL statistics."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.sampling.anisotropic_freq import build_anisotropic_frequency_library


def main() -> None:
    torch.manual_seed(0)
    Y_ref = torch.randn(48, 8)
    Y_current = Y_ref + 0.2 * torch.randn_like(Y_ref)
    library = build_anisotropic_frequency_library(
        feature_dim=8,
        num_frequencies=18,
        max_norm=6.0,
        Y_ref=Y_ref,
        Y_current=Y_current,
        band_ranges={
            'low': [0.0, 2.0],
            'medium': [2.0, 4.0],
            'high': [4.0, 6.0],
        },
        band_sample_counts={'low': 6, 'medium': 6, 'high': 6},
        candidate_scales=[0.5, 1.0, 1.5, 2.0],
    )
    for band_name, band_id in [('low', 0), ('medium', 1), ('high', 2)]:
        mask = library.band_ids == band_id
        mean_norm = float(library.norms[mask].mean().item())
        scaling = library.per_band_scaling[band_name].detach().cpu().tolist()
        score = library.per_band_score_summary[band_name]['best_score']
        print(f'{band_name}: mean_norm={mean_norm:.6f} score={score:.6f} scaling={scaling}')


if __name__ == '__main__':
    main()
