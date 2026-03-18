"""Small debug script for inspecting PD-CFD per-frequency terms."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.losses.pdcfd import pd_cfd_loss


def main() -> None:
    torch.manual_seed(7)
    Y_ref = torch.randn(32, 4, dtype=torch.float32)
    Y = Y_ref + 0.2 * torch.randn_like(Y_ref) + 0.15

    low = 0.20 * torch.randn(4, 4, dtype=torch.float32)
    mid = 1.00 * torch.randn(4, 4, dtype=torch.float32)
    high = 3.00 * torch.randn(4, 4, dtype=torch.float32)
    freqs = torch.cat([low, mid, high], dim=0)

    outputs = pd_cfd_loss(Y_ref, Y, freqs, lambda_p=1.0, alpha=0.5)

    print(f'total_loss={float(outputs.loss.item()):.6f}')
    print('idx	norm	amp_err	phase_err	lambda_phi')
    for idx in range(freqs.shape[0]):
        print(
            f'{idx}	'
            f'{float(outputs.frequency_norms[idx].item()):.6f}	'
            f'{float(outputs.per_freq_amplitude_error[idx].item()):.6f}	'
            f'{float(outputs.per_freq_phase_error[idx].item()):.6f}	'
            f'{float(outputs.lambda_phi[idx].item()):.6f}'
        )


if __name__ == '__main__':
    main()
