"""Lightweight forward-only check for Lmatch, Lgraph, and Ldiv."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import scipy.sparse as sp
import torch

from src.graph.assign import hungarian_match
from src.losses.dpp import compute_dpp_loss
from src.losses.graph_losses import compute_graph_loss, compute_match_loss


def main() -> None:
    torch.manual_seed(0)
    Y = torch.randn(6, 4)
    V_full = torch.randn(14, 4)
    degree = torch.arange(1, 15, dtype=torch.float32)

    diag = np.full(14, 2.0, dtype=np.float64)
    off = np.full(13, -1.0, dtype=np.float64)
    L_sym = sp.diags([off, diag, off], offsets=[-1, 0, 1], format='csr')

    assignment = hungarian_match(Y, V_full, degree=degree)
    Lmatch = compute_match_loss(Y, V_full, assignment.matched_indices)
    Lgraph = compute_graph_loss(Y, L_sym, assignment.matched_indices)
    Ldiv = compute_dpp_loss(Y, rff_dim=32, sigma=1.0, delta=1e-6).loss

    print(f"Lmatch={float(Lmatch.item()):.6f}")
    print(f"Lgraph={float(Lgraph.item()):.6f}")
    print(f"Ldiv={float(Ldiv.item()):.6f}")
    print(f"all_finite={bool(torch.isfinite(Lmatch) and torch.isfinite(Lgraph) and torch.isfinite(Ldiv))}")


if __name__ == "__main__":
    main()
