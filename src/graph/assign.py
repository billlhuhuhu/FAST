"""Assignment helpers for mapping continuous proxies back to discrete samples."""

from __future__ import annotations

import torch
from torch import Tensor


def compute_cost_matrix(proxies: Tensor, candidates: Tensor) -> Tensor:
    """Compute a dense assignment cost matrix.

    Args:
        proxies: Tensor of shape ``[M, D]``.
        candidates: Tensor of shape ``[N, D]``.

    Returns:
        Cost matrix with shape ``[M, N]``.
    """

    if proxies.ndim != 2 or candidates.ndim != 2:
        raise ValueError("Both proxies and candidates must be 2D tensors")
    return torch.cdist(proxies, candidates, p=2.0)


def greedy_unique_assignment(cost_matrix: Tensor) -> Tensor:
    """Greedy placeholder for unique proxy-to-sample assignment.

    Args:
        cost_matrix: Tensor of shape ``[M, N]``.

    Returns:
        Assigned sample indices with shape ``[M]``.

    TODO:
        - Replace with Hungarian or another bijective solver.
        - Add graph-aware costs once the full alignment term is implemented.
    """

    num_proxies, num_candidates = cost_matrix.shape
    assigned = []
    used = set()
    for row in range(num_proxies):
        order = torch.argsort(cost_matrix[row], dim=0)
        chosen = None
        for index in order.tolist():
            if index not in used:
                used.add(index)
                chosen = index
                break
        if chosen is None:
            chosen = int(order[0].item()) % max(num_candidates, 1)
        assigned.append(chosen)
    return torch.tensor(assigned, dtype=torch.long, device=cost_matrix.device)
