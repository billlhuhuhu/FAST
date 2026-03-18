"""Assignment helpers for mapping continuous proxies back to discrete samples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor


@dataclass
class AssignmentResult:
    """Outputs of the one-to-one matching step.

    Attributes:
        matched_indices:
            Matched dataset indices with shape ``[M]``.
        matching_cost:
            Total assignment cost as a Python float.
        cost_matrix:
            Dense cost matrix with shape ``[M, N]``.
    """

    matched_indices: Tensor
    matching_cost: float
    cost_matrix: Tensor


def _assert_feature_shapes(Y: Tensor, V_full: Tensor) -> None:
    """Validate input feature shapes.

    Args:
        Y:
            Tensor with shape ``[M, d]``.
        V_full:
            Tensor with shape ``[N, d]``.
    """

    if Y.ndim != 2 or V_full.ndim != 2:
        raise ValueError("Y and V_full must both be 2D tensors")
    if Y.shape[1] != V_full.shape[1]:
        raise ValueError("Y and V_full must have the same feature dimension")
    if Y.shape[0] <= 0 or V_full.shape[0] <= 0:
        raise ValueError("Y and V_full must both contain at least one point")
    if Y.shape[0] > V_full.shape[0]:
        raise ValueError("Hungarian one-to-one matching requires M <= N")


def compute_cost_matrix(
    Y: Tensor,
    V_full: Tensor,
    degree: Optional[Tensor] = None,
    eps: float = 1e-8,
) -> Tensor:
    """Compute the degree-aware assignment cost matrix.

    The cost is:

    ``C_ij = ||y_i - v_j||^2 / (deg(v_j) + eps)``

    Args:
        Y:
            Proxy tensor with shape ``[M, d]``.
        V_full:
            Full spectral feature tensor with shape ``[N, d]``.
        degree:
            Optional degree tensor with shape ``[N]``.
        eps:
            Small constant for numerical stability.

    Returns:
        Cost matrix with shape ``[M, N]``.
    """

    _assert_feature_shapes(Y, V_full)
    squared_distances = torch.cdist(Y, V_full, p=2.0) ** 2

    if degree is None:
        return squared_distances

    if degree.ndim != 1 or degree.shape[0] != V_full.shape[0]:
        raise ValueError("degree must have shape [N] matching V_full")

    safe_degree = torch.clamp(degree.to(device=Y.device, dtype=Y.dtype), min=0.0)
    denom = safe_degree + eps
    return squared_distances / denom.unsqueeze(0)


def hungarian_match(
    Y: Tensor,
    V_full: Tensor,
    degree: Optional[Tensor] = None,
    eps: float = 1e-8,
) -> AssignmentResult:
    """Compute a one-to-one matching using the Hungarian algorithm.

    Args:
        Y:
            Proxy tensor with shape ``[M, d]``.
        V_full:
            Full spectral feature tensor with shape ``[N, d]``.
        degree:
            Optional degree tensor with shape ``[N]``.
        eps:
            Numerical stability constant for the degree-aware denominator.

    Returns:
        An :class:`AssignmentResult` containing matched indices and total cost.
    """

    cost_matrix = compute_cost_matrix(Y=Y, V_full=V_full, degree=degree, eps=eps)
    cost_np = cost_matrix.detach().cpu().numpy().astype(np.float64, copy=False)
    row_ind, col_ind = linear_sum_assignment(cost_np)

    if row_ind.shape[0] != Y.shape[0]:
        raise AssertionError("Hungarian matching did not return one assignment per proxy")

    matched_indices_np = np.zeros(Y.shape[0], dtype=np.int64)
    matched_indices_np[row_ind] = col_ind
    matched_indices = torch.from_numpy(matched_indices_np).to(device=Y.device, dtype=torch.long)
    matching_cost = float(cost_np[row_ind, col_ind].sum())

    if int(matched_indices.min().item()) < 0 or int(matched_indices.max().item()) >= V_full.shape[0]:
        raise AssertionError("Matched indices are out of bounds")

    return AssignmentResult(
        matched_indices=matched_indices,
        matching_cost=matching_cost,
        cost_matrix=cost_matrix,
    )
