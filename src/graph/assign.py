"""Assignment helpers for mapping continuous proxies back to discrete samples."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, Optional

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
            Dense or pruned cost matrix with shape ``[M, N]``.
        candidate_stats:
            Lightweight summary for debugging/performance inspection.
        mode:
            Matching mode, either ``full`` or ``pruned``.
    """

    matched_indices: Tensor
    matching_cost: float
    cost_matrix: Tensor
    candidate_stats: Dict[str, Any]
    mode: str


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


def _resolve_mode(
    Y: Tensor,
    V_full: Tensor,
    mode: str = "auto",
    prune_topk: Optional[int] = None,
    auto_threshold: int = 20000,
) -> str:
    """Resolve the assignment mode.

    ``auto`` chooses ``pruned`` when the dense ``M x N`` matrix would be large.
    """

    if mode not in {"auto", "full", "pruned"}:
        raise ValueError("mode must be one of: auto, full, pruned")
    if mode != "auto":
        return mode

    problem_size = int(Y.shape[0] * V_full.shape[0])
    if prune_topk is None:
        prune_topk = min(max(8, Y.shape[0]), V_full.shape[0])
    if problem_size > auto_threshold and prune_topk < V_full.shape[0]:
        return "pruned"
    return "full"


def compute_cost_matrix(
    Y: Tensor,
    V_full: Tensor,
    degree: Optional[Tensor] = None,
    eps: float = 1e-8,
) -> Tensor:
    """Compute the degree-aware dense assignment cost matrix.

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
        Dense cost matrix with shape ``[M, N]``.
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


def compute_pruned_cost_matrix(
    Y: Tensor,
    V_full: Tensor,
    degree: Optional[Tensor] = None,
    prune_topk: int = 32,
    eps: float = 1e-8,
    large_cost_scale: float = 1e6,
) -> tuple[Tensor, Tensor]:
    """Compute a dense-but-pruned cost matrix using top-k candidates per row.

    Args:
        Y:
            Proxy tensor with shape ``[M, d]``.
        V_full:
            Full spectral feature tensor with shape ``[N, d]``.
        degree:
            Optional degree tensor with shape ``[N]``.
        prune_topk:
            Number of nearest candidates to keep for each proxy.
        eps:
            Stability constant for degree-aware costs.
        large_cost_scale:
            Large finite penalty used outside the candidate set.

    Returns:
        A tuple ``(pruned_cost_matrix, candidate_indices)`` where:
        - ``pruned_cost_matrix`` has shape ``[M, N]``
        - ``candidate_indices`` has shape ``[M, k_eff]``
    """

    _assert_feature_shapes(Y, V_full)
    N = int(V_full.shape[0])
    k_eff = max(1, min(int(prune_topk), N))

    dense_cost = compute_cost_matrix(Y=Y, V_full=V_full, degree=degree, eps=eps)
    candidate_costs, candidate_indices = torch.topk(dense_cost, k=k_eff, dim=1, largest=False)

    finite_max = float(torch.max(candidate_costs).detach().cpu().item()) if candidate_costs.numel() > 0 else 1.0
    fill_value = max(finite_max * float(large_cost_scale), 1.0)
    pruned_cost = torch.full_like(dense_cost, fill_value=fill_value)
    pruned_cost.scatter_(1, candidate_indices, candidate_costs)
    return pruned_cost, candidate_indices


def hungarian_match(
    Y: Tensor,
    V_full: Tensor,
    degree: Optional[Tensor] = None,
    eps: float = 1e-8,
    mode: str = "auto",
    prune_topk: Optional[int] = None,
    auto_threshold: int = 20000,
    large_cost_scale: float = 1e6,
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
        mode:
            ``full``, ``pruned``, or ``auto``.
        prune_topk:
            Candidate count for pruned mode.
        auto_threshold:
            Switch threshold used by ``auto`` mode on ``M * N``.
        large_cost_scale:
            Large finite penalty outside pruned candidates.

    Returns:
        An :class:`AssignmentResult` containing matched indices and total cost.
    """

    _assert_feature_shapes(Y, V_full)
    N = int(V_full.shape[0])
    resolved_mode = _resolve_mode(Y=Y, V_full=V_full, mode=mode, prune_topk=prune_topk, auto_threshold=auto_threshold)

    if prune_topk is None:
        prune_topk = min(max(8, Y.shape[0]), N)
    prune_topk = max(1, min(int(prune_topk), N))

    build_start = perf_counter()
    candidate_stats: Dict[str, Any] = {
        "N": N,
        "M": int(Y.shape[0]),
        "prune_topk": prune_topk,
    }

    if resolved_mode == "full":
        cost_matrix = compute_cost_matrix(Y=Y, V_full=V_full, degree=degree, eps=eps)
        candidate_stats.update({
            "candidate_count_per_row": N,
            "candidate_density": 1.0,
            "out_of_candidate_matches": 0,
        })
    else:
        cost_matrix, candidate_indices = compute_pruned_cost_matrix(
            Y=Y,
            V_full=V_full,
            degree=degree,
            prune_topk=prune_topk,
            eps=eps,
            large_cost_scale=large_cost_scale,
        )
        candidate_stats.update({
            "candidate_count_per_row": int(candidate_indices.shape[1]),
            "candidate_density": float(candidate_indices.shape[1] / max(1, N)),
            "candidate_indices_shape": [int(candidate_indices.shape[0]), int(candidate_indices.shape[1])],
        })
    build_time = perf_counter() - build_start

    match_start = perf_counter()
    cost_np = cost_matrix.detach().cpu().numpy().astype(np.float64, copy=False)
    row_ind, col_ind = linear_sum_assignment(cost_np)
    match_time = perf_counter() - match_start

    if row_ind.shape[0] != Y.shape[0]:
        raise AssertionError("Hungarian matching did not return one assignment per proxy")

    matched_indices_np = np.zeros(Y.shape[0], dtype=np.int64)
    matched_indices_np[row_ind] = col_ind
    matched_indices = torch.from_numpy(matched_indices_np).to(device=Y.device, dtype=torch.long)
    matching_cost = float(cost_np[row_ind, col_ind].sum())

    if int(matched_indices.min().item()) < 0 or int(matched_indices.max().item()) >= N:
        raise AssertionError("Matched indices are out of bounds")

    if resolved_mode == "pruned":
        candidate_mask = torch.zeros_like(cost_matrix, dtype=torch.bool)
        candidate_mask.scatter_(1, candidate_indices, True)
        row_selector = torch.arange(Y.shape[0], device=Y.device)
        out_of_candidate = (~candidate_mask[row_selector, matched_indices]).sum().item()
        candidate_stats["out_of_candidate_matches"] = int(out_of_candidate)

    candidate_stats.update({
        "cost_build_time_sec": float(build_time),
        "matching_time_sec": float(match_time),
        "problem_size": int(Y.shape[0] * N),
    })

    return AssignmentResult(
        matched_indices=matched_indices,
        matching_cost=matching_cost,
        cost_matrix=cost_matrix,
        candidate_stats=candidate_stats,
        mode=resolved_mode,
    )
