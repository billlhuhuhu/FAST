"""Reusable graph-related losses for the FAST scaffold."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from torch import Tensor


def _assert_feature_shapes(Y: Tensor, V_full: Tensor, matched_indices: Tensor) -> None:
    """Validate feature and index shapes for matching-based losses."""

    if Y.ndim != 2 or V_full.ndim != 2:
        raise ValueError("Y and V_full must both be 2D tensors")
    if Y.shape[1] != V_full.shape[1]:
        raise ValueError("Y and V_full must share the same feature dimension")
    if matched_indices.ndim != 1 or matched_indices.shape[0] != Y.shape[0]:
        raise ValueError("matched_indices must have shape [M]")


def compute_match_loss(Y: Tensor, V_full: Tensor, matched_indices: Tensor) -> Tensor:
    """Compute the matching loss ``Lmatch``.

    Args:
        Y:
            Proxy tensor with shape ``[M, d]``.
        V_full:
            Full spectral feature tensor with shape ``[N, d]``.
        matched_indices:
            Matched indices with shape ``[M]``.

    Returns:
        Scalar tensor equal to the mean squared matching error.
    """

    _assert_feature_shapes(Y, V_full, matched_indices)
    matched_points = V_full[matched_indices]
    return torch.mean((Y - matched_points) ** 2)


def compute_graph_loss(Y: Tensor, L_sym: sp.spmatrix | np.ndarray | Tensor, matched_indices: Tensor) -> Tensor:
    """Compute the graph regularization loss ``Lgraph``.

    The loss is:

    ``trace(Y^T L_sub Y)``

    where ``L_sub`` is the submatrix of ``L_sym`` induced by ``matched_indices``.

    Args:
        Y:
            Proxy tensor with shape ``[M, d]``.
        L_sym:
            Full normalized Laplacian with shape ``[N, N]`` as a scipy sparse
            matrix, NumPy array, or Torch tensor.
        matched_indices:
            Matched indices with shape ``[M]``.

    Returns:
        Scalar tensor.
    """

    if Y.ndim != 2:
        raise ValueError("Y must have shape [M, d]")
    if matched_indices.ndim != 1 or matched_indices.shape[0] != Y.shape[0]:
        raise ValueError("matched_indices must have shape [M]")

    index_np = matched_indices.detach().cpu().numpy().astype(np.int64)

    if sp.issparse(L_sym):
        L_sub_np = L_sym.tocsr()[index_np][:, index_np].toarray()
        L_sub = torch.from_numpy(L_sub_np).to(device=Y.device, dtype=Y.dtype)
    elif isinstance(L_sym, np.ndarray):
        L_sub = torch.from_numpy(L_sym[np.ix_(index_np, index_np)]).to(device=Y.device, dtype=Y.dtype)
    elif isinstance(L_sym, torch.Tensor):
        L_sub = L_sym.to(device=Y.device, dtype=Y.dtype)[matched_indices][:, matched_indices]
    else:
        raise TypeError("L_sym must be a scipy sparse matrix, NumPy array, or Torch tensor")

    if L_sub.ndim != 2 or L_sub.shape[0] != Y.shape[0] or L_sub.shape[1] != Y.shape[0]:
        raise AssertionError("L_sub must have shape [M, M]")

    loss = torch.trace(Y.T @ L_sub @ Y)
    return loss
