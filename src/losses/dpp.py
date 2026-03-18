"""DPP-style diversity loss built on random Fourier features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math
import torch
from torch import Tensor


@dataclass
class DppLossResult:
    """Outputs of the DPP diversity loss.

    Attributes:
        loss:
            Scalar tensor equal to ``-log det(K)``.
        logdet:
            Scalar tensor containing ``log |det(K)|``.
        sign:
            Scalar tensor containing the determinant sign.
        kernel:
            Kernel matrix ``K`` with shape ``[M, M]``.
        psi:
            Random Fourier feature matrix ``Psi`` with shape ``[M, rff_dim]``.
    """

    loss: Tensor
    logdet: Tensor
    sign: Tensor
    kernel: Tensor
    psi: Tensor


def _assert_Y_shape(Y: Tensor) -> None:
    """Validate the coreset variable shape.

    Args:
        Y:
            Tensor with shape ``[M, d]``.
    """

    if Y.ndim != 2:
        raise ValueError(f"Expected Y with shape [M, d], got {tuple(Y.shape)}")
    if Y.shape[0] <= 0 or Y.shape[1] <= 0:
        raise ValueError("Y must have positive shape in both dimensions")


def sample_rff_parameters(
    input_dim: int,
    rff_dim: int,
    sigma: float = 1.0,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> tuple[Tensor, Tensor]:
    """Sample random Fourier feature parameters.

    Args:
        input_dim:
            Input feature dimension ``d``.
        rff_dim:
            Number of random Fourier features.
        sigma:
            Bandwidth parameter. Larger ``sigma`` means smoother kernels.
        device:
            Torch device for sampled parameters.
        dtype:
            Torch dtype for sampled parameters.

    Returns:
        ``(omega, bias)`` where:
        - ``omega`` has shape ``[d, rff_dim]``
        - ``bias`` has shape ``[rff_dim]``
    """

    if input_dim <= 0 or rff_dim <= 0:
        raise ValueError("input_dim and rff_dim must both be positive")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")

    omega = torch.randn(input_dim, rff_dim, device=device, dtype=dtype) / sigma
    bias = 2.0 * math.pi * torch.rand(rff_dim, device=device, dtype=dtype)
    return omega, bias


def compute_rff_features(
    Y: Tensor,
    rff_dim: int = 128,
    sigma: float = 1.0,
    omega: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
) -> Tensor:
    """Construct random Fourier features ``Psi`` from ``Y``.

    Args:
        Y:
            Tensor with shape ``[M, d]``.
        rff_dim:
            Number of random Fourier features.
        sigma:
            Kernel bandwidth.
        omega:
            Optional pre-sampled frequency matrix with shape ``[d, rff_dim]``.
        bias:
            Optional pre-sampled phase vector with shape ``[rff_dim]``.

    Returns:
        Tensor ``Psi`` with shape ``[M, rff_dim]``.
    """

    _assert_Y_shape(Y)
    M, d = Y.shape
    _ = M
    if omega is None or bias is None:
        omega, bias = sample_rff_parameters(
            input_dim=d,
            rff_dim=rff_dim,
            sigma=sigma,
            device=Y.device,
            dtype=Y.dtype,
        )
    else:
        if omega.shape != (d, rff_dim):
            raise ValueError("omega must have shape [d, rff_dim]")
        if bias.shape != (rff_dim,):
            raise ValueError("bias must have shape [rff_dim]")
        omega = omega.to(device=Y.device, dtype=Y.dtype)
        bias = bias.to(device=Y.device, dtype=Y.dtype)

    projection = Y @ omega + bias.unsqueeze(0)
    scale = math.sqrt(2.0 / float(rff_dim))
    psi = scale * torch.cos(projection)
    return psi


def compute_dpp_loss(
    Y: Tensor,
    rff_dim: int = 128,
    sigma: float = 1.0,
    delta: float = 1e-6,
    omega: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
) -> DppLossResult:
    """Compute the DPP-style diversity loss.

    The construction is:

    1. build random Fourier features ``Psi``
    2. compute ``K = Psi Psi^T + delta I``
    3. compute ``Ldiv = -log det(K)``

    Args:
        Y:
            Tensor with shape ``[M, d]``.
        rff_dim:
            Number of random Fourier features.
        sigma:
            Kernel bandwidth for the RFF mapping.
        delta:
            Diagonal jitter for numerical stability.
        omega:
            Optional fixed RFF frequency matrix with shape ``[d, rff_dim]``.
        bias:
            Optional fixed RFF phase vector with shape ``[rff_dim]``.

    Returns:
        A :class:`DppLossResult` bundle.
    """

    _assert_Y_shape(Y)
    if delta <= 0.0:
        raise ValueError("delta must be positive")

    psi = compute_rff_features(Y=Y, rff_dim=rff_dim, sigma=sigma, omega=omega, bias=bias)
    M = Y.shape[0]
    identity = torch.eye(M, device=Y.device, dtype=Y.dtype)
    kernel = psi @ psi.T + delta * identity
    kernel = 0.5 * (kernel + kernel.T)
    sign, logabsdet = torch.linalg.slogdet(kernel)

    if torch.isnan(sign) or torch.isnan(logabsdet) or torch.isinf(logabsdet):
        stabilized_kernel = kernel + (10.0 * delta) * identity
        sign, logabsdet = torch.linalg.slogdet(stabilized_kernel)
        kernel = stabilized_kernel

    if sign <= 0:
        stabilized_kernel = kernel + (100.0 * delta) * identity
        sign, logabsdet = torch.linalg.slogdet(stabilized_kernel)
        kernel = stabilized_kernel

    loss = -logabsdet
    return DppLossResult(loss=loss, logdet=logabsdet, sign=sign, kernel=kernel, psi=psi)
