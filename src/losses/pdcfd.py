"""Phase-decoupled CFD utilities for the FAST scaffold."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class PdCfdOutputs:
    """Outputs of the PD-CFD computation.

    Attributes:
        loss:
            Scalar tensor for the mean PD-CFD loss.
        per_frequency_loss:
            Tensor with shape ``[K]`` containing the loss at each frequency.
        ref_cf:
            Reference empirical characteristic function with shape ``[K]``.
        y_cf:
            Current-set empirical characteristic function with shape ``[K]``.
        amplitude_diff:
            Amplitude difference with shape ``[K]``.
        phase_diff:
            Wrapped phase difference with shape ``[K]``.
        attenuation:
            Frequency attenuation weights with shape ``[K]``.
    """

    loss: Tensor
    per_frequency_loss: Tensor
    ref_cf: Tensor
    y_cf: Tensor
    amplitude_diff: Tensor
    phase_diff: Tensor
    attenuation: Tensor


def _assert_shapes(Y_ref: Tensor, Y: Tensor, freqs: Tensor) -> None:
    """Validate PD-CFD input shapes.

    Args:
        Y_ref:
            Tensor with shape ``[N, d]``.
        Y:
            Tensor with shape ``[M, d]``.
        freqs:
            Tensor with shape ``[K, d]``.
    """

    if Y_ref.ndim != 2 or Y.ndim != 2 or freqs.ndim != 2:
        raise ValueError("Y_ref, Y, and freqs must all be 2D tensors")
    if Y_ref.shape[1] != Y.shape[1] or Y_ref.shape[1] != freqs.shape[1]:
        raise ValueError("Y_ref, Y, and freqs must share the same feature dimension")
    if Y_ref.shape[0] <= 0 or Y.shape[0] <= 0 or freqs.shape[0] <= 0:
        raise ValueError("Y_ref, Y, and freqs must all contain at least one row")


def empirical_characteristic_function(points: Tensor, freqs: Tensor) -> Tensor:
    """Compute the empirical characteristic function (ECF).

    Args:
        points:
            Tensor with shape ``[S, d]``.
        freqs:
            Tensor with shape ``[K, d]``.

    Returns:
        Complex tensor with shape ``[K]`` where
        ``phi(w) = mean(exp(i * <w, y>))``.
    """

    if points.ndim != 2 or freqs.ndim != 2:
        raise ValueError("points and freqs must both be 2D tensors")
    if points.shape[1] != freqs.shape[1]:
        raise ValueError("points and freqs must share the same feature dimension")

    real_dtype = points.dtype
    projections = points @ freqs.T
    complex_dtype = torch.complex64 if real_dtype == torch.float32 else torch.complex128
    phases = torch.complex(torch.zeros_like(projections), projections).to(complex_dtype)
    return torch.exp(phases).mean(dim=0)


def wrapped_phase_difference(theta_ref: Tensor, theta_y: Tensor) -> Tensor:
    """Compute a wrapped phase difference using a stable atan2 expression.

    Args:
        theta_ref:
            Tensor with shape ``[K]``.
        theta_y:
            Tensor with shape ``[K]``.

    Returns:
        Tensor with shape ``[K]`` in ``[-pi, pi]``.
    """

    delta = theta_ref - theta_y
    return torch.atan2(torch.sin(delta), torch.cos(delta))


def frequency_attenuation(freqs: Tensor, lambda_p: float = 1.0, alpha: float = 1.0) -> Tensor:
    """Compute the phase attenuation weight per frequency.

    Args:
        freqs:
            Tensor with shape ``[K, d]``.
        lambda_p:
            Base phase weight.
        alpha:
            Frequency decay coefficient.

    Returns:
        Tensor with shape ``[K]``.
    """

    if lambda_p < 0.0 or alpha < 0.0:
        raise ValueError("lambda_p and alpha must be non-negative")
    squared_norm = torch.sum(freqs ** 2, dim=1)
    return lambda_p / (1.0 + alpha * squared_norm)


def pd_cfd_loss(
    Y_ref: Tensor,
    Y: Tensor,
    freqs: Tensor,
    lambda_p: float = 1.0,
    alpha: float = 1.0,
) -> PdCfdOutputs:
    """Compute the phase-decoupled CFD loss.

    Args:
        Y_ref:
            Reference tensor with shape ``[N, d]``.
        Y:
            Current tensor with shape ``[M, d]``.
        freqs:
            Frequency tensor with shape ``[K, d]``.
        lambda_p:
            Base phase-decoupling weight.
        alpha:
            Frequency attenuation coefficient.

    Returns:
        A :class:`PdCfdOutputs` bundle.
    """

    _assert_shapes(Y_ref, Y, freqs)

    freqs = freqs.to(device=Y_ref.device, dtype=Y_ref.dtype)
    Y = Y.to(device=Y_ref.device, dtype=Y_ref.dtype)

    ref_cf = empirical_characteristic_function(Y_ref, freqs)
    y_cf = empirical_characteristic_function(Y, freqs)

    base_cfd = torch.abs(ref_cf - y_cf) ** 2

    ref_amp = torch.abs(ref_cf)
    y_amp = torch.abs(y_cf)
    amplitude_diff = ref_amp - y_amp

    theta_ref = torch.angle(ref_cf)
    theta_y = torch.angle(y_cf)
    phase_diff = wrapped_phase_difference(theta_ref, theta_y)

    attenuation = frequency_attenuation(freqs=freqs, lambda_p=lambda_p, alpha=alpha)
    per_frequency_loss = base_cfd.real + attenuation * (phase_diff ** 2)
    loss = per_frequency_loss.mean()

    return PdCfdOutputs(
        loss=loss,
        per_frequency_loss=per_frequency_loss,
        ref_cf=ref_cf,
        y_cf=y_cf,
        amplitude_diff=amplitude_diff,
        phase_diff=phase_diff,
        attenuation=attenuation,
    )
