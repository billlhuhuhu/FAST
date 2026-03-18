"""Phase-decoupled CFD utilities for the FAST scaffold.

This module keeps the public ``pd_cfd_loss`` entry point stable for the
current pipeline while making the internal decomposition closer to FAST
Section 3.3:

- empirical characteristic function (ECF)
- amplitude discrepancy
- phase discrepancy with wrapped phase handling
- frequency-decayed phase weighting
- total per-frequency loss
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class PdCfdOutputs:
    """Outputs of the PD-CFD computation.

    Shape notation:
        - ``N``: number of reference points
        - ``M``: number of current coreset points
        - ``K``: number of frequencies
        - ``d``: feature dimension

    Attributes:
        loss:
            Scalar tensor containing the mean total PD-CFD loss.
        total_per_freq_loss:
            Tensor with shape ``[K]`` containing the final loss per frequency.
        ecf_ref:
            Complex ECF of the reference set with shape ``[K]``.
        ecf_y:
            Complex ECF of the current set with shape ``[K]``.
        per_freq_amplitude_error:
            Tensor with shape ``[K]`` containing squared amplitude mismatch.
        per_freq_phase_error:
            Tensor with shape ``[K]`` containing stabilized wrapped phase error.
        lambda_phi:
            Tensor with shape ``[K]`` containing
            ``lambda_p / (1 + alpha * ||omega||^2)``.
        phase_confidence:
            Tensor with shape ``[K]`` that downweights phase error when ECF
            amplitudes are tiny.
        raw_phase_difference:
            Wrapped phase difference in radians with shape ``[K]``.
        amplitude_difference:
            Signed amplitude difference ``|phi_ref| - |phi_y|`` with shape ``[K]``.
        frequency_norms:
            Tensor with shape ``[K]`` containing ``||omega||``.

    Compatibility aliases retained for the existing pipeline:
        - ``per_frequency_loss`` -> ``total_per_freq_loss``
        - ``ref_cf`` -> ``ecf_ref``
        - ``y_cf`` -> ``ecf_y``
        - ``amplitude_diff`` -> ``amplitude_difference``
        - ``phase_diff`` -> ``raw_phase_difference``
        - ``attenuation`` -> ``lambda_phi``
    """

    loss: Tensor
    total_per_freq_loss: Tensor
    ecf_ref: Tensor
    ecf_y: Tensor
    per_freq_amplitude_error: Tensor
    per_freq_phase_error: Tensor
    lambda_phi: Tensor
    phase_confidence: Tensor
    raw_phase_difference: Tensor
    amplitude_difference: Tensor
    frequency_norms: Tensor

    @property
    def per_frequency_loss(self) -> Tensor:
        return self.total_per_freq_loss

    @property
    def ref_cf(self) -> Tensor:
        return self.ecf_ref

    @property
    def y_cf(self) -> Tensor:
        return self.ecf_y

    @property
    def amplitude_diff(self) -> Tensor:
        return self.amplitude_difference

    @property
    def phase_diff(self) -> Tensor:
        return self.raw_phase_difference

    @property
    def attenuation(self) -> Tensor:
        return self.lambda_phi


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
    """Compute the empirical characteristic function.

    Args:
        points:
            Real tensor with shape ``[S, d]``.
        freqs:
            Real tensor with shape ``[K, d]``.

    Returns:
        Complex tensor with shape ``[K]`` where
        ``phi(omega) = mean(exp(i * <omega, x>))``.
    """

    if points.ndim != 2 or freqs.ndim != 2:
        raise ValueError("points and freqs must both be 2D tensors")
    if points.shape[1] != freqs.shape[1]:
        raise ValueError("points and freqs must share the same feature dimension")

    projections = points @ freqs.T
    complex_dtype = torch.complex64 if points.dtype == torch.float32 else torch.complex128
    complex_phases = torch.complex(torch.zeros_like(projections), projections).to(complex_dtype)
    return torch.exp(complex_phases).mean(dim=0)


def wrapped_phase_difference(theta_ref: Tensor, theta_y: Tensor) -> Tensor:
    """Compute a stable wrapped phase difference.

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
    """Compute the FAST phase attenuation weight.

    Args:
        freqs:
            Tensor with shape ``[K, d]``.
        lambda_p:
            Base phase weight.
        alpha:
            Decay coefficient.

    Returns:
        Tensor with shape ``[K]`` implementing
        ``lambda_phi(omega) = lambda_p / (1 + alpha * ||omega||^2)``.
    """

    if lambda_p < 0.0 or alpha < 0.0:
        raise ValueError("lambda_p and alpha must be non-negative")
    squared_norm = torch.sum(freqs ** 2, dim=1)
    return lambda_p / (1.0 + alpha * squared_norm)


def amplitude_discrepancy(ecf_ref: Tensor, ecf_y: Tensor) -> tuple[Tensor, Tensor]:
    """Compute signed and squared amplitude discrepancy.

    Args:
        ecf_ref:
            Complex tensor with shape ``[K]``.
        ecf_y:
            Complex tensor with shape ``[K]``.

    Returns:
        A tuple ``(signed_difference, squared_error)`` with shape ``[K]`` each.
    """

    ref_amp = torch.abs(ecf_ref)
    y_amp = torch.abs(ecf_y)
    amplitude_difference = ref_amp - y_amp
    amplitude_error = amplitude_difference.square()
    return amplitude_difference, amplitude_error


def phase_discrepancy(
    ecf_ref: Tensor,
    ecf_y: Tensor,
    phase_amplitude_floor: float = 1e-3,
) -> tuple[Tensor, Tensor]:
    """Compute a stabilized phase discrepancy.

    The wrapped phase difference is always computed, but its contribution is
    downweighted when either ECF amplitude is tiny. This reduces noise in the
    low-amplitude or high-frequency regime where phase is numerically unstable.

    Args:
        ecf_ref:
            Complex tensor with shape ``[K]``.
        ecf_y:
            Complex tensor with shape ``[K]``.
        phase_amplitude_floor:
            Positive scalar controlling when phase becomes unreliable.

    Returns:
        A tuple ``(wrapped_phase, stabilized_phase_error)`` with shape ``[K]`` each.
    """

    if phase_amplitude_floor <= 0.0:
        raise ValueError("phase_amplitude_floor must be positive")

    theta_ref = torch.angle(ecf_ref)
    theta_y = torch.angle(ecf_y)
    wrapped_phase = wrapped_phase_difference(theta_ref, theta_y)

    ref_amp = torch.abs(ecf_ref)
    y_amp = torch.abs(ecf_y)
    min_amp = torch.minimum(ref_amp, y_amp)
    phase_confidence = min_amp / (min_amp + phase_amplitude_floor)
    phase_error = phase_confidence * wrapped_phase.square()
    return wrapped_phase, phase_error


def pd_cfd_loss(
    Y_ref: Tensor,
    Y: Tensor,
    freqs: Tensor,
    lambda_p: float = 1.0,
    alpha: float = 1.0,
    phase_amplitude_floor: float = 1e-3,
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
        phase_amplitude_floor:
            Stabilization floor for low-amplitude phase terms.

    Returns:
        A :class:`PdCfdOutputs` bundle with detailed per-frequency diagnostics.
    """

    _assert_shapes(Y_ref, Y, freqs)

    freqs = freqs.to(device=Y_ref.device, dtype=Y_ref.dtype)
    Y = Y.to(device=Y_ref.device, dtype=Y_ref.dtype)

    ecf_ref = empirical_characteristic_function(Y_ref, freqs)
    ecf_y = empirical_characteristic_function(Y, freqs)

    amplitude_difference, amplitude_error = amplitude_discrepancy(ecf_ref, ecf_y)
    raw_phase_difference, phase_error = phase_discrepancy(
        ecf_ref,
        ecf_y,
        phase_amplitude_floor=phase_amplitude_floor,
    )

    lambda_phi = frequency_attenuation(freqs=freqs, lambda_p=lambda_p, alpha=alpha)
    total_per_freq_loss = amplitude_error + lambda_phi * phase_error
    loss = total_per_freq_loss.mean().real

    ref_amp = torch.abs(ecf_ref)
    y_amp = torch.abs(ecf_y)
    min_amp = torch.minimum(ref_amp, y_amp)
    phase_confidence = min_amp / (min_amp + phase_amplitude_floor)
    frequency_norms = torch.linalg.norm(freqs, dim=1)

    return PdCfdOutputs(
        loss=loss,
        total_per_freq_loss=total_per_freq_loss.real,
        ecf_ref=ecf_ref,
        ecf_y=ecf_y,
        per_freq_amplitude_error=amplitude_error.real,
        per_freq_phase_error=phase_error.real,
        lambda_phi=lambda_phi.real,
        phase_confidence=phase_confidence.real,
        raw_phase_difference=raw_phase_difference.real,
        amplitude_difference=amplitude_difference.real,
        frequency_norms=frequency_norms.real,
    )
