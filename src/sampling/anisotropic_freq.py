"""Anisotropic frequency initialization for the FAST scaffold.

This module provides a more paper-aligned first-pass implementation of the
Anisotropic Frequency Library (AFL):

1. split the frequency space into low / medium / high bands by norm,
2. maintain a band-specific anisotropic diagonal scaling,
3. search over candidate scaling magnitudes using a data-aware score,
4. build a final frequency library with band ids, norms, and score summaries.

The implementation is intentionally clarity-first and keeps the old top-level
function signature compatible with the current pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import torch
from torch import Tensor

from src.losses.pdcfd import pd_cfd_loss


@dataclass
class FrequencyLibrary:
    """Container for candidate frequencies.

    Attributes:
        omega:
            Frequency tensor with shape ``[F, D]``.
        norms:
            Norm tensor with shape ``[F]``.
        band_ids:
            Band identifier tensor with shape ``[F]`` where
            ``0=low, 1=medium, 2=high``.
        per_band_scaling:
            Mapping from band name to diagonal scaling tensor with shape ``[D]``.
        per_band_score_summary:
            Mapping from band name to small score diagnostics.
    """

    omega: Tensor
    norms: Tensor
    band_ids: Tensor
    per_band_scaling: Dict[str, Tensor] = field(default_factory=dict)
    per_band_score_summary: Dict[str, Dict[str, float]] = field(default_factory=dict)


BAND_ORDER = ('low', 'medium', 'high')
BAND_TO_ID = {'low': 0, 'medium': 1, 'high': 2}


def _default_band_ranges(max_norm: float) -> Dict[str, tuple[float, float]]:
    """Create default low / medium / high band ranges from ``max_norm``."""

    one_third = max_norm / 3.0
    two_third = 2.0 * max_norm / 3.0
    return {
        'low': (0.0, one_third),
        'medium': (one_third, two_third),
        'high': (two_third, max_norm),
    }


def _normalize_band_ranges(
    max_norm: float,
    band_ranges: Optional[Dict[str, Sequence[float]]] = None,
) -> Dict[str, tuple[float, float]]:
    """Normalize band range configuration into a canonical dictionary."""

    ranges = _default_band_ranges(max_norm) if band_ranges is None else {
        name: (float(values[0]), float(values[1])) for name, values in band_ranges.items()
    }
    for band in BAND_ORDER:
        if band not in ranges:
            raise ValueError(f'Missing band range for {band}')
        low, high = ranges[band]
        if low < 0.0 or high <= low:
            raise ValueError(f'Invalid band range for {band}: {(low, high)}')
    return ranges


def _resolve_band_counts(
    num_frequencies: int,
    band_sample_counts: Optional[Dict[str, int]] = None,
) -> Dict[str, int]:
    """Resolve how many frequencies to allocate to each band."""

    if num_frequencies <= 0:
        raise ValueError('num_frequencies must be positive')

    if band_sample_counts is None:
        base = num_frequencies // 3
        remainder = num_frequencies % 3
        counts = {'low': base, 'medium': base, 'high': base}
        for band in BAND_ORDER[:remainder]:
            counts[band] += 1
        return counts

    counts = {band: int(band_sample_counts.get(band, 0)) for band in BAND_ORDER}
    if sum(counts.values()) != num_frequencies:
        raise ValueError('band_sample_counts must sum to num_frequencies')
    if any(v <= 0 for v in counts.values()):
        raise ValueError('each band must receive at least one sample')
    return counts


def _candidate_scale_values(
    candidate_scales: Optional[Sequence[float]] = None,
    scaling_search_steps: Optional[int] = None,
) -> Tensor:
    """Return candidate scalar multipliers for anisotropic scaling search."""

    if candidate_scales is not None:
        values = torch.tensor([float(v) for v in candidate_scales], dtype=torch.float32)
    else:
        steps = 5 if scaling_search_steps is None else int(scaling_search_steps)
        if steps <= 0:
            raise ValueError('scaling_search_steps must be positive')
        values = torch.linspace(0.5, 2.5, steps=steps, dtype=torch.float32)
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError('candidate scales must be a non-empty 1D sequence')
    return values


def _compute_dimension_importance(
    Y_ref: Tensor,
    Y_current: Optional[Tensor] = None,
) -> Tensor:
    """Build a per-dimension importance vector for anisotropic scaling.

    If ``Y_current`` is available, the vector is discrepancy-aware. Otherwise it
    falls back to a reference-only structural proxy using spread statistics.
    """

    if Y_ref.ndim != 2:
        raise ValueError('Y_ref must have shape [N, d]')

    ref_mean = Y_ref.mean(dim=0)
    ref_std = Y_ref.std(dim=0, unbiased=False)

    if Y_current is None:
        importance = ref_std + (Y_ref - ref_mean).abs().mean(dim=0)
    else:
        if Y_current.ndim != 2 or Y_current.shape[1] != Y_ref.shape[1]:
            raise ValueError('Y_current must have shape [M, d] compatible with Y_ref')
        cur_mean = Y_current.mean(dim=0)
        cur_std = Y_current.std(dim=0, unbiased=False)
        importance = (ref_mean - cur_mean).abs() + (ref_std - cur_std).abs() + 1e-6

    importance = torch.clamp(importance, min=1e-6)
    importance = importance / torch.clamp(importance.mean(), min=1e-6)
    return importance


def _sample_band_frequencies(
    feature_dim: int,
    num_samples: int,
    band_range: tuple[float, float],
    scaling: Tensor,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Sample frequencies inside one band using anisotropic direction scaling.

    The band range constrains the frequency norm, while ``scaling`` biases the
    direction distribution through a diagonal covariance surrogate.
    """

    low, high = float(band_range[0]), float(band_range[1])
    low = max(low, 1e-6)
    gaussian = torch.randn(num_samples, feature_dim, device=device, dtype=dtype)
    scaled = gaussian * scaling.unsqueeze(0)
    scaled = scaled / torch.clamp(torch.linalg.norm(scaled, dim=1, keepdim=True), min=1e-12)
    radii = low + (high - low) * torch.rand(num_samples, 1, device=device, dtype=dtype)
    return scaled * radii


def _score_band_scaling(
    Y_ref: Tensor,
    Y_current: Optional[Tensor],
    candidate_freqs: Tensor,
    lambda_p: float,
    alpha: float,
) -> float:
    """Score one band-specific scaling candidate.

    If ``Y_current`` is available, use mean PD-CFD discrepancy over the sampled
    frequencies. Otherwise use a reference-only projection-variance proxy.
    """

    if Y_current is None:
        projections = Y_ref @ candidate_freqs.T
        return float(projections.var(dim=0, unbiased=False).mean().detach().cpu().item())

    outputs = pd_cfd_loss(
        Y_ref=Y_ref,
        Y=Y_current,
        freqs=candidate_freqs,
        lambda_p=lambda_p,
        alpha=alpha,
    )
    return float(outputs.per_frequency_loss.mean().detach().cpu().item())


def _build_isotropic_baseline_library(
    feature_dim: int,
    num_frequencies: int,
    max_norm: float,
    device: torch.device | None = None,
    band_ranges: Optional[Dict[str, Sequence[float]]] = None,
    band_sample_counts: Optional[Dict[str, int]] = None,
    dtype: torch.dtype = torch.float32,
) -> FrequencyLibrary:
    """Build a simple isotropic baseline library for comparison/debugging."""

    ranges = _normalize_band_ranges(max_norm=max_norm, band_ranges=band_ranges)
    counts = _resolve_band_counts(num_frequencies=num_frequencies, band_sample_counts=band_sample_counts)
    omega_parts = []
    band_ids = []
    per_band_scaling = {}
    summary = {}
    ones = torch.ones(feature_dim, device=device, dtype=dtype)

    for band in BAND_ORDER:
        freqs = _sample_band_frequencies(
            feature_dim=feature_dim,
            num_samples=counts[band],
            band_range=ranges[band],
            scaling=ones,
            device=device,
            dtype=dtype,
        )
        omega_parts.append(freqs)
        band_ids.append(torch.full((counts[band],), BAND_TO_ID[band], device=device, dtype=torch.long))
        per_band_scaling[band] = ones.clone()
        summary[band] = {'best_score': 0.0, 'mean_norm': float(torch.linalg.norm(freqs, dim=1).mean().item())}

    omega = torch.cat(omega_parts, dim=0)
    norms = torch.linalg.norm(omega, dim=1)
    band_ids_tensor = torch.cat(band_ids, dim=0)
    return FrequencyLibrary(
        omega=omega,
        norms=norms,
        band_ids=band_ids_tensor,
        per_band_scaling=per_band_scaling,
        per_band_score_summary=summary,
    )


def build_anisotropic_frequency_library(
    feature_dim: int,
    num_frequencies: int,
    max_norm: float,
    device: torch.device | None = None,
    Y_ref: Optional[Tensor] = None,
    Y_current: Optional[Tensor] = None,
    band_ranges: Optional[Dict[str, Sequence[float]]] = None,
    band_sample_counts: Optional[Dict[str, int]] = None,
    candidate_scales: Optional[Sequence[float]] = None,
    scaling_search_steps: Optional[int] = None,
    lambda_p: float = 0.5,
    alpha: float = 0.5,
) -> FrequencyLibrary:
    """Build the Anisotropic Frequency Library (AFL).

    This function keeps the old top-level signature valid while supporting a
    more paper-aligned initialization path when ``Y_ref`` is provided.

    Args:
        feature_dim:
            Feature dimension ``d``.
        num_frequencies:
            Total number of frequencies ``F``.
        max_norm:
            Maximum frequency norm.
        device:
            Target device.
        Y_ref:
            Optional reference data with shape ``[N, d]``.
        Y_current:
            Optional current coreset data with shape ``[M, d]`` used for
            discrepancy-aware scaling search.
        band_ranges:
            Optional mapping for low / medium / high norm ranges.
        band_sample_counts:
            Optional per-band sample counts.
        candidate_scales:
            Optional scalar multipliers searched for each band.
        scaling_search_steps:
            Number of candidate scales if ``candidate_scales`` is not provided.
        lambda_p:
            Phase attenuation base weight for PD-CFD scoring.
        alpha:
            Frequency attenuation coefficient for PD-CFD scoring.

    Returns:
        A :class:`FrequencyLibrary` containing ``omega``, ``norms``,
        ``band_ids``, ``per_band_scaling``, and ``per_band_score_summary``.
    """

    dtype = torch.float32 if Y_ref is None else Y_ref.dtype
    ranges = _normalize_band_ranges(max_norm=max_norm, band_ranges=band_ranges)
    counts = _resolve_band_counts(num_frequencies=num_frequencies, band_sample_counts=band_sample_counts)
    candidate_values = _candidate_scale_values(candidate_scales=candidate_scales, scaling_search_steps=scaling_search_steps)
    candidate_values = candidate_values.to(device=device, dtype=dtype)

    if Y_ref is None:
        return _build_isotropic_baseline_library(
            feature_dim=feature_dim,
            num_frequencies=num_frequencies,
            max_norm=max_norm,
            device=device,
            band_ranges=band_ranges,
            band_sample_counts=band_sample_counts,
            dtype=dtype,
        )

    if Y_ref.ndim != 2 or Y_ref.shape[1] != feature_dim:
        raise ValueError('Y_ref must have shape [N, feature_dim]')
    Y_ref = Y_ref.to(device=device, dtype=dtype)
    if Y_current is not None:
        Y_current = Y_current.to(device=device, dtype=dtype)

    importance = _compute_dimension_importance(Y_ref=Y_ref, Y_current=Y_current)
    omega_parts = []
    band_ids = []
    per_band_scaling: Dict[str, Tensor] = {}
    per_band_score_summary: Dict[str, Dict[str, float]] = {}

    for band in BAND_ORDER:
        best_score = None
        best_scaling = None
        candidate_scores = []
        for scale_value in candidate_values:
            scaling = torch.clamp(scale_value * importance, min=0.25, max=4.0)
            candidate_freqs = _sample_band_frequencies(
                feature_dim=feature_dim,
                num_samples=counts[band],
                band_range=ranges[band],
                scaling=scaling,
                device=device,
                dtype=dtype,
            )
            score = _score_band_scaling(
                Y_ref=Y_ref,
                Y_current=Y_current,
                candidate_freqs=candidate_freqs,
                lambda_p=lambda_p,
                alpha=alpha,
            )
            candidate_scores.append(score)
            if best_score is None or score > best_score:
                best_score = score
                best_scaling = scaling.detach().clone()

        assert best_scaling is not None and best_score is not None
        final_freqs = _sample_band_frequencies(
            feature_dim=feature_dim,
            num_samples=counts[band],
            band_range=ranges[band],
            scaling=best_scaling,
            device=device,
            dtype=dtype,
        )
        omega_parts.append(final_freqs)
        band_ids.append(torch.full((counts[band],), BAND_TO_ID[band], device=device, dtype=torch.long))
        per_band_scaling[band] = best_scaling
        per_band_score_summary[band] = {
            'best_score': float(best_score),
            'mean_norm': float(torch.linalg.norm(final_freqs, dim=1).mean().item()),
            'min_norm': float(torch.linalg.norm(final_freqs, dim=1).min().item()),
            'max_norm': float(torch.linalg.norm(final_freqs, dim=1).max().item()),
            'num_candidates': float(candidate_values.numel()),
            'score_min': float(min(candidate_scores)),
            'score_max': float(max(candidate_scores)),
        }

    omega = torch.cat(omega_parts, dim=0)
    norms = torch.linalg.norm(omega, dim=1)
    band_ids_tensor = torch.cat(band_ids, dim=0)
    return FrequencyLibrary(
        omega=omega,
        norms=norms,
        band_ids=band_ids_tensor,
        per_band_scaling=per_band_scaling,
        per_band_score_summary=per_band_score_summary,
    )
