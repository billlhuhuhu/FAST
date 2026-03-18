"""Progressive discrepancy-aware sampling (PDAS) utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
from torch import Tensor

from src.losses.pdcfd import pd_cfd_loss
from src.sampling.anisotropic_freq import FrequencyLibrary


@dataclass
class PdasState:
    """State bundle for progressive discrepancy-aware sampling.

    Attributes:
        step:
            Current optimization step.
        selected_indices:
            Selected frequency indices with shape ``[S]``.
        selected_frequencies:
            Selected frequencies with shape ``[S, d]``.
        per_freq_lcf:
            Candidate-pool per-frequency LCF values with shape ``[C]``.
        per_freq_diversity:
            Candidate-pool per-frequency diversity values with shape ``[C]``.
        final_scores:
            Candidate-pool final scores with shape ``[C]``.
        candidate_pool_stats:
            Small dictionary with tau/candidate-pool diagnostics.
    """

    step: int
    selected_indices: Tensor
    selected_frequencies: Tensor
    per_freq_lcf: Tensor
    per_freq_diversity: Tensor
    final_scores: Tensor
    candidate_pool_stats: Dict[str, Any] = field(default_factory=dict)


def _resolve_sampling_count(library: FrequencyLibrary, step: int, total_steps: int, config: Optional[Dict[str, Any]]) -> int:
    """Resolve how many frequencies to select at the current step."""

    if config is not None and 'pdas_frequencies_per_iter' in config:
        count = int(config['pdas_frequencies_per_iter'])
    else:
        fraction = min(max((step + 1) / total_steps, 0.0), 1.0)
        count = max(1, int(round(fraction * library.omega.shape[0])))
    return min(max(count, 1), int(library.omega.shape[0]))


def _resolve_tau(library: FrequencyLibrary, step: int, total_steps: int, config: Optional[Dict[str, Any]]) -> float:
    """Resolve the current norm threshold ``tau_t``.

    The candidate pool is:
    ``C_t = {omega in AFL | ||omega|| <= tau_t}``
    """

    norms = library.norms.detach()
    norm_min = float(norms.min().item())
    norm_max = float(norms.max().item())

    tau_start_ratio = float(config.get('tau_start_ratio', 0.2)) if config is not None else 0.2
    tau_end_ratio = float(config.get('tau_end_ratio', 1.0)) if config is not None else 1.0
    tau_start_ratio = min(max(tau_start_ratio, 0.0), 1.0)
    tau_end_ratio = min(max(tau_end_ratio, tau_start_ratio), 1.0)

    progress = min(max(step / max(total_steps - 1, 1), 0.0), 1.0)
    current_ratio = tau_start_ratio + (tau_end_ratio - tau_start_ratio) * progress
    tau = norm_min + current_ratio * (norm_max - norm_min)
    return float(tau)


def _build_candidate_pool(library: FrequencyLibrary, tau_t: float) -> Tensor:
    """Return indices of AFL frequencies whose norm is below ``tau_t``."""

    mask = library.norms <= tau_t + 1e-12
    indices = torch.nonzero(mask, as_tuple=False).reshape(-1)
    if indices.numel() == 0:
        smallest = int(torch.argmin(library.norms).item())
        indices = torch.tensor([smallest], device=library.omega.device, dtype=torch.long)
    return indices


def _compute_candidate_lcf(
    candidate_freqs: Tensor,
    Y_ref: Optional[Tensor],
    Y: Optional[Tensor],
    config: Optional[Dict[str, Any]],
) -> Tensor:
    """Compute single-frequency LCF values on the candidate pool.

    If ``Y_ref`` and ``Y`` are available, use PD-CFD per-frequency loss.
    Otherwise fall back to a norm-based proxy to preserve backward compatibility.
    """

    if Y_ref is None or Y is None:
        return torch.linalg.norm(candidate_freqs, dim=1)

    lambda_p = float(config.get('lcf_lambda_p', 0.5)) if config is not None else 0.5
    alpha = float(config.get('lcf_alpha', 0.5)) if config is not None else 0.5
    outputs = pd_cfd_loss(Y_ref=Y_ref, Y=Y, freqs=candidate_freqs, lambda_p=lambda_p, alpha=alpha)
    return outputs.per_frequency_loss.detach()


def _compute_diversity_vector(candidate_freqs: Tensor, selected_freqs: Tensor, diversity_beta: float) -> Tensor:
    """Compute diversity penalties relative to already selected frequencies.

    We use an exponential penalty based on the maximum absolute cosine similarity.
    This yields values in ``(0, 1]`` and is stable even for tiny candidate pools.
    """

    num_candidates = int(candidate_freqs.shape[0])
    if selected_freqs.numel() == 0:
        return torch.ones(num_candidates, device=candidate_freqs.device, dtype=candidate_freqs.dtype)

    cand_norm = candidate_freqs / torch.clamp(torch.linalg.norm(candidate_freqs, dim=1, keepdim=True), min=1e-12)
    sel_norm = selected_freqs / torch.clamp(torch.linalg.norm(selected_freqs, dim=1, keepdim=True), min=1e-12)
    cosine = torch.abs(cand_norm @ sel_norm.T)
    max_corr = cosine.max(dim=1).values
    return torch.exp(-diversity_beta * (max_corr ** 2))


def select_progressive_frequencies(
    library: FrequencyLibrary,
    step: int,
    total_steps: int,
    Y_ref: Optional[Tensor] = None,
    Y: Optional[Tensor] = None,
    config: Optional[Dict[str, Any]] = None,
) -> PdasState:
    """Select frequencies using progressive discrepancy-aware sampling.

    Args:
        library:
            Frequency library from AFL.
        step:
            Current optimization step.
        total_steps:
            Planned number of optimization steps.
        Y_ref:
            Optional reference feature set with shape ``[N, d]``.
        Y:
            Optional current coreset feature set with shape ``[M, d]``.
        config:
            Optional PDAS-related config dictionary.

    Returns:
        A :class:`PdasState` containing the selected frequencies and debugging
        information over the candidate pool.
    """

    if total_steps <= 0:
        raise ValueError('total_steps must be positive')

    tau_t = _resolve_tau(library=library, step=step, total_steps=total_steps, config=config)
    candidate_indices = _build_candidate_pool(library=library, tau_t=tau_t)
    candidate_freqs = library.omega[candidate_indices]
    candidate_norms = library.norms[candidate_indices]

    if Y_ref is not None:
        Y_ref = Y_ref.to(device=candidate_freqs.device, dtype=candidate_freqs.dtype)
    if Y is not None:
        Y = Y.to(device=candidate_freqs.device, dtype=candidate_freqs.dtype)

    per_freq_lcf = _compute_candidate_lcf(candidate_freqs=candidate_freqs, Y_ref=Y_ref, Y=Y, config=config)
    diversity_beta = float(config.get('diversity_beta', 4.0)) if config is not None else 4.0
    selected_count = _resolve_sampling_count(library=library, step=step, total_steps=total_steps, config=config)
    selected_count = min(selected_count, int(candidate_indices.shape[0]))

    selected_local_indices = []
    selected_freqs_list = []
    current_diversity = torch.ones(candidate_freqs.shape[0], device=candidate_freqs.device, dtype=candidate_freqs.dtype)
    final_scores = torch.zeros(candidate_freqs.shape[0], device=candidate_freqs.device, dtype=candidate_freqs.dtype)

    available_mask = torch.ones(candidate_freqs.shape[0], device=candidate_freqs.device, dtype=torch.bool)
    for _ in range(selected_count):
        selected_freqs = (
            torch.stack(selected_freqs_list, dim=0)
            if len(selected_freqs_list) > 0
            else torch.empty((0, candidate_freqs.shape[1]), device=candidate_freqs.device, dtype=candidate_freqs.dtype)
        )
        current_diversity = _compute_diversity_vector(
            candidate_freqs=candidate_freqs,
            selected_freqs=selected_freqs,
            diversity_beta=diversity_beta,
        )
        scores = per_freq_lcf * current_diversity
        masked_scores = scores.clone()
        masked_scores[~available_mask] = -1.0
        next_local = int(torch.argmax(masked_scores).item())
        if not available_mask[next_local]:
            break
        selected_local_indices.append(next_local)
        selected_freqs_list.append(candidate_freqs[next_local])
        available_mask[next_local] = False
        final_scores = scores

    if len(selected_local_indices) == 0:
        selected_local_indices = [int(torch.argmax(per_freq_lcf).item())]
        final_scores = per_freq_lcf.clone()
        current_diversity = torch.ones_like(per_freq_lcf)

    selected_local_tensor = torch.tensor(selected_local_indices, device=candidate_indices.device, dtype=torch.long)
    selected_indices = candidate_indices[selected_local_tensor]
    selected_frequencies = library.omega[selected_indices]

    stats = {
        'tau_t': float(tau_t),
        'candidate_count': int(candidate_indices.shape[0]),
        'selected_count': int(selected_indices.shape[0]),
        'candidate_norm_min': float(candidate_norms.min().item()),
        'candidate_norm_max': float(candidate_norms.max().item()),
    }
    return PdasState(
        step=step,
        selected_indices=selected_indices,
        selected_frequencies=selected_frequencies,
        per_freq_lcf=per_freq_lcf,
        per_freq_diversity=current_diversity,
        final_scores=final_scores,
        candidate_pool_stats=stats,
    )
