"""Minimal coreset optimization loop for the FAST scaffold."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from src.graph.assign import compute_cost_matrix, greedy_unique_assignment
from src.losses.dpp import dpp_diversity_loss
from src.losses.graph_losses import topology_loss_bundle
from src.losses.pdcfd import pd_cfd_loss
from src.sampling.anisotropic_freq import FrequencyLibrary
from src.sampling.pdas import select_progressive_frequencies


@dataclass
class CoresetOptimizationResult:
    """Outputs from the minimal optimization loop."""

    proxies: Tensor
    subset_indices: Tensor
    history: list[dict[str, float]]


def initialize_proxies(reference: Tensor, keep_ratio: float) -> Tensor:
    """Initialize proxies from a subset of real points.

    Args:
        reference: Tensor of shape ``[N, D]``.
        keep_ratio: Fraction in ``(0, 1]``.

    Returns:
        Tensor of shape ``[M, D]`` where ``M = max(1, round(N * keep_ratio))``.
    """

    num_points = reference.shape[0]
    num_keep = max(1, int(round(num_points * keep_ratio)))
    indices = torch.randperm(num_points, device=reference.device)[:num_keep]
    return reference[indices].clone()


def optimize_coreset(
    reference: Tensor,
    frequency_library: FrequencyLibrary,
    keep_ratio: float,
    num_steps: int,
    lr: float,
    lambda_div: float = 0.1,
    lambda_align: float = 1.0,
) -> CoresetOptimizationResult:
    """Run a very small, shape-safe coreset optimization loop.

    Args:
        reference: Tensor of shape ``[N, D]``.
        frequency_library: Candidate frequencies.
        keep_ratio: Coreset keep ratio.
        num_steps: Number of optimization steps.
        lr: Optimizer learning rate.
        lambda_div: Weight for diversity loss.
        lambda_align: Weight for alignment loss.

    Returns:
        A :class:`CoresetOptimizationResult`.

    TODO:
        - Add graph Laplacian terms.
        - Add paper-faithful assignment and topology-aware costs.
        - Support logging tensors and richer debug diagnostics.
    """

    proxies = initialize_proxies(reference, keep_ratio).detach().clone()
    proxies.requires_grad_(True)
    optimizer = torch.optim.Adam([proxies], lr=lr)
    history: list[dict[str, float]] = []

    for step in range(num_steps):
        pdas_state = select_progressive_frequencies(frequency_library, step=step, total_steps=num_steps)
        pdcfd_outputs = pd_cfd_loss(reference=reference, proxies=proxies, frequencies=pdas_state.selected_frequencies)

        with torch.no_grad():
            cost = compute_cost_matrix(proxies.detach(), reference)
            subset_indices = greedy_unique_assignment(cost)
            matched_points = reference[subset_indices]

        graph_terms = topology_loss_bundle(proxies=proxies, matched_points=matched_points)
        div_loss = dpp_diversity_loss(proxies)
        total_loss = pdcfd_outputs.loss + lambda_div * div_loss + lambda_align * graph_terms["align"] + graph_terms["graph"]

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        history.append(
            {
                "step": float(step),
                "loss_total": float(total_loss.detach().cpu().item()),
                "loss_pdcfd": float(pdcfd_outputs.loss.detach().cpu().item()),
                "loss_div": float(div_loss.detach().cpu().item()),
                "loss_align": float(graph_terms["align"].detach().cpu().item()),
            }
        )

    with torch.no_grad():
        final_cost = compute_cost_matrix(proxies.detach(), reference)
        final_subset_indices = greedy_unique_assignment(final_cost)

    return CoresetOptimizationResult(
        proxies=proxies.detach(),
        subset_indices=final_subset_indices,
        history=history,
    )
