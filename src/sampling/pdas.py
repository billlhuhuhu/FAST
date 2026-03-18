"""Progressive discrepancy-aware sampling schedule placeholders."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from src.sampling.anisotropic_freq import FrequencyLibrary


@dataclass
class PdasState:
    """State bundle for the progressive frequency schedule."""

    step: int
    selected_indices: Tensor
    selected_frequencies: Tensor


def select_progressive_frequencies(
    library: FrequencyLibrary,
    step: int,
    total_steps: int,
) -> PdasState:
    """Select a progressively expanding set of frequencies.

    Args:
        library: Frequency library with ``F`` candidates.
        step: Current optimization step.
        total_steps: Planned number of optimization steps.

    Returns:
        A :class:`PdasState` with selected rows from the library.

    TODO:
        - Add discrepancy-aware ranking instead of norm-only progression.
    """

    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    fraction = min(max((step + 1) / total_steps, 0.0), 1.0)
    count = max(1, int(round(fraction * library.omega.shape[0])))
    order = torch.argsort(library.norms)
    selected_indices = order[:count]
    selected_frequencies = library.omega[selected_indices]
    return PdasState(step=step, selected_indices=selected_indices, selected_frequencies=selected_frequencies)
