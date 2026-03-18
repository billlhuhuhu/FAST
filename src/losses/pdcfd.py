"""Phase-decoupled CFD placeholders for the FAST scaffold."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class PdCfdOutputs:
    """Outputs of the placeholder PD-CFD computation.

    Attributes:
        loss: Scalar tensor.
        ref_cf: Reference characteristic function values with shape ``[F]``.
        proxy_cf: Proxy characteristic function values with shape ``[F]``.
    """

    loss: Tensor
    ref_cf: Tensor
    proxy_cf: Tensor


def empirical_characteristic_function(features: Tensor, frequencies: Tensor) -> Tensor:
    """Compute a basic empirical characteristic function.

    Args:
        features: Tensor of shape ``[N, D]``.
        frequencies: Tensor of shape ``[F, D]``.

    Returns:
        Complex tensor of shape ``[F]``.
    """

    phases = features @ frequencies.T
    return torch.exp(1j * phases).mean(dim=0)


def pd_cfd_loss(reference: Tensor, proxies: Tensor, frequencies: Tensor) -> PdCfdOutputs:
    """Compute a minimal PD-CFD-style loss.

    This is intentionally simplified. It only aims to keep interfaces, shapes,
    and complex-number flow correct for later refinement.

    Args:
        reference: Tensor of shape ``[N, D]``.
        proxies: Tensor of shape ``[M, D]``.
        frequencies: Tensor of shape ``[F, D]``.

    Returns:
        A :class:`PdCfdOutputs` bundle.

    TODO:
        - Split amplitude and phase according to the paper.
        - Add attenuation and frequency scheduling details from FAST.
    """

    ref_cf = empirical_characteristic_function(reference, frequencies)
    proxy_cf = empirical_characteristic_function(proxies, frequencies)
    loss = torch.mean(torch.abs(ref_cf - proxy_cf) ** 2).real
    return PdCfdOutputs(loss=loss, ref_cf=ref_cf, proxy_cf=proxy_cf)
