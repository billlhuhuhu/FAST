"""Coreset initialization and first-pass FAST joint optimization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import json
import numpy as np
import torch
from torch import Tensor, nn

from src.graph.assign import hungarian_match
from src.losses.dpp import compute_dpp_loss
from src.losses.graph_losses import compute_graph_loss, compute_match_loss
from src.losses.pdcfd import pd_cfd_loss
from src.sampling.anisotropic_freq import FrequencyLibrary, build_anisotropic_frequency_library
from src.sampling.pdas import select_progressive_frequencies


InitMode = Literal["random_subset", "kmeans++"]


@dataclass
class CoresetInitResult:
    """Outputs of the continuous coreset initialization step."""

    Y: nn.Parameter
    init_indices: Tensor
    M: int


@dataclass
class DiscreteSubsetExport:
    """Outputs of final discrete subset export.

    Attributes:
        selected_indices:
            Final unique selected indices with shape ``[M]``.
        matched_indices:
            Raw matched indices before deduplication with shape ``[M]``.
        stats:
            Export statistics dictionary, including counts before and after deduplication.
    """

    selected_indices: Tensor
    matched_indices: Tensor
    stats: Dict[str, Any]


@dataclass
class JointOptimizationResult:
    """Outputs of the first-pass FAST joint optimization loop."""

    Y: Tensor
    matched_indices: Tensor
    selected_indices: Tensor
    export_stats: Dict[str, Any]
    logs: Dict[str, List[Any]]


def resolve_num_coreset_points(
    num_points: int,
    keep_ratio: Optional[float] = None,
    M: Optional[int] = None,
) -> int:
    """Resolve the coreset size ``M`` from either ``keep_ratio`` or ``M``."""

    if num_points <= 0:
        raise ValueError("num_points must be positive")
    if M is None and keep_ratio is None:
        raise ValueError("Either keep_ratio or M must be provided")
    if M is not None:
        if M <= 0 or M > num_points:
            raise ValueError("M must satisfy 1 <= M <= N")
        return int(M)

    assert keep_ratio is not None
    if keep_ratio <= 0.0 or keep_ratio > 1.0:
        raise ValueError("keep_ratio must satisfy 0 < keep_ratio <= 1")
    resolved = max(1, int(round(num_points * keep_ratio)))
    return min(resolved, num_points)


def _assert_feature_matrix(V_full: Tensor) -> None:
    """Validate the spectral feature matrix shape.

    Args:
        V_full:
            Tensor with shape ``[N, d]``.
    """

    if V_full.ndim != 2:
        raise ValueError(f"Expected V_full with shape [N, d], got {tuple(V_full.shape)}")
    if V_full.shape[0] <= 0 or V_full.shape[1] <= 0:
        raise ValueError("V_full must have positive shape in both dimensions")


def random_subset_init_indices(V_full: Tensor, M: int) -> Tensor:
    """Sample unique initialization indices uniformly at random."""

    _assert_feature_matrix(V_full)
    N = int(V_full.shape[0])
    if M <= 0 or M > N:
        raise ValueError("M must satisfy 1 <= M <= N")
    return torch.randperm(N, device=V_full.device)[:M]


def kmeans_plus_plus_init_indices(V_full: Tensor, M: int) -> Tensor:
    """Compute a simplified kmeans++-style initialization."""

    _assert_feature_matrix(V_full)
    N = int(V_full.shape[0])
    if M <= 0 or M > N:
        raise ValueError("M must satisfy 1 <= M <= N")

    first_index = torch.randint(low=0, high=N, size=(1,), device=V_full.device)
    selected = [int(first_index.item())]

    while len(selected) < M:
        selected_tensor = torch.tensor(selected, dtype=torch.long, device=V_full.device)
        centers = V_full[selected_tensor]
        distances = torch.cdist(V_full, centers, p=2.0) ** 2
        min_dist_sq = distances.min(dim=1).values
        min_dist_sq[selected_tensor] = 0.0

        total = min_dist_sq.sum()
        if not torch.isfinite(total) or float(total.item()) <= 0.0:
            remaining_mask = torch.ones(N, dtype=torch.bool, device=V_full.device)
            remaining_mask[selected_tensor] = False
            remaining = torch.arange(N, device=V_full.device)[remaining_mask]
            next_index = int(remaining[0].item())
        else:
            probs = min_dist_sq / total
            next_index = int(torch.multinomial(probs, num_samples=1).item())
            if next_index in selected:
                remaining_mask = torch.ones(N, dtype=torch.bool, device=V_full.device)
                remaining_mask[selected_tensor] = False
                remaining = torch.arange(N, device=V_full.device)[remaining_mask]
                next_index = int(remaining[0].item())
        selected.append(next_index)

    return torch.tensor(selected, dtype=torch.long, device=V_full.device)


def initialize_coreset_variable(
    V_full: Tensor,
    keep_ratio: Optional[float] = None,
    M: Optional[int] = None,
    init_mode: InitMode = "random_subset",
) -> CoresetInitResult:
    """Initialize the continuous coreset variable ``Y`` from ``V_full``."""

    _assert_feature_matrix(V_full)
    num_points, feature_dim = int(V_full.shape[0]), int(V_full.shape[1])
    resolved_M = resolve_num_coreset_points(num_points=num_points, keep_ratio=keep_ratio, M=M)

    if init_mode == "random_subset":
        init_indices = random_subset_init_indices(V_full, M=resolved_M)
    elif init_mode == "kmeans++":
        init_indices = kmeans_plus_plus_init_indices(V_full, M=resolved_M)
    else:
        raise ValueError(f"Unsupported init_mode: {init_mode}")

    if init_indices.ndim != 1 or init_indices.shape[0] != resolved_M:
        raise AssertionError("init_indices must have shape [M]")
    if torch.unique(init_indices).shape[0] != resolved_M:
        raise AssertionError("Initialization indices must be unique")
    if int(init_indices.min().item()) < 0 or int(init_indices.max().item()) >= num_points:
        raise AssertionError("Initialization indices are out of bounds")

    Y_tensor = V_full[init_indices].clone()
    if Y_tensor.shape != (resolved_M, feature_dim):
        raise AssertionError("Initialized Y has an unexpected shape")

    Y = nn.Parameter(Y_tensor)
    return CoresetInitResult(Y=Y, init_indices=init_indices, M=resolved_M)


def _coerce_laplacian_type(L_sym: Any, device: torch.device, dtype: torch.dtype) -> Any:
    """Return a laplacian object acceptable by the graph loss helper."""

    if isinstance(L_sym, torch.Tensor):
        return L_sym.to(device=device, dtype=dtype)
    return L_sym


def _extract_degree_from_config(config: Dict[str, Any], N: int, device: torch.device, dtype: torch.dtype) -> Optional[Tensor]:
    """Get optional degree information from the config dictionary."""

    degree = config.get("degree", None)
    if degree is None:
        return None
    if isinstance(degree, Tensor):
        tensor = degree.to(device=device, dtype=dtype)
    else:
        tensor = torch.as_tensor(degree, device=device, dtype=dtype)
    if tensor.ndim != 1 or tensor.shape[0] != N:
        raise ValueError("config['degree'] must have shape [N]")
    return tensor


def _extract_assignment_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract assignment configuration while keeping backward compatibility."""

    assignment_cfg = config.get("assignment", {})
    return {
        "mode": assignment_cfg.get("mode", "auto"),
        "prune_topk": assignment_cfg.get("prune_topk", None),
        "auto_threshold": int(assignment_cfg.get("auto_threshold", 20000)),
        "large_cost_scale": float(assignment_cfg.get("large_cost_scale", 1e6)),
        "eps": float(assignment_cfg.get("eps", 1e-8)),
    }


def _build_frequency_library_from_config(
    V_full: Tensor,
    Y_current: Optional[Tensor],
    config: Dict[str, Any],
) -> FrequencyLibrary:
    """Create the AFL frequency library for PDAS/PD-CFD.

    The function accepts either a flat config dictionary or a nested
    ``config['sampling']`` dictionary and keeps the current pipeline compatible.
    """

    sampling_cfg = config.get("sampling", config)
    num_frequencies = int(sampling_cfg.get("num_frequencies", 64))
    max_frequency_norm = float(sampling_cfg.get("max_frequency_norm", 8.0))
    band_ranges = sampling_cfg.get("band_ranges", None)
    band_sample_counts = sampling_cfg.get("band_sample_counts", None)
    if band_sample_counts is not None:
        resolved_sum = int(sum(int(v) for v in band_sample_counts.values()))
        if resolved_sum != num_frequencies:
            band_sample_counts = None
    candidate_scales = sampling_cfg.get("candidate_scales", None)
    scaling_search_steps = sampling_cfg.get("scaling_search_steps", None)
    lambda_p = float(sampling_cfg.get("lcf_lambda_p", sampling_cfg.get("lambda_p", 0.5)))
    alpha = float(sampling_cfg.get("lcf_alpha", sampling_cfg.get("alpha", 0.5)))

    return build_anisotropic_frequency_library(
        feature_dim=int(V_full.shape[1]),
        num_frequencies=num_frequencies,
        max_norm=max_frequency_norm,
        device=V_full.device,
        Y_ref=V_full,
        Y_current=None if Y_current is None else Y_current.detach(),
        band_ranges=band_ranges,
        band_sample_counts=band_sample_counts,
        candidate_scales=candidate_scales,
        scaling_search_steps=scaling_search_steps,
        lambda_p=lambda_p,
        alpha=alpha,
    )


def _init_logs() -> Dict[str, List[Any]]:
    """Initialize the joint-optimization log dictionary."""

    return {
        "step": [],
        "loss_total": [],
        "loss_match": [],
        "loss_graph": [],
        "loss_div": [],
        "loss_pdcfd": [],
        "matching_cost": [],
        "matched_min": [],
        "matched_max": [],
        "matched_head": [],
        "num_freqs": [],
        "tau_t": [],
        "candidate_count": [],
        "assignment_mode": [],
        "assignment_build_time_sec": [],
        "assignment_matching_time_sec": [],
    }


def export_selected_subset(
    matched_indices: Tensor,
    Y: Tensor,
    V_full: Tensor,
    M: Optional[int] = None,
) -> DiscreteSubsetExport:
    """Export a final unique discrete subset from matched indices.

    Args:
        matched_indices:
            Raw matched indices with shape ``[M]``.
        Y:
            Final optimized coreset variable with shape ``[M, d]``.
        V_full:
            Full candidate feature matrix with shape ``[N, d]``.
        M:
            Optional target subset size. Defaults to ``Y.shape[0]``.

    Returns:
        A :class:`DiscreteSubsetExport` containing unique selected indices and stats.
    """

    _assert_feature_matrix(V_full)
    if Y.ndim != 2:
        raise ValueError("Y must have shape [M, d]")
    if matched_indices.ndim != 1:
        raise ValueError("matched_indices must have shape [M]")

    target_M = int(Y.shape[0] if M is None else M)
    if target_M <= 0:
        raise ValueError("M must be positive")
    if matched_indices.shape[0] != target_M:
        raise ValueError("matched_indices length must match target M")
    if Y.shape[0] != target_M:
        raise ValueError("Y first dimension must match target M")

    N = int(V_full.shape[0])
    if int(matched_indices.min().item()) < 0 or int(matched_indices.max().item()) >= N:
        raise AssertionError("matched_indices are out of bounds")

    raw_list = [int(x) for x in matched_indices.detach().cpu().tolist()]
    unique_list: List[int] = []
    seen = set()
    for idx in raw_list:
        if idx not in seen:
            unique_list.append(idx)
            seen.add(idx)

    dedup_count = len(unique_list)

    if dedup_count < target_M:
        distances = torch.cdist(Y.detach(), V_full.detach(), p=2.0) ** 2
        ranked = torch.argsort(distances, dim=1)
        for row in range(target_M):
            for cand in ranked[row].detach().cpu().tolist():
                cand_i = int(cand)
                if cand_i not in seen:
                    unique_list.append(cand_i)
                    seen.add(cand_i)
                    break
            if len(unique_list) >= target_M:
                break

    if len(unique_list) < target_M:
        for cand_i in range(N):
            if cand_i not in seen:
                unique_list.append(cand_i)
                seen.add(cand_i)
            if len(unique_list) >= target_M:
                break

    selected_indices = torch.tensor(unique_list[:target_M], dtype=torch.long, device=matched_indices.device)

    if selected_indices.shape[0] != target_M:
        raise AssertionError("selected_indices length is incorrect")
    if torch.unique(selected_indices).shape[0] != target_M:
        raise AssertionError("selected_indices must be unique")
    if int(selected_indices.min().item()) < 0 or int(selected_indices.max().item()) >= N:
        raise AssertionError("selected_indices are out of bounds")

    stats: Dict[str, Any] = {
        "target_M": target_M,
        "raw_match_count": int(matched_indices.shape[0]),
        "unique_before_fill": int(dedup_count),
        "filled_count": int(target_M - dedup_count),
        "final_unique_count": int(selected_indices.shape[0]),
    }
    return DiscreteSubsetExport(
        selected_indices=selected_indices,
        matched_indices=matched_indices.detach().clone(),
        stats=stats,
    )


def save_selected_indices(
    selected_indices: Tensor,
    output_path: str | Path,
) -> Path:
    """Save selected indices as ``.npy`` or ``.json``.

    Args:
        selected_indices:
            Tensor with shape ``[M]``.
        output_path:
            Destination file path ending in ``.npy`` or ``.json``.

    Returns:
        The resolved output path.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = selected_indices.detach().cpu().numpy().astype(np.int64)

    if path.suffix.lower() == ".npy":
        np.save(path, values)
    elif path.suffix.lower() == ".json":
        path.write_text(json.dumps(values.tolist(), ensure_ascii=True), encoding="utf-8")
    else:
        raise ValueError("output_path must end with .npy or .json")
    return path


def optimize_coreset(
    V_full: Tensor,
    L_sym: Any,
    config: Dict[str, Any],
) -> JointOptimizationResult:
    """Run the first-pass FAST joint optimization loop.

    Args:
        V_full:
            Spectral feature tensor with shape ``[N, d]``.
        L_sym:
            Normalized Laplacian with shape ``[N, N]``.
        config:
            Plain configuration dictionary. Expected keys include:
            ``keep_ratio`` or ``M``, ``init_mode``, ``iterations``, ``lr``,
            ``lambda_match``, ``lambda_graph``, ``lambda_div``,
            ``lambda_pdcfd``, ``num_frequencies``, ``max_frequency_norm``,
            ``lambda_p``, ``alpha``, ``rff_dim``, ``dpp_sigma``, ``dpp_delta``.

    Returns:
        A :class:`JointOptimizationResult` with optimized ``Y``, final matched
        indices, and a per-iteration log dictionary.
    """

    _assert_feature_matrix(V_full)
    N, d = int(V_full.shape[0]), int(V_full.shape[1])
    device = V_full.device
    dtype = V_full.dtype

    sampling_cfg = config.get("sampling", config)

    keep_ratio = config.get("keep_ratio", None)
    M = config.get("M", None)
    init_mode = config.get("init_mode", "random_subset")
    iterations = int(config.get("iterations", config.get("steps", 5)))
    lr = float(config.get("lr", 1e-3))
    lambda_match = float(config.get("lambda_match", 1.0))
    lambda_graph = float(config.get("lambda_graph", 0.1))
    lambda_div = float(config.get("lambda_div", 0.1))
    lambda_pdcfd = float(config.get("lambda_pdcfd", 1.0))
    lambda_p = float(sampling_cfg.get("lambda_p", config.get("lambda_p", 1.0)))
    alpha = float(sampling_cfg.get("alpha", config.get("alpha", 1.0)))
    rff_dim = int(config.get("rff_dim", 128))
    dpp_sigma = float(config.get("dpp_sigma", 1.0))
    dpp_delta = float(config.get("dpp_delta", 1e-6))
    verbose = bool(config.get("verbose", False))
    log_every = max(1, int(config.get("log_every", 10)))

    if iterations <= 0:
        raise ValueError("iterations must be positive")

    init_result = initialize_coreset_variable(
        V_full=V_full,
        keep_ratio=keep_ratio,
        M=M,
        init_mode=init_mode,
    )
    Y = init_result.Y
    if Y.shape[1] != d:
        raise AssertionError("Initialized Y has incompatible feature dimension")

    degree = _extract_degree_from_config(config=config, N=N, device=device, dtype=dtype)
    assignment_kwargs = _extract_assignment_kwargs(config=config)
    laplacian = _coerce_laplacian_type(L_sym=L_sym, device=device, dtype=dtype)
    rebuild_afl_each_iter = bool(sampling_cfg.get("rebuild_afl_each_iter", True))
    frequency_library = _build_frequency_library_from_config(V_full=V_full, Y_current=Y.detach(), config=config)

    optimizer = torch.optim.Adam([Y], lr=lr)
    logs = _init_logs()

    for step in range(iterations):
        if Y.ndim != 2 or Y.shape[1] != d:
            raise AssertionError("Y must keep shape [M, d] throughout optimization")

        assignment = hungarian_match(Y=Y, V_full=V_full, degree=degree, **assignment_kwargs)
        matched_indices = assignment.matched_indices
        if matched_indices.shape[0] != Y.shape[0]:
            raise AssertionError("matched_indices must have shape [M]")

        loss_match = compute_match_loss(Y=Y, V_full=V_full, matched_indices=matched_indices)
        loss_graph = compute_graph_loss(Y=Y, L_sym=laplacian, matched_indices=matched_indices)
        loss_div_result = compute_dpp_loss(Y=Y, rff_dim=rff_dim, sigma=dpp_sigma, delta=dpp_delta)
        loss_div = loss_div_result.loss

        if rebuild_afl_each_iter and step > 0:
            frequency_library = _build_frequency_library_from_config(V_full=V_full, Y_current=Y.detach(), config=config)

        pdas_state = select_progressive_frequencies(
            library=frequency_library,
            step=step,
            total_steps=iterations,
            Y_ref=V_full,
            Y=Y,
            config=sampling_cfg,
        )
        if pdas_state.selected_frequencies.ndim != 2 or pdas_state.selected_frequencies.shape[1] != d:
            raise AssertionError("Selected frequencies must have shape [K, d]")

        pdcfd_outputs = pd_cfd_loss(
            Y_ref=V_full,
            Y=Y,
            freqs=pdas_state.selected_frequencies.to(device=device, dtype=dtype),
            lambda_p=lambda_p,
            alpha=alpha,
        )
        loss_pdcfd = pdcfd_outputs.loss

        total_loss = (
            lambda_match * loss_match
            + lambda_graph * loss_graph
            + lambda_div * loss_div
            + lambda_pdcfd * loss_pdcfd
        )
        total_loss = torch.real(total_loss)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        logs["step"].append(step)
        logs["loss_total"].append(float(total_loss.detach().cpu().item()))
        logs["loss_match"].append(float(loss_match.detach().cpu().item()))
        logs["loss_graph"].append(float(loss_graph.detach().cpu().item()))
        logs["loss_div"].append(float(loss_div.detach().cpu().item()))
        logs["loss_pdcfd"].append(float(loss_pdcfd.detach().cpu().item()))
        logs["matching_cost"].append(float(assignment.matching_cost))
        logs["matched_min"].append(int(matched_indices.min().detach().cpu().item()))
        logs["matched_max"].append(int(matched_indices.max().detach().cpu().item()))
        logs["matched_head"].append(matched_indices[: min(5, matched_indices.shape[0])].detach().cpu().tolist())
        logs["num_freqs"].append(int(pdas_state.selected_frequencies.shape[0]))
        logs["tau_t"].append(float(pdas_state.candidate_pool_stats.get("tau_t", 0.0)))
        logs["candidate_count"].append(int(pdas_state.candidate_pool_stats.get("candidate_count", pdas_state.selected_frequencies.shape[0])))
        logs["assignment_mode"].append(str(assignment.mode))
        logs["assignment_build_time_sec"].append(float(assignment.candidate_stats.get("cost_build_time_sec", 0.0)))
        logs["assignment_matching_time_sec"].append(float(assignment.candidate_stats.get("matching_time_sec", 0.0)))

        should_print = verbose and (step == 0 or (step + 1) % log_every == 0 or step == iterations - 1)
        if should_print:
            print(
                "[FAST] iter={step}/{total} total={loss_total:.6f} match={loss_match:.6f} graph={loss_graph:.6f} div={loss_div:.6f} pdcfd={loss_pdcfd:.6f} tau={tau:.6f} cand={cand} assign={assign}".format(
                    step=step + 1,
                    total=iterations,
                    loss_total=logs["loss_total"][-1],
                    loss_match=logs["loss_match"][-1],
                    loss_graph=logs["loss_graph"][-1],
                    loss_div=logs["loss_div"][-1],
                    loss_pdcfd=logs["loss_pdcfd"][-1],
                    tau=logs["tau_t"][-1],
                    cand=logs["candidate_count"][-1],
                    assign=logs["assignment_mode"][-1],
                ),
                flush=True,
            )

    final_assignment = hungarian_match(Y=Y.detach(), V_full=V_full, degree=degree, **assignment_kwargs)
    exported = export_selected_subset(
        matched_indices=final_assignment.matched_indices.detach(),
        Y=Y.detach(),
        V_full=V_full,
        M=Y.shape[0],
    )
    return JointOptimizationResult(
        Y=Y.detach(),
        matched_indices=final_assignment.matched_indices.detach(),
        selected_indices=exported.selected_indices.detach(),
        export_stats=exported.stats,
        logs=logs,
    )
