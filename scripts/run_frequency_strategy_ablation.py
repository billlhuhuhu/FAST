"""Lightweight ablation runner for frequency-selection strategies.

Strategies:
- progressive_discrepancy_aware
- non_progressive_discrepancy_aware
- progressive_uniform
- non_progressive_uniform
- collinear_selection

Outputs per-step main / total losses for later curve plotting.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import Tensor

from src.data.cifar import prepare_cifar10_data
from src.graph.assign import hungarian_match
from src.graph.knn_graph import build_multiscale_knn_graph
from src.graph.spectral import spectral_decomposition
from src.losses.dpp import compute_dpp_loss
from src.losses.graph_losses import compute_graph_loss, compute_match_loss
from src.losses.pdcfd import pd_cfd_loss
from src.optimize.optimize_coreset import export_selected_subset, initialize_coreset_variable
from src.sampling.anisotropic_freq import FrequencyLibrary, build_anisotropic_frequency_library
from src.sampling.pdas import select_progressive_frequencies
from src.utils.io import ensure_dir, load_yaml
from src.utils.seed import set_seed


def _build_runtime(config: Dict[str, Any], debug: bool) -> Dict[str, Any]:
    data_cfg = config.get('data', {})
    coreset_cfg = config.get('coreset', {})
    graph_cfg = config.get('graph', {})
    sampling_cfg = copy.deepcopy(config.get('sampling', {}))
    assignment_cfg = copy.deepcopy(config.get('assignment', {}))
    optimize_cfg = config.get('optimize', {})

    runtime = {
        'seed': int(config.get('seed', 42)),
        'data_root': str(data_cfg.get('root', './data')),
        'flatten': bool(data_cfg.get('flatten', True)),
        'pca_dim': int(data_cfg.get('pca_dim', 128)),
        'manifold_dim': int(data_cfg.get('manifold_dim', 32)),
        'keep_ratio': float(coreset_cfg.get('keep_ratio', 0.1)),
        'init_mode': str(coreset_cfg.get('init_mode', 'random_subset')),
        'k_list': list(graph_cfg.get('k_list', [5, 10, 20])),
        'sampling_cfg': sampling_cfg,
        'assignment_cfg': assignment_cfg,
        'iterations': int(optimize_cfg.get('iterations', optimize_cfg.get('steps', 400))),
        'opt_lr': float(optimize_cfg.get('lr', 1e-3)),
        'lambda_match': float(optimize_cfg.get('lambda_match', optimize_cfg.get('lambda_align', 1.0))),
        'lambda_graph': float(optimize_cfg.get('lambda_graph', 0.1)),
        'lambda_div': float(optimize_cfg.get('lambda_div', 0.1)),
        'lambda_main': float(optimize_cfg.get('lambda_pdcfd', 1.0)),
        'output_dir': Path(config.get('output_dir', './outputs')) / 'ablations' / 'frequency_strategy',
        'train_num_samples': None,
        'test_num_samples': None,
    }
    if debug:
        runtime['pca_dim'] = min(runtime['pca_dim'], 32)
        runtime['manifold_dim'] = min(runtime['manifold_dim'], 16)
        runtime['iterations'] = min(runtime['iterations'], 5)
        runtime['train_num_samples'] = 256
        runtime['test_num_samples'] = 64
        runtime['sampling_cfg']['num_frequencies'] = min(int(runtime['sampling_cfg'].get('num_frequencies', 64)), 32)
        runtime['sampling_cfg']['pdas_frequencies_per_iter'] = min(int(runtime['sampling_cfg'].get('pdas_frequencies_per_iter', 64)), 16)
    return runtime


def _prepare_spectral_features(runtime: Dict[str, Any]) -> tuple[Tensor, Any]:
    prepared = prepare_cifar10_data(
        root=runtime['data_root'],
        flatten=runtime['flatten'],
        standardize=True,
        pca_dim=runtime['pca_dim'],
        train_num_samples=runtime['train_num_samples'],
        test_num_samples=runtime['test_num_samples'],
        download=True,
    )
    graph = build_multiscale_knn_graph(prepared.train.X, k_list=runtime['k_list'])
    spectral = spectral_decomposition(graph.combined_graph, d=runtime['manifold_dim'])
    return torch.from_numpy(spectral.eigenvectors).to(dtype=torch.float32), spectral.laplacian


def _build_library(V_full: Tensor, Y_current: Tensor, sampling_cfg: Dict[str, Any]) -> FrequencyLibrary:
    num_frequencies = int(sampling_cfg.get('num_frequencies', 64))
    band_sample_counts = sampling_cfg.get('band_sample_counts', None)
    if band_sample_counts is not None:
        resolved_sum = int(sum(int(v) for v in band_sample_counts.values()))
        if resolved_sum != num_frequencies:
            band_sample_counts = None
    return build_anisotropic_frequency_library(
        feature_dim=int(V_full.shape[1]),
        num_frequencies=num_frequencies,
        max_norm=float(sampling_cfg.get('max_frequency_norm', 8.0)),
        device=V_full.device,
        Y_ref=V_full,
        Y_current=Y_current.detach(),
        band_ranges=sampling_cfg.get('band_ranges', None),
        band_sample_counts=band_sample_counts,
        candidate_scales=sampling_cfg.get('candidate_scales', None),
        scaling_search_steps=sampling_cfg.get('scaling_search_steps', None),
        lambda_p=float(sampling_cfg.get('lcf_lambda_p', 0.5)),
        alpha=float(sampling_cfg.get('lcf_alpha', 0.5)),
    )


def _resolve_tau(norms: Tensor, step: int, total_steps: int, cfg: Dict[str, Any], progressive: bool) -> float:
    if not progressive:
        return float(norms.max().item())
    tau_start_ratio = float(cfg.get('tau_start_ratio', 0.2))
    tau_end_ratio = float(cfg.get('tau_end_ratio', 1.0))
    progress = min(max(step / max(total_steps - 1, 1), 0.0), 1.0)
    ratio = tau_start_ratio + (tau_end_ratio - tau_start_ratio) * progress
    nmin = float(norms.min().item())
    nmax = float(norms.max().item())
    return nmin + ratio * (nmax - nmin)


def _candidate_pool(library: FrequencyLibrary, tau_t: float) -> Tensor:
    idx = torch.nonzero(library.norms <= tau_t + 1e-12, as_tuple=False).reshape(-1)
    if idx.numel() == 0:
        idx = torch.tensor([int(torch.argmin(library.norms).item())], device=library.omega.device, dtype=torch.long)
    return idx


def _uniform_select(pool_indices: Tensor, library: FrequencyLibrary, count: int, generator: torch.Generator) -> tuple[Tensor, Tensor]:
    count = min(count, int(pool_indices.shape[0]))
    perm = torch.randperm(int(pool_indices.shape[0]), generator=generator, device=pool_indices.device)[:count]
    selected_indices = pool_indices[perm]
    return selected_indices, library.omega[selected_indices]


def _collinear_select(pool_indices: Tensor, library: FrequencyLibrary, Y_ref: Tensor, count: int) -> tuple[Tensor, Tensor]:
    count = min(count, int(pool_indices.shape[0]))
    candidate_freqs = library.omega[pool_indices]
    mean_dir = Y_ref.mean(dim=0)
    mean_dir = mean_dir / torch.clamp(torch.linalg.norm(mean_dir), min=1e-12)
    cand_norm = candidate_freqs / torch.clamp(torch.linalg.norm(candidate_freqs, dim=1, keepdim=True), min=1e-12)
    scores = torch.abs(cand_norm @ mean_dir)
    top = torch.topk(scores, k=count, largest=True).indices
    selected_indices = pool_indices[top]
    return selected_indices, library.omega[selected_indices]


def _select_strategy(strategy: str, library: FrequencyLibrary, step: int, total_steps: int, Y_ref: Tensor, Y: Tensor, cfg: Dict[str, Any], generator: torch.Generator):
    count = min(int(cfg.get('pdas_frequencies_per_iter', 16)), int(library.omega.shape[0]))
    if strategy == 'progressive_discrepancy_aware':
        state = select_progressive_frequencies(library, step, total_steps, Y_ref=Y_ref, Y=Y, config=cfg)
        return state.selected_indices, state.selected_frequencies, state.candidate_pool_stats
    if strategy == 'non_progressive_discrepancy_aware':
        local_cfg = dict(cfg)
        local_cfg['tau_start_ratio'] = 1.0
        local_cfg['tau_end_ratio'] = 1.0
        state = select_progressive_frequencies(library, step, total_steps, Y_ref=Y_ref, Y=Y, config=local_cfg)
        return state.selected_indices, state.selected_frequencies, state.candidate_pool_stats

    progressive = strategy == 'progressive_uniform'
    tau_t = _resolve_tau(library.norms, step, total_steps, cfg, progressive=progressive)
    pool_indices = _candidate_pool(library, tau_t)

    if strategy in {'progressive_uniform', 'non_progressive_uniform'}:
        selected_indices, selected_freqs = _uniform_select(pool_indices, library, count, generator)
    elif strategy == 'collinear_selection':
        selected_indices, selected_freqs = _collinear_select(pool_indices, library, Y_ref, count)
    else:
        raise ValueError(f'Unsupported strategy: {strategy}')

    return selected_indices, selected_freqs, {
        'tau_t': float(tau_t),
        'candidate_count': int(pool_indices.shape[0]),
        'selected_count': int(selected_indices.shape[0]),
    }


def run_strategy(strategy: str, runtime: Dict[str, Any], V_full: Tensor, L_sym: Any, output_root: Path) -> Dict[str, Any]:
    set_seed(runtime['seed'])
    run_dir = ensure_dir(output_root / strategy)
    generator = torch.Generator(device=V_full.device).manual_seed(runtime['seed'])
    init = initialize_coreset_variable(V_full=V_full, keep_ratio=runtime['keep_ratio'], init_mode=runtime['init_mode'])
    Y = init.Y
    optimizer = torch.optim.Adam([Y], lr=runtime['opt_lr'])
    logs = {'step': [], 'loss_total': [], 'loss_main': [], 'num_freqs': [], 'tau_t': []}

    for step in range(runtime['iterations']):
        assignment = hungarian_match(Y=Y, V_full=V_full, degree=None, **runtime['assignment_cfg'])
        loss_match = compute_match_loss(Y=Y, V_full=V_full, matched_indices=assignment.matched_indices)
        loss_graph = compute_graph_loss(Y=Y, L_sym=L_sym, matched_indices=assignment.matched_indices)
        loss_div = compute_dpp_loss(Y=Y, rff_dim=32, sigma=1.0, delta=1e-5).loss
        library = _build_library(V_full, Y.detach(), runtime['sampling_cfg'])
        _selected_indices, selected_freqs, stats = _select_strategy(strategy, library, step, runtime['iterations'], V_full, Y, runtime['sampling_cfg'], generator)
        main_loss = pd_cfd_loss(
            Y_ref=V_full,
            Y=Y,
            freqs=selected_freqs,
            lambda_p=float(runtime['sampling_cfg'].get('lcf_lambda_p', 0.5)),
            alpha=float(runtime['sampling_cfg'].get('lcf_alpha', 0.5)),
        ).loss
        total_loss = runtime['lambda_main'] * main_loss + runtime['lambda_match'] * loss_match + runtime['lambda_graph'] * loss_graph + runtime['lambda_div'] * loss_div
        total_loss = torch.real(total_loss)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        logs['step'].append(step)
        logs['loss_total'].append(float(total_loss.detach().cpu().item()))
        logs['loss_main'].append(float(main_loss.detach().cpu().item()))
        logs['num_freqs'].append(int(selected_freqs.shape[0]))
        logs['tau_t'].append(float(stats.get('tau_t', float(library.norms.max().item()))))
        print(f"strategy={strategy} step={step} total={logs['loss_total'][-1]:.6f} main={logs['loss_main'][-1]:.6f}")

    exported = export_selected_subset(hungarian_match(Y.detach(), V_full, degree=None, **runtime['assignment_cfg']).matched_indices, Y.detach(), V_full, M=Y.shape[0])
    result = {
        'strategy': strategy,
        'selected_count': int(exported.selected_indices.shape[0]),
        'logs': logs,
        'main_loss_summary': {
            'first': logs['loss_main'][0],
            'last': logs['loss_main'][-1],
            'min': min(logs['loss_main']),
            'max': max(logs['loss_main']),
        },
    }
    (run_dir / 'result.json').write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding='utf-8')
    return result


def _save_combined_csv(results: List[Dict[str, Any]], output_root: Path) -> Path:
    csv_path = output_root / 'curves.csv'
    with csv_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['strategy', 'step', 'loss_total', 'loss_main', 'num_freqs', 'tau_t'])
        for result in results:
            strategy = result['strategy']
            logs = result['logs']
            for i, step in enumerate(logs['step']):
                writer.writerow([strategy, step, logs['loss_total'][i], logs['loss_main'][i], logs['num_freqs'][i], logs['tau_t'][i]])
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description='Run lightweight frequency-strategy ablation')
    parser.add_argument('--config', type=str, default=str(PROJECT_ROOT / 'configs' / 'cifar10_fast.yaml'))
    parser.add_argument('--debug', action='store_true')
    parser.add_argument(
        '--strategies',
        nargs='+',
        default=[
            'progressive_discrepancy_aware',
            'non_progressive_discrepancy_aware',
            'progressive_uniform',
            'non_progressive_uniform',
            'collinear_selection',
        ],
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    runtime = _build_runtime(config, debug=args.debug)
    output_root = ensure_dir(runtime['output_dir'] / ('debug' if args.debug else 'full'))

    print('[ablation] preparing shared CIFAR-10 spectral features')
    V_full, L_sym = _prepare_spectral_features(runtime)

    results = [run_strategy(strategy, runtime, V_full, L_sym, output_root) for strategy in args.strategies]
    summary = {'strategies': results}
    (output_root / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding='utf-8')
    csv_path = _save_combined_csv(results, output_root)
    print('csv_path=' + str(csv_path))


if __name__ == '__main__':
    main()
