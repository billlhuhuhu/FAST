"""Lightweight ablation runner for comparing distribution metrics.

Pilot metrics:
- MSE
- KL
- CE
- PD-CFD

The script keeps graph / assignment / diversity / classifier settings as aligned
as possible and only swaps the main distribution-matching loss.
"""

from __future__ import annotations

import argparse
import copy
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
from src.eval.train_classifier import train_classifier_on_subset
from src.graph.assign import hungarian_match
from src.graph.knn_graph import build_multiscale_knn_graph
from src.graph.spectral import spectral_decomposition
from src.losses.dpp import compute_dpp_loss
from src.losses.graph_losses import compute_graph_loss, compute_match_loss
from src.losses.pdcfd import empirical_characteristic_function, pd_cfd_loss
from src.optimize.optimize_coreset import export_selected_subset, initialize_coreset_variable, save_selected_indices
from src.sampling.anisotropic_freq import build_anisotropic_frequency_library
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
    eval_cfg = config.get('eval', {})

    runtime = {
        'seed': int(config.get('seed', 42)),
        'device': str(config.get('device', 'cpu')),
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
        'backbone': str(eval_cfg.get('backbone', 'resnet18')),
        'optimizer_name': str(eval_cfg.get('optimizer', 'sgd')),
        'eval_lr': float(eval_cfg.get('lr', 0.1)),
        'momentum': float(eval_cfg.get('momentum', 0.9)),
        'weight_decay': float(eval_cfg.get('weight_decay', 5e-4)),
        'epochs': int(eval_cfg.get('epochs', 200)),
        'batch_size': int(eval_cfg.get('batch_size', 128)),
        'output_dir': Path(config.get('output_dir', './outputs')) / 'ablations' / 'distribution_metric',
        'train_num_samples': None,
        'test_num_samples': None,
        'train_max_batches': None,
        'eval_max_batches': None,
    }

    if debug:
        runtime['pca_dim'] = min(runtime['pca_dim'], 32)
        runtime['manifold_dim'] = min(runtime['manifold_dim'], 16)
        runtime['iterations'] = min(runtime['iterations'], 4)
        runtime['epochs'] = 1
        runtime['batch_size'] = min(runtime['batch_size'], 16)
        runtime['train_num_samples'] = 256
        runtime['test_num_samples'] = 64
        runtime['train_max_batches'] = 2
        runtime['eval_max_batches'] = 1
        runtime['sampling_cfg']['num_frequencies'] = min(int(runtime['sampling_cfg'].get('num_frequencies', 64)), 32)
        runtime['sampling_cfg']['pdas_frequencies_per_iter'] = min(int(runtime['sampling_cfg'].get('pdas_frequencies_per_iter', 64)), 16)
        runtime['sampling_cfg']['rebuild_afl_each_iter'] = True
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
    V_full = torch.from_numpy(spectral.eigenvectors).to(dtype=torch.float32)
    return V_full, spectral.laplacian


def _build_frequency_library(V_full: Tensor, Y_current: Tensor, sampling_cfg: Dict[str, Any]):
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


def _metric_main_loss(metric: str, Y_ref: Tensor, Y: Tensor, freqs: Tensor, sampling_cfg: Dict[str, Any]) -> Tensor:
    metric_name = metric.lower()
    if metric_name == 'pdcfd':
        return pd_cfd_loss(
            Y_ref=Y_ref,
            Y=Y,
            freqs=freqs,
            lambda_p=float(sampling_cfg.get('lcf_lambda_p', 0.5)),
            alpha=float(sampling_cfg.get('lcf_alpha', 0.5)),
        ).loss

    ecf_ref = empirical_characteristic_function(Y_ref, freqs)
    ecf_y = empirical_characteristic_function(Y, freqs)
    if metric_name == 'mse':
        diff = torch.cat([(ecf_ref.real - ecf_y.real), (ecf_ref.imag - ecf_y.imag)], dim=0)
        return torch.mean(diff.square())

    ref_logits = torch.abs(ecf_ref)
    y_logits = torch.abs(ecf_y)
    target_probs = torch.softmax(ref_logits, dim=0)
    log_probs = torch.log_softmax(y_logits, dim=0)

    if metric_name == 'kl':
        return torch.sum(target_probs * (torch.log(torch.clamp(target_probs, min=1e-12)) - log_probs))
    if metric_name == 'ce':
        return -torch.sum(target_probs * log_probs)
    raise ValueError(f'Unsupported metric: {metric}')


def _loss_summary(values: List[float]) -> Dict[str, float]:
    return {
        'first': float(values[0]) if values else 0.0,
        'last': float(values[-1]) if values else 0.0,
        'min': float(min(values)) if values else 0.0,
        'max': float(max(values)) if values else 0.0,
    }


def run_metric(metric: str, runtime: Dict[str, Any], V_full: Tensor, L_sym: Any, output_root: Path) -> Dict[str, Any]:
    set_seed(runtime['seed'])
    run_dir = ensure_dir(output_root / metric.lower())
    init = initialize_coreset_variable(V_full=V_full, keep_ratio=runtime['keep_ratio'], init_mode=runtime['init_mode'])
    Y = init.Y
    optimizer = torch.optim.Adam([Y], lr=runtime['opt_lr'])

    logs = {'step': [], 'loss_total': [], 'loss_main': [], 'loss_match': [], 'loss_graph': [], 'loss_div': []}
    frequency_library = _build_frequency_library(V_full, Y.detach(), runtime['sampling_cfg'])

    for step in range(runtime['iterations']):
        assignment = hungarian_match(Y=Y, V_full=V_full, degree=None, **runtime['assignment_cfg'])
        loss_match = compute_match_loss(Y=Y, V_full=V_full, matched_indices=assignment.matched_indices)
        loss_graph = compute_graph_loss(Y=Y, L_sym=L_sym, matched_indices=assignment.matched_indices)
        loss_div = compute_dpp_loss(Y=Y, rff_dim=32, sigma=1.0, delta=1e-5).loss
        if step > 0:
            frequency_library = _build_frequency_library(V_full, Y.detach(), runtime['sampling_cfg'])
        pdas_state = select_progressive_frequencies(
            library=frequency_library,
            step=step,
            total_steps=runtime['iterations'],
            Y_ref=V_full,
            Y=Y,
            config=runtime['sampling_cfg'],
        )
        loss_main = _metric_main_loss(metric=metric, Y_ref=V_full, Y=Y, freqs=pdas_state.selected_frequencies, sampling_cfg=runtime['sampling_cfg'])
        total_loss = runtime['lambda_main'] * loss_main + runtime['lambda_match'] * loss_match + runtime['lambda_graph'] * loss_graph + runtime['lambda_div'] * loss_div
        total_loss = torch.real(total_loss)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        logs['step'].append(step)
        logs['loss_total'].append(float(total_loss.detach().cpu().item()))
        logs['loss_main'].append(float(loss_main.detach().cpu().item()))
        logs['loss_match'].append(float(loss_match.detach().cpu().item()))
        logs['loss_graph'].append(float(loss_graph.detach().cpu().item()))
        logs['loss_div'].append(float(loss_div.detach().cpu().item()))
        print(f"metric={metric} step={step} total={logs['loss_total'][-1]:.6f} main={logs['loss_main'][-1]:.6f}")

    final_assignment = hungarian_match(Y=Y.detach(), V_full=V_full, degree=None, **runtime['assignment_cfg'])
    exported = export_selected_subset(final_assignment.matched_indices.detach(), Y.detach(), V_full, M=Y.shape[0])
    selected_path = save_selected_indices(exported.selected_indices, run_dir / 'selected_indices.npy')

    clf = train_classifier_on_subset(
        selected_indices=exported.selected_indices,
        backbone=runtime['backbone'],
        root=runtime['data_root'],
        epochs=runtime['epochs'],
        batch_size=runtime['batch_size'],
        optimizer_name=runtime['optimizer_name'],
        lr=runtime['eval_lr'],
        momentum=runtime['momentum'],
        weight_decay=runtime['weight_decay'],
        seed=runtime['seed'],
        device=runtime['device'],
        download=True,
        train_max_batches=runtime['train_max_batches'],
        eval_max_batches=runtime['eval_max_batches'],
        num_workers=0,
        output_dir=run_dir / 'classifier',
        extra_config={'ablation': 'distribution_metric', 'metric': metric},
    )

    result = {
        'metric': metric,
        'final_accuracy': float(clf.test_accuracy),
        'selected_count': int(exported.selected_indices.shape[0]),
        'main_loss_summary': _loss_summary(logs['loss_main']),
        'total_loss_summary': _loss_summary(logs['loss_total']),
        'selected_indices_path': str(selected_path),
        'classifier_result_path': clf.result_path,
        'logs': logs,
    }
    (run_dir / 'result.json').write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding='utf-8')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description='Run lightweight distribution-metric ablation')
    parser.add_argument('--config', type=str, default=str(PROJECT_ROOT / 'configs' / 'cifar10_fast.yaml'))
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--metrics', nargs='+', default=['mse', 'kl', 'ce', 'pdcfd'])
    args = parser.parse_args()

    config = load_yaml(args.config)
    runtime = _build_runtime(config, debug=args.debug)
    output_root = ensure_dir(runtime['output_dir'] / ('debug' if args.debug else 'full'))

    print('[ablation] preparing shared CIFAR-10 spectral features')
    V_full, L_sym = _prepare_spectral_features(runtime)

    summary = {'metrics': []}
    for metric in args.metrics:
        summary['metrics'].append(run_metric(metric, runtime, V_full, L_sym, output_root))

    summary_path = output_root / 'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding='utf-8')
    print('summary_path=' + str(summary_path))


if __name__ == '__main__':
    main()
