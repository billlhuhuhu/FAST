"""Lightweight FAST pipeline runner for CIFAR-10.

This script keeps the project on a simple, configuration-driven path while
supporting a paper-like evaluation loop with:

- method selection (FAST / random)
- keep-ratio override
- seed override
- repeat runs
- per-run artifact saving
- debug mode for quick smoke checks
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Keep BLAS/OpenMP thread usage conservative before importing NumPy/sklearn users.
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.data.cifar import prepare_cifar10_data
from src.eval.train_classifier import load_cifar10_datasets, sample_random_subset, train_classifier_on_subset
from src.graph.knn_graph import build_multiscale_knn_graph
from src.graph.spectral import spectral_decomposition
from src.optimize.optimize_coreset import optimize_coreset, save_selected_indices
from src.utils.io import ensure_dir, load_yaml
from src.utils.seed import set_seed


def build_runtime_config(
    config: Dict[str, Any],
    debug: bool,
    method: str | None = None,
    keep_ratio: float | None = None,
    seed: int | None = None,
    repeat: int | None = None,
) -> Dict[str, Any]:
    """Build runtime settings from config and CLI overrides."""

    data_cfg = config.get('data', {})
    coreset_cfg = config.get('coreset', {})
    graph_cfg = config.get('graph', {})
    sampling_cfg = config.get('sampling', {})
    assignment_cfg = config.get('assignment', {})
    optimize_cfg = config.get('optimize', {})
    experiment_cfg = config.get('experiment', {})
    eval_cfg = config.get('eval', {})

    runtime = {
        'seed': int(config.get('seed', 42) if seed is None else seed),
        'device': str(config.get('device', 'cpu')),
        'output_dir': str(config.get('output_dir', './outputs')),
        'method': str(experiment_cfg.get('method', 'fast') if method is None else method).lower(),
        'repeat': int(experiment_cfg.get('repeat', 1) if repeat is None else repeat),
        'data_root': str(data_cfg.get('root', './data')),
        'flatten': bool(data_cfg.get('flatten', True)),
        'pca_dim': int(data_cfg.get('pca_dim', 128)),
        'manifold_dim': int(data_cfg.get('manifold_dim', 32)),
        'keep_ratio': float(coreset_cfg.get('keep_ratio', 0.1) if keep_ratio is None else keep_ratio),
        'init_mode': str(coreset_cfg.get('init_mode', 'random_subset')),
        'k_list': list(graph_cfg.get('k_list', [5, 10, 20])),
        'sampling_cfg': dict(sampling_cfg),
        'assignment_cfg': dict(assignment_cfg),
        'iterations': int(optimize_cfg.get('iterations', optimize_cfg.get('steps', 400))),
        'opt_lr': float(optimize_cfg.get('lr', 1e-3)),
        'lambda_div': float(optimize_cfg.get('lambda_div', 0.1)),
        'lambda_graph': float(optimize_cfg.get('lambda_graph', 0.1)),
        'lambda_match': float(optimize_cfg.get('lambda_match', optimize_cfg.get('lambda_align', 1.0))),
        'lambda_pdcfd': float(optimize_cfg.get('lambda_pdcfd', 1.0)),
        'backbone': str(eval_cfg.get('backbone', 'resnet18')),
        'optimizer_name': str(eval_cfg.get('optimizer', 'sgd')),
        'scheduler_name': str(eval_cfg.get('scheduler', 'none')),
        'scheduler_milestones': list(eval_cfg.get('scheduler_milestones', [])),
        'scheduler_gamma': float(eval_cfg.get('scheduler_gamma', 0.1)),
        'eval_lr': float(eval_cfg.get('lr', 0.1)),
        'momentum': float(eval_cfg.get('momentum', 0.9)),
        'weight_decay': float(eval_cfg.get('weight_decay', 5e-4)),
        'epochs': int(eval_cfg.get('epochs', 200)),
        'batch_size': int(eval_cfg.get('batch_size', 128)),
        'train_max_batches': None,
        'eval_max_batches': None,
        'download': True,
        'num_threads': int(config.get('num_threads', 1)),
        'train_num_samples': data_cfg.get('train_num_samples', None),
        'test_num_samples': data_cfg.get('test_num_samples', None),
    }

    if runtime['method'] not in {'fast', 'random'}:
        raise ValueError('method must be one of: fast, random')
    if runtime['repeat'] <= 0:
        raise ValueError('repeat must be positive')

    if debug:
        runtime['pca_dim'] = min(runtime['pca_dim'], 32)
        runtime['manifold_dim'] = min(runtime['manifold_dim'], 16)
        runtime['iterations'] = min(runtime['iterations'], 5)
        runtime['epochs'] = min(runtime['epochs'], 1)
        runtime['batch_size'] = min(runtime['batch_size'], 16)
        runtime['train_num_samples'] = 256
        runtime['test_num_samples'] = 64
        runtime['train_max_batches'] = 2
        runtime['eval_max_batches'] = 2
        runtime['output_dir'] = str(Path(runtime['output_dir']) / 'debug')
    else:
        runtime['output_dir'] = str(Path(runtime['output_dir']) / 'full')

    return runtime




def _configure_runtime_threads(num_threads: int) -> None:
    """Apply conservative thread settings for BLAS/OpenMP and PyTorch."""

    resolved = max(1, int(num_threads))
    os.environ['OPENBLAS_NUM_THREADS'] = str(resolved)
    os.environ['OMP_NUM_THREADS'] = str(resolved)
    os.environ['MKL_NUM_THREADS'] = str(resolved)
    os.environ['NUMEXPR_NUM_THREADS'] = str(resolved)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(resolved)
    torch.set_num_threads(resolved)
    if hasattr(torch, 'set_num_interop_threads'):
        try:
            torch.set_num_interop_threads(max(1, min(resolved, 4)))
        except RuntimeError:
            pass

def _format_keep_ratio(keep_ratio: float) -> str:
    return f'{keep_ratio:.3f}'


def _build_run_dir(output_root: Path, method: str, keep_ratio: float, seed: int, run_index: int) -> Path:
    return ensure_dir(output_root / f'method_{method}' / f'keep_{_format_keep_ratio(keep_ratio)}' / f'seed_{seed}' / f'run_{run_index:02d}')


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding='utf-8')


def _select_indices_fast(runtime: Dict[str, Any], run_seed: int, run_dir: Path) -> Dict[str, Any]:
    """Run FAST selection and return selection artifacts."""

    print(f"[1/7] Loading CIFAR-10 from {runtime['data_root']}")
    prepared = prepare_cifar10_data(
        root=runtime['data_root'],
        flatten=runtime['flatten'],
        standardize=True,
        pca_dim=runtime['pca_dim'],
        train_num_samples=runtime['train_num_samples'],
        test_num_samples=runtime['test_num_samples'],
        download=runtime['download'],
    )
    print(f"train_images={tuple(prepared.train.images.shape)} train_features={tuple(prepared.train.X.shape)}")

    print(f"[2/7] Building multiscale graph with k_list={runtime['k_list']}")
    graph = build_multiscale_knn_graph(prepared.train.X, k_list=runtime['k_list'])
    print(f"graph_shape={graph.combined_graph.shape} nnz={graph.combined_graph.nnz} components={graph.stats['connected_components']}")

    print(f"[3/7] Computing spectral embedding with manifold_dim={runtime['manifold_dim']}")
    spectral = spectral_decomposition(graph.combined_graph, d=runtime['manifold_dim'])
    print(f"L_sym_shape={spectral.laplacian.shape} V_full_shape={tuple(spectral.eigenvectors.shape)}")

    print(f"[4/7] Running FAST joint optimization for {runtime['iterations']} iterations")
    V_full = torch.from_numpy(spectral.eigenvectors).to(dtype=torch.float32)
    sampling_cfg = dict(runtime['sampling_cfg'])
    if runtime['train_num_samples'] is not None:
        sampling_cfg['num_frequencies'] = min(int(sampling_cfg.get('num_frequencies', 64)), 32)
        sampling_cfg['pdas_frequencies_per_iter'] = min(int(sampling_cfg.get('pdas_frequencies_per_iter', 64)), 16)
        sampling_cfg['rebuild_afl_each_iter'] = True

    optimize_config = {
        'keep_ratio': runtime['keep_ratio'],
        'init_mode': runtime['init_mode'],
        'iterations': runtime['iterations'],
        'lr': runtime['opt_lr'],
        'lambda_match': runtime['lambda_match'],
        'lambda_graph': runtime['lambda_graph'],
        'lambda_div': runtime['lambda_div'],
        'lambda_pdcfd': runtime['lambda_pdcfd'],
        'rff_dim': 32 if runtime['train_num_samples'] is not None else 64,
        'dpp_sigma': 1.0,
        'dpp_delta': 1e-5,
        'sampling': sampling_cfg,
        'assignment': runtime['assignment_cfg'],
        'verbose': True,
        'log_every': 1 if runtime['train_num_samples'] is not None else 10,
    }
    opt_result = optimize_coreset(V_full=V_full, L_sym=spectral.laplacian, config=optimize_config)
    for idx in range(len(opt_result.logs['loss_total'])):
        print(
            'iter={step} total={total:.6f} match={match:.6f} graph={graph_loss:.6f} div={div:.6f} pdcfd={pdcfd:.6f} tau={tau:.6f} cand={cand} assign={assign}'.format(
                step=opt_result.logs['step'][idx],
                total=opt_result.logs['loss_total'][idx],
                match=opt_result.logs['loss_match'][idx],
                graph_loss=opt_result.logs['loss_graph'][idx],
                div=opt_result.logs['loss_div'][idx],
                pdcfd=opt_result.logs['loss_pdcfd'][idx],
                tau=opt_result.logs['tau_t'][idx],
                cand=opt_result.logs['candidate_count'][idx],
                assign=opt_result.logs['assignment_mode'][idx],
            )
        )

    print('[5/7] Exporting selected indices')
    selected_path = save_selected_indices(opt_result.selected_indices, run_dir / 'selected_indices.npy')
    _save_json(run_dir / 'selected_indices_stats.json', opt_result.export_stats)
    _save_json(run_dir / 'graph_stats.json', graph.stats)
    print(f"selected_count={int(opt_result.selected_indices.shape[0])} saved_to={selected_path}")

    return {
        'selected_indices': opt_result.selected_indices,
        'selected_path': str(selected_path),
        'selection_stats': opt_result.export_stats,
        'graph_stats': graph.stats,
        'opt_logs': opt_result.logs,
        'train_images_shape': tuple(prepared.train.images.shape),
        'train_features_shape': tuple(prepared.train.X.shape),
    }


def _select_indices_random(runtime: Dict[str, Any], run_seed: int, run_dir: Path) -> Dict[str, Any]:
    """Sample a random subset baseline and save it."""

    print(f"[1/7] Loading CIFAR-10 metadata from {runtime['data_root']}")
    train_dataset, _ = load_cifar10_datasets(root=runtime['data_root'], download=runtime['download'])
    selected_indices = sample_random_subset(train_size=len(train_dataset), keep_ratio=runtime['keep_ratio'], seed=run_seed)
    print('[2/7] Skipping graph / spectral / optimization because method=random')
    selected_path = save_selected_indices(selected_indices, run_dir / 'selected_indices.npy')
    selection_stats = {
        'target_M': int(selected_indices.shape[0]),
        'raw_match_count': int(selected_indices.shape[0]),
        'unique_before_fill': int(selected_indices.shape[0]),
        'filled_count': 0,
        'final_unique_count': int(selected_indices.shape[0]),
    }
    _save_json(run_dir / 'selected_indices_stats.json', selection_stats)
    print(f"selected_count={int(selected_indices.shape[0])} saved_to={selected_path}")
    return {
        'selected_indices': selected_indices,
        'selected_path': str(selected_path),
        'selection_stats': selection_stats,
        'graph_stats': {},
        'opt_logs': {},
        'train_images_shape': None,
        'train_features_shape': None,
    }


def run_single_pipeline(runtime: Dict[str, Any], run_index: int) -> Dict[str, Any]:
    """Run one FAST or random experiment and save per-run artifacts."""

    run_seed = int(runtime['seed'] + run_index)
    set_seed(run_seed)
    output_root = ensure_dir(runtime['output_dir'])
    run_dir = _build_run_dir(output_root, runtime['method'], runtime['keep_ratio'], run_seed, run_index)

    print(f'===== run {run_index + 1}/{runtime["repeat"]} method={runtime["method"]} seed={run_seed} keep_ratio={runtime["keep_ratio"]:.3f} =====')

    if runtime['method'] == 'fast':
        selection = _select_indices_fast(runtime=runtime, run_seed=run_seed, run_dir=run_dir)
    else:
        selection = _select_indices_random(runtime=runtime, run_seed=run_seed, run_dir=run_dir)

    print(f"[6/7] Training {runtime['backbone']} on selected subset")
    classifier_output_dir = ensure_dir(run_dir / 'classifier')
    clf_result = train_classifier_on_subset(
        selected_indices=selection['selected_indices'],
        backbone=runtime['backbone'],
        root=runtime['data_root'],
        epochs=runtime['epochs'],
        batch_size=runtime['batch_size'],
        optimizer_name=runtime['optimizer_name'],
        scheduler_name=runtime['scheduler_name'],
        lr=runtime['eval_lr'],
        momentum=runtime['momentum'],
        weight_decay=runtime['weight_decay'],
        scheduler_milestones=runtime['scheduler_milestones'],
        scheduler_gamma=runtime['scheduler_gamma'],
        seed=run_seed,
        device=runtime['device'],
        download=runtime['download'],
        train_max_batches=runtime['train_max_batches'],
        eval_max_batches=runtime['eval_max_batches'],
        num_workers=0,
        output_dir=classifier_output_dir,
        extra_config={
            'method': runtime['method'],
            'keep_ratio': runtime['keep_ratio'],
            'run_index': run_index,
        },
    )
    print(f"[7/7] Final test accuracy={clf_result.test_accuracy:.6f} best_accuracy={clf_result.best_accuracy:.6f} test_loss={clf_result.test_loss:.6f}")

    summary = {
        'debug': runtime['train_num_samples'] is not None,
        'method': runtime['method'],
        'seed': run_seed,
        'run_index': run_index,
        'keep_ratio': runtime['keep_ratio'],
        'selected_count': int(selection['selected_indices'].shape[0]),
        'selected_indices_path': selection['selected_path'],
        'selection_stats': selection['selection_stats'],
        'graph_stats': selection['graph_stats'],
        'train_images_shape': selection['train_images_shape'],
        'train_features_shape': selection['train_features_shape'],
        'test_accuracy': float(clf_result.test_accuracy),
        'best_accuracy': float(clf_result.best_accuracy),
        'best_epoch': int(clf_result.best_epoch),
        'test_loss': float(clf_result.test_loss),
        'train_loss_summary': clf_result.train_loss_summary,
        'classifier_result_path': clf_result.result_path,
        'run_dir': str(run_dir),
    }
    _save_json(run_dir / 'run_summary.json', summary)
    return summary


def run_pipeline(
    config_path: str | Path,
    debug: bool = False,
    method: str | None = None,
    keep_ratio: float | None = None,
    seed: int | None = None,
    repeat: int | None = None,
) -> Dict[str, Any]:
    """Run one or multiple experiments and save aggregate summary."""

    config = load_yaml(config_path)
    runtime = build_runtime_config(config, debug=debug, method=method, keep_ratio=keep_ratio, seed=seed, repeat=repeat)
    _configure_runtime_threads(runtime['num_threads'])

    run_summaries: List[Dict[str, Any]] = []
    for run_index in range(runtime['repeat']):
        run_summaries.append(run_single_pipeline(runtime=runtime, run_index=run_index))

    accuracies = [float(item['test_accuracy']) for item in run_summaries]
    best_accuracies = [float(item['best_accuracy']) for item in run_summaries]
    aggregate = {
        'debug': debug,
        'method': runtime['method'],
        'keep_ratio': runtime['keep_ratio'],
        'base_seed': runtime['seed'],
        'repeat': runtime['repeat'],
        'mean_accuracy': float(sum(accuracies) / max(len(accuracies), 1)),
        'std_accuracy': float(torch.tensor(accuracies, dtype=torch.float32).std(unbiased=False).item() if accuracies else 0.0),
        'mean_best_accuracy': float(sum(best_accuracies) / max(len(best_accuracies), 1)),
        'runs': run_summaries,
    }

    output_root = ensure_dir(runtime['output_dir'])
    aggregate_path = output_root / f'aggregate_{runtime["method"]}_keep_{_format_keep_ratio(runtime["keep_ratio"])}.json'
    _save_json(aggregate_path, aggregate)
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description='Run the lightweight FAST pipeline')
    parser.add_argument('--config', type=str, default=str(PROJECT_ROOT / 'configs' / 'cifar10_fast.yaml'))
    parser.add_argument('--debug', action='store_true', help='Run a small debug-mode pipeline')
    parser.add_argument('--method', type=str, default=None, help='Subset method: fast or random')
    parser.add_argument('--keep-ratio', type=float, default=None, help='Override keep ratio, e.g. 0.1 / 0.2 / 0.3')
    parser.add_argument('--seed', type=int, default=None, help='Override random seed')
    parser.add_argument('--repeat', type=int, default=None, help='Number of repeated runs')
    args = parser.parse_args()

    summary = run_pipeline(
        config_path=args.config,
        debug=args.debug,
        method=args.method,
        keep_ratio=args.keep_ratio,
        seed=args.seed,
        repeat=args.repeat,
    )
    print('pipeline_summary=' + json.dumps(summary, ensure_ascii=True))


if __name__ == '__main__':
    main()
