"""Minimal end-to-end FAST pipeline script.

Pipeline:
1. load CIFAR-10
2. build PCA features
3. build graph and spectral features
4. run first-pass FAST joint optimization
5. export selected indices
6. train ResNet18 on the selected subset
7. report final test accuracy

The script supports a lightweight debug mode and a full-run mode.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cifar import prepare_cifar10_data
from src.eval.train_classifier import train_classifier_on_subset
from src.graph.knn_graph import build_multiscale_knn_graph
from src.graph.spectral import spectral_decomposition
from src.optimize.optimize_coreset import optimize_coreset, save_selected_indices
from src.utils.io import ensure_dir, load_yaml
from src.utils.seed import set_seed


def _get_nested(config: Dict[str, Any], key: str, default: Any) -> Any:
    """Read a top-level config key with a default."""

    return config.get(key, default)


def build_runtime_config(config: Dict[str, Any], debug: bool) -> Dict[str, Any]:
    """Build runtime settings from the YAML config and mode flag."""

    data_cfg = config.get('data', {})
    coreset_cfg = config.get('coreset', {})
    graph_cfg = config.get('graph', {})
    sampling_cfg = config.get('sampling', {})
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
        'num_frequencies': int(sampling_cfg.get('num_frequencies', 64)),
        'max_frequency_norm': float(sampling_cfg.get('max_frequency_norm', 8.0)),
        'iterations': int(optimize_cfg.get('iterations', optimize_cfg.get('steps', 400))),
        'lr': float(optimize_cfg.get('lr', 1e-3)),
        'lambda_div': float(optimize_cfg.get('lambda_div', 0.1)),
        'lambda_graph': float(optimize_cfg.get('lambda_graph', 0.1)),
        'lambda_match': float(optimize_cfg.get('lambda_match', optimize_cfg.get('lambda_align', 1.0))),
        'lambda_pdcfd': float(optimize_cfg.get('lambda_pdcfd', 1.0)),
        'backbone': str(eval_cfg.get('backbone', 'resnet18')),
        'epochs': int(eval_cfg.get('epochs', 1)),
        'batch_size': int(eval_cfg.get('batch_size', 32)),
        'output_dir': str(config.get('output_dir', './outputs')),
        'train_num_samples': None,
        'test_num_samples': None,
        'train_max_batches': None,
        'eval_max_batches': None,
    }

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


def run_pipeline(config_path: str | Path, debug: bool = False) -> Dict[str, Any]:
    """Run the minimal FAST pipeline end to end."""

    config = load_yaml(config_path)
    runtime = build_runtime_config(config, debug=debug)
    set_seed(runtime['seed'])

    output_dir = ensure_dir(runtime['output_dir'])
    print(f"[1/7] Loading CIFAR-10 from {runtime['data_root']}")
    prepared = prepare_cifar10_data(
        root=runtime['data_root'],
        flatten=runtime['flatten'],
        standardize=True,
        pca_dim=runtime['pca_dim'],
        train_num_samples=runtime['train_num_samples'],
        test_num_samples=runtime['test_num_samples'],
        download=True,
    )
    print(f"train_images={tuple(prepared.train.images.shape)} train_features={tuple(prepared.train.X.shape)}")

    print(f"[2/7] Building multiscale graph with k_list={runtime['k_list']}")
    graph = build_multiscale_knn_graph(prepared.train.X, k_list=runtime['k_list'])
    print(f"graph_shape={graph.combined_graph.shape} nnz={graph.combined_graph.nnz}")

    print(f"[3/7] Computing spectral embedding with manifold_dim={runtime['manifold_dim']}")
    spectral = spectral_decomposition(graph.combined_graph, d=runtime['manifold_dim'])
    print(f"L_sym_shape={spectral.laplacian.shape} V_full_shape={tuple(spectral.eigenvectors.shape)}")

    print(f"[4/7] Running FAST joint optimization for {runtime['iterations']} iterations")
    import torch
    V_full = torch.from_numpy(spectral.eigenvectors).to(dtype=torch.float32)
    optimize_config = {
        'keep_ratio': runtime['keep_ratio'],
        'init_mode': runtime['init_mode'],
        'iterations': runtime['iterations'],
        'lr': runtime['lr'],
        'lambda_match': runtime['lambda_match'],
        'lambda_graph': runtime['lambda_graph'],
        'lambda_div': runtime['lambda_div'],
        'lambda_pdcfd': runtime['lambda_pdcfd'],
        'num_frequencies': runtime['num_frequencies'],
        'max_frequency_norm': runtime['max_frequency_norm'],
        'lambda_p': 0.5,
        'alpha': 0.5,
        'rff_dim': 32 if debug else 64,
        'dpp_sigma': 1.0,
        'dpp_delta': 1e-5,
    }
    opt_result = optimize_coreset(V_full=V_full, L_sym=spectral.laplacian, config=optimize_config)
    for idx in range(len(opt_result.logs['loss_total'])):
        print(
            'iter={step} total={total:.6f} match={match:.6f} graph={graph_loss:.6f} div={div:.6f} pdcfd={pdcfd:.6f}'.format(
                step=opt_result.logs['step'][idx],
                total=opt_result.logs['loss_total'][idx],
                match=opt_result.logs['loss_match'][idx],
                graph_loss=opt_result.logs['loss_graph'][idx],
                div=opt_result.logs['loss_div'][idx],
                pdcfd=opt_result.logs['loss_pdcfd'][idx],
            )
        )

    print('[5/7] Exporting selected indices')
    selected_path = save_selected_indices(opt_result.selected_indices, output_dir / 'selected_indices.npy')
    stats_path = output_dir / 'selected_indices_stats.json'
    stats_path.write_text(json.dumps(opt_result.export_stats, indent=2), encoding='utf-8')
    print(f'selected_count={int(opt_result.selected_indices.shape[0])} saved_to={selected_path}')

    print(f"[6/7] Training {runtime['backbone']} on selected subset")
    clf_result = train_classifier_on_subset(
        selected_indices=opt_result.selected_indices,
        backbone=runtime['backbone'],
        root=runtime['data_root'],
        epochs=runtime['epochs'],
        batch_size=runtime['batch_size'],
        device=runtime['device'],
        download=True,
        train_max_batches=runtime['train_max_batches'],
        eval_max_batches=runtime['eval_max_batches'],
        num_workers=0,
    )
    print(f"[7/7] Final test accuracy={clf_result.test_accuracy:.6f} test_loss={clf_result.test_loss:.6f}")

    summary = {
        'debug': debug,
        'selected_count': int(opt_result.selected_indices.shape[0]),
        'selected_indices_path': str(selected_path),
        'test_accuracy': float(clf_result.test_accuracy),
        'test_loss': float(clf_result.test_loss),
        'export_stats': opt_result.export_stats,
    }
    summary_path = output_dir / 'pipeline_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description='Run the minimal FAST pipeline')
    parser.add_argument('--config', type=str, default=str(PROJECT_ROOT / 'configs' / 'cifar10_fast.yaml'))
    parser.add_argument('--debug', action='store_true', help='Run a small debug-mode pipeline')
    args = parser.parse_args()

    summary = run_pipeline(config_path=args.config, debug=args.debug)
    print('pipeline_summary=' + json.dumps(summary, ensure_ascii=True))


if __name__ == '__main__':
    main()
