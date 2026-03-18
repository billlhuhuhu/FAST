"""Lightweight CIFAR subset run for first-pass FAST optimization."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cifar import prepare_cifar10_data
from src.graph.knn_graph import build_multiscale_knn_graph
from src.graph.spectral import spectral_decomposition
from src.optimize.optimize_coreset import optimize_coreset


def main() -> None:
    prepared = prepare_cifar10_data(
        root='./data',
        flatten=True,
        standardize=True,
        pca_dim=32,
        train_num_samples=256,
        test_num_samples=32,
        download=True,
    )
    V_input = prepared.train.X
    graph = build_multiscale_knn_graph(V_input, k_list=[5, 10, 20])
    spectral = spectral_decomposition(graph.combined_graph, d=16)

    config = {
        'keep_ratio': 0.1,
        'init_mode': 'random_subset',
        'iterations': 5,
        'lr': 1e-2,
        'lambda_match': 1.0,
        'lambda_graph': 0.05,
        'lambda_div': 0.01,
        'lambda_pdcfd': 1.0,
        'num_frequencies': 32,
        'max_frequency_norm': 4.0,
        'lambda_p': 0.5,
        'alpha': 0.5,
        'rff_dim': 32,
        'dpp_sigma': 1.0,
        'dpp_delta': 1e-5,
    }

    result = optimize_coreset(V_full=torch_from_numpy(spectral.eigenvectors), L_sym=spectral.laplacian, config=config)

    for idx in range(len(result.logs['loss_total'])):
        print(
            'step={step} total={total:.6f} match={match:.6f} graph={graph:.6f} div={div:.6f} pdcfd={pdcfd:.6f}'.format(
                step=result.logs['step'][idx],
                total=result.logs['loss_total'][idx],
                match=result.logs['loss_match'][idx],
                graph=result.logs['loss_graph'][idx],
                div=result.logs['loss_div'][idx],
                pdcfd=result.logs['loss_pdcfd'][idx],
            )
        )

    print(f'selected_indices={result.matched_indices.cpu().tolist()}')


def torch_from_numpy(array):
    import torch
    return torch.from_numpy(array).to(dtype=torch.float32)


if __name__ == '__main__':
    main()
