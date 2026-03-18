"""Lightweight integration script for CIFAR -> PCA -> graph -> spectral pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cifar import prepare_cifar10_data
from src.graph.knn_graph import build_multiscale_knn_graph
from src.graph.spectral import spectral_decomposition


def main() -> None:
    prepared = prepare_cifar10_data(
        root="./data",
        flatten=True,
        standardize=True,
        pca_dim=32,
        train_num_samples=256,
        test_num_samples=32,
        download=True,
    )
    X = prepared.train.X
    graph = build_multiscale_knn_graph(X, k_list=[5, 10, 20])
    result = spectral_decomposition(graph.combined_graph, d=16)

    print(f"train_images_shape={tuple(prepared.train.images.shape)}")
    print(f"train_feature_shape={tuple(prepared.train.X.shape)}")
    print(f"graph_shape={graph.combined_graph.shape}")
    print(f"laplacian_shape={result.laplacian.shape}")
    print(f"eigenvalues_shape={tuple(result.eigenvalues.shape)}")
    print(f"V_full_shape={tuple(result.eigenvectors.shape)}")


if __name__ == "__main__":
    main()
