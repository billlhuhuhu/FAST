# FAST Reproduction Scaffold

This repository is a staged reproduction scaffold for the paper:
`FAST: Topology-Aware Frequency-Domain Distribution Matching for Coreset Selection`.

The current goal is the first runnable CIFAR-10 reproduction pass. We prioritize
clear module boundaries, stable tensor shapes, and a simple workflow over full
paper fidelity in the first iteration.

## First-Pass CIFAR-10 Default Setup

The default experiment target is:

- dataset: CIFAR-10
- keep ratio: `0.1`
- PCA dimension: `128`
- manifold dimension: `32`
- graph scales: `k_list = [5, 10, 20]`
- coreset optimization iterations: `400`
- PDAS frequencies per iteration: `64`
- downstream classifier: `ResNet18`

## Overall Workflow

The project follows a strict two-stage workflow:

1. Run subset selection on the full training set to produce a discrete coreset.
2. Train a downstream classifier only on the selected subset.

This split is important because FAST itself is a coreset selection method, not
the downstream classifier.

## Stage 1: Subset Selection

Goal:

- load CIFAR-10 training data,
- reduce image features with PCA,
- build a multi-scale manifold graph with `k_list=[5,10,20]`,
- optimize a continuous coreset proxy for `400` iterations,
- apply PDAS with `64` frequencies each iteration,
- map optimized proxies back to discrete sample indices.

Expected output:

- selected training subset indices,
- optimization logs,
- basic diagnostics for losses and tensor shapes.

## Stage 2: Downstream Training

Goal:

- construct the selected discrete subset from Stage 1,
- train a `ResNet18` classifier only on that subset,
- evaluate on the CIFAR-10 test set.

Expected output:

- training loss curves,
- final test accuracy,
- a later comparison against simple subset baselines.

## Module Overview

- `src/data`: dataset materialization and vectorization helpers.
- `src/graph`: kNN graphs, spectral utilities, and assignment logic.
- `src/losses`: diversity, topology, and frequency-domain loss functions.
- `src/sampling`: frequency-library creation and progressive frequency selection.
- `src/optimize`: coreset proxy initialization and optimization loop.
- `src/eval`: downstream classifier training and evaluation stubs.
- `src/utils`: config and reproducibility helpers.
- `tests`: smoke tests for imports and basic tensor shapes.

## How To Run

Install dependencies:

1. `pip install -r requirements.txt`

Run the current smoke tests:

1. `python -m unittest discover -s tests`

Planned subset selection flow:

1. Load config from `configs/cifar10_fast.yaml`
2. Build CIFAR-10 features
3. Run coreset optimization
4. Save selected subset indices

Planned downstream training flow:

1. Load selected subset indices from Stage 1
2. Build a CIFAR-10 subset dataloader
3. Train `ResNet18`
4. Evaluate on the test set

The command-line runners are not added yet, but the intended order is fixed:

1. subset selection first
2. downstream training second

## Notes

- The current implementation prioritizes `runnable + modular + shape-safe`.
- Several functions intentionally use placeholder approximations and include TODOs.
- We should only upgrade one algorithmic block at a time to keep debugging easy.
