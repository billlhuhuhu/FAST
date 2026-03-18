# FAST Reproduction Scaffold

This repository is a staged reproduction scaffold for the paper:
`FAST: Topology-Aware Frequency-Domain Distribution Matching for Coreset Selection`.

The current goal is not full paper fidelity yet. We are first building a clean,
debuggable Python 3.10 + PyTorch project skeleton with stable interfaces, shape
contracts, and minimal smoke tests.

## Current Status

This stage provides:

- a minimal config file,
- CIFAR data loading helpers,
- simple kNN graph construction,
- spectral graph placeholders,
- assignment placeholders for mapping continuous proxies to discrete samples,
- placeholder loss modules for DPP, topology alignment, and PD-CFD,
- placeholder frequency sampling modules for AFL/PDAS,
- a tiny coreset optimization loop,
- a minimal downstream classifier training stub,
- smoke tests for imports and tensor shapes.

This stage intentionally does **not** implement the full FAST algorithm.

## Planned Reproduction Route

1. Build and verify the project skeleton.
2. Keep all module boundaries stable and test shape correctness.
3. Add a runnable tiny coreset optimization loop on toy tensors.
4. Add CIFAR-10 end-to-end wiring with debug-sized runs.
5. Replace placeholder graph logic with a closer topology-aware formulation.
6. Replace placeholder PD-CFD with a paper-faithful implementation.
7. Add anisotropic frequency initialization closer to AFL.
8. Replace simple progressive scheduling with discrepancy-aware PDAS.
9. Improve continuous-to-discrete assignment.
10. Run small reproducibility experiments before any large training.

## Module Overview

- `src/data`: dataset materialization and vectorization helpers.
- `src/graph`: kNN graphs, spectral utilities, and assignment logic.
- `src/losses`: diversity, topology, and frequency-domain loss functions.
- `src/sampling`: frequency-library creation and progressive frequency selection.
- `src/optimize`: coreset proxy initialization and optimization loop.
- `src/eval`: downstream classifier training and evaluation stubs.
- `src/utils`: config and reproducibility helpers.
- `tests`: smoke tests for imports and basic tensor shapes.

## Minimal Development Workflow

1. Install dependencies from `requirements.txt`.
2. Run `python -m unittest discover -s tests`.
3. Fill in one module at a time.
4. After each change, rerun smoke tests before attempting larger experiments.

## Notes

- The current implementation prioritizes `runnable + modular + shape-safe`.
- Several functions intentionally use placeholder approximations and include TODOs.
- We should only upgrade one algorithmic block at a time to keep debugging easy.
