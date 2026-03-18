"""Summarize repeated FAST experiment results.

The script recursively reads ``run_summary.json`` files (preferred) or
``classifier_result.json`` files under an input directory and produces:

- summary.json
- summary.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def collect_result_files(input_dir: str | Path) -> List[Path]:
    """Collect repeat result files under a directory."""

    root = Path(input_dir)
    run_files = sorted(root.rglob('run_summary.json'))
    if run_files:
        return run_files
    return sorted(root.rglob('classifier_result.json'))


def summarize_result_files(result_files: List[Path]) -> Dict[str, Any]:
    """Aggregate mean/std/per-run accuracy from result files."""

    if not result_files:
        raise ValueError('No result files found to summarize')

    runs: List[Dict[str, Any]] = []
    accuracies: List[float] = []
    for path in result_files:
        payload = _load_json(path)
        if 'test_accuracy' in payload:
            accuracy = float(payload['test_accuracy'])
        else:
            raise ValueError(f'Missing test_accuracy in {path}')
        run_item = {
            'path': str(path),
            'test_accuracy': accuracy,
            'best_accuracy': float(payload.get('best_accuracy', accuracy)),
            'seed': payload.get('seed', payload.get('config_snapshot', {}).get('seed', None)),
            'method': payload.get('method', payload.get('config_snapshot', {}).get('extra_config', {}).get('method', None)),
            'keep_ratio': payload.get('keep_ratio', payload.get('config_snapshot', {}).get('extra_config', {}).get('keep_ratio', None)),
        }
        runs.append(run_item)
        accuracies.append(accuracy)

    mean_accuracy = sum(accuracies) / len(accuracies)
    variance = sum((value - mean_accuracy) ** 2 for value in accuracies) / len(accuracies)
    std_accuracy = variance ** 0.5
    return {
        'num_runs': len(runs),
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'per_run_accuracy': accuracies,
        'runs': runs,
    }


def build_markdown_table(summary: Dict[str, Any]) -> str:
    """Create a small markdown table for repeat results."""

    lines = [
        '| run | seed | method | keep_ratio | test_accuracy | best_accuracy |',
        '| --- | --- | --- | --- | --- | --- |',
    ]
    for idx, run in enumerate(summary['runs']):
        lines.append(
            f"| {idx} | {run.get('seed', '')} | {run.get('method', '')} | {run.get('keep_ratio', '')} | {run['test_accuracy']:.6f} | {run['best_accuracy']:.6f} |"
        )
    lines.append('')
    lines.append(f"mean accuracy: {summary['mean_accuracy']:.6f}")
    lines.append(f"std accuracy: {summary['std_accuracy']:.6f}")
    return '\n'.join(lines)


def save_summary_outputs(summary: Dict[str, Any], output_dir: str | Path) -> Dict[str, str]:
    """Save JSON and markdown summaries."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = output_dir / 'summary.json'
    summary_md = output_dir / 'summary.md'
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding='utf-8')
    summary_md.write_text(build_markdown_table(summary), encoding='utf-8')
    return {'summary_json': str(summary_json), 'summary_md': str(summary_md)}


def main() -> None:
    parser = argparse.ArgumentParser(description='Summarize repeated experiment results')
    parser.add_argument('--input-dir', type=str, required=True, help='Directory containing run results')
    parser.add_argument('--output-dir', type=str, default=None, help='Directory to save summary outputs')
    args = parser.parse_args()

    result_files = collect_result_files(args.input_dir)
    summary = summarize_result_files(result_files)
    output_dir = args.output_dir if args.output_dir is not None else args.input_dir
    saved = save_summary_outputs(summary, output_dir)

    print('summary=' + json.dumps(summary, ensure_ascii=True))
    print('saved=' + json.dumps(saved, ensure_ascii=True))


if __name__ == '__main__':
    main()
