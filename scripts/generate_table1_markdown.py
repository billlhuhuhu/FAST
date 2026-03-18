"""Generate a minimal Table-1-style markdown table from experiment JSON files.

Supported inputs:
- aggregate_*.json produced by ``scripts/run_fast_pipeline.py``
- summary.json produced by ``scripts/summarize_repeat_results.py``
- run_summary.json for a single run

The script writes ``table.md`` under the requested output directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def _normalize_method(value: Any) -> str:
    method = str(value if value is not None else 'unknown').strip()
    if method.lower() == 'fast':
        return 'FAST'
    if method.lower() == 'random':
        return 'random'
    return method


def _infer_record(payload: Dict[str, Any], source_path: Path) -> Dict[str, Any]:
    dataset = str(payload.get('dataset', 'CIFAR-10'))
    backbone = str(
        payload.get('backbone')
        or payload.get('config_snapshot', {}).get('backbone')
        or 'ResNet18'
    )
    keep_ratio = payload.get('keep_ratio', payload.get('config_snapshot', {}).get('extra_config', {}).get('keep_ratio', 0.1))
    method = payload.get('method', payload.get('config_snapshot', {}).get('extra_config', {}).get('method', source_path.stem))

    if 'mean_accuracy' in payload and 'std_accuracy' in payload:
        mean_acc = float(payload['mean_accuracy'])
        std_acc = float(payload['std_accuracy'])
    elif 'test_accuracy' in payload:
        mean_acc = float(payload['test_accuracy'])
        std_acc = 0.0
    else:
        raise ValueError(f'Unsupported result payload in {source_path}')

    return {
        'Method': _normalize_method(method),
        'Dataset': dataset,
        'Keep Ratio': float(keep_ratio),
        'Backbone': backbone,
        'Mean Acc': mean_acc,
        'Std': std_acc,
        'Source': str(source_path),
    }


def load_records(json_paths: List[str]) -> List[Dict[str, Any]]:
    """Load experiment records from json files."""

    records: List[Dict[str, Any]] = []
    for item in json_paths:
        path = Path(item)
        if not path.exists():
            raise FileNotFoundError(f'Result file not found: {path}')
        payload = _load_json(path)
        records.append(_infer_record(payload, path))
    return records


def build_markdown_table(records: List[Dict[str, Any]]) -> str:
    """Build a minimal markdown table for paper-aligned reporting."""

    if not records:
        raise ValueError('No records provided for table generation')

    lines = [
        '| Method | Dataset | Keep Ratio | Backbone | Mean Acc | Std |',
        '| --- | --- | --- | --- | --- | --- |',
    ]
    for record in records:
        lines.append(
            '| {Method} | {Dataset} | {Keep Ratio:.0%} | {Backbone} | {Mean Acc:.4f} | {Std:.4f} |'.format(**record)
        )
    return '\n'.join(lines) + '\n'


def save_table(markdown: str, output_dir: str | Path) -> Path:
    """Save the markdown table to ``table.md``."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / 'table.md'
    table_path.write_text(markdown, encoding='utf-8')
    return table_path


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate a minimal Table-1 markdown summary')
    parser.add_argument('--results', nargs='+', required=True, help='Input result json files')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save table.md')
    args = parser.parse_args()

    records = load_records(args.results)
    markdown = build_markdown_table(records)
    table_path = save_table(markdown, args.output_dir)

    print('records=' + json.dumps(records, ensure_ascii=True))
    print('table_path=' + str(table_path))


if __name__ == '__main__':
    main()
