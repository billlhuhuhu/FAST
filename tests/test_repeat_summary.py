"""Lightweight tests for repeat result summarization."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_repeat_results import build_markdown_table, collect_result_files, save_summary_outputs, summarize_result_files


class RepeatSummaryTestCase(unittest.TestCase):
    """Small tests for repeat summary aggregation."""

    def test_repeat_summary_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_a = root / 'run_00'
            run_b = root / 'run_01'
            run_a.mkdir(parents=True, exist_ok=True)
            run_b.mkdir(parents=True, exist_ok=True)

            (run_a / 'run_summary.json').write_text(json.dumps({'test_accuracy': 0.70, 'best_accuracy': 0.72, 'seed': 1, 'method': 'fast', 'keep_ratio': 0.1}), encoding='utf-8')
            (run_b / 'run_summary.json').write_text(json.dumps({'test_accuracy': 0.80, 'best_accuracy': 0.81, 'seed': 2, 'method': 'fast', 'keep_ratio': 0.1}), encoding='utf-8')

            files = collect_result_files(root)
            summary = summarize_result_files(files)
            markdown = build_markdown_table(summary)
            saved = save_summary_outputs(summary, root / 'summary_out')

            self.assertEqual(len(files), 2)
            self.assertAlmostEqual(summary['mean_accuracy'], 0.75, places=6)
            self.assertAlmostEqual(summary['std_accuracy'], 0.05, places=6)
            self.assertEqual(summary['per_run_accuracy'], [0.7, 0.8])
            self.assertIn('| run | seed | method | keep_ratio | test_accuracy | best_accuracy |', markdown)
            self.assertTrue(Path(saved['summary_json']).exists())
            self.assertTrue(Path(saved['summary_md']).exists())


if __name__ == '__main__':
    unittest.main()
