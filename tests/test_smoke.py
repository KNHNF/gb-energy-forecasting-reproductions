"""Check the saved comparison inputs and regenerate the root figure."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_json(*parts: str) -> dict:
    return json.loads((ROOT.joinpath(*parts)).read_text(encoding="utf-8"))


class SavedResultsSmokeTests(unittest.TestCase):
    def test_saved_result_inputs_have_the_metrics_used_by_the_figure(self) -> None:
        lucas_main = load_json("lucas-2020-reproduction", "results", "paper_comparison.json")
        lucas_paper = load_json("lucas-2020-reproduction", "results", "paper_comparison_paper_window.json")
        deng = load_json("deng-2023-reproduction", "results", "model_comparison.json")
        bunn = load_json("bunn-ganesh-2021-2024-reproduction", "results", "bunn_2021_comparison.json")

        self.assertIsInstance(lucas_main["paper_xgboost"]["MAE"], (int, float))
        self.assertIsInstance(lucas_paper["this_reproduction_xgboost"]["MAE"], (int, float))
        self.assertIsInstance(deng["point_forecast_metrics"]["SeasonalAttentionBiLSTM"]["RMSE"], (int, float))
        self.assertIsInstance(bunn["this_reproduction"]["out_of_sample_RMSE"]["regime_switching"], (int, float))

    def test_comparison_figure_regenerates_from_saved_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "reported_vs_reproduced.png"
            subprocess.run(
                [sys.executable, "make_summary_figure.py", "--output", str(output)],
                cwd=ROOT,
                check=True,
            )
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
