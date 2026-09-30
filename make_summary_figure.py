"""Draw docs/reported_vs_reproduced.png from the saved result files in each reproduction."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def load(*parts):
    return json.loads((ROOT.joinpath(*parts)).read_text(encoding="utf-8"))


lucas_main = load("lucas-2020-reproduction", "results", "paper_comparison.json")
lucas_paper = load("lucas-2020-reproduction", "results", "paper_comparison_paper_window.json")
deng = load("deng-2023-reproduction", "results", "model_comparison.json")
bunn = load("bunn-ganesh-2021-2024-reproduction", "results", "bunn_2021_comparison.json")

PAPER, REPRO = "#9ca3af", "#0f6b5c"

panels = [
    ("Lucas et al. 2020\nXGBoost, MAE", [
        ("Paper", lucas_main["paper_xgboost"]["MAE"], PAPER),
        ("Mine, paper's\nwindow 2018-19", lucas_paper["this_reproduction_xgboost"]["MAE"], REPRO),
        ("Mine, recent\nwindow 2023-25", lucas_main["this_reproduction_xgboost"]["MAE"], REPRO),
    ]),
    ("Deng et al. 2023\n1-step RMSE", [
        ("Paper", deng["paper_reported_rmse_1step_full_sample"], PAPER),
        ("Mine,\nBiLSTM", deng["point_forecast_metrics"]["BiLSTM_baseline"]["RMSE"], REPRO),
        ("Mine, seasonal\nattention", deng["point_forecast_metrics"]["SeasonalAttentionBiLSTM"]["RMSE"], REPRO),
    ]),
    ("Bunn et al. 2021\nout-of-sample RMSE, regime switching", [
        ("Paper", bunn["paper"]["out_of_sample_RMSE"]["regime_switching"], PAPER),
        ("Mine", bunn["this_reproduction"]["out_of_sample_RMSE"]["regime_switching"], REPRO),
    ]),
]

fig, axes = plt.subplots(1, 3, figsize=(11, 4), dpi=110, gridspec_kw={"width_ratios": [3, 3, 2]})
fig.patch.set_facecolor("#faf9f6")
for ax, (title, bars) in zip(axes, panels):
    ax.set_facecolor("#faf9f6")
    labels = [b[0] for b in bars]
    vals = [b[1] for b in bars]
    ax.bar(labels, vals, color=[b[2] for b in bars], width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.02, f"{v:.1f}", ha="center", fontsize=9)
    ax.set_title(title, fontsize=9, loc="left")
    ax.set_ylim(0, max(vals) * 1.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7.5)
axes[0].set_ylabel("GBP per MWh (lower is better)", fontsize=8)
fig.suptitle("Three reproductions against the numbers the papers report", x=0.01, ha="left", fontsize=11)
fig.text(0.01, 0.01, "Grey is the paper, green is my reproduction. Only the Lucas MAE on the paper's own window matches. Lucas R2 and the other two papers do not.",
         fontsize=7.5, color="#555555")
fig.tight_layout(rect=(0, 0.04, 1, 0.94))
out = ROOT / "docs" / "reported_vs_reproduced.png"
fig.savefig(out)
print("saved", out, out.stat().st_size // 1024, "KB")
