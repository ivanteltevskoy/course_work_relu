import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import RESULTS_DIR

def _scenario_order(name):
    if name == "baseline":
        return 0
    if name.startswith("wd="):
        return 10
    if name.startswith("sampler="):
        return 20
    if name.startswith("grid_size="):
        return 30
    if name.startswith("noise="):
        return 40
    if name.startswith("seed_under_noise="):
        return 50
    return 999

def plot_ablation_results():
    csv_path = os.path.join(RESULTS_DIR, "data", "exp3_slopes.csv")
    if not os.path.exists(csv_path):
        print(f"Missing: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("Empty exp3_slopes.csv")
        return

    agg = (
        df.groupby(["scenario", "function"])
          .agg(
              abs_alpha_l2_mean=("abs_alpha_l2", "mean"),
              abs_alpha_l2_std=("abs_alpha_l2", "std"),
              abs_alpha_linf_mean=("abs_alpha_linf", "mean"),
              abs_alpha_linf_std=("abs_alpha_linf", "std"),
          )
          .reset_index()
    )

    agg["abs_alpha_l2_std"] = agg["abs_alpha_l2_std"].fillna(0.0)
    agg["abs_alpha_linf_std"] = agg["abs_alpha_linf_std"].fillna(0.0)

    out_dir = os.path.join(RESULTS_DIR, "figures")
    os.makedirs(out_dir, exist_ok=True)

    functions = sorted(agg["function"].unique())

    for func in functions:
        a = agg[agg["function"] == func].copy()
        a["order"] = a["scenario"].map(_scenario_order)
        a = a.sort_values(["order", "scenario"])

        labels = a["scenario"].tolist()
        x = np.arange(len(labels))

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        ax1, ax2 = axes

        ax1.bar(x, a["abs_alpha_l2_mean"].values, yerr=a["abs_alpha_l2_std"].values, capsize=3)
        ax1.set_title(f"{func}: |alpha| по L2")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha="right")
        ax1.set_ylabel("|alpha|")
        ax1.grid(True, alpha=0.3)

        ax2.bar(x, a["abs_alpha_linf_mean"].values, yerr=a["abs_alpha_linf_std"].values, capsize=3)
        ax2.set_title(f"{func}: |alpha| по L∞")
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=45, ha="right")
        ax2.set_ylabel("|alpha|")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        out = os.path.join(out_dir, f"exp3_ablation_{func}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Saved: {out}")

    summary_path = os.path.join(RESULTS_DIR, "data", "exp3_slopes_summary.csv")
    agg.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

if __name__ == "__main__":
    plot_ablation_results()
