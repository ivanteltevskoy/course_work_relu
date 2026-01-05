import os
import pandas as pd
import matplotlib.pyplot as plt
from config import RESULTS_DIR


def plot_depth_comparison():
    csv_path = os.path.join(RESULTS_DIR, "data", "depth_results.csv")
    df = pd.read_csv(csv_path)

    if df.empty:
        print("No data")
        return

    df = df.sort_values(["function", "depth"])
    funcs = list(df["function"].unique())
    n = len(funcs)

    fig, axes = plt.subplots(3, n, figsize=(5 * n, 12))

    if n == 1:
        axes = axes.reshape(3, 1)

    for j, func_name in enumerate(funcs):
        d = df[df["function"] == func_name].sort_values("depth")
        depths = d["depth"].tolist()

        ax = axes[0, j]
        ax.plot(d["depth"], d["l2"], "o-", linewidth=2, markersize=7)
        ax.set_yscale("log")
        ax.set_title(f"{func_name}: L2 vs depth")
        ax.set_xlabel("Depth (number of Linear layers)")
        ax.set_ylabel("L2 error (log scale)")
        ax.set_xticks(depths)
        ax.grid(True, alpha=0.3)

        ax = axes[1, j]
        ax.plot(d["depth"], d["linf"], "s--", linewidth=2, markersize=7)
        ax.set_yscale("log")
        ax.set_title(f"{func_name}: L∞ vs depth")
        ax.set_xlabel("Depth (number of Linear layers)")
        ax.set_ylabel("L∞ error (log scale)")
        ax.set_xticks(depths)
        ax.grid(True, alpha=0.3)

        ax = axes[2, j]
        ax.plot(d["depth"], d["width"], "o-", linewidth=2, markersize=7)
        ax.set_title(f"{func_name}: width vs depth (fixed N)")
        ax.set_xlabel("Depth (number of Linear layers)")
        ax.set_ylabel("Width")
        ax.set_xticks(depths)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    out = os.path.join(RESULTS_DIR, "figures", "depth_comparison.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out}")


if __name__ == "__main__":
    plot_depth_comparison()
