import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import RESULTS_DIR


V2_FUNCTIONS = ["smooth", "piecewise_kink_poly", "besov_faber_schauder"]


def fit_power_law_tail(N, err, tail_k=3):
    N = np.asarray(N, dtype=float)
    E = np.asarray(err, dtype=float)
    E = np.maximum(E, 1e-12)

    order = np.argsort(N)
    N = N[order]
    E = E[order]

    k = min(tail_k, len(N))
    N_tail = N[-k:]
    E_tail = E[-k:]

    x = np.log10(N_tail)
    y = np.log10(E_tail)
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def plot_one(df_exp, exp_type, show_fit=True, tail_k=3):
    df_exp = df_exp.sort_values(["function", "N"])
    funcs = sorted(df_exp["function"].unique())

    fig, axes = plt.subplots(len(funcs), 2, figsize=(13, 4 * len(funcs)))
    if len(funcs) == 1:
        axes = np.array([axes])

    for i, func in enumerate(funcs):
        d = df_exp[df_exp["function"] == func].sort_values("N")

        N = d["N"].to_numpy(dtype=float)
        l2 = d["l2"].to_numpy(dtype=float)
        linf = d["linf"].to_numpy(dtype=float)

        ax = axes[i, 0]
        ax.plot(N, l2, "o-", linewidth=2, markersize=6)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{func} / {exp_type}: L2 vs N (log-log)")
        ax.set_xlabel("N (params)")
        ax.set_ylabel("L2 error")
        ax.grid(True, alpha=0.3)

        if show_fit and len(N) >= 2:
            slope, intercept = fit_power_law_tail(N, l2, tail_k=tail_k)
            N_line = np.logspace(np.log10(N.min()), np.log10(N.max()), 200)
            y_line = (10 ** intercept) * (N_line ** slope)
            ax.plot(N_line, y_line, "--", linewidth=2, label=f"slope={slope:.3f}, α≈{-slope:.3f}")
            ax.legend()

        ax = axes[i, 1]
        ax.plot(N, linf, "o-", linewidth=2, markersize=6)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{func} / {exp_type}: L∞ vs N (log-log)")
        ax.set_xlabel("N (params)")
        ax.set_ylabel("L∞ error")
        ax.grid(True, alpha=0.3)

        if show_fit and len(N) >= 2:
            slope, intercept = fit_power_law_tail(N, linf, tail_k=tail_k)
            N_line = np.logspace(np.log10(N.min()), np.log10(N.max()), 200)
            y_line = (10 ** intercept) * (N_line ** slope)
            ax.plot(N_line, y_line, "--", linewidth=2, label=f"slope={slope:.3f}, α≈{-slope:.3f}")
            ax.legend()

    plt.tight_layout()

    out_dir = os.path.join(RESULTS_DIR, "figures")
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, f"scaling_{exp_type}_v2.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {fig_path}")


def plot_scaling_results_v2(show_fit=True, tail_k=3):

    csv_path = os.path.join(RESULTS_DIR, "data", "scaling_results_v2.csv")
    if not os.path.exists(csv_path):
        print(f"Missing: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    df = df[df["function"].isin(V2_FUNCTIONS)].copy()

    for exp_type in ["fixed_depth", "fixed_width"]:
        df_exp = df[df["experiment_type"] == exp_type].copy()
        if len(df_exp) == 0:
            continue
        plot_one(df_exp, exp_type, show_fit=show_fit, tail_k=tail_k)


if __name__ == "__main__":
    plot_scaling_results_v2(show_fit=True, tail_k=3)
