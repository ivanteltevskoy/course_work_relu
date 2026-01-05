import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import (
    RESULTS_DIR, device,
    TRAIN_SIZE, TEST_SIZE,
    SCALING_DEPTH,
    EXP3_FUNCTIONS, EXP3_WIDTHS, EXP3_SEEDS,
    EXP3_NOISE_SIGMAS, EXP3_GRID_SIZES, EXP3_SAMPLERS,
    EXP3_WEIGHT_DECAYS, EXP3_MAX_EPOCHS, EXP3_LEARNING_RATE,
    SEED
)

from functions.functions import get_function_dict
from models.relu_network import ReLUNetwork
from models.model_utils import count_parameters
from training.trainer import train_model
from training.metrics import compute_l2, compute_linf


def _set_seed(seed=SEED):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def sample_x(n, sampler, g):
    if sampler == "grid":
        return torch.linspace(0.0, 1.0, n).reshape(-1, 1)
    if sampler == "random":
        return torch.rand(n, 1, generator=g)
    raise ValueError(f"Unknown sampler='{sampler}'")


def make_dataset(func, train_size, val_size, grid_size, sampler, seed, noise_sigma):
    g = torch.Generator()
    g.manual_seed(seed)

    X_train = sample_x(train_size, sampler, g).to(device)
    X_val = sample_x(val_size, sampler, g).to(device)
    X_grid = torch.linspace(0.0, 1.0, grid_size).reshape(-1, 1).to(device)

    with torch.no_grad():
        y_train = func(X_train)
        y_val = func(X_val)
        y_grid = func(X_grid)

    if noise_sigma and noise_sigma > 0:
        noise = torch.randn(y_train.shape, generator=g).to(device)
        y_train = y_train + noise_sigma * noise

    return (X_train, y_train), (X_val, y_val), (X_grid, y_grid)


def fit_slope(N_vals, err_vals):
    N = np.asarray(N_vals, dtype=float)
    E = np.asarray(err_vals, dtype=float)
    E = np.maximum(E, 1e-12)
    slope, _ = np.polyfit(np.log10(N), np.log10(E), 1)
    return float(slope)


def scenario_list():
    base = {
        "sampler": "random",
        "grid_size": 1000,
        "noise_sigma": 0.0,
        "weight_decay": 0.0,
        "lr": EXP3_LEARNING_RATE,
    }

    scenarios = []

    for seed in EXP3_SEEDS:
        s = dict(base)
        s["name"] = "baseline"
        s["seed"] = seed
        scenarios.append(s)

    for wd in EXP3_WEIGHT_DECAYS:
        s = dict(base)
        s["name"] = f"wd={wd:g}"
        s["seed"] = EXP3_SEEDS[0]
        s["weight_decay"] = wd
        scenarios.append(s)

    for smp in EXP3_SAMPLERS:
        s = dict(base)
        s["name"] = f"sampler={smp}"
        s["seed"] = EXP3_SEEDS[0]
        s["sampler"] = smp
        scenarios.append(s)

    for gs in EXP3_GRID_SIZES:
        s = dict(base)
        s["name"] = f"grid_size={gs}"
        s["seed"] = EXP3_SEEDS[0]
        s["grid_size"] = gs
        scenarios.append(s)

    for ns in EXP3_NOISE_SIGMAS:
        s = dict(base)
        s["name"] = f"noise={ns:g}"
        s["seed"] = EXP3_SEEDS[0]
        s["noise_sigma"] = ns
        scenarios.append(s)

    noise_ref = 0.01
    if noise_ref in EXP3_NOISE_SIGMAS:
        for seed in EXP3_SEEDS:
            s = dict(base)
            s["name"] = f"seed_under_noise={noise_ref:g}"
            s["seed"] = seed
            s["noise_sigma"] = noise_ref
            scenarios.append(s)

    uniq, seen = [], set()
    for s in scenarios:
        key = (s["name"], s["seed"], s["sampler"], s["grid_size"], s["noise_sigma"], s["weight_decay"], s["lr"])
        if key not in seen:
            seen.add(key)
            uniq.append(s)
    return uniq


def run_ablation_experiment():
    all_funcs = get_function_dict()
    funcs = {k: all_funcs[k] for k in EXP3_FUNCTIONS if k in all_funcs}
    if not funcs:
        raise ValueError(f"No functions found for EXP3_FUNCTIONS={EXP3_FUNCTIONS}")

    scenarios = scenario_list()
    point_rows = []

    for sc in tqdm(scenarios, desc="Scenarios"):
        seed = sc["seed"]

        for func_name, func in funcs.items():
            (X_train, y_train), (X_val, y_val), (X_grid, y_grid) = make_dataset(
                func=func,
                train_size=TRAIN_SIZE,
                val_size=TEST_SIZE,
                grid_size=sc["grid_size"],
                sampler=sc["sampler"],
                seed=seed,
                noise_sigma=sc["noise_sigma"],
            )

            for width in EXP3_WIDTHS:
                _set_seed(seed)

                model = ReLUNetwork(depth=SCALING_DEPTH, width=int(width)).to(device)
                N = int(count_parameters(model))

                model, _ = train_model(
                    model,
                    X_train, y_train,
                    X_val=X_val, y_val=y_val,
                    lr=sc["lr"],
                    weight_decay=sc["weight_decay"],
                    max_epochs=EXP3_MAX_EPOCHS,
                    clip_grad_norm=None,
                    verbose=False
                )

                l2 = float(compute_l2(model, X_grid, y_grid))
                linf = float(compute_linf(model, X_grid, y_grid))

                point_rows.append({
                    "scenario": sc["name"],
                    "seed": int(seed),
                    "function": func_name,
                    "sampler": sc["sampler"],
                    "grid_size": int(sc["grid_size"]),
                    "noise_sigma": float(sc["noise_sigma"]),
                    "weight_decay": float(sc["weight_decay"]),
                    "lr": float(sc["lr"]),
                    "depth": int(SCALING_DEPTH),
                    "width": int(width),
                    "N": int(N),
                    "l2": l2,
                    "linf": linf,
                })

    df_points = pd.DataFrame(point_rows)

    slope_rows = []
    group_cols = ["scenario", "seed", "function", "sampler", "grid_size", "noise_sigma", "weight_decay", "lr"]

    for key, g in df_points.groupby(group_cols):
        g = g.sort_values("N")

        slope_l2 = fit_slope(g["N"].values, g["l2"].values)
        slope_linf = fit_slope(g["N"].values, g["linf"].values)

        row = dict(zip(group_cols, key))
        row.update({
            "slope_l2": slope_l2,
            "slope_linf": slope_linf,
            "abs_alpha_l2": abs(slope_l2),
            "abs_alpha_linf": abs(slope_linf),
        })
        slope_rows.append(row)

    df_slopes = pd.DataFrame(slope_rows)

    out_dir = os.path.join(RESULTS_DIR, "data")
    os.makedirs(out_dir, exist_ok=True)

    p_path = os.path.join(out_dir, "exp3_points.csv")
    s_path = os.path.join(out_dir, "exp3_slopes.csv")

    df_points.to_csv(p_path, index=False)
    df_slopes.to_csv(s_path, index=False)

    print(f"Saved: {p_path}")
    print(f"Saved: {s_path}")
    return df_points, df_slopes


if __name__ == "__main__":
    run_ablation_experiment()
