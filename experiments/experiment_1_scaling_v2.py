import os
import torch
import pandas as pd
from tqdm import tqdm

from config import (
    SEED,
    SCALING_DEPTH, SCALING_WIDTHS2, DEPTH_VALUES,
    device, RESULTS_DIR, MAX_EPOCHS,
    EXP1_LR, EXP1_WEIGHT_DECAY, EXP1_CLIP_NORM
)

from functions.functions import get_function_dict
from functions.dataset_utils import create_all_datasets

from models.relu_network import ReLUNetwork
from models.model_utils import count_parameters

from training.trainer import train_model
from training.metrics import compute_l2, compute_linf


NEW_FUNCS = ["smooth_sine", "piecewise_kink_poly", "besov_faber_schauder"]


def _set_seed(seed=SEED):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _train_val_split(X, y, val_frac=0.2, seed=SEED):
    n = X.shape[0]
    n_val = int(val_frac * n)

    g = torch.Generator(device=X.device)
    g.manual_seed(seed)

    perm = torch.randperm(n, generator=g, device=X.device)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    X_val, y_val = X[val_idx], y[val_idx]
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    return (X_tr, y_tr), (X_val, y_val)


def run_scaling_experiment_v2():
    results = []
    all_funcs = get_function_dict()

    functions = {k: all_funcs[k] for k in NEW_FUNCS if k in all_funcs}
    if len(functions) != len(NEW_FUNCS):
        missing = [k for k in NEW_FUNCS if k not in all_funcs]
        raise ValueError(f"Missing functions in get_function_dict(): {missing}")

    FIXED_WIDTH = 40

    for func_name, func in functions.items():
        dataset = create_all_datasets(func, func_name)
        X_train, y_train = dataset["train"]
        X_grid, y_grid = dataset["grid"]

        (X_train, y_train), (X_val, y_val) = _train_val_split(X_train, y_train, val_frac=0.2, seed=SEED)

        for width in tqdm(SCALING_WIDTHS2, desc=f"[fixed_depth v2] {func_name}"):
            _set_seed(SEED)

            model = ReLUNetwork(depth=SCALING_DEPTH, width=width).to(device)
            N = count_parameters(model)

            model, _ = train_model(
                model, X_train, y_train,
                X_val=X_val, y_val=y_val,
                lr=EXP1_LR,
                weight_decay=EXP1_WEIGHT_DECAY,
                clip_grad_norm=EXP1_CLIP_NORM,
                max_epochs=MAX_EPOCHS,
                verbose=False
            )

            l2 = compute_l2(model, X_grid, y_grid)
            linf = compute_linf(model, X_grid, y_grid)

            results.append({
                "experiment_type": "fixed_depth",
                "function": func_name,
                "depth": SCALING_DEPTH,
                "width": width,
                "N": N,
                "l2": l2,
                "linf": linf
            })

        for depth in tqdm(DEPTH_VALUES, desc=f"[fixed_width v2] {func_name}"):
            _set_seed(SEED)

            model = ReLUNetwork(depth=depth, width=FIXED_WIDTH).to(device)
            N = count_parameters(model)

            model, _ = train_model(
                model, X_train, y_train,
                X_val=X_val, y_val=y_val,
                lr=EXP1_LR,
                weight_decay=EXP1_WEIGHT_DECAY,
                clip_grad_norm=EXP1_CLIP_NORM,
                max_epochs=MAX_EPOCHS,
                verbose=False
            )

            l2 = compute_l2(model, X_grid, y_grid)
            linf = compute_linf(model, X_grid, y_grid)

            results.append({
                "experiment_type": "fixed_width",
                "function": func_name,
                "depth": depth,
                "width": FIXED_WIDTH,
                "N": N,
                "l2": l2,
                "linf": linf
            })

    df = pd.DataFrame(results).sort_values(["experiment_type", "function", "N"])

    out_path = os.path.join(RESULTS_DIR, "data", "scaling_results_v2.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Saved: {out_path} | rows={len(df)}")
    return df


if __name__ == "__main__":
    run_scaling_experiment_v2()
