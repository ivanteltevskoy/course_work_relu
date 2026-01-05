import os
import torch
import pandas as pd
from tqdm import tqdm

from config import (
    SEED,
    DEPTH_VALUES, FIXED_PARAMS, device, RESULTS_DIR,
    EXP2_LR, EXP2_WEIGHT_DECAY, EXP2_CLIP_NORM,
    EXP2_MAX_EPOCHS
)

from functions.functions import get_function_dict
from functions.dataset_utils import create_all_datasets

from models.relu_network import ReLUNetwork
from models.model_utils import count_parameters, find_width_for_params

from training.trainer import train_model
from training.metrics import compute_l2, compute_linf


KEEP_FUNCS = ["smooth", "piecewise_kink_poly", "besov_faber_schauder"]


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

    return (X[tr_idx], y[tr_idx]), (X[val_idx], y[val_idx])


def run_depth_experiment(val_frac=0.2):
    results = []
    all_functions = get_function_dict()

    missing = [k for k in KEEP_FUNCS if k not in all_functions]
    if missing:
        raise ValueError(f"Missing functions in get_function_dict(): {missing}")

    functions = {k: all_functions[k] for k in KEEP_FUNCS}

    for func_name, func in functions.items():
        dataset = create_all_datasets(func, func_name)

        X_train_full, y_train_full = dataset["train"]
        X_grid, y_grid = dataset["grid"]

        (X_train, y_train), (X_val, y_val) = _train_val_split(
            X_train_full, y_train_full, val_frac=val_frac, seed=SEED
        )

        for depth in tqdm(DEPTH_VALUES, desc=f"{func_name}"):
            _set_seed(SEED)

            width = int(find_width_for_params(depth, FIXED_PARAMS))
            model = ReLUNetwork(depth=int(depth), width=width).to(device)
            N = int(count_parameters(model))

            model, _ = train_model(
                model,
                X_train, y_train,
                X_val=X_val, y_val=y_val,
                lr=EXP2_LR,
                weight_decay=EXP2_WEIGHT_DECAY,
                clip_grad_norm=EXP2_CLIP_NORM,
                max_epochs=EXP2_MAX_EPOCHS,
                verbose=False
            )

            l2 = float(compute_l2(model, X_grid, y_grid))
            linf = float(compute_linf(model, X_grid, y_grid))

            results.append({
                "function": func_name,
                "depth": int(depth),
                "width": int(width),
                "N": int(N),
                "l2": l2,
                "linf": linf,
            })

            print(
                f"func={func_name:>18} | depth={int(depth):>2} | width={int(width):>4} | "
                f"N={int(N):>6} | l2={l2:.6g} | linf={linf:.6g}"
            )

    df = pd.DataFrame(results).sort_values(["function", "depth"])

    out_path = os.path.join(RESULTS_DIR, "data", "depth_results.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Saved: {out_path} | rows={len(df)}")
    return df


if __name__ == "__main__":
    run_depth_experiment()
