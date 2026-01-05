import torch
from config import SEED, TRAIN_SIZE, TEST_SIZE, GRID_SIZE, device

def create_train_test_data(func, n_train=TRAIN_SIZE, n_test=TEST_SIZE, seed=SEED):
    g = torch.Generator()
    g.manual_seed(seed)

    X_train = torch.rand(n_train, 1, generator=g).to(device)
    X_test = torch.rand(n_test, 1, generator=g).to(device)

    y_train = func(X_train)
    y_test = func(X_test)

    return X_train, y_train, X_test, y_test

def create_uniform_grid(func, n_points=GRID_SIZE):
    X = torch.linspace(0.0, 1.0, n_points).reshape(-1, 1).to(device)
    y = func(X)
    return X, y

def create_all_datasets(func, func_name):
    X_train, y_train, X_test, y_test = create_train_test_data(func)
    X_grid, y_grid = create_uniform_grid(func)

    return {
        "train": (X_train, y_train),
        "test": (X_test, y_test),
        "grid": (X_grid, y_grid),
        "name": func_name,
    }
