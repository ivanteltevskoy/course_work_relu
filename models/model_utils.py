import math
import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_params_for_relu_mlp(depth, width, input_dim=1, output_dim=1):
    k = max(depth - 2, 0)
    first = width * input_dim + width
    middle = k * (width * width + width)
    last = output_dim * width + output_dim
    return first + middle + last

def find_width_for_params(depth, target_params, input_dim=1, output_dim=1):
    k = max(depth - 2, 0)

    a = k
    b = (input_dim + 1) + k + output_dim
    c = output_dim - target_params

    if a == 0:
        w_float = (target_params - output_dim) / b
    else:
        disc = b * b - 4 * a * c
        w_float = (-b + math.sqrt(max(disc, 0.0))) / (2 * a)

    base = max(int(w_float), 10)

    candidates = sorted(set([max(base - 2, 10), max(base - 1, 10), base, base + 1, base + 2]))

    best_w = candidates[0]
    best_err = abs(estimate_params_for_relu_mlp(depth, best_w, input_dim, output_dim) - target_params)

    for w in candidates[1:]:
        err = abs(estimate_params_for_relu_mlp(depth, w, input_dim, output_dim) - target_params)
        if err < best_err:
            best_err = err
            best_w = w

    return best_w
