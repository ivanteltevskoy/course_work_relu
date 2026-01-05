import torch
import math


def smooth_quadratic(x):
    return x ** 2

def smooth_sine(x):
    return torch.sin(2 * math.pi * x)

def piecewise_kink_poly(x):
    left = x**2
    right = x**2 + 0.3 * (x - 0.5)
    return torch.where(x < 0.5, left, right)

def besov_faber_schauder(x):
    result = torch.zeros_like(x)
    for k in range(1, 8):
        t = (2 ** k) * (x - 0.5)
        hat = torch.relu(1.0 - torch.abs(t)) 
        result = result + (1.0 / (k ** 2)) * hat
    return result

def piecewise_smooth(x):
    left = torch.sin(6 * math.pi * x)
    right = torch.sin(6 * math.pi * x) + 0.3 * (x - 0.5)
    return torch.where(x < 0.5, left, right)


def besov_haar_simplified(x):
    result = torch.zeros_like(x)
    for k in range(1, 6):
        arg = (2 ** k) * (x - 0.5)
        wavelet = torch.where(
            (arg >= 0) & (arg < 0.5), 1.0,
            torch.where((arg >= 0.5) & (arg < 1.0), -1.0, 0.0)
        )
        result = result + (1.0 / (k ** 2)) * wavelet
    return result



def get_function_dict():
    return {
        "smooth": smooth_quadratic,
        "piecewise": piecewise_smooth,
        "besov": besov_haar_simplified,

        "smooth_sine": smooth_sine,
        "piecewise_kink_poly": piecewise_kink_poly,
        "besov_faber_schauder": besov_faber_schauder,

    }
