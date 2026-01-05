import torch

def compute_l2(model, X, y_true):
    model.eval()
    with torch.no_grad():
        y_pred = model(X)
        err2 = (y_pred - y_true).pow(2).squeeze()

        x = X.squeeze()
        val = torch.trapz(err2, x)
        return float(torch.sqrt(torch.clamp(val, min=1e-12)).item())

def compute_linf(model, X, y_true):
    model.eval()
    with torch.no_grad():
        y_pred = model(X)
        return float(torch.max(torch.abs(y_pred - y_true)).item())
