import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from config import device, MAX_EPOCHS, BATCH_SIZE

def train_model(
    model,
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    lr=1e-3,
    weight_decay=0.0,
    max_epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    clip_grad_norm=None,
    verbose=False,
):
    model = model.to(device)
    X_train, y_train = X_train.to(device), y_train.to(device)
    if X_val is not None and y_val is not None:
        X_val, y_val = X_val.to(device), y_val.to(device)

    dl = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"train": [], "val": []}
    best_state = None
    best_val = float("inf")

    for epoch in range(max_epochs):
        model.train()
        total, nb = 0.0, 0

        for xb, yb in dl:
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()

            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

            optimizer.step()
            total += float(loss.item())
            nb += 1

        train_loss = total / max(1, nb)
        history["train"].append(train_loss)

        if X_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                val_loss = float(criterion(model(X_val), y_val).item())
            history["val"].append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())

        if verbose and (epoch % 500 == 0 or epoch == max_epochs - 1):
            if history["val"]:
                print(f"Epoch {epoch+1}/{max_epochs}: train={train_loss:.3e}, val={history['val'][-1]:.3e}")
            else:
                print(f"Epoch {epoch+1}/{max_epochs}: train={train_loss:.3e}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history
