import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import os

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

def load_dataset(name):
    X = np.load(f"dataset/{name}_X.npy")
    y = np.load(f"dataset/{name}_y.npy")

    print(f"[DEBUG] X: min={X.min()}, max={X.max()}, mean={X.mean()}")
    print(f"[DEBUG] y: min={y.min()}, max={y.max()}, mean={y.mean()}")
    print(f"[DEBUG] Any NaNs in X: {np.isnan(X).any()}, Infs: {np.isinf(X).any()}")
    print(f"[DEBUG] Any NaNs in y: {np.isnan(y).any()}, Infs: {np.isinf(y).any()}")

    # Normalize y
    y = (y - y.mean()) / y.std()

    return X, y

def train_model(name="convection", epochs=1000, lr=1e-3):
    X, y = load_dataset(name)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).to(device)

    model = MLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0 or epoch == epochs - 1:
            model.eval()
            val_loss = criterion(model(X_val), y_val).item()
            print(f"Epoch {epoch}: Train Loss = {loss.item():.5f}, Val Loss = {val_loss:.5f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), f"models/{name}_mlp.pth")
    print(f"✅ Model saved to models/{name}_mlp.pth")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("equation", type=str, help="Name of the equation: convection, conv_diff, or burgers")
    args = parser.parse_args()

    train_model(name=args.equation)

