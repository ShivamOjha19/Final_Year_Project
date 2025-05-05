import torch
import torch.nn as nn
import numpy as np

class MLP(nn.Module):
    def __init__(self, in_features=2, hidden=128, out_features=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features)
        )

    def forward(self, x):
        return self.net(x)

def load_model_and_predict(equation, X_input):
    model = MLP()
    model.load_state_dict(torch.load(f"models/{equation}_mlp.pth", map_location='cpu'))
    model.eval()

    X_tensor = torch.tensor(X_input, dtype=torch.float32)
    with torch.no_grad():
        y_pred = model(X_tensor).numpy().flatten().tolist()

    return y_pred
