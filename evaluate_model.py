# evaluate_model.py

import numpy as np
import torch
import matplotlib.pyplot as plt
from train_model import MLP
import os


def load_model(model_path):
    model = MLP()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def evaluate_and_plot(model_name="convection"):
    # Load saved x, t values
    x = np.load(f"data/{model_name}_x.npy")
    t = np.load(f"data/{model_name}_t.npy")
    X, T = np.meshgrid(x, t)

    x_flat = X.flatten()
    t_flat = T.flatten()
    inputs = np.stack([x_flat, t_flat], axis=1)
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)

    model = load_model(f"models/{model_name}_mlp.pth")
    with torch.no_grad():
        preds = model(inputs_tensor).numpy().reshape(len(t), len(x))

    plt.figure(figsize=(10, 6))
    plt.imshow(preds, extent=[x.min(), x.max(), t.max(), t.min()], aspect='auto', cmap='inferno')
    plt.colorbar(label='u')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title(f"Predicted solution for {model_name.capitalize()} Equation")
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/{model_name}_prediction.png")
    plt.show()
    print(f"✅ Prediction graph saved as results/{model_name}_prediction.png")


if __name__ == '__main__':
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "convection"
    evaluate_and_plot(name)
