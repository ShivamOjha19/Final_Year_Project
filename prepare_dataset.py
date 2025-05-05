# prepare_dataset.py

import numpy as np
import os
from sklearn.model_selection import train_test_split

def load_and_prepare_data(u_file, x_file, t_file):
    u = np.load(u_file)  # shape: (nt, nx)
    x = np.load(x_file)  # shape: (nx,)
    t = np.load(t_file)  # shape: (nt,)

    nt, nx = u.shape

    # Create meshgrid of (x, t) pairs
    X, T = np.meshgrid(x, t)

    # Flatten and create input-output pairs
    X_input = np.stack([X.flatten(), T.flatten()], axis=1)  # shape: (nt * nx, 2)
    y_output = u.flatten().reshape(-1, 1)                   # shape: (nt * nx, 1)

    return X_input, y_output

def save_dataset(X, y, out_folder, prefix):
    os.makedirs(out_folder, exist_ok=True)
    np.save(os.path.join(out_folder, f"{prefix}_X.npy"), X)
    np.save(os.path.join(out_folder, f"{prefix}_y.npy"), y)

def generate_all():
    input_folder = "data"
    output_folder = "dataset"

    equations = [
        ("convection", "convection_u.npy", "convection_x.npy", "convection_t.npy"),
        ("conv_diff", "conv_diff_u.npy", "conv_diff_x.npy", "conv_diff_t.npy"),
        ("burgers", "burgers_u.npy", "burgers_x.npy", "burgers_t.npy")
    ]

    for name, u_f, x_f, t_f in equations:
        print(f"📦 Preparing dataset for {name} equation")
        X, y = load_and_prepare_data(
            os.path.join(input_folder, u_f),
            os.path.join(input_folder, x_f),
            os.path.join(input_folder, t_f),
        )
        save_dataset(X, y, output_folder, name)

    print("✅ All datasets saved in ./dataset folder")

if __name__ == '__main__':
    generate_all()
