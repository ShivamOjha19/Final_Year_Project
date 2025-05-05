# pde_solver_data_gen.py

import numpy as np
import matplotlib.pyplot as plt
import os

def convection_1d(c=1.0, L=1.0, T=1.0, nx=100, nt=100, u0_fn=lambda x: np.sin(2 * np.pi * x)):
    dx = L / (nx - 1)
    dt = T / nt
    x = np.linspace(0, L, nx)
    t = np.linspace(0, T, nt)
    u = u0_fn(x)
    u_hist = [u.copy()]
    
    for n in range(1, nt):
        u_new = u.copy()
        u_new[1:] = u[1:] - c * dt / dx * (u[1:] - u[:-1])
        u = u_new.copy()
        u_hist.append(u)

    return np.array(u_hist), x, t

def convection_diffusion_1d(c=1.0, D=0.01, L=1.0, T=1.0, nx=100, nt=100, u0_fn=lambda x: np.sin(2 * np.pi * x)):
    dx = L / (nx - 1)
    dt = T / nt
    x = np.linspace(0, L, nx)
    t = np.linspace(0, T, nt)
    u = u0_fn(x)
    u_hist = [u.copy()]
    
    for n in range(1, nt):
        u_new = u.copy()
        u_new[1:-1] = u[1:-1] - c * dt / (2 * dx) * (u[2:] - u[:-2]) + D * dt / dx**2 * (u[2:] - 2*u[1:-1] + u[:-2])
        u = u_new.copy()
        u_hist.append(u)

    return np.array(u_hist), x, t

def burgers_1d(nu=0.01, L=1.0, T=1.0, nx=100, nt=100, u0_fn=lambda x: -np.sin(np.pi * x)):
    dx = L / (nx - 1)
    dt = T / nt
    x = np.linspace(0, L, nx)
    t = np.linspace(0, T, nt)
    u = u0_fn(x)
    u_hist = [u.copy()]

    for n in range(1, nt):
        u_new = u.copy()
        u_new[1:-1] = (u[1:-1] - u[1:-1] * dt / (2 * dx) * (u[2:] - u[:-2]) 
                      + nu * dt / dx**2 * (u[2:] - 2 * u[1:-1] + u[:-2]))
        u = u_new.copy()
        u_hist.append(u)

    return np.array(u_hist), x, t

def save_data(folder="data"):
    os.makedirs(folder, exist_ok=True)

    # Convection
    u_hist, x, t = convection_1d()
    np.save(os.path.join(folder, "convection_u.npy"), u_hist)
    np.save(os.path.join(folder, "convection_x.npy"), x)
    np.save(os.path.join(folder, "convection_t.npy"), t)

    # Convection-Diffusion
    u_hist, x, t = convection_diffusion_1d()
    np.save(os.path.join(folder, "conv_diff_u.npy"), u_hist)
    np.save(os.path.join(folder, "conv_diff_x.npy"), x)
    np.save(os.path.join(folder, "conv_diff_t.npy"), t)

    # Burgers'
    u_hist, x, t = burgers_1d()
    np.save(os.path.join(folder, "burgers_u.npy"), u_hist)
    np.save(os.path.join(folder, "burgers_x.npy"), x)
    np.save(os.path.join(folder, "burgers_t.npy"), t)

if __name__ == '__main__':
    save_data()
    print("✅ Data saved in ./data folder")
