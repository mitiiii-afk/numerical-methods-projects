import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Global parameters
# -----------------------------
Nx, Ny = 120, 120         # grid size (increase for smoother patterns)
alpha = 1.0               # thermal diffusivity
Lx, Ly = 1.0, 1.0         # physical size of plate

dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1) 
dt = 0.2 * dx**2          # time step (stability condition)
n_steps = 600             # number of time steps

# Choose what to simulate:
initial_pattern    = "ring"      # "square", "circle", "two_spots", "ring"
boundary_condition = "periodic"       # "fixed", "insulated", "periodic"


# -----------------------------
# Initial condition functions
# -----------------------------
def init_temperature(pattern: str) -> np.ndarray:
    """
    Create initial temperature distribution u(x, y).
    pattern: "square", "circle", "two_spots", or "ring".
    """
    u0 = np.zeros((Nx, Ny))

    if pattern == "square":
        hot_value = 100.0
        x_start = int(0.4 * Nx)
        x_end   = int(0.6 * Nx)
        y_start = int(0.4 * Ny)
        y_end   = int(0.6 * Ny)
        u0[x_start:x_end, y_start:y_end] = hot_value

    elif pattern == "circle":
        hot_value = 100.0
        cx, cy = 0.5, 0.5      # center in physical coords
        r = 0.15               # radius
        x = np.linspace(0, Lx, Nx)
        y = np.linspace(0, Ly, Ny)
        X, Y = np.meshgrid(x, y, indexing="ij")
        mask = (X - cx)**2 + (Y - cy)**2 <= r**2
        u0[mask] = hot_value

    elif pattern == "two_spots":
        hot_value = 100.0
        size = 6
        # left hot spot
        cx1, cy1 = int(0.3 * Nx), int(0.4 * Ny)
        u0[cx1-size:cx1+size, cy1-size:cy1+size] = hot_value
        # right hot spot
        cx2, cy2 = int(0.7 * Nx), int(0.6 * Ny)
        u0[cx2-size:cx2+size, cy2-size:cy2+size] = hot_value

    elif pattern == "ring":
        hot_value = 100.0
        cx, cy = 0.5, 0.5
        r_inner = 0.12
        r_outer = 0.18
        x = np.linspace(0, Lx, Nx)
        y = np.linspace(0, Ly, Ny)
        X, Y = np.meshgrid(x, y, indexing="ij")
        R2 = (X - cx)**2 + (Y - cy)**2
        mask = (R2 >= r_inner**2) & (R2 <= r_outer**2)
        u0[mask] = hot_value

    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    return u0


# -----------------------------
# Boundary conditions
# -----------------------------
def apply_boundary(u: np.ndarray) -> None:
    """
    Apply boundary conditions in-place.
    boundary_condition:
        - "fixed": edges held at temperature 0 (Dirichlet)
        - "insulated": zero normal derivative at edges (Neumann)
    Periodic BC is handled differently inside step().
    """
    if boundary_condition == "fixed":
        u[0, :]  = 0.0
        u[-1, :] = 0.0
        u[:, 0]  = 0.0
        u[:, -1] = 0.0

    elif boundary_condition == "insulated":
        # zero gradient: copy neighbour values
        u[0, :]  = u[1, :]
        u[-1, :] = u[-2, :]
        u[:, 0]  = u[:, 1]
        u[:, -1] = u[:, -2]


# -----------------------------
# One time step of the PDE
# -----------------------------
def step(u: np.ndarray) -> None:
    """
    Advance the solution u by one time step using explicit finite differences
    for the 2D heat equation.
    """
    if boundary_condition == "periodic":
        # Periodic: wrap around using np.roll in both directions
        u_xx = (np.roll(u, -1, axis=0) - 2*u + np.roll(u, 1, axis=0)) / dx**2
        u_yy = (np.roll(u, -1, axis=1) - 2*u + np.roll(u, 1, axis=1)) / dy**2
        u[:, :] += dt * alpha * (u_xx + u_yy)

    else:
        # Interior points Laplacian (non-periodic)
        laplacian = (
            (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx**2 +
            (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) / dy**2
        )

        u[1:-1, 1:-1] += dt * alpha * laplacian
        apply_boundary(u)


# -----------------------------
# Main simulation + animation
# -----------------------------
def run_simulation():
    print(f"Pattern           : {initial_pattern}")
    print(f"Boundary condition: {boundary_condition}")
    print(f"Nx = {Nx}, Ny = {Ny}, dt = {dt:.3e}, steps = {n_steps}")

    # initial condition
    u = init_temperature(initial_pattern)

    # figure & colormap
    plt.ion()
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(
        u.T,
        origin="lower",
        extent=[0, Lx, 0, Ly],
        cmap="inferno",           # nicer colormap
        vmin=0.0,
        vmax=100.0                # keep color scale fixed
    )
    cbar = plt.colorbar(im, ax=ax, label="Temperature")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Heat diffusion")
    plt.tight_layout()

    # time loop
    for n in range(n_steps):
        step(u)

        if n % 5 == 0:
            im.set_data(u.T)
            ax.set_title(
                f"Heat diffusion ({initial_pattern}, {boundary_condition}), "
                f"step {n}, t = {n*dt:.4f}"
            )
            plt.pause(0.001)

    plt.ioff()
    plt.show()
    print("Simulation finished.")


if __name__ == "__main__":
    run_simulation()
