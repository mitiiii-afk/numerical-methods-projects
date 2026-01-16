import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# -----------------------------
# Global parameters
# -----------------------------
Nx, Ny = 150, 150        # higher resolution for beautiful video
alpha = 1.0
Lx, Ly = 1.0, 1.0

dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)
dt = 0.2 * dx**2         # stability condition
n_frames = 300           # number of frames in the GIF
steps_per_frame = 5      # how many PDE steps between frames

initial_pattern    = "ring"
boundary_condition = "periodic"


# -----------------------------
# Initial condition (ring shape)
# -----------------------------
def init_temperature() -> np.ndarray:
    u0 = np.zeros((Nx, Ny))

    hot_value = 100.0
    cx, cy = 0.5, 0.5     # center
    r_inner = 0.12
    r_outer = 0.18

    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    R2 = (X - cx)**2 + (Y - cy)**2
    mask = (R2 >= r_inner**2) & (R2 <= r_outer**2)
    u0[mask] = hot_value

    return u0


# -----------------------------
# One time step (periodic BC)
# -----------------------------
def step(u: np.ndarray) -> None:
    # Periodic Laplacian using np.roll
    u_xx = (np.roll(u, -1, axis=0) - 2*u + np.roll(u, 1, axis=0)) / dx**2
    u_yy = (np.roll(u, -1, axis=1) - 2*u + np.roll(u, 1, axis=1)) / dy**2

    u[:, :] += dt * alpha * (u_xx + u_yy)


# -----------------------------
# Create and save GIF
# -----------------------------
def run_video():
    print("Generating GIF...")

    u = init_temperature()

    # Create figure
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(
        u.T,
        origin="lower",
        extent=[0, Lx, 0, Ly],
        cmap="inferno",
        vmin=0.0,
        vmax=100.0,
        animated=True
    )
    cbar = plt.colorbar(im, ax=ax, label="Temperature")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Update function for animation
    def update(frame):
        for _ in range(steps_per_frame):
            step(u)
        im.set_data(u.T)
        ax.set_title(
            f"Heat Diffusion (ring, periodic)  t = {frame * steps_per_frame * dt:.4f}"
        )
        return [im]

    # Create animation
    ani = FuncAnimation(fig, update, frames=n_frames, blit=True)

    # Save as GIF
    writer = PillowWriter(fps=25)
    ani.save("heat_diffusion_demo.gif", writer=writer)
    plt.close(fig)

    print("Saved GIF as heat_diffusion_demo.gif")


if __name__ == "__main__":
    run_video()
