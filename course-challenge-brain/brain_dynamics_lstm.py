import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# -----------------------------
# 1. Generate synthetic "brain-like" multichannel signals
# -----------------------------

def generate_brain_signals(T=3000, n_channels=5, noise_level=0.05, seed=0):
    """
    Generate synthetic multichannel time series that look like coupled neural activity.
    Each channel is a combination of sinusoidal oscillations with slightly different
    frequencies and some coupling + noise.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 60, T)  # time axis (arbitrary units)

    signals = np.zeros((T, n_channels))

    # base frequencies for each "region"
    base_freqs = np.linspace(0.5, 1.5, n_channels)  # different rhythms
    phases = rng.uniform(0, 2*np.pi, size=n_channels)

    for k in range(n_channels):
        signals[:, k] = np.sin(base_freqs[k] * t + phases[k])

    # Add some cross-channel coupling: each channel influenced by neighbours
    for k in range(n_channels):
        if k > 0:
            signals[:, k] += 0.2 * signals[:, k-1]
        if k < n_channels - 1:
            signals[:, k] += 0.2 * signals[:, k+1]

    # Add small random noise
    signals += noise_level * rng.normal(size=signals.shape)

    # Normalise each channel (zero mean, unit variance)
    signals = (signals - signals.mean(axis=0, keepdims=True)) / signals.std(axis=0, keepdims=True)

    return t, signals  # t: (T,), signals: (T, n_channels)


# -----------------------------
# 2. Build sequences for supervised learning
#    given past seq_len steps (for all channels) -> predict next step (all channels)
# -----------------------------

def create_multichannel_sequences(data, seq_len):
    """
    data: array of shape (T, n_channels)
    returns:
      X: (N, seq_len, n_channels)
      y: (N, n_channels)
    """
    xs, ys = [], []
    T, n_channels = data.shape
    for i in range(T - seq_len):
        x = data[i:i+seq_len, :]     # seq_len x n_channels
        y = data[i+seq_len, :]       # n_channels
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


# -----------------------------
# 3. Define LSTM model for brain dynamics
# -----------------------------

class BrainLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_size, input_size)  # predict all channels at once

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)           # out: (batch, seq_len, hidden_size)
        last = out[:, -1, :]            # (batch, hidden_size)
        yhat = self.fc(last)            # (batch, input_size)
        return yhat


# -----------------------------
# 4. Main training + evaluation routine
# -----------------------------

def main():
    # --- parameters ---
    T = 3000
    n_channels = 5
    seq_len = 30
    batch_size = 32
    n_epochs = 30
    lr = 0.001

    # --- 4.1 Generate data ---
    t, signals = generate_brain_signals(T=T, n_channels=n_channels)
    # signals shape: (T, n_channels)

    # --- 4.2 Build sequences ---
    X, y = create_multichannel_sequences(signals, seq_len)
    # X: (N, seq_len, n_channels), y: (N, n_channels)

    # train/test split
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test,  y_test  = X[split:], y[split:]

    # convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test  = torch.tensor(X_test,  dtype=torch.float32)
    y_test  = torch.tensor(y_test,  dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test,  y_test  = X_test.to(device),  y_test.to(device)

    # --- 4.3 Model, loss, optimiser ---
    model = BrainLSTM(input_size=n_channels, hidden_size=64, num_layers=1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # --- 4.4 Training loop ---
    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(X_train.size(0))
        X_train_epoch = X_train[perm]
        y_train_epoch = y_train[perm]

        total_loss = 0.0
        for i in range(0, X_train_epoch.size(0), batch_size):
            xb = X_train_epoch[i:i+batch_size]
            yb = y_train_epoch[i:i+batch_size]

            optimizer.zero_grad()
            y_pred = model(xb)
            loss = criterion(y_pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        avg_train_loss = total_loss / X_train.size(0)

        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            y_pred_test = model(X_test)
            test_loss = criterion(y_pred_test, y_test).item()

        print(f"Epoch {epoch+1}/{n_epochs} - "
              f"train MSE: {avg_train_loss:.6f} - test MSE: {test_loss:.6f}")

    # --- 4.5 Final evaluation: per-channel metrics ---
    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test)  # (N_test, n_channels)

    # move back to CPU numpy
    y_pred_np = y_pred_test.cpu().numpy()
    y_test_np = y_test.cpu().numpy()

    # MSE per channel
    mse_per_channel = ((y_pred_np - y_test_np) ** 2).mean(axis=0)

    # R^2 per channel: 1 - SS_res / SS_tot
    ss_res = ((y_test_np - y_pred_np) ** 2).sum(axis=0)
    ss_tot = ((y_test_np - y_test_np.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    r2_per_channel = 1.0 - ss_res / ss_tot

    print("\nPer-channel MSE:", mse_per_channel)
    print("Per-channel R^2:", r2_per_channel)

    # --- 4.6 Visualise some channels: true vs predicted ---
    n_test = y_test_np.shape[0]
    time_test = np.arange(n_test)  # test time index

    n_plot = min(3, n_channels)  # plot up to 3 channels
    plt.figure(figsize=(10, 6))
    for ch in range(n_plot):
        plt.subplot(n_plot, 1, ch+1)
        plt.plot(time_test, y_test_np[:, ch], label=f"True (ch {ch})")
        plt.plot(time_test, y_pred_np[:, ch], label=f"Pred (ch {ch})", alpha=0.7)
        plt.ylabel("Activity")
        plt.legend(loc="upper right")
    plt.xlabel("Time step (test)")
    plt.tight_layout()
    plt.savefig("plots/channels_true_vs_pred.png", dpi=200)
    plt.show()


    # --- 4.7 Heatmap visualisation ---
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(y_test_np.T, aspect="auto", origin="lower", cmap="viridis")
    plt.title("True activity (test)")
    plt.xlabel("Time step")
    plt.ylabel("Channel")

    plt.subplot(1, 2, 2)
    plt.imshow(y_pred_np.T, aspect="auto", origin="lower", cmap="viridis")
    plt.title("Predicted activity (test)")
    plt.xlabel("Time step")
    plt.ylabel("Channel")

    plt.tight_layout()
    plt.savefig("plots/heatmap_true_vs_pred.png", dpi=200)
    plt.show()




if __name__ == "__main__":
    main()
