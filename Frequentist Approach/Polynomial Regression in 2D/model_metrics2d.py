import numpy as np
import matplotlib.pyplot as plt


def plot_fit_2d_poisson(x, y, z, model, grid_size=100):

    x_lin = np.linspace(min(x), max(x), grid_size)
    y_lin = np.linspace(min(y), max(y), grid_size)

    X_grid, Y_grid = np.meshgrid(x_lin, y_lin)

    # flatten → predict → reshape
    Z_pred = model.predict(X_grid.ravel(), Y_grid.ravel())
    Z_pred = Z_pred.reshape(X_grid.shape)

    fig = plt.figure(figsize=(12, 5))

    # --- observed ---
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(x, y, z, alpha=0.1)
    ax1.set_title("Observed Data")

    # --- fitted surface ---
    ax2 = fig.add_subplot(122, projection='3d')

    surf = ax2.plot_surface(
        X_grid,
        Y_grid,
        Z_pred,
        cmap='viridis',     # 🔥 THIS is the key line
        edgecolor='none',   # smoother surface
        antialiased=True
    )

    # add color bar (very useful)
    ax2.set_title("Poisson Fit (λ)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("λ")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def plot_fit_2d_log(x, y, z, model, grid_size=100):
    """
    Plot log-space comparison: log(z) vs log(λ)
    """

    x_lin = np.linspace(min(x), max(x), grid_size)
    y_lin = np.linspace(min(y), max(y), grid_size)

    X, Y = np.meshgrid(x_lin, y_lin)

    Z_pred = model.predict(X, Y)

    # avoid log(0)
    z_safe = np.maximum(z, 1e-12)

    fig = plt.figure(figsize=(12, 4.5))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(x, y, np.log(z_safe), alpha=0.1)
    ax1.set_title("Observed log(z)")
    ax1.set_zlabel("log(z)")

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X, Y, np.log(Z_pred), alpha=0.8)
    ax2.set_title("Fitted log(λ)")
    ax2.set_zlabel("log(λ)")

    plt.tight_layout()
    plt.show()

def plot_projection_poisson(x, y, z, model):

    z_pred = model.predict(x, y)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # --- x projection ---
    ax[0].scatter(x, z, alpha=0.1, label="Observed")
    ax[0].scatter(x, z_pred, alpha=0.1, label="Predicted")
    ax[0].set_title("Projection on x")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("Counts")
    ax[0].legend()

    # --- y projection ---
    ax[1].scatter(y, z, alpha=0.1, label="Observed")
    ax[1].scatter(y, z_pred, alpha=0.1, label="Predicted")
    ax[1].set_title("Projection on y")
    ax[1].set_xlabel("y")
    ax[1].set_ylabel("Counts")
    ax[1].legend()

    plt.tight_layout()
    plt.show()

def plot_projection_log(x, y, z, model):

    z_pred = model.predict(x, y)

    z_safe = np.maximum(z, 1e-12)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].scatter(x, np.log(z_safe), alpha=0.1, label="Observed")
    ax[0].scatter(x, np.log(z_pred), alpha=0.1, label="Predicted")
    ax[0].set_title("log Projection on x")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("log(z)")
    ax[0].legend()

    ax[1].scatter(y, np.log(z_safe), alpha=0.1, label="Observed")
    ax[1].scatter(y, np.log(z_pred), alpha=0.1, label="Predicted")
    ax[1].set_title("log Projection on y")
    ax[1].set_xlabel("y")
    ax[1].set_ylabel("log(z)")
    ax[1].legend()

    plt.tight_layout()
    plt.show()
