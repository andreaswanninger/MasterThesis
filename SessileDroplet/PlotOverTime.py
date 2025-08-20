import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configuration for each angle
cases = {
    75: {
        "file": "interface_positions_75.csv",
        "north_y_analytical": 0.00542,
        "x_left_analytical": 0.00794,
        "x_right_analytical": 0.02206,
    },
    90: {
        "file": "interface_positions_90.csv",
        "north_y_analytical": 0.006,
        "x_left_analytical": 0.009,
        "x_right_analytical": 0.021,
    },
    105: {
        "file": "interface_positions_105.csv",
        "north_y_analytical": 0.00656,
        "x_left_analytical": 0.00965,
        "x_right_analytical": 0.02035,
    }
}

# Global color map
colors = {
    "left_x": "red",
    "north_y": "blue",
    "right_x": "green"
}

# Loop through each case
for angle, cfg in cases.items():
    # Load numerical data
    df = pd.read_csv(cfg["file"])
    time_np = df["time"].to_numpy()
    north_y_np = df["north_y"].to_numpy()
    left_x_np = df["left_x"].to_numpy()
    right_x_np = df["right_x"].to_numpy()

    # ====== Position Plot ======
    plt.figure(figsize=(10, 6))

    # Numerical data (solid)
    plt.plot(time_np, north_y_np, label='droplet height (numerical result)', color=colors["north_y"], linestyle='-')
    plt.plot(time_np, left_x_np, label='left contact line point (numerical result)', color=colors["left_x"], linestyle='-')
    plt.plot(time_np, right_x_np, label='right contact line point (numerical result)', color=colors["right_x"], linestyle='-')

    # Analytical references (dotted)
    plt.hlines(cfg["north_y_analytical"], xmin=time_np.min(), xmax=time_np.max(),
               colors=colors["north_y"], linestyles='dotted', label='droplet height (analytical result)')
    plt.hlines(cfg["x_left_analytical"], xmin=time_np.min(), xmax=time_np.max(),
               colors=colors["left_x"], linestyles='dotted', label='left contact line point (analytical result)')
    plt.hlines(cfg["x_right_analytical"], xmin=time_np.min(), xmax=time_np.max(),
               colors=colors["right_x"], linestyles='dotted', label='right contact line point (analytical result)')

    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.title(f"Interface Positions vs Time (θ={angle}°)")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"interface_comparison_{angle}.png", dpi=300)
    plt.close()

    # ====== Error Plot ======
    plt.figure(figsize=(10, 6))

    # Compute absolute relative errors (as percentages)
    err_north = 100 * np.abs((north_y_np - cfg["north_y_analytical"]) / cfg["north_y_analytical"])
    err_left = 100 * np.abs((left_x_np - cfg["x_left_analytical"]) / cfg["x_left_analytical"])
    err_right = 100 * np.abs((right_x_np - cfg["x_right_analytical"]) / cfg["x_right_analytical"])

    # Plot errors
    plt.plot(time_np, err_left, label="left contact line point", color=colors["left_x"])
    plt.plot(time_np, err_north, label="droplet height", color=colors["north_y"])
    plt.plot(time_np, err_right, label="right contact line point", color=colors["right_x"])

    plt.xlabel("Time (s)")
    plt.ylabel("Absolute Relative Error (%)")
    plt.title(f"Interface Position Errors vs Time (θ={angle}°)")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"interface_error_{angle}.png", dpi=300)
    plt.close()
