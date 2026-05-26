"""
Based on how we access data in scenario/sim_analyzer.py from the simulations in sim_data/test/simulations, create a figure that demonstrate 
the impact of current on path tracking accuracy.

For this purpose, create a 2D plot in polar coordinates where each point represents the tip of the current vector (speed=length, direction=bearing angle)
and the color represent the average path tracking error (hence each point represents a single simulation).

Make it as simple as possible and go straight to the point.

"""

import glob, os, json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

SIM_DATA_DIR = os.path.join("sim_data", "test")
SCENARIOS_DIR = os.path.join(SIM_DATA_DIR, "scenarios")
SIMULATIONS_DIR = os.path.join(SIM_DATA_DIR, "simulations")


def _load_data(scenarios_dir: str, simulations_dir: str):
    """Yield (config, mean_pos_error, mean_speed_error) for each matched scenario/simulation pair."""
    for json_path in sorted(glob.glob(os.path.join(scenarios_dir, "*.json"))):
        sim_folder = os.path.join(simulations_dir, Path(json_path).stem)
        if not Path(sim_folder).exists():
            continue

        with open(json_path) as f:
            config = json.load(f)

        x_npz = np.load(os.path.join(sim_folder, "navigation_actual_states.npz"), allow_pickle=True)
        x_des_npz = np.load(os.path.join(sim_folder, "guidance_states_des.npz"), allow_pickle=True)
        x_data, x_des_data = x_npz["data"], x_des_npz["data"]
        x_npz.close()
        x_des_npz.close()

        n = min(len(x_data), len(x_des_data))
        pos_error = np.hypot(x_data[:n, 0] - x_des_data[:n, 0], x_data[:n, 1] - x_des_data[:n, 1])
        actual_speed = np.hypot(x_data[:n, 6], x_data[:n, 7])  # sqrt(surge^2 + sway^2)
        speed_error = actual_speed - x_des_data[:n, 6]

        yield config, float(np.quantile(pos_error, 0.8)), float(np.quantile(np.abs(speed_error), 0.5))


def _polar_tracking_figure(speeds, angles, errors, title, speed_label, error_label, out_path):
    """Plot a polar scatter where each point is the tip of a disturbance vector, colored by tracking error."""
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)  # clockwise like a compass

    sc = ax.scatter(angles, speeds, c=errors, cmap="plasma", s=40, alpha=0.85)

    cbar = fig.colorbar(sc, ax=ax, pad=0.12, shrink=0.75)
    cbar.set_label(error_label)
    ax.set_xlabel(speed_label, labelpad=15)
    ax.set_title(title, pad=20)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {out_path}")
    return fig


def plot_current_vs_tracking_error(sim_data_dir: str = SIM_DATA_DIR):
    speeds, angles, errors = [], [], []
    for config, pos_err, _ in _load_data(
        os.path.join(sim_data_dir, "scenarios"),
        os.path.join(sim_data_dir, "simulations"),
    ):
        current = config["scenario_generation"]["operational_domain"]["current"]
        speeds.append(current["speed"]["value"])
        angles.append(current["angle"]["value"])
        errors.append(pos_err)

    return _polar_tracking_figure(
        np.array(speeds), np.array(angles), np.array(errors),
        title="Impact of current on path tracking accuracy",
        speed_label="Current speed [m/s]",
        error_label="Median position tracking error [m]",
        out_path=os.path.join(sim_data_dir, "figures", "current_vs_pos_tracking_error.png"),
    )


def plot_current_vs_speed_tracking_error(sim_data_dir: str = SIM_DATA_DIR):
    speeds, angles, errors = [], [], []
    for config, _, speed_err in _load_data(
        os.path.join(sim_data_dir, "scenarios"),
        os.path.join(sim_data_dir, "simulations"),
    ):
        current = config["scenario_generation"]["operational_domain"]["current"]
        speeds.append(current["speed"]["value"])
        angles.append(current["angle"]["value"])
        errors.append(speed_err)

    return _polar_tracking_figure(
        np.array(speeds), np.array(angles), np.array(errors),
        title="Impact of current on speed tracking accuracy",
        speed_label="Current speed [m/s]",
        error_label="Median speed tracking error [m/s]",
        out_path=os.path.join(sim_data_dir, "figures", "current_vs_speed_tracking_error.png"),
    )


def plot_wind_vs_tracking_error(sim_data_dir: str = SIM_DATA_DIR):
    speeds, angles, errors = [], [], []
    for config, pos_err, _ in _load_data(
        os.path.join(sim_data_dir, "scenarios"),
        os.path.join(sim_data_dir, "simulations"),
    ):
        wind = config["scenario_generation"]["operational_domain"]["wind"]
        speeds.append(wind["speed"]["value"])
        angles.append(wind["angle"]["value"])
        errors.append(pos_err)

    return _polar_tracking_figure(
        np.array(speeds), np.array(angles), np.array(errors),
        title="Impact of wind on path tracking accuracy",
        speed_label="Wind speed [m/s]",
        error_label="Median position tracking error [m]",
        out_path=os.path.join(sim_data_dir, "figures", "wind_vs_pos_tracking_error.png"),
    )


def plot_wind_vs_speed_tracking_error(sim_data_dir: str = SIM_DATA_DIR):
    speeds, angles, errors = [], [], []
    for config, _, speed_err in _load_data(
        os.path.join(sim_data_dir, "scenarios"),
        os.path.join(sim_data_dir, "simulations"),
    ):
        wind = config["scenario_generation"]["operational_domain"]["wind"]
        speeds.append(wind["speed"]["value"])
        angles.append(wind["angle"]["value"])
        errors.append(speed_err)

    return _polar_tracking_figure(
        np.array(speeds), np.array(angles), np.array(errors),
        title="Impact of wind on speed tracking accuracy",
        speed_label="Wind speed [m/s]",
        error_label="Median speed tracking error [m/s]",
        out_path=os.path.join(sim_data_dir, "figures", "wind_vs_speed_tracking_error.png"),
    )


if __name__ == "__main__":
    plot_current_vs_tracking_error()
    plot_current_vs_speed_tracking_error()
    plot_wind_vs_tracking_error()
    plot_wind_vs_speed_tracking_error()
    plt.show()
