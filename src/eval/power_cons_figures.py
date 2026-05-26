import glob, os, json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from python_vehicle_simulator.utils.math_fn import ssa

SIM_DATA_DIR = os.path.join("sim_data", "test")

MU = 1e-3  # thrust power weight, same as sim_analyzer.py


def _load_data(scenarios_dir: str, simulations_dir: str):
    """Yield (config, median_power) for each matched scenario/simulation pair."""
    for json_path in sorted(glob.glob(os.path.join(scenarios_dir, "*.json"))):
        sim_folder = os.path.join(simulations_dir, Path(json_path).stem)
        if not Path(sim_folder).exists():
            continue

        with open(json_path) as f:
            config = json.load(f)

        x_npz = np.load(os.path.join(sim_folder, "navigation_actual_states.npz"), allow_pickle=True)
        u_npz = np.load(os.path.join(sim_folder, "control_u.npz"), allow_pickle=True)
        x_data = x_npz["data"]
        u_data = u_npz["data"]
        x_npz.close()
        u_npz.close()

        # Same indexing as sim_analyzer.py::power_cons()
        azimuth_commands = u_data[1:, 0:4]
        azimuth_angle = x_data[:, 12:16]
        thruster_speed = x_data[:, 16:20]

        n = min(len(azimuth_commands), len(azimuth_angle))
        delta_azimuth = np.abs(ssa(azimuth_commands[:n] - azimuth_angle[:n]))
        power = np.sum(delta_azimuth ** 2, axis=1) + MU * np.sum(thruster_speed[:n] ** 2, axis=1)

        yield config, float(np.mean(power) + np.std(power))


def _polar_power_figure(speeds, angles, power, title, speed_label, out_path):
    """Polar scatter: each point is the tip of a disturbance vector, colored by power consumption."""
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)  # clockwise like a compass

    sc = ax.scatter(angles, speeds, c=power, cmap="plasma", s=40, alpha=0.85)

    cbar = fig.colorbar(sc, ax=ax, pad=0.12, shrink=0.75)
    cbar.set_label("Power consumption (mean + std)")
    ax.set_xlabel(speed_label, labelpad=15)
    ax.set_title(title, pad=20)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {out_path}")
    return fig


def plot_current_vs_power(sim_data_dir: str = SIM_DATA_DIR):
    speeds, angles, power = [], [], []
    for config, p in _load_data(
        os.path.join(sim_data_dir, "scenarios"),
        os.path.join(sim_data_dir, "simulations"),
    ):
        current = config["scenario_generation"]["operational_domain"]["current"]
        speeds.append(current["speed"]["value"])
        angles.append(current["angle"]["value"])
        power.append(p)

    return _polar_power_figure(
        np.array(speeds), np.array(angles), np.array(power),
        title="Impact of current on power consumption",
        speed_label="Current speed [m/s]",
        out_path=os.path.join(sim_data_dir, "figures", "current_vs_power.png"),
    )


def plot_wind_vs_power(sim_data_dir: str = SIM_DATA_DIR):
    speeds, angles, power = [], [], []
    for config, p in _load_data(
        os.path.join(sim_data_dir, "scenarios"),
        os.path.join(sim_data_dir, "simulations"),
    ):
        wind = config["scenario_generation"]["operational_domain"]["wind"]
        speeds.append(wind["speed"]["value"])
        angles.append(wind["angle"]["value"])
        power.append(p)

    return _polar_power_figure(
        np.array(speeds), np.array(angles), np.array(power),
        title="Impact of wind on power consumption",
        speed_label="Wind speed [m/s]",
        out_path=os.path.join(sim_data_dir, "figures", "wind_vs_power.png"),
    )


if __name__ == "__main__":
    plot_current_vs_power()
    plot_wind_vs_power()
    plt.show()
