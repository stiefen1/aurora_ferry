from enum import Enum
import numpy as np, numpy.typing as npt
from python_vehicle_simulator.utils.math_fn import ssa
from typing import Tuple, Dict

def get_detection_probability(
        os_ne: npt.NDArray,
        ts_neyaw: npt.NDArray,
        loa: float,
        beam: float,
        visibility: float,
        illumination: float,
    ) -> Tuple[float | npt.NDArray, Dict]:
    """
    Compute the probability of detection using a camera.

    os_ne: N-E position of the own ship with shape (2,) or (2, N)
    ts_neyaw: N-E-Yaw pose of the own ship with shape (3,) or (3, N)
    loa: Lenght-Over-All [m] of the target ship 
    beam: Beam [m] of the target ship 
    visibility: scalar ranging from 0 (dense fog) to 1 (clear)
    illumination: scalar randing from 0 (night) to 1 (daylight)
    
    """
    os_ne = np.reshape(os_ne, (-1, 2))
    ts_neyaw = np.reshape(ts_neyaw, (-1, 3))
    yaw_ts = ssa(ts_neyaw[:, 2])
    xy_os, xy_ts = np.array([os_ne[:, 1], os_ne[:, 0]]).T, np.array([ts_neyaw[:, 1], ts_neyaw[:, 0]]).T
    xy_rel = xy_ts - xy_os
    rel_angle = ssa(np.atan2(xy_rel[:, 0], xy_rel[:, 1]))
    delta_angle_abs = abs(ssa(yaw_ts - rel_angle))
    rel_distance = np.linalg.norm(xy_rel, axis=1)

    # Actual size of the TS seen by the own ship
    corrected_size =  0.5 * (beam + loa) - 0.5 * np.cos(2*delta_angle_abs) * (loa - beam) # beam when 0 and loa when pi/2
    
    # FOV
    half_fov_rad = np.atan(corrected_size / 2 / rel_distance)
    fov = 2 * np.rad2deg(half_fov_rad)

    # Impact of visibility, illumination
    sqrt_vis_ill = np.sqrt(visibility * illumination)
    scale = 0.7 - 0.35 * sqrt_vis_ill
    offset = 3 - 2 * sqrt_vis_ill

    # Probability of detecting target ship
    p = 1 / (1 + 1 * np.exp(-(fov-offset)/scale) )

    # p -> 0 when FOV -> 0
    # p -> 1 when FOV -> 30
    # p -> 0 when sqrt_vis_ill -> 0
    return p, {}

def is_target_detected(
        os_ne: npt.NDArray,
        ts_neyaw: npt.NDArray,
        loa: float,
        beam: float,
        visibility: float,
        illumination: float
    ) -> Tuple[bool, Dict]:
    p, info = get_detection_probability(os_ne, ts_neyaw, loa, beam, visibility, illumination)
    val = np.random.uniform(low=0, high=1)
    return bool(val <= p), info
        
if __name__ == "__main__":
    """
    Modify the get_detection_probability (lines 34-52) function to change the behavior 
    """
    import matplotlib.pyplot as plt, numpy as np
    from matplotlib import cm, colors

    # OS states
    os_neyaw = np.array([100, 100, np.deg2rad(45)])
    
    # Create list of states for target ships that will be tested.
    distances = np.linspace(10, 3000, 300)
    rel_dir_vec = np.repeat(np.array([1, 1, 0])[None, :], distances.shape[0], axis=0) 
    ts_neyaw = os_neyaw + rel_dir_vec * distances[:, None]
    ts_neyaw[:, 2] = np.deg2rad(45) # heading of TS -> affect relative size on camera
    
    # Size (loa, beam) of target ships to be tested
    size = [(12, 6), (60, 20), (100, 40)]
    size_to_linestyle_hashmap = {size[0]: '-', size[1]: '--', size[2]: '-.'}

    # Visibility and illumination values to be tested
    vis_and_illum = set([(1, 1), (0.7, 0.7), (0.4, 0.4), (0.1, 0.1)])
    
    # Figure and cmap
    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.get_cmap("viridis")
    norm = colors.Normalize(vmin=0.0, vmax=1.0)

    # Create plot
    for v, i in sorted(vis_and_illum):
        sqrt_vi = float(np.sqrt(v * i))
        color = cmap(norm(sqrt_vi))
        for s in size:
            p, _ = get_detection_probability(np.repeat(os_neyaw[None, 0:2], distances.shape[0], axis=0), ts_neyaw, s[0], s[1], v, i)
            ax.plot(distances, p, c=color, linestyle=size_to_linestyle_hashmap[s], label=f"(loa, beam) = {s}")
    
    fig.suptitle(f"Probability of detecting a target ship as a function of its distance, visibility (v), illumination (i), LOA and beam\nResults are shown for a relative angle of ${np.rad2deg(ts_neyaw[0, 2]):.1f}$ degrees")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Distance to target vessel [m]")
    ax.set_ylabel("Detection probability [-]")
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="sqrt(visibility*illumination)")
    plt.legend()
    plt.show()