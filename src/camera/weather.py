from enum import Enum
import numpy as np, numpy.typing as npt
from python_vehicle_simulator.utils.math_fn import ssa
from typing import Tuple, Dict

class Weather(Enum):
    SUNNY = 0
    CLOUDY = 1
    FOGGY = 2


def get_detection_probability_interactive(
        os_neyaw: npt.NDArray,
        ts_neyaw: npt.NDArray,
        loa: float,
        beam: float,
        visibility: float,
        illumination: float,
        scale: float,
        offset: float
    ) -> float | npt.NDArray:
    """
    Modified version of get_detection_probability with explicit scale and offset parameters.
    """
    os_neyaw = np.reshape(os_neyaw, (-1, 3))
    ts_neyaw = np.reshape(ts_neyaw, (-1, 3))
    yaw_os, yaw_ts = ssa(os_neyaw[:, 2]), ssa(ts_neyaw[:, 2])
    yaw_rel = ssa(yaw_ts - yaw_os)
    xy_os, xy_ts = np.array([os_neyaw[:, 1], os_neyaw[:, 0]]).T, np.array([ts_neyaw[:, 1], ts_neyaw[:, 0]]).T
    xy_rel = xy_ts - xy_os
    rel_angle = ssa(np.atan2(xy_rel[:, 0], xy_rel[:, 1]))
    delta_angle_abs = ssa(yaw_rel - rel_angle)

    corrected_size = 0.5 * (beam + loa) - 0.5 * np.cos(2*delta_angle_abs) * (loa - beam)
    fov = 2 * np.rad2deg(np.atan(corrected_size / 2 / np.linalg.norm(xy_rel, axis=1)))
    
    return 1 / (1 + np.exp(-(fov-offset)/scale))


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

    corrected_size =  0.5 * (beam + loa) - 0.5 * np.cos(2*delta_angle_abs) * (loa - beam) # beam when 0 and loa when pi/2
    half_fov_rad = np.atan(corrected_size / 2 / np.linalg.norm(xy_rel, axis=1))
    fov = 2*np.rad2deg(half_fov_rad)
    sqrt_vis_ill = np.sqrt(visibility * illumination)
    scale = 0.5 - 0.4 * sqrt_vis_ill
    offset = 8 - 6 * sqrt_vis_ill

    # p -> 0 when FOV -> 0
    # p -> 1 when FOV -> 30
    # p -> 0 when sqrt_vis_ill -> 0
    # return 2/(1+np.exp(-(sqrt_vis_ill**2-1)/1)) * 1 / (1 + 1 * np.exp(-(fov-offset)/scale) )
    # return np.clip((fov)**2, 0, 1)
    return 1 / (1 + 1 * np.exp(-(fov-offset)/scale) ), {"corrected_size": corrected_size, "yaw_ts": yaw_ts, "rel_angle": rel_angle, "delta_angle_abs": delta_angle_abs, "b": beam, "l": loa}

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
    import matplotlib.pyplot as plt, numpy as np
    
    fig, ax = plt.subplots()

    d = np.linspace(10, 3000, 300)
    os_neyaw = np.array([100, 100, np.deg2rad(45)])
    rel_dir_vec = np.repeat(np.array([1, 1, 0])[None, :], d.shape[0], axis=0)
    ts_neyaw = os_neyaw + rel_dir_vec * d[:, None]
    print("SHAPE: ", ts_neyaw.shape)

    ts_neyaw[:, 2] = np.deg2rad(135)

    # ts_neyaw = np.array([400, 300, np.deg2rad(45)])
    size = [(50, 20), (75, 30), (100, 40)]
    weathers = [(1, 1), (0.6, 0.6), (0.1, 0.1)]
    color_map = {weathers[0]: 'blue', weathers[1]: 'red', weathers[2]: 'green'}
    linestyle_map = {size[0]: '-', size[1]: '--', size[2]: '-.'}
    xline = 1000

    for v,i in weathers:
        for s in size:
            p, _ = get_detection_probability(np.repeat(os_neyaw[None, 0:2], d.shape[0], axis=0), ts_neyaw, s[0], s[1], v, i)
            ax.plot(d, p, c=color_map[v, i], linestyle=linestyle_map[s], label=f"{s} - {v}_{i}")
            # print(f"Detected (v={v}, i={i} - s={s}): ", is_target_detected(1000, s/np.sqrt(2), s/np.sqrt(2), v, i))
    ax.vlines(xline, -2, 2, 'black', linestyles=':')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("distance [m]")
    ax.set_ylabel("detection probability [-]")
    plt.legend()
    plt.show()
    
    # Test interactive plot
    # print("\nTo use interactive plot, call: create_interactive_detection_plot()")
    # create_interactive_detection_plot()  # Uncomment this line to run interactive plot



# def create_interactive_detection_plot():
#     """
#     Create an interactive plot that reproduces the __main__ section behavior
#     with sliders to control the affine parameters for scale and offset functions.
#     scale = a_scale + b_scale * (visibility * illumination)
#     offset = a_offset + b_offset * (visibility * illumination)
#     """
#     import matplotlib.pyplot as plt
#     from matplotlib.widgets import Slider
    
#     # Setup data (same as __main__ section)
#     d = np.linspace(10, 3000, 300)
#     os_neyaw = np.array([100, 100, np.deg2rad(0)])
#     rel_dir_vec = np.repeat(np.array([1, 1, 0])[None, :], d.shape[0], axis=0)
#     ts_neyaw = os_neyaw + rel_dir_vec * d[:, None]
#     ts_neyaw[:, 2] = np.deg2rad(135)
    
#     size = [(80, 20), (90, 30), (100, 40)]
#     weathers = [(1, 1), (0.7, 0.7), (0.4, 0.4)]
#     color_map = {weathers[0]: 'blue', weathers[1]: 'red', weathers[2]: 'green'}
#     linestyle_map = {size[0]: '-', size[1]: '--', size[2]: '-.'}
#     xline = 1000
    
#     # Create figure and axis
#     fig, ax = plt.subplots(figsize=(12, 8))
#     plt.subplots_adjust(bottom=0.35)
    
#     # Initial parameters from original code
#     initial_a_scale = 0.8   # scale = 0.8 - 0.7 * (vis * illum)
#     initial_b_scale = 0.7
#     initial_a_offset = 6.0  # offset = 6 - 2.5 * (vis * illum)  
#     initial_b_offset = 2.5
    
#     # Create sliders
#     ax_a_scale = plt.axes([0.15, 0.25, 0.7, 0.025])
#     ax_b_scale = plt.axes([0.15, 0.20, 0.7, 0.025])
#     ax_a_offset = plt.axes([0.15, 0.15, 0.7, 0.025])
#     ax_b_offset = plt.axes([0.15, 0.10, 0.7, 0.025])
    
#     slider_a_scale = Slider(ax_a_scale, 'a_scale', 0.1, 10.0, valinit=initial_a_scale, valfmt='%.2f')
#     slider_b_scale = Slider(ax_b_scale, 'b_scale', 0.0, 10.0, valinit=initial_b_scale, valfmt='%.2f')
#     slider_a_offset = Slider(ax_a_offset, 'a_offset', 1.0, 50.0, valinit=initial_a_offset, valfmt='%.1f')
#     slider_b_offset = Slider(ax_b_offset, 'b_offset', 0.0, 50.0, valinit=initial_b_offset, valfmt='%.1f')
    
#     def plot_detection_curves(a_scale, b_scale, a_offset, b_offset):
#         """Plot all detection curves with given affine parameters."""
#         ax.clear()
        
#         for v, i in weathers:
#             vis_illum = np.sqrt(v * i)
#             # Calculate scale and offset using affine functions
#             scale_val = a_scale - b_scale * vis_illum
#             offset_val = a_offset - b_offset * vis_illum
            
#             for s in size:
#                 p = get_detection_probability_interactive(
#                     np.repeat(os_neyaw[None, :], d.shape[0], axis=0), 
#                     ts_neyaw, s[0], s[1], v, i, scale_val, offset_val
#                 )
#                 ax.plot(d, p, c=color_map[v, i], linestyle=linestyle_map[s], 
#                        label=f"{s} - vis={v}_illum={i}")
        
#         ax.vlines(xline, -0.05, 1.05, 'black', linestyles=':')
#         ax.set_ylim(-0.05, 1.05)
#         ax.set_xlabel("distance [m]")
#         ax.set_ylabel("detection probability [-]")
#         ax.legend()
#         ax.grid(True, alpha=0.3)
        
#         # Show current function equations in title
#         ax.set_title(f'Detection Probability\n' +
#                     f'scale = {a_scale:.2f} + {b_scale:.2f} × (vis×illum)\n' +
#                     f'offset = {a_offset:.1f} + {b_offset:.1f} × (vis×illum)')
    
#     # Initial plot
#     plot_detection_curves(initial_a_scale, initial_b_scale, initial_a_offset, initial_b_offset)
    
#     # Update function for sliders
#     def update(val):
#         a_scale = slider_a_scale.val
#         b_scale = slider_b_scale.val
#         a_offset = slider_a_offset.val
#         b_offset = slider_b_offset.val
#         plot_detection_curves(a_scale, b_scale, a_offset, b_offset)
#         fig.canvas.draw()
    
#     # Connect sliders to update function
#     slider_a_scale.on_changed(update)
#     slider_b_scale.on_changed(update)
#     slider_a_offset.on_changed(update)
#     slider_b_offset.on_changed(update)
    
#     plt.tight_layout()
#     plt.show()