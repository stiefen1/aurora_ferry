"""
Interactive AIS map with time slider (minimal)

- Draws the HelsingborgMap (UTM) as background
- Shows the latest position for each ship at or before the selected time
- Slider selects timestep index (mapped to timestamps present in AIS)

Minimal dependencies: matplotlib, pandas, pyproj (already in repo)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Polygon
from matplotlib.text import Text
from typing import List
from src.map import HelsingborgMap
from src.ais import AIS, Vessel


def get_vessels_at_time(ais: AIS, timestamp: pd.Timestamp) -> List[Vessel]:
    """Return vessel objects at or before timestamp.

    Args:
        ais: AIS object instance
        timestamp: pandas Timestamp to query

    Returns:
        List of Vessel objects
    """
    # Use the AIS method to get vessels at the specified time
    vessels = ais.get_vessels_at_time(timestamp=timestamp, max_age_seconds=10)
    return vessels


def main():
    # Load map and AIS
    env = HelsingborgMap()
    ais = AIS(path_to_folder='data', filename='AIS2.csv')

    # Ensure timestamp parsed
    ais['timestamp_dt'] = pd.to_datetime(ais['timestamp'])

    # Unique sorted timestamps in the dataset (we'll use indices)
    timestamps = np.sort(ais['timestamp_dt'].unique())
    if len(timestamps) == 0:
        raise RuntimeError('No timestamps found in AIS data')

    # Create figure and plot base map
    fig, ax = plt.subplots(figsize=(10, 8))
    env.__plot__(ax, routes=True, terminals=True)

    # Initial vessels (first timestamp)
    idx0 = 0
    ts0 = timestamps[idx0]
    vessels = get_vessels_at_time(ais, ts0)

    # Store vessel patches and text labels for updating
    vessel_patches = []
    vessel_texts = []
    
    # Plot initial vessels as polygons
    for vessel in vessels:
        if vessel.geometry is not None:
            # Extract vessel outline coordinates (east=x, north=y)
            vessel_coords = np.column_stack((vessel.geometry[1, :], vessel.geometry[0, :]))
            patch = Polygon(vessel_coords, facecolor='blue', edgecolor='darkblue', alpha=0.7, zorder=10)
            ax.add_patch(patch)
            vessel_patches.append(patch)
            
            # Add vessel label text
            text = ax.text(vessel.east, vessel.north + 100, vessel.name, 
                          fontsize=8, ha='center', va='center', 
                          bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7),
                          zorder=11)
            vessel_texts.append(text)
        else:
            scatter = ax.scatter(vessel.east, vessel.north, s=50, c='red')
            text = ax.text(vessel.east, vessel.north + 100, f"Invalid geometry, heading={vessel.heading}")
            vessel_patches.append(scatter)  # Treat scatter as a patch-like object
            vessel_texts.append(text)
            

    # Annotation for timestamp and count
    timestamp_text = ax.text(0.01, 0.98, f'Time: {pd.Timestamp(ts0)}', transform=ax.transAxes,
                             va='top', ha='left', fontsize=10, bbox=dict(boxstyle='round', fc='white', alpha=0.7))
    count_text = ax.text(0.01, 0.94, f'Ships: {len(vessels)}', transform=ax.transAxes,
                         va='top', ha='left', fontsize=10, bbox=dict(boxstyle='round', fc='white', alpha=0.7))

    # Slider axes
    axcolor = 'lightgoldenrodyellow'
    ax_slider = plt.axes((0.15, 0.03, 0.7, 0.03), facecolor=axcolor)

    slider = Slider(ax_slider, 'Timestep', valmin=0, valmax=len(timestamps) - 1,
                    valinit=0, valstep=1)

    def update(val):
        idx = int(slider.val)
        ts = timestamps[idx]
        vessels = get_vessels_at_time(ais, ts)
        
        # Remove old patches and texts
        for patch in vessel_patches:
            patch.remove()
        for text in vessel_texts:
            text.remove()
        vessel_patches.clear()
        vessel_texts.clear()
        
        # Add new patches and texts for vessels
        for vessel in vessels:
            if vessel.geometry is not None:
                print(vessel.name, vessel.mmsi, vessel.length, vessel.width, "sog: ", vessel.sog, "cog: ", vessel.cog, "dt: ", vessel.timestamp)
                # Create vessel patch
                vessel_coords = np.column_stack((vessel.geometry[1, :], vessel.geometry[0, :]))
                patch = Polygon(vessel_coords, facecolor='blue', edgecolor='darkblue', alpha=0.7, zorder=10)
                ax.add_patch(patch)
                vessel_patches.append(patch)
                
                # Create vessel text label
                text = ax.text(vessel.east, vessel.north + 100, vessel.name, 
                              fontsize=8, ha='center', va='center', 
                              bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7),
                              zorder=11)
                vessel_texts.append(text)
            else:
                # Handle invalid geometry
                scatter = ax.scatter(vessel.east, vessel.north, s=50, c='red')
                text = ax.text(vessel.east, vessel.north + 100, f"Invalid geometry, heading={vessel.heading}")
                vessel_patches.append(scatter)
                vessel_texts.append(text)
        
        # Update annotation
        timestamp_text.set_text(f'Time: {pd.Timestamp(ts)}')
        count_text.set_text(f'Ships: {len(vessels)}')
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.suptitle('Interactive AIS Map - use slider to select time')
    plt.show()


if __name__ == '__main__':
    main()
