from python_vehicle_simulator.lib.simulator import Simulator
from python_vehicle_simulator.lib.env import NavEnv
from python_vehicle_simulator.lib.weather import Current
from python_vehicle_simulator.utils.unit_conversion import DEG2RAD
from python_vehicle_simulator.lib.map import RandomMapGenerator
from python_vehicle_simulator.lib.actuator import AzimuthThruster
from python_vehicle_simulator.lib.path import PWLPath
import numpy as np, matplotlib.pyplot as plt
from src.aurora import SingleAzimuthThrusterParameters, AuroraFerry
from src.navigation import NavigationAurora
from src.ais import AIS, Vessel
from datetime import datetime, timedelta
from src.map import HelsingborgMap


dt = 1
helsingborg = HelsingborgMap()
center = helsingborg.center_utm_ne
eta0 = (center[0], center[1], 0, 0, 0, 0)
nu0 = (0, 0, 0, 0, 0, 0)

ferry = AuroraFerry(
        eta0=eta0,
        nu0=nu0,
        dt=dt,
        actuators=[
            AzimuthThruster(xy=(-35, -9.4), length=2, width=1, **vars(SingleAzimuthThrusterParameters())),
            AzimuthThruster(xy=(-35, 9.4), length=2, width=1, **vars(SingleAzimuthThrusterParameters())),
            AzimuthThruster(xy=(35, -9.4), length=2, width=1, **vars(SingleAzimuthThrusterParameters())),
            AzimuthThruster(xy=(35, 9.4), length=2, width=1, **vars(SingleAzimuthThrusterParameters()))
        ],
        navigation=NavigationAurora(
            eta=np.array(eta0),
            nu=np.array(nu0),
            dt=dt,
            max_age_seconds=dt/2
        )
    )

ais = AIS()
# t0 = ais.get_first_timestamp()
t0 = datetime(2023, 4, 6, hour=5, minute=7)
# 2023-04-06 05:07:04
tf = ais.get_last_timestamp()
n_sim = int((tf - t0).seconds / dt)
t = t0
print("final time: ", tf, " sec: ", tf.second)
print("initial time: ", t0, "sec: ", t0.second)

print(n_sim)



# Enable interactive mode for real-time plotting
plt.ion()

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 8))
fig.suptitle('Aurora Ferry Target Tracker Simulation')

# Plot static elements once
helsingborg.plot(ax=ax)
ax.set_xlim(helsingborg.xlim)
ax.set_ylim(helsingborg.ylim)

# Initialize dynamic plot elements
vessels_artists = []

print("Starting simulation...")

for step in range(n_sim):
    print(f"Step {step}/{n_sim} | t={t} | n={ferry.eta.n:.1f} | e={ferry.eta.e:.1f}")
    
    # Update simulation
    ferry.step(None, None, [], [], control_commands=4*[np.array([0, 1e6])], timestamp=t)
    
    # Remove old vessel artists
    for artist in vessels_artists:
        artist.remove()
    vessels_artists.clear()
    
    # Clear and redraw (simple approach)
    ax.cla()
    helsingborg.plot(ax=ax)
    xlim_focus = (352000, 355000)
    ylim_focus = (6_212_000, 6_214_000)
    ax.set_xlim(xlim_focus) #helsingborg.xlim)
    ax.set_ylim(ylim_focus) #helsingborg.ylim)
    
    # Plot ferry
    ferry.plot(ax=ax)

    for vessel in ferry.navigation.last_info["vessels_ais"]:
        ax.scatter(vessel.east, vessel.north)
    
    # Get and plot vessels from AIS
    vessels = ais.get_vessels_at_time(timestamp=t, max_age_seconds=30)
    for vessel in vessels:
        if vessel.geometry is not None:
            vessel_line, = ax.plot(vessel.geometry[1, :], vessel.geometry[0, :], 
                                 'r-', linewidth=1.5, alpha=0.7)
            vessels_artists.append(vessel_line)
    
    # Advance time
    t = t + timedelta(seconds=dt)
    
    # Update display
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.01)

plt.ioff()  # Turn off interactive mode
print("Simulation completed.")

