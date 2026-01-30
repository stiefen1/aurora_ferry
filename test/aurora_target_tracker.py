from python_vehicle_simulator.lib.simulator import Simulator
from python_vehicle_simulator.lib.env import NavEnv
from python_vehicle_simulator.lib.weather import Current
from python_vehicle_simulator.utils.unit_conversion import DEG2RAD
from python_vehicle_simulator.lib.map import RandomMapGenerator
from python_vehicle_simulator.lib.actuator import AzimuthThruster
from python_vehicle_simulator.lib.path import PWLPath
from python_vehicle_simulator.utils.unit_conversion import knot_to_m_per_sec
import numpy as np, matplotlib.pyplot as plt
from src.aurora import SingleAzimuthThrusterParameters, AuroraFerry
from src.navigation import NavigationAurora
from src.ais import AIS
from datetime import datetime, timedelta
from src.map import HelsingborgMap
from colav.planner import TimeSpaceColav
from colav.obstacles.moving import MovingShip
import colav, logging
colav.configure_logging(level=logging.INFO)


dt = 2
helsingborg = HelsingborgMap()
center = helsingborg.center_utm_ne
goal_ne = helsingborg.helsingborg_coords_ne
start_ne = helsingborg.helsingor_coords_ne
u_des = 5
eta0 = (start_ne[0], start_ne[1]-100, 0, 0, 0, np.deg2rad(80)) # (center[0], center[1], 0, 0, 0, 0)
nu0 = (u_des, 0, 0, 0, 0, 0)

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
            max_age_seconds=dt
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



# Enable interactive mode for real-time plotting
plt.ion()

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 8))
fig.suptitle('Aurora Ferry Target Tracker Simulation')

# Plot static elements once
helsingborg.plot(ax=ax)
ax.set_xlim(helsingborg.xlim)
ax.set_ylim(helsingborg.ylim)

for name, route in helsingborg.get_ferry_routes().items():
    route.plot(ax=ax, label=name)

# Initialize dynamic plot elements
vessels_artists = []

print("Starting simulation...")

# desired speed
# projector = TimeSpaceProjector(u_des)
planner = TimeSpaceColav(u_des, shore=[poly.simplify(3e2) for poly in helsingborg.polygons])

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
    helsingborg.get_ferry_routes()['Helsingør (DK) - Helsingborg (S)'].plot(ax=ax)
    helsingborg.get_ferry_routes()['Helsingør (DK) - Helsingborg (SE)'].plot(ax=ax)
    # for name, route in helsingborg.get_ferry_routes().items():
    #     route.plot(ax=ax, label=name)
    # xlim_focus = (352000, 355000)
    # ylim_focus = (6_212_000, 6_214_000)

    # ax.set_xlim(xlim_focus) #helsingborg.xlim)
    # ax.set_ylim(ylim_focus) #helsingborg.ylim)
    ax.set_xlim(helsingborg.xlim) #helsingborg.xlim)
    ax.set_ylim(helsingborg.ylim) #helsingborg.ylim)
    
    # Plot ferry
    ferry.plot(ax=ax)

    ships_for_projection = []
    for vessel in ferry.navigation.last_observation["vessels_ais"]:
        ax.scatter(vessel.east, vessel.north)

        if vessel.heading is not None and vessel.cog is not None and vessel.sog is not None:
            ships_for_projection.append(MovingShip.from_csog((vessel.east, vessel.north), vessel.heading, vessel.cog, knot_to_m_per_sec(vessel.sog), vessel.length, vessel.width, degrees=True, mmsi=vessel.mmsi).buffer(200, join_style='mitre'))
        
    # Get and plot vessels from AIS
    vessels = ais.get_vessels_at_time(timestamp=t, max_age_seconds=30)
    
    for vessel in vessels:
        
        if vessel.geometry is not None:
            vessel_line, = ax.plot(vessel.geometry[1, :], vessel.geometry[0, :], 
                                 'r-', linewidth=1.5, alpha=0.7)
            vessels_artists.append(vessel_line)
    
    # Project target ships
    planner.desired_speed = ferry.nu.u
    projected_ships = planner.projector.get((ferry.eta.e, ferry.eta.n), (goal_ne[1], goal_ne[0]), ships_for_projection)
    for projected_ship in projected_ships:
        ax.fill(*projected_ship.exterior.xy, c='black', alpha=0.5)

    traj, info = planner.get(
        (ferry.eta.e, ferry.eta.n),
        (goal_ne[1], goal_ne[0]), 
        ships_for_projection,
        heading=ferry.eta.yaw,
        degrees=False
    )

    if traj is not None:
        print(traj)
        traj.plot(ax=ax)

    # Advance time
    t = t + timedelta(seconds=dt)
    
    # Update display
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.01)

plt.ioff()  # Turn off interactive mode
print("Simulation completed.")

