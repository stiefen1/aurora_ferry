from python_vehicle_simulator.states import states

from src.ferry.aurora import AuroraFerry, AuroraFerryActuatorsParameters, AuroraFerryParameters
from python_vehicle_simulator.lib.simulator import Simulator
from python_vehicle_simulator.lib.env import NavEnv
from python_vehicle_simulator.lib.weather import Current
from python_vehicle_simulator.utils.unit_conversion import DEG2RAD
from python_vehicle_simulator.lib.map import RandomMapGenerator
from python_vehicle_simulator.lib.actuator import AzimuthThruster
from python_vehicle_simulator.lib.path import PWLPath
from python_vehicle_simulator.utils.unit_conversion import knot_to_m_per_sec
import numpy as np, matplotlib.pyplot as plt
from src.ferry.aurora import SingleAzimuthThrusterParameters, AuroraFerry
from src.ferry.navigation import NavigationAurora
from datetime import datetime, timedelta
from src.environment.map import HelsingborgMap
import logging
from csnlp import wrappers


dt = 5
center = (0, 0)
nepsi0 = (-150, -150, np.deg2rad(0))
u_des = 5
eta0 = (nepsi0[0], nepsi0[1], 0, 0, 0, nepsi0[2]) # (center[0], center[1], 0, 0, 0, 0)
nu0 = (u_des, 0, 0, 0, 0, 0)
xlim = (-1000, 1000)
ylim = (-200, 2500)
T_sim = 500.0  # 80 seconds simulation (increased to see more of the trajectory)
H = 10  # MPC horizon

ferry = AuroraFerry(
        eta=eta0,
        nu=nu0,
        dt=dt,
        navigation=NavigationAurora(
            states=np.array([*eta0, *nu0]),
            dt=dt,
            max_age_seconds=dt
        )
    )


n_sim = int(T_sim / dt)


# Get discrete dynamics function
# discrete_dynamics = get_discrete_3dof_dynamics_as_fn(dt)

# Create MPC controller
# mpc = DiffMPCTrajectoryTracking(discrete_dynamics, H, u_des=u_des)
from src.mpcrl.trajectory_tracker import ParametricTrajectoryTracker
mpc = ParametricTrajectoryTracker(H, dt)


path = PWLPath.sample(T_sim * u_des, max_turn_deg=30, seg_len_range=(200, 500), start=nepsi0[0:2])

# Enable interactive mode for real-time plotting
plt.ion()

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 8))
fig.suptitle('Aurora Ferry Target Tracker Simulation')

# Plot static elements once
ax.set_xlim(xlim)
ax.set_ylim(ylim)


# Initialize dynamic plot elements
vessels_artists = []

print("Starting simulation...")

print(ferry.alpha_actual, ferry.n_actual)

# desired speed
# projector = TimeSpaceProjector(u_des)
# mpc_ = wrappers.NlpSensitivity(mpc, mpc.nu_des)

t = timedelta(seconds=0)
t0 = datetime.now()
for step in range(n_sim):
    print(f"Step {step}/{n_sim} | t={t} | n={ferry.eta.n:.1f} | e={ferry.eta.e:.1f}")

    wpts = path.get_target_wpts_from(ferry.eta.n, ferry.eta.e, dp=u_des*dt, N=H+1) # [(x, y) for (x, y, psi) in path.get_target_wpts_from(ferry.eta.n, ferry.eta.e, dp=u_des*dt, N=H+1)]

    # mpc_commands = mpc.__get__(wpts, ferry.eta.to_numpy(dofs=3), ferry.nu.to_numpy(dofs=3), np.array(ferry.alpha_actual), np.array(ferry.n_actual), None, None)
    x0 = np.concatenate([ferry.eta.to_numpy(dofs=3), ferry.nu.to_numpy(dofs=3), np.array(ferry.alpha_actual), np.array(ferry.n_actual)]).reshape(14, 1)
    mpc_commands = mpc.forward(x0, nonlearnable_params={"wpts": np.array(wpts).T, "nu_des": np.array([u_des, 0, 0])}, learnable_params={"Q": np.array([1e3, 10, 10])})
    control_commands = [np.array([mpc_commands[i], mpc_commands[i+4]]) for i in range(4)]
    print("Sensitivities: ", mpc.get_mpc_sensitivities())
    # print("Commands: ", control_commands)

    # Update simulation
    ferry.step(None, None, [], [], control_commands=control_commands, timestamp=t0 + t)

    # Advance time
    t += timedelta(seconds=dt)
    # print(t.seconds % 5)

    if t.seconds % 5 == 0:
        
        # Remove old vessel artists
        for artist in vessels_artists:
            artist.remove()
        vessels_artists.clear()
        
        # Clear and redraw (simple approach)
        ax.cla()

        ax.set_xlim(xlim) #helsingborg.xlim)
        ax.set_ylim(ylim) #helsingborg.ylim)

        for wpt in wpts:
            ax.scatter(wpt[1], wpt[0], c='red')

        # for k in range(mpc.x_prev.shape[1]):
        #     ax.scatter(mpc.x_prev[1, k], mpc.x_prev[0, k], c='blue')
        
        # Plot ferry
        ferry.plot(ax=ax, verbose=10)

        ships_for_projection = []
        # Get and plot vessels from AIS
        vessels = []
        
        for vessel in vessels:
            
            if vessel.geometry is not None:
                vessel_line, = ax.plot(vessel.geometry[1, :], vessel.geometry[0, :], 
                                    'r-', linewidth=1.5, alpha=0.7)
                vessels_artists.append(vessel_line)
    



        # if traj is not None:
        #     traj.plot(ax=ax)
        path.plot(ax=ax)

    
        # Update display
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.01)
    
    

plt.ioff()  # Turn off interactive mode
print("Simulation completed.")
