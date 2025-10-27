from python_vehicle_simulator.lib.simulator import Simulator
from python_vehicle_simulator.lib.env import NavEnv
from python_vehicle_simulator.vehicles.revolt3 import Revolt3DOF, RevoltBowThrusterParams, RevoltSternThrusterParams, RevoltParameters3DOF, RevoltThrusterParameters
from python_vehicle_simulator.lib.navigation import NavigationRevolt3WithEKF
from python_vehicle_simulator.lib.weather import Current
from python_vehicle_simulator.utils.unit_conversion import DEG2RAD
from python_vehicle_simulator.lib.map import RandomMapGenerator
from python_vehicle_simulator.lib.actuator import AzimuthThruster
from python_vehicle_simulator.lib.mpc import MPCPathTrackingRevolt
from python_vehicle_simulator.lib.path import PWLPath
from python_vehicle_simulator.lib.guidance import PathFollowingGuidance
import numpy as np, matplotlib.pyplot as plt
import os

dt = 0.2
horizon = 30

Q_diagnosis = np.eye(9)*1e-5
Q_diagnosis[6, 6] = 1e-3
Q_diagnosis[7, 7] = 1e-3
Q_diagnosis[8, 8] = 1e-3
vessel = Revolt3DOF(
        params=RevoltParameters3DOF(),
        eta=np.array([0, 0, 0, 0, 0, 0]),
        dt=dt,
        actuators=[
            AzimuthThruster(xy=(-1.65, -0.15), **vars(RevoltSternThrusterParams())),
            AzimuthThruster(xy=(-1.65, 0.15), **vars(RevoltSternThrusterParams())),
            AzimuthThruster(xy=(1.15, 0.0), **vars(RevoltBowThrusterParams()))
        ],
        control=MPCPathTrackingRevolt(
            vessel_params=RevoltParameters3DOF(),
            actuator_params=RevoltThrusterParameters(),
            dt=dt,
            horizon=horizon,
            gamma=0.99
        ),
        guidance=PathFollowingGuidance(
            path=PWLPath.sample(d_tot=100, max_turn_deg=45, seg_len_range=(1, 5)),
            horizon=horizon,
            dt=dt,
            desired_speed=0.5
        ),
        navigation=NavigationRevolt3WithEKF(np.array([0, 0, 0, 0, 0, 0]), np.array([0, 0, 0, 0, 0, 0]), dt=dt),
    )

map_generator = RandomMapGenerator(
        (-100, 100),
        (-100, 100),
        (20, 30),
        min_dist=5
    )

env = NavEnv(
    own_vessel=vessel,
    target_vessels=[],
    obstacles=[], # map_generator.get([(vessel.eta[0], vessel.eta[1])], min_density=0.3)[0],
    dt=dt,
    current=Current(beta=-30.0*DEG2RAD, v=0.)
)

sim = Simulator(
        env,
        dt=dt,
        render_mode="human",
        verbose=2,
        skip_frames=1,
        window_size=(20, 20)
    )

print("=== Running Simulation (no rendering) ===")
sim.run(tf=40, render=False, store_data=True)
sim.save_animation(os.path.join('videos', 'vessel_simulation'), format="gif", fps=15)

fig1 = sim.plot_gnc_data_multi([
    'navigation.eta[0]',
    'vessel.eta[0]'
    ], x_path=['navigation.eta[1]', 'vessel.eta[1]'])
vessel.guidance.path.plot(ax=fig1.axes[0])
fig2 = sim.plot_gnc_data_multi([
    'vessel.nu[0]', 'vessel.nu[1]'
])
fig3 = sim.plot_gnc_data_multi([
    'control.info.psi_des', 'vessel.eta[5]'
])

plt.show(block=True)
