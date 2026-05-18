from matplotlib.pylab import True_

from src.ferry.aurora import AuroraFerry, AuroraFerryParameters
from src.rl.ppo_controller import PPOTrajTrackingController
from src.rl.sac_controller import SACTrajTrackingController
from src.ferry.navigation import NavigationAurora
from src.ferry.guidance import TimespaceGuidance
from src.environment.map import HelsingborgMap
from src.ais.ais import AIS
from src.camera.camera import Camera

from python_vehicle_simulator.lib.simulator import Simulator
from python_vehicle_simulator.lib.env import NavEnv
from python_vehicle_simulator.lib.weather import Wind, Current
from python_vehicle_simulator.lib.obstacle import Obstacle

import numpy as np, pandas as pd, os
from datetime import timedelta

helsingborg = HelsingborgMap()
duration = 1000
day_str = "2023-04-15" # 2023-04-06
t0_str = "11:40:00.000Z" # "05:05:00.000Z"
t0 = pd.to_datetime(day_str + "T" + t0_str)
time_window = (t0, t0 + timedelta(seconds=duration))


u_des = 6
dt = 0.2
path_to_sac_params = 'models\\sac\\2026_05_07_09_24_00'
start_ne = helsingborg.get_ferry_routes()['Helsingør (DK) - Helsingborg (SE)'].waypoints[0] 
start_ne[1] += 200
states = np.array([*start_ne] + 3 * [0] + [np.deg2rad(70)] + 14*[0])

n_passengers = 1000 # 10-1250
n_cars = 200 # 0-240

AURORA_AF_HELSINGBORG_MMSI = 265041000
SEED = 42

# TODO: Find the best way to check collision with target ships
# TODO: forward uncertainty from kalman filter pose estimation to timespace colav ?
# TODO: Implement realistic matching between camera data and already existing targets
# TODO: Solve distance to target error (discrete jumps)

aurora = AuroraFerry(
    dt,
    eta = (states[0], states[1], states[5]),
    nu = (states[6], states[7], states[11]),
    control=SACTrajTrackingController( # TODO: Check if action_repeat creates an issue
        path_to_params=path_to_sac_params
    ),
    navigation=NavigationAurora(
        states,
        dt,
        sensors={
            'ais': AIS(
                os.path.join('data', 'raw', day_str.replace('-', '_') + '.csv'),
                t0=time_window[0],
                tf=time_window[1],
                mmsi_to_exclude=[AURORA_AF_HELSINGBORG_MMSI], # Aurora AF Helsingborg, since that's us
                update_every_sec=1,
                ),
            'camera': Camera(
                os.path.join('data', 'smooth_interp', day_str.replace('-', '_') + '.csv'),
                t0=time_window[0],
                tf=time_window[1],
                mmsi_to_exclude=[AURORA_AF_HELSINGBORG_MMSI],
                update_every_sec=1,
                )
            }
    ),
    guidance=TimespaceGuidance(
        global_path=helsingborg.get_ferry_routes()['Helsingør (DK) - Helsingborg (SE)'],
        u_des=u_des,
        update_every_sec=30,
        lookahead_distance=1000,
        colregs=True,
        good_seamanship=True,
        abort_colregs_after_iter=1,
        buffer_target_ships=200,
    ),
    n_cars=n_cars,
    n_passengers=n_passengers
)

current_params = {
        "angle": {
            "min": -3.14159,
            "max": 3.14159,
            "ornstein-uhlenbeck": {
                "attraction": 0.01,
                "amplitude": 0.05
            }
        },
        "speed": {
            "min": 0,
            "max": 2,
            "ornstein-uhlenbeck": {
                "attraction": 0.01,
                "amplitude": 0.1
            }
        }
    }

wind_params = {
        "angle":{
            "min": -3.14159,
            "max": 3.14159,
            "ornstein-uhlenbeck": {
                "attraction": 0.01,
                "amplitude": 0.05
            }
        },
        "speed": {
            "min": 0,
            "max": 20,
            "ornstein-uhlenbeck": {
                "attraction": 0.01,
                "amplitude": 0.1
            }
        }
    }

# Sample wind current values
wind = Wind(
    np.random.uniform(wind_params["angle"]["min"], wind_params["angle"]["max"]),
    np.random.uniform(wind_params["speed"]["min"], wind_params["speed"]["max"]),
    attraction_beta=wind_params["angle"]["ornstein-uhlenbeck"]["attraction"],
    amplitude_beta=wind_params["angle"]["ornstein-uhlenbeck"]["amplitude"],
    attraction_norm=wind_params["speed"]["ornstein-uhlenbeck"]["attraction"],
    amplitude_norm=wind_params["angle"]["ornstein-uhlenbeck"]["amplitude"],
    dt=dt,
    seed=SEED
)

current = Current(
    np.random.uniform(current_params["angle"]["min"], current_params["angle"]["max"]),
    np.random.uniform(current_params["speed"]["min"], current_params["speed"]["max"]),
    attraction_beta=current_params["angle"]["ornstein-uhlenbeck"]["attraction"],
    amplitude_beta=current_params["angle"]["ornstein-uhlenbeck"]["amplitude"],
    attraction_norm=current_params["speed"]["ornstein-uhlenbeck"]["attraction"],
    amplitude_norm=current_params["angle"]["ornstein-uhlenbeck"]["amplitude"],
    dt=dt,
    seed=SEED
)


env = NavEnv(aurora, [], [Obstacle(geometry=list(zip(*poly.exterior.coords.xy[::-1]))) for poly in helsingborg.polygons], dt, wind=wind, current=current)
sim = Simulator(env, dt=dt, skip_frames=10, render_mode='human', window_size=(6000, 2000), verbose=7)
sim.run(duration, render=True, store_data=True, m_tot_estimated=aurora.vessel_params.m_tot_estimated, visibility=1.0, illumination=1.0, t0=time_window[0])

import matplotlib.pyplot as plt
        
fig1 = sim.plot_gnc_data_multi([
    'guidance.eta_des[0]',
    'vessel.eta[0]',
    'navigation.eta[0]',
    ], x_path=['guidance.eta_des[1]', 'vessel.eta[1]', 'navigation.eta[1]'])

fig1.axes[0].set_aspect('equal')

plt.show(block=True)