from python_vehicle_simulator.lib.path import PWLPath

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

import numpy as np, pandas as pd, os, json
from datetime import timedelta
from typing import LiteralString


class SimLauncher:
    def __init__(
            self
    ):
        self.map = HelsingborgMap()
        self.ferry_route = self.map.get_ferry_routes()['Helsingør (DK) - Helsingborg (SE)']


    def run_single_sim(self, path_to_config: LiteralString) -> None:
        with open(path_to_config) as f:
            config = json.load(f)

        scenario_generation = config["scenario_generation"]
        odm = scenario_generation["operational_domain"]
        seed = config["seed"]
        duration = scenario_generation["simulation"]["duration_sec"]
        t0 = pd.to_datetime(scenario_generation["simulation"]["start_time_iso_utc"])
        dt = scenario_generation["simulation"]["dt"]
        guidance = scenario_generation["guidance"]
        navigation = scenario_generation["navigation"]
        control = scenario_generation["control"]
        n_passengers = odm["ferry"]["passengers"]["number"]
        n_cars = odm["ferry"]["cars"]["number"]
        mmsi_to_exclude = scenario_generation["mmsi_to_exclude"]
        time_window = (t0, t0 + timedelta(seconds=duration))
        wind_params, current_params = odm["wind"], odm["current"]

        match control["algorithm"]:
            case "sac":
                controller = SACTrajTrackingController( # TODO: Check if action_repeat creates an issue
                        path_to_params=control["path_to_weights"]
                    )
            case "ppo":
                controller = PPOTrajTrackingController( # TODO: Check if action_repeat creates an issue
                        path_to_params=control["path_to_weights"]
                    )
            case _:
                raise ValueError(f"Invalid RL algorithm")
            
        match scenario_generation["start"]:
            case "Helsingor":
                start_ne = self.ferry_route.waypoints[0]
                start_ne[1] += 200
                yaw = np.deg2rad(70)
                ferry_route = self.ferry_route
            case "Helsingborg":
                start_ne = self.ferry_route.waypoints[-1]
                start_ne[1] -= 400
                start_ne[0] -= 200
                yaw = np.deg2rad(-145)
                ferry_route = PWLPath(self.ferry_route.waypoints.tolist(), flip=True)
            case _:
                raise ValueError(f"Invalid start place")

        states = np.array([*start_ne] + 3 * [0] + [yaw] + 14*[0])

        aurora = AuroraFerry(
            dt,
            eta = (states[0], states[1], states[5]),
            nu = (states[6], states[7], states[11]),
            control=controller,
            navigation=NavigationAurora(
                states,
                dt,
                sensors={
                    'ais': AIS(
                        scenario_generation["ais_data_paths"],
                        t0=time_window[0],
                        tf=time_window[1],
                        mmsi_to_exclude=mmsi_to_exclude, # Aurora AF Helsingborg, since that's us
                        update_every_sec=navigation["update_ais_every_sec"],
                        ),
                    'camera': Camera(
                        scenario_generation["ais_data_paths"].replace("raw", "smooth_interp"),
                        t0=time_window[0],
                        tf=time_window[1],
                        mmsi_to_exclude=mmsi_to_exclude,
                        update_every_sec=navigation["update_camera_every_sec"],
                        )
                    }
            ),
            guidance=TimespaceGuidance(
                global_path=ferry_route,
                u_des=odm["ferry"]["target_speed"],
                **guidance
            ),
            n_cars=n_cars,
            n_passengers=n_passengers
        )

        wind = Wind(
            wind_params["angle"]["value"],
            wind_params["speed"]["value"],
            attraction_beta=wind_params["angle"]["ornstein_uhlenbeck"]["attraction"],
            amplitude_beta=wind_params["angle"]["ornstein_uhlenbeck"]["amplitude"],
            attraction_norm=wind_params["speed"]["ornstein_uhlenbeck"]["attraction"],
            amplitude_norm=wind_params["angle"]["ornstein_uhlenbeck"]["amplitude"],
            dt=dt,
            seed=seed
        )

        current = Current(
            current_params["angle"]["value"],
            current_params["speed"]["value"],
            attraction_beta=current_params["angle"]["ornstein_uhlenbeck"]["attraction"],
            amplitude_beta=current_params["angle"]["ornstein_uhlenbeck"]["amplitude"],
            attraction_norm=current_params["speed"]["ornstein_uhlenbeck"]["attraction"],
            amplitude_norm=current_params["angle"]["ornstein_uhlenbeck"]["amplitude"],
            dt=dt,
            seed=seed
        )
        
        env = NavEnv(aurora, [], [Obstacle(geometry=list(zip(*poly.exterior.coords.xy[::-1]))) for poly in self.map.polygons], dt, wind=wind, current=current)
        sim = Simulator(env, dt=dt, skip_frames=10, render_mode='human', window_size=(6000, 2000), verbose=7)
        sim.run(duration, render=True, store_data=False, m_tot_estimated=aurora.vessel_params.m_tot_estimated, visibility=odm["visibility"], illumination=odm["illumination"], t0=time_window[0])

if __name__ == "__main__":
    import os
    launcher = SimLauncher()
    launcher.run_single_sim(os.path.join("src", "scenarios", "default_config.json"))



        
