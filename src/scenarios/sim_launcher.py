from copy import deepcopy
import pathlib

from python_vehicle_simulator.lib.path import PWLPath

from src.ferry.aurora import AuroraFerry, AuroraFerryParameters
from src.rl.ppo_controller import PPOTrajTrackingController
from src.rl.sac_controller import SACTrajTrackingController
from src.ferry.navigation import NavigationAurora
from src.ferry.guidance import TimespaceGuidance
from src.environment.map import HelsingborgMap
from src.ais.ais import AIS, Vessel
from src.camera.camera import Camera

from python_vehicle_simulator.lib.simulator import Simulator
from python_vehicle_simulator.lib.env import NavEnv
from python_vehicle_simulator.lib.weather import Wind, Current
from python_vehicle_simulator.lib.obstacle import Obstacle

import numpy as np, pandas as pd, os, json
from datetime import timedelta, datetime, timezone
from typing import LiteralString, List


class SimLauncher:
    def __init__(
            self
    ):
        self.map = HelsingborgMap()
        self.ferry_route = self.map.get_ferry_routes()['Helsingør (DK) - Helsingborg (SE)']


    def run_single_sim(self, path_to_config: LiteralString | str, render: bool = False) -> None:
        purepath_to_config = pathlib.Path(path_to_config) # type: ignore
        with open(purepath_to_config) as f:
            config = json.load(f)

        scenario_generation = config["scenario_generation"]
        odm = scenario_generation["operational_domain"]
        seed = config["seed"]
        duration = scenario_generation["simulation"]["duration_sec"]
        t0 = pd.to_datetime(scenario_generation["simulation"]["start_time_iso_utc"])
        dt = scenario_generation["simulation"]["dt"]
        guidance = scenario_generation["guidance"]
        navigation = scenario_generation["navigation"]
        camera = scenario_generation["sensors"]["camera"]
        ais = scenario_generation["sensors"]["ais"]
        control = scenario_generation["control"]
        n_passengers = odm["ferry"]["passengers"]["number"]
        n_cars = odm["ferry"]["cars"]["number"]
        mmsi_to_exclude = scenario_generation["mmsi_to_exclude"]
        time_window = (t0, t0 + timedelta(seconds=duration))
        wind_params, current_params = odm["wind"], odm["current"]

        match control["algorithm"]:
            case "sac":
                controller = SACTrajTrackingController(
                        path_to_params=control["path_to_weights"]
                    )
            case "ppo":
                controller = PPOTrajTrackingController(
                        path_to_params=control["path_to_weights"]
                    )
            case _:
                raise ValueError(f"Invalid RL algorithm")
            
        match scenario_generation["start"]:
            case "Helsingor":
                start_ne = deepcopy(self.ferry_route.waypoints[0])
                start_ne[1] += 300
                yaw = np.deg2rad(70)
                ferry_route = deepcopy(self.ferry_route)
            case "Helsingborg":
                start_ne = deepcopy(self.ferry_route.waypoints[-1])
                start_ne[1] -= 400
                start_ne[0] -= 200
                yaw = np.deg2rad(-145)
                ferry_route = PWLPath(self.ferry_route.waypoints.tolist(), flip=True)
            case _:
                raise ValueError(f"Invalid start place")

        states = np.array([*start_ne] + 3 * [0] + [yaw] + 14*[0])

        sensors = {}
        if scenario_generation["sensors"]["ais"]["enabled"]:
            sensors["ais"] = AIS(
                scenario_generation["ais_data_paths"],
                t0=time_window[0],
                tf=time_window[1],
                mmsi_to_exclude=mmsi_to_exclude, # Aurora AF Helsingborg, since that's us
                update_every_sec=ais["update_every_sec"],
                )
        if scenario_generation["sensors"]["camera"]["enabled"]:
            sensors["camera"] = Camera(
                scenario_generation["ais_data_paths"].replace("raw", "smooth_interp"),
                t0=time_window[0],
                tf=time_window[1],
                mmsi_to_exclude=mmsi_to_exclude,
                update_every_sec=camera["update_every_sec"],
                failure=camera["failure"],
                seed=seed
                )

        aurora = AuroraFerry(
            dt,
            eta = (states[0], states[1], states[5]),
            nu = (states[6], states[7], states[11]),
            control=controller,
            navigation=NavigationAurora(
                states,
                dt,
                sensors=sensors,
                seed=seed,
                **navigation
            ),
            guidance=TimespaceGuidance(
                global_path=ferry_route,
                u_des=odm["ferry"]["target_speed"],
                **guidance
            ),
            n_cars=n_cars,
            n_passengers=n_passengers,
            
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
        sim.run(duration, render=render, store_data=True, m_tot_estimated=aurora.vessel_params.m_tot_estimated, visibility=odm["visibility"], illumination=odm["illumination"], t0=time_window[0])

        """
        Data to save
        # Guidance
        target position (prev['info']['ne_des']), target speed (prev['info']['V_des'])

        # Navigation
        actual state (prev['info']['actual_states'])
        state estimation (prev['info']['states'])
        target pose estimation -> x & P

        # Control
        commands (prev['u'])

        """
        

        # Desired states
        t_x_des, x_des, valid_indices, y_index_from_path = sim._extract_data_from_path('guidance.states_des')
        x_des = np.array([e for i, e in enumerate(x_des) if i in valid_indices])
        t_x_des = np.take(t_x_des, valid_indices)

        # Estimated states
        t_x_est, x_est, valid_indices, y_index_from_path = sim._extract_data_from_path('navigation.states')
        x_est = np.array([e for i, e in enumerate(x_est) if i in valid_indices])
        t_x_est = np.take(t_x_est, valid_indices)

        # Control commands
        t_u, u, valid_indices, y_index_from_path = sim._extract_data_from_path('control.u')
        us = []
        for i, ui in enumerate(u):
            if ui is not None:
                u_prev = ui.copy()
            else:
                ui = u_prev.copy()

            us.append(ui)
                
            
        # u = np.array([ui for i, ui in enumerate(u) if i in valid_indices])
        u = np.array(us)
        # t_u = np.take(t_u, valid_indices)

        # Actual states
        t_x, x, valid_indices, y_index_from_path = sim._extract_data_from_path('navigation.actual_states')
        x = np.array([xi for i, xi in enumerate(x) if i in valid_indices])
        t_x = np.take(t_x, valid_indices)

        # Target vessels
        out = sim._extract_data_from_path('navigation.target_vessels')
        t_vessels, valid_indices, y_index_from_path = out[0], out[2], out[3]
        detected_vessels_of_t: List[List[Vessel]] = out[1] # vessels(t) 
        vessels_states = []
        for i, (t, vessels) in enumerate(zip(t_vessels, detected_vessels_of_t)):
            if i in valid_indices:
                vessels_states.append([])
                for v in vessels:
                    vessels_states[-1].append((v.mmsi, v.north, v.east, v.cog, v.sog))
            else:
                vessels_states.append(None)
            
        t_vessels = np.take(t_vessels, valid_indices)
        vessels_states = [vessels_states[i] for i in valid_indices]


        # Save simulation data as npz/jsonl files
        out_dir = os.path.join(purepath_to_config.parent.parent, 
                               "simulations",
                               purepath_to_config.name.split('.')[0])
        os.makedirs(out_dir, exist_ok=True)

        def save_array_npz(time_data, data, file_name):
            data = np.asarray(data, dtype=np.float64)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            np.savez_compressed(os.path.join(out_dir, file_name), time=time_data, data=data)

        save_array_npz(t_x_des, x_des, "guidance_states_des.npz")
        save_array_npz(t_x_est, x_est, "navigation_states.npz")
        save_array_npz(t_u, u, "control_u.npz")
        save_array_npz(t_x, x, "navigation_actual_states.npz")

        df_vessels = pd.DataFrame({
            "time": t_vessels,
            "vessels": vessels_states,
        })
        df_vessels.to_json(os.path.join(out_dir, "navigation_target_vessels.jsonl"), orient="records", lines=True)
        print(f"Saved simulation data in: {out_dir}")

if __name__ == "__main__":
    import os
    launcher = SimLauncher()
    launcher.run_single_sim(os.path.join("sim_data", "test", "scenarios", "test_0.json"), render=True)
