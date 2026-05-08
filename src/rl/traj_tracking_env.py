import gymnasium as gym, numpy as np, numpy.typing as npt, sys, os
from typing import Dict, Optional, Tuple, List, Literal
from python_vehicle_simulator.vehicles.vessel import IVessel
from python_vehicle_simulator.lib.obstacle import Obstacle
from python_vehicle_simulator.lib.weather import Wind, Current
from python_vehicle_simulator.utils.math_fn import ssa
from python_vehicle_simulator.utils.math_fn import Rzyx
from python_vehicle_simulator.lib.thruster import ROTATION_MATRIX
from python_vehicle_simulator.lib.path import PWLPath
import json

from src.ferry.aurora import AuroraFerry, AuroraFerryActuatorsParameters
from src.utils.normalize import normalize
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from src.odm import ODM
from src.ferry.navigation import NavigationAurora


"""
IMPORTANT NOTES:

- Current and Wind are not handled in the dynamics, i.e. using non-zero values won't have any effect
- This environment is made to train the Revolt to reach a target point
- Action repeat  = number of time a single action is applied to the system (see "step" method)
- To adapt this environment to other tasks, you must modify the following methods:
    - reward()
    - init_action_space()
    - init_observation_space()
    - get_obs()                 ->              extract observation from the simulation
    - map_action_to_command()

"""

DEFAULT_CENTER_NE = (0, 0) # (6.212e6, 351900.0)

DEFAULT_PATH_PARAMS = {
    "d_tot": 10000, "max_turn_deg": 45, "seg_len_range":(500, 1000), "start":DEFAULT_CENTER_NE, "N":1
}

class TrajTrackingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}
    actuators_params = AuroraFerryActuatorsParameters()

    def __init__(
            self,
            dt: float,
            target_vessels:List[IVessel] = [],
            obstacles:List[Obstacle] = [], 
            render_mode:Optional[Literal['human']] = None,
            map_bounds: float = 1e3,
            path_params: Dict = DEFAULT_PATH_PARAMS,
            n_wpts: int = 2,
            wpts_space_multiplicator: int = 25,
            initial_angle_range: Tuple[float, float] = (-180, 180),
            odm: Optional[ODM] = None,
            action_repeat: int = 10,
            path_to_obs_ranges: Optional[str] = None,
    ):
        """
        Gymnasium navigation environment for vessel control.
        
        own_vessel:     Vessel to be controlled
        target_vessels: List of other vessels in environment
        obstacles:      List of obstacles to avoid
        wind:           Wind disturbance model
        current:        Current disturbance model
        render_mode:    Visualization mode ('human' or None)
        map_bounds:     Map boundary limits for rendering (±map_bounds for both x and y)
        """
        self.dt = dt
        self.target_vessels = target_vessels
        self.obstacles = obstacles
        self.map_bounds = map_bounds
        self.path_params = path_params
        self.n_wpts = n_wpts
        self.wpts_space_multiplicator = wpts_space_multiplicator
        self.initial_angle_range = initial_angle_range
        self.odm = odm or ODM()

        self.wind_speed_range = self.odm.wind["speed"] # range for observations -> fine even though actual wind will sometimes gets out of the range
        self.wind_angle_range = self.odm.wind["angle"]
        self.current_speed_range = self.odm.current["speed"]
        self.current_angle_range = self.odm.current["angle"]
        self.V_range = tuple(self.odm.ferry["target-speed"].values())

        self.init_action_space()
        self.init_observation_space(path_to_ranges=path_to_obs_ranges)

        self.safety_radius = 2.5
        self.action_repeat = action_repeat # if dt is 0.02, this leads the RL frequency to 1/(10*0.02) = 1/0.2 = 5Hz

        # Rendering 
        self.render_mode = render_mode
        self.fig = None
        self.ax = None
        self.vessel_plot = None

        # Current step (for plot purpose)
        self._step = 0
        self.max_steps = 500 # i.e. 100 seconds for dt=0.02 and action_repeat=10

    def reset(self, seed: int | None = None, options: Dict | None = None) -> Tuple[Dict, Dict]:
        """
        Start a new episode.

        seed:       Random seed for reproducible episodes
        options:    Additional configuration (unused)

        Returns:
            Tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed, options=options)
        self.np_random, _ = gym.utils.seeding.np_random(seed) # type: ignore

        # Reset own vessel
        ### MASS 
        n_passengers = int(self.np_random.uniform(self.odm.ferry["passengers"]["number"]["min"], self.odm.ferry["passengers"]["number"]["max"]))
        n_cars = int(self.np_random.uniform(self.odm.ferry["cars"]["number"]["min"], self.odm.ferry["cars"]["number"]["max"]))
        # dmass are computed such that average mass stays within the min-max values defined in the ODM
        passenger_dmass = np.clip(
            self.np_random.normal(self.odm.ferry["passengers"]["mass"]["mean"], self.odm.ferry["passengers"]["mass"]["std"]),
            self.odm.ferry["passengers"]["mass"]["min"],
            self.odm.ferry["passengers"]["mass"]["max"]
        ) - self.odm.ferry["passengers"]["mass"]["mean"]
        car_dmass = np.clip(
            self.np_random.normal(self.odm.ferry["cars"]["mass"]["mean"], self.odm.ferry["cars"]["mass"]["std"]),
            self.odm.ferry["cars"]["mass"]["min"],
            self.odm.ferry["cars"]["mass"]["max"]
        ) - self.odm.ferry["cars"]["mass"]["mean"]
        self.own_vessel = AuroraFerry(self.dt, mass=self.odm.ferry["mass"], n_passengers=n_passengers, n_cars=n_cars, passenger_dmass=passenger_dmass, car_dmass=car_dmass)

        # print(f"{self.own_vessel.vessel_params.m_tot_estimated:.1f} in [{self.total_mass_range["min"]:.1f}; {self.total_mass_range["max"]:.1f}] ?")
        # 351900.0, 6.212e6
        # self.map_bounds
        x_init_min = np.array([-300+DEFAULT_CENTER_NE[0], -300+DEFAULT_CENTER_NE[1], 0, 0, 0, -np.pi, 
                               -self.V_range[1], -self.V_range[1]/10, 0, 0, 0, -0.1,
                               *self.azimuth_angles_range["min"],
                               *self.thruster_speeds_range["min"]])
        x_init_max = np.array([300+DEFAULT_CENTER_NE[0], 300+DEFAULT_CENTER_NE[1], 0, 0, 0, np.pi, 
                               self.V_range[1], self.V_range[1]/10, 0, 0, 0, 0.1,
                               *self.azimuth_angles_range["max"],
                               *self.thruster_speeds_range["max"]])
        
        self.own_vessel.reset(random=True, seed=seed, x_min=x_init_min, x_max=x_init_max)
        self.prev_states = self.own_vessel.states.copy()

        R_se = np.diag(self.np_random.uniform(self.odm.sensors['states']['noise-covariance']["min"], self.odm.sensors['states']['noise-covariance']["max"]))
        self.own_vessel.navigation=NavigationAurora(
                                        self.own_vessel.states,
                                        dt=self.dt,
                                        seed=seed,
                                        odm=self.odm,
                                        R_se=R_se
                                    )

        for target_vessel in self.target_vessels:
            target_vessel.reset()

        # Sample a new target position within map bounds
        self.path = PWLPath.sample(**self.path_params, initial_angle=float(self.np_random.uniform(*self.initial_angle_range)), seed=seed)
        self.sample_new_target_speed()
        self.current_waypoint = 1

        # Sample wind current values
        self.wind = Wind(
            self.np_random.uniform(self.wind_angle_range["min"], self.wind_angle_range["max"]),
            self.np_random.uniform(self.wind_speed_range["min"], self.wind_speed_range["max"]),
            attraction_beta=self.odm.wind["angle"]["ornstein-uhlenbeck"]["attraction"],
            amplitude_beta=self.odm.wind["angle"]["ornstein-uhlenbeck"]["amplitude"],
            attraction_norm=self.odm.wind["speed"]["ornstein-uhlenbeck"]["attraction"],
            amplitude_norm=self.odm.wind["angle"]["ornstein-uhlenbeck"]["amplitude"],
            dt=self.dt,
            seed=seed
        )
        self.current = Current(
            self.np_random.uniform(self.current_angle_range["min"], self.current_angle_range["max"]),
            self.np_random.uniform(self.current_speed_range["min"], self.current_speed_range["max"]),
            attraction_beta=self.odm.current["angle"]["ornstein-uhlenbeck"]["attraction"],
            amplitude_beta=self.odm.current["angle"]["ornstein-uhlenbeck"]["amplitude"],
            attraction_norm=self.odm.current["speed"]["ornstein-uhlenbeck"]["attraction"],
            amplitude_norm=self.odm.current["angle"]["ornstein-uhlenbeck"]["amplitude"],
            dt=self.dt,
            seed=seed
        )

        observation = self._get_obs()
        info = self._get_info()

        # Reset figure
        self.fig = None
        self.ax = None
        self.vessel_plot = None

        # Reset step (for plot purpose)
        self._step = 0

        return observation, info

    def step(self, action) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one timestep within the environment.

        action:     Action to take (normalized to [-1,1])    (6,)

        Returns:
            Tuple: (observation, reward, terminated, truncated, info)
        """
        self._step += 1

        # Step vessels
        for _ in range(self.action_repeat):
            for vessel in self.target_vessels:
                vessel.step(self.current, self.wind, self.obstacles, [])
            self.own_vessel.step(self.current, self.wind, self.obstacles, self.target_vessels, control_commands=self.map_action_to_command(action), theta=np.array(8*[1.0]))
            self.wind.step()
            self.current.step()
        
        # Reward result of action
        reward = self.reward()

        # Get observation
        observation = self._get_obs()

        # Check for collisions
        terminated = self.collision()

        # We don't use truncation in this simple environment
        # (could add a step limit here if desired)
        truncated = self._step >= self.max_steps
        
        info = self._get_info()

        if self.path.get_current_waypoint(*self.own_vessel.states[0:2]) > self.current_waypoint: # Update speed when we switch to next waypoint
            self.sample_new_target_speed()
            self.current_waypoint += 1

        self.prev_states = self.own_vessel.states.copy()

        return observation, reward, terminated, truncated, info

    def sample_new_target_speed(self) -> None:
        self.V_des = float(
            np.clip(
                self.np_random.normal(
                    np.mean(self.V_range), # mean = center of V_range
                    (self.V_range[1]-self.V_range[0])/6 # standard deviation = range / 6
                    ), 
                self.V_range[0],
                self.V_range[1]
            )
        )

    def reward(self) -> float:
        """
        Compute reward based on distance to target, power consumption and fault diagnosis performance (to be implemented)
        
        Returns:
            float: Reward value (higher is better)
        """
        return (
            1 -
            (self.dist_to_target()/1300) - # / 500
            self.speed_error() / self.own_vessel.vessel_params.surge_speed_max -
            self.weighted_power_consumption() / 5 # 25 # 50 # 100
        )
    
    def speed_error(self) -> float:
        return np.abs(np.linalg.norm(self.own_vessel.states[6:8]).astype(float) - self.V_des)

    def dist_to_target(self) -> float:
        """
        Calculate Euclidean distance to target path.
        
        Returns:
            float: Distance to path in meters
        """
        ne = np.array(self.own_vessel.eta.neyaw[0:2])
        closest_ne = self.path.closest_point(ne[0], ne[1]) # type: ignore
        target = np.array([closest_ne[1], closest_ne[0]])
        return np.linalg.norm(ne-target) # type: ignore
    
    def weighted_power_consumption(self, w: Optional[npt.NDArray] = None) -> float:
        if w is None:
            w = np.diag(np.concatenate([
                50 / (self.actuators_params.alpha_max - self.actuators_params.alpha_min)**2, # 10 / ...
                1 / (self.actuators_params.speed_max - self.actuators_params.speed_min)**2,
            ])) / 100.0

        dx = np.concatenate([self.own_vessel.states[12:16] - self.prev_states[12:16], self.own_vessel.states[16:20]])
        return float((dx @ w @ dx))
    
    def collision(self) -> bool:
        """
        Check for collisions with boundaries or obstacles.
        
        Returns:
            bool: True if collision detected, False otherwise
        """
        for obs in self.obstacles:
            if obs.distance(*self.own_vessel.eta.neyaw[0:2]) < self.safety_radius:
                return True
        return False
    
    def get_target_wpts(self) -> List[Tuple[float, float, float]]:
        # v_average = np.mean(self.V_range).astype(float)
        return self.path.get_target_wpts_from( # type: ignore
            self.own_vessel.states[0],
            self.own_vessel.states[1],
            self.action_repeat*self.dt*self.V_des*self.wpts_space_multiplicator,
            self.n_wpts
        )

    def _get_obs(self) -> Dict:
        """
        Convert internal state to normalized observation format.
        
        Returns:
            Dict: Normalized observations with keys 'ne', 'uvr', 'rel_target', 'rel_yaw'
        """
        states = self.own_vessel.navigation.prev["states"] # type: ignore
        wind: Wind = self.own_vessel.navigation.prev["wind"] or self.wind
        current: Current = self.own_vessel.navigation.prev["current"] or self.current
        u_wind_0, v_wind_0 = wind.uv0(states[5])
        u_current_0, v_current_0 = current.uv0(states[5])

        
        # Extract elements of observation space
        eta = states[0:6]
        nu = states[6:12]
        ne = eta[0:2]
        uvr = np.take(nu, (0, 1, 5))
        yaw = eta[5]
        azimuth_angles = states[12:16]  # The outcome of a thruster depends on the azimuth angle -> it's probably needed here
        thruster_speeds = states[16:20]
        uv_wind_rel_0 = np.array([u_wind_0, v_wind_0]) - nu[0:2]
        uv_current_rel_0 = np.array([u_current_0, v_current_0]) - nu[0:2]
        rel_wind_angle_0 = ssa(np.atan2(uv_wind_rel_0[1], uv_wind_rel_0[0]))
        rel_current_angle_0 = ssa(np.atan2(uv_current_rel_0[1], uv_current_rel_0[0]))
        rel_wind_norm_0 = np.linalg.norm(uv_wind_rel_0)
        rel_current_norm_0 = np.linalg.norm(uv_current_rel_0)
 
        # Compute distances and yaw angles relative to target waypoints
        distances, rel_yaws = [], []
        try:
            # targets = self.path.get_target_wpts_from(ne[0], ne[1], self.action_repeat*self.dt*self.V_des*self.wpts_space_multiplicator, self.n_wpts)
            targets = self.get_target_wpts() #self.path.get_target_wpts_from(ne[0], ne[1], self.action_repeat*self.dt*v_average*self.wpts_space_multiplicator, self.n_wpts)
        except:
            print(f"An error occured at n,e = {ne} with path = {self.path.waypoints}") # type: ignore
            print(
                "x: ", self.own_vessel.navigation.state_estimator.x,
                "Q: ", self.own_vessel.navigation.state_estimator.Q,
                "R: ", self.own_vessel.navigation.state_estimator.R,
                # "S: ", self.own_vessel.navigation.state_estimator.S,
                "P: ", self.own_vessel.navigation.state_estimator.P
            )
            targets = self.get_target_wpts() #self.path.get_target_wpts_from(ne[0], ne[1], self.action_repeat*self.dt*v_average*self.wpts_space_multiplicator, self.n_wpts)
            

        for target in targets: # type: ignore
            delta = target[0:2] - ne
            distance = float(np.linalg.norm(delta))
            rel_yaw = ssa(yaw + np.atan2(-delta[1], delta[0]))
            distances.append(distance)
            rel_yaws.append(rel_yaw)

        # Normalize each and cast to float32
        uvr_norm = normalize(uvr, self.uvr_range["min"], self.uvr_range["max"]).astype(np.float32)
        rel_target_norm = normalize(np.array(distances), self.rel_target_range["min"], self.rel_target_range["max"]).astype(np.float32)
        rel_yaw_norm = normalize(np.array(rel_yaws), self.rel_yaw_range["min"], self.rel_yaw_range["max"]).astype(np.float32)
        speed_error_norm = normalize(np.array([np.linalg.norm(uvr[0:2]) - self.V_des]), self.speed_error_range["min"], self.speed_error_range["max"]).astype(np.float32)
        azimuth_angles_norm = normalize(azimuth_angles, self.azimuth_angles_range["min"], self.azimuth_angles_range["max"]).astype(np.float32)
        thruster_speeds_norm = normalize(thruster_speeds, self.thruster_speeds_range["min"], self.thruster_speeds_range["max"]).astype(np.float32)
        rel_wind_speed_norm = normalize(np.array([rel_wind_norm_0]), self.rel_wind_speed_range["min"], self.rel_wind_speed_range["max"]).astype(np.float32)
        rel_wind_angle_norm = normalize(np.array([rel_wind_angle_0]), self.rel_wind_angle_range["min"], self.rel_wind_angle_range["max"]).astype(np.float32)
        rel_current_speed_norm = normalize(np.array([rel_current_norm_0]), self.rel_current_speed_range["min"], self.rel_current_speed_range["max"]).astype(np.float32)
        rel_current_angle_norm = normalize(np.array([rel_current_angle_0]), self.rel_current_angle_range["min"], self.rel_current_angle_range["max"]).astype(np.float32)
        total_mass_norm = normalize(np.array([self.own_vessel.vessel_params.m_tot_estimated]), self.total_mass_range["min"], self.total_mass_range["max"]).astype(np.float32)

        return {
            "uvr": uvr_norm,
            "rel_target": rel_target_norm,
            "rel_yaw": rel_yaw_norm,
            "speed_error": speed_error_norm,
            "azimuth_angles": azimuth_angles_norm,
            "thruster_speeds": thruster_speeds_norm,
            "rel_wind_speed": rel_wind_speed_norm,
            "rel_wind_angle": rel_wind_angle_norm,
            "rel_current_speed": rel_current_speed_norm,
            "rel_current_angle": rel_current_angle_norm,
            "total_mass": total_mass_norm
        }

    def init_observation_space(self, path_to_ranges: Optional[str] = None) -> None:
        """
        Initialize observation space with normalized ranges.
        
        Sets up Dict observation space with bounds [-1,1] for all components
        and defines mapping ranges for normalization.
        """
        # Observation space is normalized to enhance learning stability
        self.observation_space = gym.spaces.Dict(
            {
                "uvr": gym.spaces.Box(-1.0, 1.0, shape=(3,)),           # Surge-Sway-YawRate
                "rel_target": gym.spaces.Box(-1.0, 1.0, shape=(self.n_wpts,)),    # Easier to figure out using relative pose
                "rel_yaw": gym.spaces.Box(-1.0, 1.0, shape=(self.n_wpts,)),
                "speed_error": gym.spaces.Box(-1.0, 1.0, shape=(1,)),
                "azimuth_angles": gym.spaces.Box(-1.0, 1.0, shape=(4,)),
                "thruster_speeds": gym.spaces.Box(-1.0, 1.0, shape=(4,)),
                "rel_wind_speed": gym.spaces.Box(-1.0, 1.0, shape=(1,)),
                "rel_wind_angle": gym.spaces.Box(-1.0, 1.0, shape=(1,)),
                "rel_current_speed": gym.spaces.Box(-1.0, 1.0, shape=(1,)),
                "rel_current_angle": gym.spaces.Box(-1.0, 1.0, shape=(1,)),
                "total_mass": gym.spaces.Box(-1.0, 1.0, shape=(1,))
            }
        )

        if path_to_ranges is not None:
            self.load_ranges_from_json(path_to_ranges)
        else:
            # Used to map normalized observations to actual values (see method get_obs)
            self.uvr_range = {"min": np.array([-10, -10, -10]), "max": np.array([10, 10, 10])}
            self.rel_target_range = {"min":np.array(self.n_wpts*[0]), "max": np.array(self.n_wpts*[self.path_params['d_tot']])} # relative distance to a point of the horizon
            self.rel_yaw_range = {"min": np.array(self.n_wpts*[-np.pi]), "max": np.array(self.n_wpts*[np.pi])} # relative bearing angle to a point of the horizon
            self.speed_error_range = {"min": np.array([-3*self.V_range[1]]), "max": np.array([3*self.V_range[1]])}
            self.azimuth_angles_range = {"min": self.actuators_params.alpha_min, "max": self.actuators_params.alpha_max}
            self.thruster_speeds_range = {"min": self.actuators_params.speed_min, "max": self.actuators_params.speed_max}
            self.rel_wind_speed_range = {"min": np.array([0.0]), "max": np.array([self.wind_speed_range["max"] + self.V_range[1]])}
            self.rel_wind_angle_range = {"min": np.array([-np.pi]), "max": np.array([np.pi])}
            self.rel_current_angle_range = {"min": np.array([-np.pi]), "max": np.array([np.pi])}
            self.rel_current_speed_range = {"min": np.array([0.0]), "max": np.array([self.current_speed_range["max"] + self.V_range[1]])}
            self.total_mass_range = {
                "min": np.array(self.odm.ferry["mass"]),
                "max": np.array(self.odm.ferry["mass"] + \
                        self.odm.ferry["passengers"]["number"]["max"] * self.odm.ferry["passengers"]["mass"]["max"] + \
                        self.odm.ferry["cars"]["number"]["max"] * self.odm.ferry["cars"]["mass"]["max"])
            }

    def load_ranges_from_json(self, path: str) -> None:
        with open(path, 'r') as f:
            ranges_config = json.load(f)
        
        # Convert lists to numpy arrays
        self.uvr_range = {"min": np.array(ranges_config["uvr_range"]["min"]), 
                          "max": np.array(ranges_config["uvr_range"]["max"])}
        self.rel_target_range = {"min": np.array(ranges_config["rel_target_range"]["min"]), 
                                 "max": np.array(ranges_config["rel_target_range"]["max"])}
        self.rel_yaw_range = {"min": np.array(ranges_config["rel_yaw_range"]["min"]), 
                              "max": np.array(ranges_config["rel_yaw_range"]["max"])}
        self.speed_error_range = {"min": np.array(ranges_config["speed_error_range"]["min"]), 
                                  "max": np.array(ranges_config["speed_error_range"]["max"])}
        self.azimuth_angles_range = {"min": np.array(ranges_config["azimuth_angles_range"]["min"]), 
                                     "max": np.array(ranges_config["azimuth_angles_range"]["max"])}
        self.thruster_speeds_range = {"min": np.array(ranges_config["thruster_speeds_range"]["min"]), 
                                      "max": np.array(ranges_config["thruster_speeds_range"]["max"])}
        self.rel_wind_speed_range = {"min": np.array(ranges_config["rel_wind_speed_range"]["min"]), 
                                      "max": np.array(ranges_config["rel_wind_speed_range"]["max"])}
        self.rel_wind_angle_range = {"min": np.array(ranges_config["rel_wind_angle_range"]["min"]), 
                                      "max": np.array(ranges_config["rel_wind_angle_range"]["max"])}
        self.rel_current_speed_range = {"min": np.array(ranges_config["rel_current_speed_range"]["min"]), 
                                      "max": np.array(ranges_config["rel_current_speed_range"]["max"])}
        self.rel_current_angle_range = {"min": np.array(ranges_config["rel_current_angle_range"]["min"]), 
                                      "max": np.array(ranges_config["rel_current_angle_range"]["max"])}
        self.total_mass_range = {"min": np.array(ranges_config["total_mass_range"]["min"]), 
                                      "max": np.array(ranges_config["total_mass_range"]["max"])}
        
        # Load environment parameters
        self.action_repeat = ranges_config["action_repeat"]
        self.dt = ranges_config["dt"]
        self.wpts_space_multiplicator = ranges_config["wpts_space_multiplicator"]
        self.n_wpts = ranges_config["n_wpts"]
        print(f"Loaded observation space ranges from {path}")



    
    def export_observation_space_ranges_to(self, path: str) -> None:
        """
        Export observation space ranges to a JSON configuration file.
        
        Args:
            path: File path where to save the JSON configuration
        """
        ranges_config = {
            "uvr_range": {
                "min": self.uvr_range["min"].tolist(),
                "max": self.uvr_range["max"].tolist()
            },
            "rel_target_range": {
                "min": self.rel_target_range["min"].tolist(),
                "max": self.rel_target_range["max"].tolist()
            },
            "rel_yaw_range": {
                "min": self.rel_yaw_range["min"].tolist(),
                "max": self.rel_yaw_range["max"].tolist()
            },
            "speed_error_range": {
                "min": self.speed_error_range["min"].tolist(),
                "max": self.speed_error_range["max"].tolist()
            },
            "azimuth_angles_range": {
                "min": self.azimuth_angles_range["min"].tolist(),
                "max": self.azimuth_angles_range["max"].tolist()
            },
            "thruster_speeds_range": {
                "min": self.thruster_speeds_range["min"].tolist(),
                "max": self.thruster_speeds_range["max"].tolist()
            },
            "rel_wind_speed_range": {
                "min": self.rel_wind_speed_range["min"].tolist(),
                "max": self.rel_wind_speed_range["max"].tolist()
            },
            "rel_wind_angle_range": {
                "min": self.rel_wind_angle_range["min"].tolist(),
                "max": self.rel_wind_angle_range["max"].tolist()
            },
            "rel_current_speed_range": {
                "min": self.rel_current_speed_range["min"].tolist(),
                "max": self.rel_current_speed_range["max"].tolist()
            },
            "rel_current_angle_range": {
                "min": self.rel_current_angle_range["min"].tolist(),
                "max": self.rel_current_angle_range["max"].tolist()
            },
            "total_mass_range": {
                "min": self.total_mass_range["min"].tolist(),
                "max": self.total_mass_range["max"].tolist()
            },
            "action_repeat": self.action_repeat,
            "dt": self.dt,
            "wpts_space_multiplicator": self.wpts_space_multiplicator,
            "n_wpts": self.n_wpts
        }
        
        with open(path, 'w') as f:
            json.dump(ranges_config, f, indent=4)
        
        print(f"Observation space ranges exported to: {path}")


    def _get_info(self) -> Dict:
        """
        Compute auxiliary information for debugging.

        Returns:
            Dict: Empty info dictionary (can be extended)
        """
        return {

        }
    
    def map_action_to_command(self, action) -> npt.NDArray:
        """
        Map normalized action [-1,1] to actuator command range.
        
        action:     Normalized action                   (6,)
        
        Returns:
            Command for vessel actuators [azimuth, speeds]
        """
        command = np.zeros_like(action)
        alpha_min, alpha_max = self.azimuth_angles_range["min"], self.azimuth_angles_range["max"]
        thruster_speed_min, thruster_speed_max = self.thruster_speeds_range["min"], self.thruster_speeds_range["max"]
        command[0:4] = action[0:4] * (alpha_max - alpha_min) / 2 + (alpha_min + alpha_max) / 2
        command[4:8] = action[4:8] * (thruster_speed_max - thruster_speed_min) / 2 + (thruster_speed_min + thruster_speed_max) / 2
        return command

    def init_action_space(self) -> None:
        """
        Initialize action space with normalized range [-1,1].
        
        Sets up Box action space matching vessel's control input dimensions.
        """
        # Observation space is normalized to enhance learning stability
        self.action_space = gym.spaces.Box(
            
            -np.ones(shape=(2 * len(self.actuators_params.thrusters),)), # 2 action per azimuth thrusters
            +np.ones(shape=(2 * len(self.actuators_params.thrusters),))
        ) # action space is -1, +1

    def render(self, mode=None):
        """
        Render the environment for visualization.
        
        mode:       Render mode ('human' for matplotlib visualization)
        
        Creates and updates a 2D plot showing vessel position and target.
        """
        mode = mode or self.render_mode
        if mode not in ("human",):
            return

        if self.fig is None or self.ax is None:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            self.vessel_plot, = self.ax.plot([], [], label='Vessel')
            self.actuators_plot: List[Line2D] = [self.ax.plot([], [], label=f'Th{i}')[0] for i in range(4)]
            self.waypoints_plot = self.ax.scatter([], [], label='waypoints')
            self.ax.set_xlim(-self.map_bounds + DEFAULT_CENTER_NE[1], self.map_bounds + DEFAULT_CENTER_NE[1]) # 351900.0, 6.212e6
            self.ax.set_ylim(-self.map_bounds + DEFAULT_CENTER_NE[0], self.map_bounds + DEFAULT_CENTER_NE[0])
            self.ax.plot(self.path.waypoints[:, 1], self.path.waypoints[:, 0], c='red') # type: ignore
            self.ax.set_xlabel('East')
            self.ax.set_ylabel('North')

            # Create arrow plots for wind and current using axis coordinates
            wind_coords = self.wind.get_arrow_coords(self.ax, 0)
            current_coords = self.current.get_arrow_coords(self.ax, 1)
            
            self.wind_arrow = self.ax.annotate('', xy=(wind_coords[2], wind_coords[3]), 
                                              xytext=(wind_coords[0], wind_coords[1]),
                                              xycoords='axes fraction', textcoords='axes fraction',
                                              arrowprops=dict(arrowstyle='->', color='red', lw=2))
            self.current_arrow = self.ax.annotate('', xy=(current_coords[2], current_coords[3]), 
                                                 xytext=(current_coords[0], current_coords[1]),
                                                 xycoords='axes fraction', textcoords='axes fraction',
                                                 arrowprops=dict(arrowstyle='->', color='blue', lw=2))
            
            # Add text labels for wind and current
            self.wind_text = self.ax.text(wind_coords[2] + 0.02, wind_coords[3] + 0.02,
                                         f'Wind: {self.wind.norm:.1f} m/s',
                                         transform=self.ax.transAxes, fontsize=10, color='red',
                                         bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            self.current_text = self.ax.text(current_coords[2] + 0.02, current_coords[3] + 0.02,
                                            f'Current: {self.current.norm:.1f} m/s',
                                            transform=self.ax.transAxes, fontsize=10, color='blue',
                                            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
                
            self.ax.legend()
            plt.ion()
            plt.show()

        waypoints = np.array(self.get_target_wpts())
        # print(np.rad2deg(self.wind.beta), self.wind.norm, np.rad2deg(self.current.beta), self.current.norm)

        # waypoints should be in format [[x1,y1], [x2,y2], ...] but path returns [[n1,e1], [n2,e2], ...]
        # Convert from North,East to East,North for plotting
        self.waypoints_plot.set_offsets(np.flip(waypoints[:, 0:2]))
        self.vessel_plot.set_data(*self.own_vessel.geometry_for_2D_plot) # type: ignore
        for i, actuator_plot in enumerate(self.actuators_plot):
            envelope = (ROTATION_MATRIX(self.own_vessel.states[12 + i]) @ self.actuators_params.geometries[i].T) + self.actuators_params.xy[i].reshape(-1, 1)
            envelope_in_ned_frame = Rzyx(*self.own_vessel.eta.to_numpy()[3:6].tolist())[0:2, 0:2] @ envelope + self.own_vessel.eta.to_numpy()[0:2, None]
            actuator_plot.set_data(envelope_in_ned_frame[1, :], envelope_in_ned_frame[0, :])

        # Update wind and current arrows
        wind_coords = self.wind.get_arrow_coords(self.ax, 0)
        current_coords = self.current.get_arrow_coords(self.ax, 1)
        
        self.wind_arrow.xy = (wind_coords[2], wind_coords[3])
        self.wind_arrow.xytext = (wind_coords[0], wind_coords[1])
        self.current_arrow.xy = (current_coords[2], current_coords[3])
        self.current_arrow.xytext = (current_coords[0], current_coords[1])
        
        # Update text labels
        self.wind_text.set_position((wind_coords[2] + 0.02, wind_coords[3] + 0.02))
        self.wind_text.set_text(f'Wind: {self.wind.norm:.1f} m/s')
        self.current_text.set_position((current_coords[2] + 0.02, current_coords[3] + 0.02))
        self.current_text.set_text(f'Current: {self.current.norm:.1f} m/s')

        self.ax.set_title(f"Step: {self._step} | Time: {self._step * self.action_repeat * self.dt} | V_des: {self.V_des:.1f} | V: {np.linalg.norm(self.own_vessel.states[6:8]):.1f}")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


# # Register the environment so we can create it with gym.make()
# gym.register(
#     id="gymnasium_env/GymNavEnv-v0",
#     entry_point=GymNavEnv,
#     max_episode_steps=300,  # Prevent infinite episodes
# )

def check_environment() -> None:
    """
    Validate environment implementation using gymnasium checker.
    
    Creates test environment and runs standard validation checks
    to ensure compatibility with RL training frameworks.
    """
    from gymnasium.utils.env_checker import check_env
    from python_vehicle_simulator.utils.unit_conversion import DEG2RAD
    from src.odm import ODM

    dt = 0.2
    
    env = TrajTrackingEnv(
        dt,
        render_mode='human',
        odm=ODM()
    )

    # This will catch many common issues
    try:
        check_env(env)
        print("Environment passes all checks!")
    except Exception as e:
        print(f"Environment has issues: {e}")

    # Run an episode
    obs, info = env.reset()
    print("first observation returned by reset: ", obs)
    
    # Export ranges for controller (optional)
    env.export_observation_space_ranges_to("observation_space_ranges.json")
    
    for step in range(env.max_steps):
        action = env.action_space.sample()  # Random action
        # action = np.array(8*[-1.])
        # action = np.array([0, 0, 0.0, 0.0, 1, -1, -1, -1])
        # action = np.array([0, 0, 0.0, 0.0, 1, 1, 1, 1])
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        print(f"Step {step}: reward={reward:.3f}, distance={env.dist_to_target():.1f}m")
        
        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break
    
    plt.show(block=True)  # Keep plot open

if __name__=="__main__":
    check_environment()
