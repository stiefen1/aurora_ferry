from python_vehicle_simulator.lib.control import IControl
from python_vehicle_simulator.lib.obstacle import Obstacle
from python_vehicle_simulator.lib.weather import Current, Wind
from python_vehicle_simulator.lib.path import PWLPath
from python_vehicle_simulator.utils.math_fn import ssa
from src.ferry.aurora import AuroraFerryActuatorsParameters
from src.utils.normalize import normalize
from typing import List, Tuple, Dict, Optional, Any
from matplotlib.axes import Axes
import numpy as np, numpy.typing as npt, json, os, glob
from abc import ABC, abstractmethod

class RLTrajTrackingController(IControl, ABC):
    model: Any
    actuators: AuroraFerryActuatorsParameters = AuroraFerryActuatorsParameters()

    def __init__(
            self,
            path_to_params: str,
            weights_filename: Optional[str] = None,
            env_config_filename: Optional[str] = None,
            initial_commands: Tuple[float, ...] = tuple(8*[0.0])
    ):
        super().__init__(initial_commands=np.array(initial_commands))
        
        # Load ranges from JSON config. If no filename is provided, use the first json/.zip files
        if env_config_filename is None:
            # Find first JSON file in the path_to_params folder
            json_files = glob.glob(os.path.join(path_to_params, "*.json"))
            if not json_files:
                raise FileNotFoundError(f"No JSON config files found in {path_to_params}")
            ranges_config_path = json_files[0]
            print(f"Auto-detected ranges config: {ranges_config_path}")
        else:
            ranges_config_path = os.path.join(path_to_params, env_config_filename)
        
        if weights_filename is None:
            # Find first ZIP file in the path_to_params folder
            zip_files = glob.glob(os.path.join(path_to_params, "*.zip"))
            if not zip_files:
                raise FileNotFoundError(f"No ZIP model files found in {path_to_params}")
            path_to_model = zip_files[0]
            print(f"Auto-detected model weights: {path_to_model}")
        else:
            path_to_model = os.path.join(path_to_params, weights_filename)
        
        self.load_env_config(ranges_config_path)
        self.load_model(path_to_model)
        self.targets = []
        
    def reset(self, initial_commands: npt.NDArray, seed: Optional[int] = None):
        self.prev = {'u': initial_commands, 'info': None}

    def load_env_config(self, config_path: str) -> None:
        """
        Load observation ranges from JSON configuration file.
        
        Args:
            config_path: Path to JSON file containing ranges
        """
        with open(config_path, 'r') as f:
            ranges_config = json.load(f)
        
        # Convert lists to numpy arrays
        self.uvr_range = {"min": np.array(ranges_config["uvr_range"]["min"]), 
                          "max": np.array(ranges_config["uvr_range"]["max"])}
        self.rel_target_range = {"min": np.array(ranges_config["rel_target_range"]["min"]), 
                                 "max": np.array(ranges_config["rel_target_range"]["max"])}
        self.rel_yaw_cos_range = {"min": np.array(ranges_config["rel_yaw_cos_range"]["min"]), 
                              "max": np.array(ranges_config["rel_yaw_cos_range"]["max"])}
        self.rel_yaw_sin_range = {"min": np.array(ranges_config["rel_yaw_sin_range"]["min"]), 
                              "max": np.array(ranges_config["rel_yaw_sin_range"]["max"])}
        self.speed_error_range = {"min": np.array(ranges_config["speed_error_range"]["min"]), 
                                  "max": np.array(ranges_config["speed_error_range"]["max"])}
        self.azimuth_angles_cos_range = {"min": np.array(ranges_config["azimuth_angles_cos_range"]["min"]), 
                                     "max": np.array(ranges_config["azimuth_angles_cos_range"]["max"])}
        self.azimuth_angles_sin_range = {"min": np.array(ranges_config["azimuth_angles_sin_range"]["min"]), 
                                     "max": np.array(ranges_config["azimuth_angles_sin_range"]["max"])}
        self.thruster_speeds_range = {"min": np.array(ranges_config["thruster_speeds_range"]["min"]), 
                                      "max": np.array(ranges_config["thruster_speeds_range"]["max"])}
        self.rel_wind_speed_range = {"min": np.array(ranges_config["rel_wind_speed_range"]["min"]), 
                                      "max": np.array(ranges_config["rel_wind_speed_range"]["max"])}
        self.rel_wind_angle_cos_range = {"min": np.array(ranges_config["rel_wind_angle_cos_range"]["min"]), 
                                      "max": np.array(ranges_config["rel_wind_angle_cos_range"]["max"])}
        self.rel_wind_angle_sin_range = {"min": np.array(ranges_config["rel_wind_angle_sin_range"]["min"]), 
                                      "max": np.array(ranges_config["rel_wind_angle_sin_range"]["max"])}
        self.rel_current_speed_range = {"min": np.array(ranges_config["rel_current_speed_range"]["min"]), 
                                      "max": np.array(ranges_config["rel_current_speed_range"]["max"])}
        self.rel_current_angle_cos_range = {"min": np.array(ranges_config["rel_current_angle_cos_range"]["min"]), 
                                      "max": np.array(ranges_config["rel_current_angle_cos_range"]["max"])}
        self.rel_current_angle_sin_range = {"min": np.array(ranges_config["rel_current_angle_sin_range"]["min"]), 
                                      "max": np.array(ranges_config["rel_current_angle_sin_range"]["max"])}
        self.total_mass_range = {"min": np.array(ranges_config["total_mass_range"]["min"]), 
                                      "max": np.array(ranges_config["total_mass_range"]["max"])}
        
        # Load environment parameters
        self.action_repeat = ranges_config["action_repeat"]
        self.dt = ranges_config["dt"]
        self.wpts_space_multiplicator = ranges_config["wpts_space_multiplicator"]
        self.n_wpts = ranges_config["n_wpts"]
        self.counter: int = self.action_repeat - 1

        print(f"Loaded environment configuration from {config_path}")

    @abstractmethod
    def load_model(self, path_to_model: str) -> None:
        """
        Load model from file.
        
        Args:
            path_to_model: Path to saved PPO model (.zip file)
        """
        pass

    def _get_obs(self, states: np.ndarray, current: Current, wind: Wind, obstacles: List[Obstacle], target_vessels: List, path: PWLPath, V_des: float, m_tot_estimated: float, *args, **kwargs) -> npt.NDArray:
        """
        Convert internal state to normalized observation format.
        
        Returns:
            Dict: Normalized observations with keys 'ne', 'uvr', 'rel_target', 'rel_yaw'
        """
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
        distances = []
        rel_yaws_cos, rel_yaws_sin = [], []
        self.targets = path.get_target_wpts_from( # type: ignore
            ne[0],
            ne[1],
            self.action_repeat*self.dt*V_des*self.wpts_space_multiplicator,
            self.n_wpts
        )

        for target in self.targets: # type: ignore
            delta = target[0:2] - ne
            distance = float(np.linalg.norm(delta))
            rel_yaw = ssa(yaw + np.atan2(-delta[1], delta[0]))
            rel_yaws_cos.append(np.cos(rel_yaw))
            rel_yaws_sin.append(np.sin(rel_yaw))
            distances.append(distance)

        # Normalize each and cast to float32
        uvr_norm = normalize(uvr, self.uvr_range["min"], self.uvr_range["max"]).astype(np.float32)
        rel_target_norm = normalize(np.array(distances), self.rel_target_range["min"], self.rel_target_range["max"]).astype(np.float32)
        rel_yaws_cos_norm = normalize(np.array(rel_yaws_cos), self.rel_yaw_cos_range["min"], self.rel_yaw_cos_range["max"]).astype(np.float32)
        rel_yaws_sin_norm = normalize(np.array(rel_yaws_sin), self.rel_yaw_sin_range["min"], self.rel_yaw_sin_range["max"]).astype(np.float32)
        speed_error_norm = normalize(np.array([np.linalg.norm(uvr[0:2]) - V_des]), self.speed_error_range["min"], self.speed_error_range["max"]).astype(np.float32)
        azimuth_angles_cos_norm = normalize(np.cos(azimuth_angles), self.azimuth_angles_cos_range["min"], self.azimuth_angles_cos_range["max"]).astype(np.float32)
        azimuth_angles_sin_norm = normalize(np.sin(azimuth_angles), self.azimuth_angles_sin_range["min"], self.azimuth_angles_sin_range["max"]).astype(np.float32)
        thruster_speeds_norm = normalize(thruster_speeds, self.thruster_speeds_range["min"], self.thruster_speeds_range["max"]).astype(np.float32)
        rel_wind_speed_norm = normalize(np.array([rel_wind_norm_0]), self.rel_wind_speed_range["min"], self.rel_wind_speed_range["max"]).astype(np.float32)
        rel_wind_angle_cos_norm = normalize(np.array(np.cos(rel_wind_angle_0)), self.rel_wind_angle_cos_range["min"], self.rel_wind_angle_cos_range["max"]).astype(np.float32)
        rel_wind_angle_sin_norm = normalize(np.array(np.sin(rel_wind_angle_0)), self.rel_wind_angle_sin_range["min"], self.rel_wind_angle_sin_range["max"]).astype(np.float32)
        rel_current_speed_norm = normalize(np.array([rel_current_norm_0]), self.rel_current_speed_range["min"], self.rel_current_speed_range["max"]).astype(np.float32)
        rel_current_angle_cos_norm = normalize(np.array(np.cos(rel_current_angle_0)), self.rel_current_angle_cos_range["min"], self.rel_current_angle_cos_range["max"]).astype(np.float32)
        rel_current_angle_sin_norm = normalize(np.array(np.sin(rel_current_angle_0)), self.rel_current_angle_sin_range["min"], self.rel_current_angle_sin_range["max"]).astype(np.float32)
        total_mass_norm = normalize(np.array([m_tot_estimated]), self.total_mass_range["min"], self.total_mass_range["max"]).astype(np.float32)

        # Create dictionary with normalized observations using EXACT same keys as training environment
        obs_dict = {
            "uvr": uvr_norm,
            "rel_target": rel_target_norm,
            "rel_yaw_cos": rel_yaws_cos_norm,
            "rel_yaw_sin": rel_yaws_sin_norm,
            "speed_error": speed_error_norm,
            "azimuth_angles_cos": azimuth_angles_cos_norm,
            "azimuth_angles_sin": azimuth_angles_sin_norm,
            "thruster_speeds": thruster_speeds_norm,
            "rel_wind_speed": rel_wind_speed_norm,
            "rel_wind_angle_cos": rel_wind_angle_cos_norm,
            "rel_wind_angle_sin": rel_wind_angle_sin_norm,
            "rel_current_speed": rel_current_speed_norm,
            "rel_current_angle_cos": rel_current_angle_cos_norm,
            "rel_current_angle_sin": rel_current_angle_sin_norm,
            "total_mass": total_mass_norm
        }

        # Automatically concatenate in alphabetical order (same as FlattenObservation)
        return np.concatenate([obs_dict[key] for key in sorted(obs_dict.keys())]).astype(np.float32)

    def map_action_to_command(self, action) -> npt.NDArray:
        """
        Map normalized action [-1,1] to actuator command range.
        
        action:     Normalized action                   (6,)
        
        Returns:
            Command for vessel actuators [azimuth, speeds]
        """
        command = np.zeros_like(action)
        alpha_min, alpha_max = self.actuators.alpha_min, self.actuators.alpha_max
        thruster_speed_min, thruster_speed_max = self.actuators.speed_min, self.actuators.speed_max
        command[0:4] = action[0:4] * (alpha_max - alpha_min) / 2 + (alpha_min + alpha_max) / 2
        command[4:8] = action[4:8] * (thruster_speed_max - thruster_speed_min) / 2 + (thruster_speed_min + thruster_speed_max) / 2
        return command

    def __get__(self, states_des: np.ndarray, states: np.ndarray, current: Current, wind: Wind, obstacles: List[Obstacle], target_vessels: List, path: PWLPath, V_des: float, m_tot_estimated: float, *args, **kwargs) -> Tuple[np.ndarray, Dict]:
        """
        Compute control commands using the trained model.
        
        Args:
            states_des: Desired states
            states: Current states
            current: Current conditions
            wind: Wind conditions
            obstacles: List of obstacles
            target_vessels: List of target vessels
            path: Path object for trajectory tracking
            V_des: Desired speed
            
        Returns:
            Control commands and additional info
        """
        if path is None or V_des is None:
            return self.prev['u'], {}
        
        if not(self.counter == self.action_repeat - 1):
            self.counter += 1
            return self.prev['u'], self.prev['info']

        # print("V_DES: ", V_des)
        observation = self._get_obs(states, current, wind, obstacles, target_vessels, path, V_des, m_tot_estimated, *args, **kwargs)
        # print(observation)
        # Retrieve action from NN (azimuth angles, thruster speeds)
        action, _ = self.model.predict(observation, deterministic=True)

        # reset counter
        self.counter = 0

        return self.map_action_to_command(action), {}
    
    def __plot__(self, ax:Axes, *args, verbose:int=0, **kwargs) -> Axes:
        if verbose >= 4:
            for target in self.targets: # target are in NE frame
                ax.scatter(target[1], target[0], c='red')
        return ax