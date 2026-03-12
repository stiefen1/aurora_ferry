from python_vehicle_simulator.lib.control import IControl
from python_vehicle_simulator.lib.obstacle import Obstacle
from python_vehicle_simulator.lib.weather import Current, Wind
from python_vehicle_simulator.lib.path import PWLPath
from python_vehicle_simulator.utils.math_fn import ssa
from stable_baselines3.ppo import PPO
from src.utils.normalize import normalize
from typing import List, Tuple, Dict, Optional
import numpy as np
import json
import os
import glob

class PPOTrajTrackingController(IControl):
    def __init__(
            self,
            path_to_params: str,
            weights_filename: Optional[str] = None,
            env_config_filename: Optional[str] = None
    ):
        super().__init__()
        
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
        self.load_ppo_model(path_to_model)
        
    def reset(self):
        pass

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
        self.rel_yaw_range = {"min": np.array(ranges_config["rel_yaw_range"]["min"]), 
                              "max": np.array(ranges_config["rel_yaw_range"]["max"])}
        self.speed_error_range = {"min": np.array(ranges_config["speed_error_range"]["min"]), 
                                  "max": np.array(ranges_config["speed_error_range"]["max"])}
        self.azimuth_angles_range = {"min": np.array(ranges_config["azimuth_angles_range"]["min"]), 
                                     "max": np.array(ranges_config["azimuth_angles_range"]["max"])}
        self.thruster_speeds_range = {"min": np.array(ranges_config["thruster_speeds_range"]["min"]), 
                                      "max": np.array(ranges_config["thruster_speeds_range"]["max"])}
        
        # Load environment parameters
        self.action_repeat = ranges_config["action_repeat"]
        self.dt = ranges_config["dt"]
        self.wpts_space_multiplicator = ranges_config["wpts_space_multiplicator"]
        self.n_wpts = ranges_config["n_wpts"]

        print(f"Loaded environment configuration from {config_path}")


    def load_ppo_model(self, path_to_model: str) -> None:
        """
        Load PPO model from file.
        
        Args:
            path_to_model: Path to saved PPO model (.zip file)
        """
        self.model = PPO.load(path_to_model)
        print(f"Loaded PPO weights from {path_to_model}")

    def __get__(self, states_des: np.ndarray, states: np.ndarray, current: Current, wind: Wind, obstacles: List[Obstacle], target_vessels: List, path: PWLPath, V_des: float, *args, **kwargs) -> Tuple[np.ndarray, Dict]:
        """
        Compute control commands using the trained PPO model.
        
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
        # TODO: Implement observation extraction and model prediction
        # This should extract observations in the same format as training environment
        # and use self.model.predict() to get actions

        if path is None or V_des is None:
            return self.prev['u'], {}

        # Get raw values
        ne = states[0:2]
        uvr = np.array([states[6], states[7], states[11]])
        distances, rel_yaws = [], []
        for target in path.get_target_wpts_from(ne[0], ne[1], self.action_repeat*self.dt*V_des*self.wpts_space_multiplicator, self.n_wpts):
            delta = target[0:2] - ne
            distance = float(np.linalg.norm(delta))
            rel_yaw = ssa(states[5] + np.atan2(-delta[1], delta[0]))
            distances.append(distance)
            rel_yaws.append(rel_yaw)

        azimuth_angles = states[12:16]
        thruster_speeds = states[16:20]

        # Normalize each and cast to float32
        uvr_norm = normalize(uvr, self.uvr_range["min"], self.uvr_range["max"]).astype(np.float32)
        rel_target_norm = normalize(np.array(distances), self.rel_target_range["min"], self.rel_target_range["max"]).astype(np.float32)
        rel_yaw_norm = normalize(np.array(rel_yaws), self.rel_yaw_range["min"], self.rel_yaw_range["max"]).astype(np.float32)
        speed_error_norm = normalize(np.array([np.linalg.norm(uvr[0:2]) - V_des]), self.speed_error_range["min"], self.speed_error_range["max"]).astype(np.float32)
        azimuth_angles_norm = normalize(azimuth_angles, self.azimuth_angles_range["min"], self.azimuth_angles_range["max"]).astype(np.float32)
        thruster_speeds_norm = normalize(thruster_speeds, self.thruster_speeds_range["min"], self.thruster_speeds_range["max"]).astype(np.float32)

        # Flatten observation to match training format (Box space of shape (24,))
        observation = np.concatenate([
            uvr_norm,                   # 3 dims  
            rel_target_norm,            # n_wpts dims (5)
            rel_yaw_norm,               # n_wpts dims (5)
            speed_error_norm,           # 1 dim
            azimuth_angles_norm,        # 4 dims
            thruster_speeds_norm        # 4 dims
        ]).astype(np.float32)           # Total: 24 dims
        
        # Retrieve action from NN (azimuth angles, thruster speeds)
        action, _ = self.model.predict(observation, deterministic=True)

        # Map action to command
        command = np.zeros_like(action)
        alpha_min, alpha_max = self.azimuth_angles_range["min"], self.azimuth_angles_range["max"]
        thruster_speed_min, thruster_speed_max = self.thruster_speeds_range["min"], self.thruster_speeds_range["max"]
        command[0:4] = action[0:4] * (alpha_max - alpha_min) / 2 + (alpha_min + alpha_max) / 2
        command[4:8] = action[4:8] * (thruster_speed_max - thruster_speed_min) / 2 + (thruster_speed_min + thruster_speed_max) / 2

        return command, {}
    

if __name__ == "__main__":
    controller = PPOTrajTrackingController('C:\\dev\\aurora_ferry\\models\\ppo')