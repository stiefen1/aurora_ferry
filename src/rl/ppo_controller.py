from src.rl.rl_controller import RLTrajTrackingController
from stable_baselines3.ppo import PPO
from typing import Tuple, Optional

class PPOTrajTrackingController(RLTrajTrackingController):
    def __init__(
            self,
            path_to_params: str,
            weights_filename: Optional[str] = None,
            env_config_filename: Optional[str] = None,
            initial_commands: Tuple[float, ...] = tuple(8*[0.0])
    ):
        super().__init__(path_to_params, weights_filename, env_config_filename, initial_commands)
        
    def load_model(self, path_to_model: str) -> None:
        """
        Load PPO model from file.
        
        Args:
            path_to_model: Path to saved PPO model (.zip file)
        """
        self.model = PPO.load(path_to_model)
        print(f"Loaded PPO weights from {path_to_model}")

if __name__ == "__main__":
    controller = PPOTrajTrackingController('C:\\dev\\aurora_ferry\\models\\ppo')