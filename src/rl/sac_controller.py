from src.rl.rl_controller import RLTrajTrackingController
from stable_baselines3.sac import SAC
from typing import Tuple, Optional

class SACTrajTrackingController(RLTrajTrackingController):
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
        Load SAC model from file.
        
        Args:
            path_to_model: Path to saved SAC model (.zip file)
        """
        self.model = SAC.load(path_to_model)
        print(f"Loaded SAC weights from {path_to_model}")

if __name__ == "__main__":
    controller = SACTrajTrackingController('C:\\dev\\aurora_ferry\\models\\sac\\2026_04_30_22_20_25')