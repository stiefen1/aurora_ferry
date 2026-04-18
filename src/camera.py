from python_vehicle_simulator.lib.noise import INoise
from python_vehicle_simulator.lib.sensor import ISensor
import numpy as np, os, pandas as pd, numpy.typing as npt
from typing import Tuple, Dict, List, Optional, LiteralString, Any
from datetime import datetime
from src import weather
from src.weather import Weather, is_target_detected


"""
Uncertainties will be parameterized by:
- weather type
- distance of the target vessel w.r.t ferry

First detection should be parameterized by distance, weather and target ship size

We probably assume the detector does not provide a good distance measure of target ship, or even no distance at all -> (AIS data are needed)

"""


class Camera(ISensor):
    def __init__(self, src: LiteralString = "", *args, noise: Optional[INoise] = None, **kwargs):
        self.load_data(src)
        super().__init__(*args, noise=noise, **kwargs)

    def load_data(self, src: LiteralString) -> None:
        # with open(src, 'r') as f:
        #     self.data: Optional[pd.DataFrame] = None
        pass

    def query(self, t: datetime) -> Any:
        success = False
        return None, success

    def __get__(self, t: datetime) -> Tuple[List[Tuple[float, float]], Dict]:
        """
        Current simulation time, to be used to fetch correct data in the database.
        Output: List of target vessel's positions in body coordinates. 
        
        """
        data_at_t, success = self.query(t)
        info = {'status': success}
        return [], info
    
def get_camera_covariance(distance: float | npt.NDArray, visibility: float, illumination: float) -> Tuple[float | npt.NDArray, float | npt.NDArray]:
    """
    Return covariance of relative bearing angle (rad^2) and distance (m^2) depending on distance and weather.

    Target size is not taken into account for now. 
    """
    sqrt_vis_ill = np.sqrt(visibility * illumination)
    a_gamma = np.deg2rad(2e-7) + np.deg2rad(3e-7) * (1 - sqrt_vis_ill)
    a_dist = 1.5e-4 + 2e-4 * (1 - sqrt_vis_ill)

    c_gamma = np.deg2rad(0.1) + np.deg2rad(0.4) * (1 - sqrt_vis_ill)
    c_dist = 50 + 400 * (1 - sqrt_vis_ill)

    return (a_gamma * distance**2 + c_gamma, a_dist * distance**2 + c_dist)

if __name__ == "__main__":
    import numpy as np, matplotlib.pyplot as plt
    distance = np.linspace(0, 3000, 300)

    fig, axs = plt.subplots(1, 2)
    for w in [(1, 1), (0.7, 0.7), (0.4, 0.4)]:
        axs[0].plot(distance, np.rad2deg(get_camera_covariance(distance, *w)[0]), label=f'{w}')
        axs[1].plot(distance, get_camera_covariance(distance, *w)[1], label=f'{w}')

    axs[0].set_title(f"bearing covariance")
    axs[1].set_title(f"distance covariance")
    plt.legend()
    plt.show()
    
