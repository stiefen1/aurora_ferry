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
    

def get_camera_covariance(distance: float | npt.NDArray, weather: Weather) -> Tuple[float | npt.NDArray, float | npt.NDArray]:
    """
    Return covariance of relative bearing angle (rad^2) and distance (m^2) depending on distance and weather.

    Target size is not taken into account for now. 
    """
    slope_cov_gamma = 1e-4
    slope_cov_dist = 1e-2
    match weather:
        case Weather.SUNNY:
            cov_gamma_0 = 1e-2
            cov_dist_0 = 5
        case Weather.CLOUDY:
            cov_gamma_0 = 3e-2
            cov_dist_0 = 10
        case Weather.FOGGY:
            cov_gamma_0 = 5e-2
            cov_dist_0 = 15
        case _:
            raise ValueError(f"Invalid weather {weather}")
    return (slope_cov_gamma * distance + cov_gamma_0, slope_cov_dist * distance + cov_dist_0)

if __name__ == "__main__":
    import numpy as np, matplotlib.pyplot as plt
    distance = np.linspace(0, 3000, 300)

    fig, axs = plt.subplots(1, 2)
    for w in [Weather.SUNNY, Weather.CLOUDY, Weather.FOGGY]:
        axs[0].plot(distance, np.rad2deg(get_camera_covariance(distance, w)[0]), label=f'{w}')
        axs[1].plot(distance, get_camera_covariance(distance, w)[1], label=f'{w}')

    axs[0].set_title(f"bearing covariance")
    axs[1].set_title(f"distance covariance")
    plt.legend()
    plt.show()
    
