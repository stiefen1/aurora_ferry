from python_vehicle_simulator.lib.noise import INoise
from python_vehicle_simulator.lib.sensor import ISensor
import numpy as np, os, pandas as pd
from typing import Tuple, Dict, List, Optional, LiteralString, Any
from datetime import datetime
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
    
