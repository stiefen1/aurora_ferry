from python_vehicle_simulator.lib.guidance import IGuidance
from python_vehicle_simulator.lib.path import PWLPath
from python_vehicle_simulator.lib.weather import Current, Wind
from python_vehicle_simulator.lib.obstacle import Obstacle
from python_vehicle_simulator.utils.unit_conversion import knot_to_m_per_sec
from src.ais import Vessel

from typing import List, Tuple, Dict
import numpy as np, numpy.typing as npt

from shapely import Geometry
from datetime import datetime

from colav.planner import TimeSpaceColav
from colav.obstacles.moving import MovingShip
import colav, logging
colav.configure_logging(level=logging.INFO)

class TimespaceGuidance(IGuidance):
    def __init__(
            self,
            global_path: PWLPath,
            u_des: float,
            shore: List[Geometry] = [],
            update_every_sec: int = 10,
    ):
        self.global_path = global_path
        self.planner = TimeSpaceColav(u_des, shore=shore)
        self.update_every_sec = update_every_sec
        self.traj = None
        self.last_update_time = None
        super().__init__()


    def __get__(self, states: npt.NDArray, current:Current, wind:Wind, obstacles:List[Obstacle], target_vessels:List[Vessel], timestamp:datetime, *args, **kwargs) -> Tuple[npt.NDArray, Dict]:
        # Update trajectory every self.update_every_sec seconds
        should_update = (self.last_update_time is None or 
                        (timestamp - self.last_update_time).total_seconds() >= self.update_every_sec)
        
        if should_update:
            print("update trajectory")
            self.t0 = timestamp
            self.last_update_time = timestamp
            ships_for_projection = []
            for vessel in target_vessels:
                if vessel.heading is not None and vessel.cog is not None and vessel.sog is not None:
                    ships_for_projection.append(MovingShip.from_csog((vessel.east, vessel.north), vessel.heading, vessel.cog, knot_to_m_per_sec(vessel.sog), vessel.length, vessel.width, degrees=True, mmsi=vessel.mmsi).buffer(200, join_style='mitre'))
                
            self.traj, info = self.planner.get(
                (states[1], states[0]),
                (self.global_path.waypoints[-1][1], self.global_path.waypoints[-1][0]), 
                ships_for_projection,
                heading=states[5],
                degrees=False
            )

        if self.traj is not None:           
            elapsed_time = (timestamp-self.t0).total_seconds()
            print(f"Elapsed time: {elapsed_time:.3f} seconds", " Trajectory: ", self.traj.xyt, "ne: ", states[0:2])
            return np.array(20*[0]), {'path': PWLPath(self.traj.xy, input_format='east-north'), 'V_des': self.traj.get_speed(elapsed_time)}

        # self.prev = {'eta_des': states[0:6], 'nu_des': states[6:12], 'states_des': states, 'info': self.prev['info']}
        return states, {'path': None, 'V_des': None} # type:ignore
    
    def reset(self) :
        pass