from python_vehicle_simulator.lib.guidance import IGuidance
from python_vehicle_simulator.lib.path import PWLPath
from python_vehicle_simulator.lib.weather import Current, Wind
from python_vehicle_simulator.lib.obstacle import Obstacle
from python_vehicle_simulator.utils.unit_conversion import knot_to_m_per_sec
from src.ais.ais import Vessel

from typing import List, Tuple, Dict
import numpy as np, numpy.typing as npt

from shapely import Geometry
from datetime import datetime
from matplotlib.axes import Axes

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
            lookahead_distance: float = 1000,
            colregs: bool = True,
            good_seamanship: bool = True,
            distance_threshold: float = 3000.0,
            abort_colregs_after_iter: int = 1,
            max_course_rate: float = 0.5,
            max_speed: float = 7.5, # m/s
            **kwargs
    ):
        self.global_path = global_path
        self.planner = TimeSpaceColav(u_des, distance_threshold=distance_threshold, shore=shore, colregs=colregs, good_seamanship=good_seamanship, abort_colregs_after_iter=abort_colregs_after_iter, max_course_rate=max_course_rate, max_speed=max_speed, **kwargs)
        self.update_every_sec = update_every_sec
        self.traj = None
        self.last_update_time = None
        self.lookahead_distance = lookahead_distance
        self.good_semanship = good_seamanship
        super().__init__()


    def __get__(self, states: npt.NDArray, current:Current, wind:Wind, obstacles:List[Obstacle], target_vessels:List[Vessel], timestamp:datetime, *args, **kwargs) -> Tuple[npt.NDArray, Dict]:
        # Update trajectory every self.update_every_sec seconds
        should_update = (self.last_update_time is None or 
                        (timestamp - self.last_update_time).total_seconds() >= self.update_every_sec)
        
        if should_update:
            print("update trajectory")
            self.t0 = timestamp
            self.last_update_time = timestamp
            ships_for_projection: List[MovingShip] = []
            for vessel in target_vessels:
                if vessel.heading is not None and vessel.cog is not None and vessel.sog is not None:
                    ships_for_projection.append(MovingShip.from_csog((vessel.east, vessel.north), vessel.heading, vessel.cog, knot_to_m_per_sec(vessel.sog), vessel.length, vessel.width, degrees=True, mmsi=vessel.mmsi).buffer(200, join_style='mitre'))
                
            # distance_along_global_path = self.global_path.
            pf_ne = self.global_path.get_target_wpts_from(states[0], states[1], self.lookahead_distance, 2)[1]
            traj, info = self.planner.get(
                (states[1], states[0]),
                (pf_ne[1], pf_ne[0]),
                ships_for_projection,
                heading=states[5],
                degrees=False,
                ts_in_TSS=True,
                good_seamanship=self.good_semanship
            )

            if traj is not None: # valid trajectory was found
                self.traj = traj

        if self.traj is not None:           
            elapsed_time = (timestamp-self.t0).total_seconds()
            # print(f"Elapsed time: {elapsed_time:.3f} seconds", " Trajectory: ", self.traj.xyt, "ne: ", states[0:2])
            # return np.array(20*[0]), {'path': self.global_path, 'V_des': self.planner.desired_speed}
            return np.array(20*[0]), {'path': PWLPath(self.traj.xy, input_format='east-north'), 'V_des': self.traj.get_speed(elapsed_time)}

        # self.prev = {'eta_des': states[0:6], 'nu_des': states[6:12], 'states_des': states, 'info': self.prev['info']}
        return states, {'path': None, 'V_des': None} # type:ignore
    
    def __plot__(self, ax:Axes, *args, verbose:int=0, **kwargs) -> Axes:
        if self.traj is not None:
            PWLPath(self.traj.xy, input_format='east-north').plot(ax=ax, verbose=verbose)
        return ax

    def reset(self) :
        pass