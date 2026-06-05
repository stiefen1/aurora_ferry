from python_vehicle_simulator.lib.guidance import IGuidance
from python_vehicle_simulator.lib.path import PWLPath
from python_vehicle_simulator.lib.weather import Current, Wind
from python_vehicle_simulator.lib.obstacle import Obstacle
from python_vehicle_simulator.utils.unit_conversion import knot_to_m_per_sec
from sympy import N
from src.ais.ais import Vessel

from typing import List, Tuple, Dict, Literal, Optional
import numpy as np, numpy.typing as npt

from shapely import Geometry
from datetime import datetime
from matplotlib.axes import Axes
from math import isclose

from colav.planner import TimeSpaceColav
from colav.obstacles.moving import MovingShip
from colav.path.pwl import PWLTrajectory
import colav, logging, shapely
colav.configure_logging(level=logging.ERROR)

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
            max_course_rate: float = 1,
            max_speed: float = 7.5, # m/s
            buffer_target_ships: float = 300.0,
            speed_factor: float = 0.98,
            term_dist: float = 100.0,
            trim_path: float = 400.0,
            dchi: float = 1.0,
            du: float = 0.1,
            delay: Optional[float] = None,
            delay_type: Literal['symmetric', 'late', 'early', 'flat'] = 'symmetric',
            corridor_width: float = 0.0,
            simplify_corridor: float = 0.0,
            new_traj_offset: Optional[float] = 50, #None,
            max_iter: int = 10,
            move_p_0_allowed_after_iter: Optional[int] = 0,
            move_p_f_allowed_after_iter: Optional[int] = None,
            smooth_radius: Optional[float] = None,
            kp: float = 0.01,
            max_shrink_dist_per_step: float = 50.0,
            shrink_eps: float = 0.1,
            **kwargs
    ):
        self.global_path = global_path.trim((0, global_path.length-trim_path), normalized=False)
        self.planner = TimeSpaceColav(u_des, distance_threshold=distance_threshold, shore=shore, colregs=colregs, good_seamanship=good_seamanship, abort_colregs_after_iter=abort_colregs_after_iter, max_course_rate=max_course_rate, max_speed=max_speed, speed_factor=speed_factor, max_iter=max_iter, **kwargs)
        self.update_every_sec = update_every_sec
        self.traj = None
        self.last_update_time = None
        self.lookahead_distance = lookahead_distance
        self.good_semanship = good_seamanship
        self.buffer_target_ships = buffer_target_ships
        self.term_dist = term_dist
        self.dchi = dchi
        self.du = du
        self.delay = delay
        self.delay_type = delay_type
        self.corridor_width = corridor_width
        self.simplify_corridor = simplify_corridor
        self.new_traj_offset = new_traj_offset
        self.move_p_0_allowed_after_iter = move_p_0_allowed_after_iter
        self.move_p_f_allowed_after_iter = move_p_f_allowed_after_iter
        self.smooth_radius = smooth_radius
        self.kp = kp
        self.max_shrink_dist_per_step = max_shrink_dist_per_step
        self.shrink_eps = shrink_eps
        super().__init__()

    def terminated(self, states: npt.NDArray) -> Tuple[bool, Dict]:
        dist_to_term = np.linalg.norm(self.global_path.waypoints[-1] - states[0:2]).astype(float)
        info = {"dist_to_term": dist_to_term, "term_dist": self.term_dist}
        if self.traj is not None:
            progression = self.traj.progression(states[1], states[0], normalized=True)
            return dist_to_term <= self.term_dist or progression >= 1.0, info | {"progression": progression}
        return dist_to_term <= self.term_dist, info

    def __get__(self, states: npt.NDArray, current:Current, wind:Wind, obstacles:List[Obstacle], target_vessels:List[Vessel], timestamp:datetime, *args, **kwargs) -> Tuple[npt.NDArray, Dict]:
        # Update trajectory every self.update_every_sec seconds
        should_update = (self.last_update_time is None or 
                        (timestamp - self.last_update_time).total_seconds() >= self.update_every_sec)

        terminated, info = self.terminated(states)

        info = {}
        if should_update:
            # self.t0 = timestamp
            self.last_update_time = timestamp
            ships_for_projection: List[MovingShip] = []
            for vessel in target_vessels:
                if vessel.heading is not None and vessel.cog is not None and vessel.sog is not None:
                    # ships_for_projection.append(MovingShip.from_csog((vessel.east, vessel.north), vessel.heading, vessel.cog, knot_to_m_per_sec(vessel.sog), vessel.length, vessel.width, degrees=True, mmsi=vessel.mmsi).buffer(self.buffer_target_ships).simplify(2))
                    ships_for_projection.append(MovingShip.from_csog((vessel.east, vessel.north), vessel.heading, vessel.cog, knot_to_m_per_sec(vessel.sog), vessel.length, vessel.width, degrees=True, mmsi=vessel.mmsi, dchi=self.dchi, du=self.du).buffer(self.buffer_target_ships, join_style='mitre'))
                
            # distance_along_global_path = self.global_path.
            pf_ne = self.global_path.get_target_wpts_from(states[0], states[1], self.lookahead_distance, 2)[1]

            # Compute time-offset
            
            if self.traj is not None and self.new_traj_offset is not None:
                prog = self.traj.progression(states[1], states[0])
                x1, y1, t1 = self.traj.interpolate(self.new_traj_offset + prog) # type: ignore -> timespace vector of OS at start of new trajectory
                x0, y0, t0 = self.traj.interpolate(prog)
            else:
                x1, y1, t1 = states[1], states[0], 0.0
                t0 = 0.0

            try:
                traj, info = self.planner.get(
                    (x1, y1),
                    (pf_ne[1], pf_ne[0]),
                    ships_for_projection,
                    heading=states[5] + np.clip(states[11] * self.update_every_sec, -np.deg2rad(60), np.deg2rad(60)),
                    degrees=False,
                    ts_in_TSS=True,
                    good_seamanship=self.good_semanship,
                    delay=self.delay,
                    delay_type=self.delay_type, # type: ignore
                    corridor_width=self.corridor_width,
                    simplify_corridor=self.simplify_corridor,
                    t0=t1-t0,
                    move_p_0_allowed_after_iter=self.move_p_0_allowed_after_iter,
                    move_p_f_allowed_after_iter=self.move_p_f_allowed_after_iter,
                    smooth_radius=self.smooth_radius,
                    max_shrink_dist_per_step=self.max_shrink_dist_per_step,
                    shkrink_eps=self.shrink_eps
                )
            except Exception as e:
                print(f"Error while planning avoidance maneuver: {e}")
                traj = None

            if traj is not None: # valid trajectory was found
                if isclose(traj(0)[0], x1) and  isclose(traj(0)[1], y1): # There are small numerical errors
                    # print(f"Starting position was moved because p0 = {(states[1], states[0])} != {traj(0)}")
                    if self.traj is not None and self.new_traj_offset is not None:
                        self.traj = PWLTrajectory([(x0, y0, 0.0)] + traj.xyt) # type: ignore
                        self.traj.corridor_width = self.corridor_width
                    else:
                        self.traj = traj
                    self.update_time = timestamp

        if self.traj is not None:           
            elapsed_time = (timestamp-self.update_time).total_seconds()
            # print(f"Elapsed time: {elapsed_time:.3f} seconds", " Trajectory: ", self.traj.xyt, "ne: ", states[0:2])
            # return np.array(20*[0]), {'path': self.global_path, 'V_des': self.planner.desired_speed}

            p_des = self.traj.get_closest_point(states[1], states[0]) # east-north
            #  'ne_des': (p_des[1], p_des[0])

            x, y, t = self.traj.interpolate(self.traj.progression(states[1], states[0]))

            delay = elapsed_time - t

            V_des = self.traj.get_speed(elapsed_time)

            xy = np.array(self.traj(elapsed_time))

            V_command = min(max(0.0, V_des + self.kp * delay), 8.0)
            # V_command = min(max(0.0, V_des + self.kp * np.linalg.norm(xy-np.array([states[1], states[0]]))), 8.0) 

            print("Delay: ", elapsed_time - t, V_des, V_command)

            return np.array([p_des[1], p_des[0]] + 4*[0.0] + [V_command] + 13*[0.0]), {'path': PWLPath(self.traj.xy, input_format='east-north'), 'V_des': V_command, 'delay': elapsed_time - t} | info | {'term': terminated}

        # self.prev = {'eta_des': states[0:6], 'nu_des': states[6:12], 'states_des': states, 'info': self.prev['info']}
        return states, {'path': None, 'V_des': None, 'term': terminated} # type:ignore
    
    def __plot__(self, ax:Axes, *args, verbose:int=0, **kwargs) -> Axes:
        
        if verbose >= 2:
            if self.traj is not None:
                self.traj.plot(ax=ax, corridor='both')
                # PWLPath(self.traj.xy, input_format='east-north').plot(ax=ax, verbose=verbose)

        if verbose >= 6:
            if 'projected_obstacles' in self.prev['info'].keys():  
                projected_obstacles: List[shapely.Polygon] = self.prev['info']['projected_obstacles']
                for poly in projected_obstacles:
                    ax.plot(*poly.exterior.coords.xy, c='blue')

        return ax
    

    def reset(self) :
        pass