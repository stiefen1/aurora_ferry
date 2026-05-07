from copy import deepcopy

from python_vehicle_simulator.lib.navigation import INavigation
from python_vehicle_simulator.lib.obstacle import Obstacle
from python_vehicle_simulator.lib.sensor import ISensor
from python_vehicle_simulator.lib.weather import Wind, Current
from python_vehicle_simulator.vehicles.vessel import IVessel
from python_vehicle_simulator.utils.unit_conversion import knot_to_m_per_sec, m_per_sec_to_knot
from python_vehicle_simulator.utils.math_fn import ssa

from src.camera.camera import Camera
from src.ais.ais import Vessel, AIS
from src.ferry.target_tracker import TargetTrackerSequentialEKF
from src.ferry.state_estimator import StateEstimatorEKF
from src.odm import ODM
from src.ferry.aurora import AuroraFerryParameters
# from src.camera.camera import get_camera_covariance

from colav.obstacles.moving import MovingShip

from matplotlib.axes import Axes
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np, numpy.typing as npt, gymnasium as gym, os, pathlib


Q_AURORA = np.diag([0.3**2, 0.3**2, 0, 0, 0, (0.4*np.pi/180)**2, 0.02**2, 0.02**2, 0, 0, 0, 15*np.pi/180/3600, *np.array(4*[np.pi/100]), *np.array(4*[1.0])])
R_AURORA = np.diag([1e-2, 1e-2, 0.2*np.pi/180, 5e-2, 5e-2, 5e-2, *np.array(4*[np.pi/100])])

R_WIND = np.diag([np.pi/5, 0.2])    # direction [rad], speed [m/s]
R_CURRENT = np.diag([np.pi/20, 0.1]) # direction [rad], speed [m/s]

DEFAULT_DETECTION_RADIUS_METERS = 3e3
DEFAULT_TIME_TOLERANCE_SECONDS = 1

@dataclass
class TrackedTarget:
    """Clean container for a vessel and its associated tracker."""
    vessel: Vessel
    tracker: TargetTrackerSequentialEKF
    
    def predict(self, command: np.ndarray) -> None:
        """Predict the target's next state."""
        self.tracker.predict(command)
        self.vessel.north, self.vessel.east, sog, cog = self.tracker.x.tolist()
        self.vessel.sog = m_per_sec_to_knot(sog)
        self.vessel.cog = np.rad2deg(cog)
    
    def update_from_ais(self, measurement: np.ndarray) -> None:
        """Update tracker with AIS measurement."""
        self.tracker.update_ais(measurement)
        # Update vessel with tracker state
        self.vessel.north, self.vessel.east, sog, cog = self.tracker.x.tolist()
        self.vessel.sog = m_per_sec_to_knot(sog)
        self.vessel.cog = np.rad2deg(cog)
    
    def update_from_camera(self, measurement: np.ndarray, os_neyaw: Optional[np.ndarray] = None) -> None:
        """Update tracker with camera measurement."""
        self.tracker.update_camera(measurement, os_neyaw=os_neyaw)
        # Update vessel with tracker state
        self.vessel.north, self.vessel.east, sog, cog = self.tracker.x.tolist()
        # print("abc", self.vessel.north, self.vessel.east, sog, cog)
        self.vessel.sog = m_per_sec_to_knot(sog)
        self.vessel.cog = np.rad2deg(cog)

class NavigationAurora(INavigation):
    """
        According to https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/2452115, measurement uncertainties for ReVolt are:

        heading +- 0.2°
        position +- 1cm
        u, v +- 0.05 m/s
        r not specified, assuming it is very low according to graph. let's say r +- 0.05 deg/s as well
    """
    def __init__(
            self,
            states: np.ndarray,
            dt: float,
            *args,
            Q_tt: Optional[np.ndarray] = np.diag([1, 1, np.deg2rad(5), 0.1]),  # process noise     (target tracker)
            R_ais: Optional[np.ndarray] = np.diag([10, 10, np.deg2rad(5), 0.1]),        # measurement noise (AIS)
            R_camera: Optional[np.ndarray] = np.diag([np.deg2rad(1), 100]),     # measurement noise (camera)
            P0_tt: np.ndarray = np.eye(4),                  # state covariance  (target tracker)
            Q_se: Optional[np.ndarray] = Q_AURORA,          # process noise     (state estimator)
            R_se: Optional[np.ndarray] = R_AURORA,          # measurement noise (state estimator)
            P0_se: np.ndarray = np.eye(20),                 # state covariance  (state estimator)
            sensors: Optional[Dict[str, ISensor]] = None,
            max_age_seconds: Optional[float] = None,
            seed: Optional[int] = None,
            odm: Optional[ODM] = None,
            ferry_params: AuroraFerryParameters = AuroraFerryParameters(),
            **kwargs
    ):
        sensors = sensors if sensors is not None else {} # 'camera': Camera(), 'ais': AIS(path_to_ais)
        self.ferry_params = ferry_params

        self.target_tracker_params = {
            'Q': Q_tt,
            'R_ais': R_ais, # self.odm.sensors["ais"]["noise-covariance"]
            'R_camera': R_camera, # self.odm.sensors["camera"]["noise-covariance"]
            'P0': P0_tt,
            'dt': dt
        }

        self.state_estimator_params = {
            'Q': Q_se,
            'R': R_se,
            'P0': P0_se,
            'dt': dt
        }

        self.state_estimator = StateEstimatorEKF(
            **self.state_estimator_params,
            x0=states
        )

        # Clean type-safe collection of tracked targets
        self.target_collection: Dict[int, TrackedTarget] = {}
        self.max_age_seconds = max_age_seconds or dt
        
        # update timing
        self.last_ais_update_time: Optional[datetime] = None
        self.last_camera_update_time: Optional[datetime] = None
        
        super().__init__(states, sensors, *args, **kwargs)
        self.reset(states, seed=seed)

    def measure_states(self, states: npt.NDArray) -> npt.NDArray:
        noise = self.np_random.multivariate_normal(np.array(10*[0]), R_AURORA)
        noisy_states = np.array([states[0], states[1], states[5], states[6], states[7], states[11], *states[12:16]]) + noise
        return noisy_states

    def measure_wind(self, wind: Wind) -> Wind:
        # beta, norm = self.np_random.multivariate_normal(np.array([wind.beta, wind.norm]), R_WIND).flatten().tolist()
        beta, norm = wind._beta_0, wind._norm_0 # assume measurement is just a constant value, e.g. no sensors onboard but access to a low-freq API 
        return Wind(beta, norm) # EXTREMELY IMPORTANT TO CREATE A NEW OBJECT -> OTHERWISE INITIAL OBJECT WILL BE AFFECTED (MEMORY IS SHARED) 
    
    def measure_current(self, current: Current) -> Current:
        # beta, norm = self.np_random.multivariate_normal(np.array([current.beta, current.norm]), R_CURRENT).flatten().tolist()
        beta, norm = current._beta_0, current._norm_0 # assume measurement is just a constant value, e.g. no sensors onboard but access to a low-freq API
        return Current(beta, norm) # EXTREMELY IMPORTANT TO CREATE A NEW OBJECT -> OTHERWISE INITIAL OBJECT WILL BE AFFECTED (MEMORY IS SHARED)
        
    def update_nearby_vessels_from_ais(self, states: npt.NDArray, target_time: datetime, detection_radius_meters: float = DEFAULT_DETECTION_RADIUS_METERS, time_tolerance: int = DEFAULT_TIME_TOLERANCE_SECONDS) -> Tuple[List[Vessel], Dict]:
        # Get "measurements" from AIS and update EKF estimation
        # Only query AIS for new vessels every update_every_sec seconds
        should_update_ais = (
            self.last_ais_update_time is None or 
            (target_time - self.last_ais_update_time).total_seconds() >= self.sensors['ais'].update_every_sec
        )
        
        if should_update_ais:
            self.last_ais_update_time = target_time
            vessels_ais: List[Vessel] = self.sensors['ais'].get_nearby_vessels(states[0], states[1], detection_radius_meters, target_time, time_tolerance=time_tolerance)
            for vessel_ais in vessels_ais:
                # Vessel is already being tracked
                if vessel_ais.mmsi in self.target_collection:
                    tracked_target = self.target_collection[vessel_ais.mmsi]
                    tracked_target.vessel = deepcopy(vessel_ais) # update vessel data
                    measurement = np.array([vessel_ais.north, vessel_ais.east, knot_to_m_per_sec(vessel_ais.sog), np.deg2rad(vessel_ais.cog)])
                    tracked_target.update_from_ais(measurement)
                # Create new tracker for new vessel
                else:
                    new_tracker = TargetTrackerSequentialEKF(
                        **self.target_tracker_params,
                        x0=np.array([vessel_ais.north, vessel_ais.east, knot_to_m_per_sec(vessel_ais.sog), np.deg2rad(vessel_ais.cog)])
                    )
                    self.target_collection[vessel_ais.mmsi] = TrackedTarget(vessel=vessel_ais, tracker=new_tracker)
        else:
            vessels_ais: List[Vessel] = []

        for tracked_target in self.target_collection.values():
            tracked_target.vessel.update_geometry()

        info = {'raw_vessels_ais': vessels_ais}
        return [tracked_target.vessel for tracked_target in self.target_collection.values()], info

    def update_nearby_vessels_from_camera(self, states: npt.NDArray, states_estimation: npt.NDArray, target_time: datetime, visibility: float, illumination: float, time_tolerance: int = DEFAULT_TIME_TOLERANCE_SECONDS) -> Tuple[List[Vessel], Dict]:
        # Get "measurements" from camera and update EKF estimation
        # Only query Camera for new vessels every update_every_sec seconds
        should_update_camera = (
            self.last_camera_update_time is None or 
            (target_time - self.last_camera_update_time).total_seconds() >= self.sensors['camera'].update_every_sec
        )
        
        if should_update_camera:
            self.last_camera_update_time = target_time
            out = self.sensors['camera'](states, target_time, visibility=visibility, illumination=illumination, time_tolerance=time_tolerance)
            vessels_camera: List[Vessel] = out[0]
            info = out[1]
            for i, vessel_camera in enumerate(vessels_camera):
                # Vessel is already being tracked
                if vessel_camera.mmsi in self.target_collection:
                    tracked_target = self.target_collection[vessel_camera.mmsi]
                    tracked_target.vessel = deepcopy(vessel_camera) # update vessel data
                    cov_i: Tuple = self.sensors['camera'].get_camera_covariance(info["rel_distance"][i], visibility, illumination) # type: ignore
                    std_i = np.sqrt(cov_i)

                    measurement = np.array([ssa(info["rel_angle"][i] - states_estimation[5] + self.np_random.normal(0, std_i[0])), info["rel_distance"][i] + self.np_random.normal(0, std_i[1])]) # Target vessel pose estimation relies on our own state estimation
                    # print(info["rel_angle"], info["rel_distance"])
                    tracked_target.update_from_camera(measurement, os_neyaw=np.take(states_estimation, (0, 1, 5)))
                # Create new tracker for new vessel
                else:
                    new_tracker = TargetTrackerSequentialEKF( 
                        **self.target_tracker_params,
                        x0=np.array([vessel_camera.north, vessel_camera.east, knot_to_m_per_sec(vessel_camera.sog), np.deg2rad(vessel_camera.cog)]) # TODO: Don't use true value as initial guess, this is so over-confident
                    )
                    self.target_collection[vessel_camera.mmsi] = TrackedTarget(vessel=vessel_camera, tracker=new_tracker)
        else:
            vessels_camera: List[Vessel] = []

        info = {'raw_vessels_camera': vessels_camera}
        return [tracked_target.vessel for tracked_target in self.target_collection.values()], info

    def measure_nearby_vessels(self, states: npt.NDArray, states_estimation: npt.NDArray, target_time: datetime, visibility: float, illumination: float, detection_radius_meters: float = DEFAULT_DETECTION_RADIUS_METERS, time_tolerance: int = DEFAULT_TIME_TOLERANCE_SECONDS) -> Tuple[List[Vessel], Dict]:
        # Update vessel that are already being tracked
        vessels_command = np.array([0, 0])
        for tracked_target in self.target_collection.values():
            tracked_target.predict(vessels_command)

        info = {}
        if "ais" in self.sensors.keys():
            _, ais_info = self.update_nearby_vessels_from_ais(states, target_time, detection_radius_meters=detection_radius_meters, time_tolerance=time_tolerance)
            info = info | ais_info

        if "camera" in self.sensors.keys():
            _, camera_info = self.update_nearby_vessels_from_camera(states, states_estimation, target_time, visibility, illumination, time_tolerance=time_tolerance)
            info = info | camera_info

        # Update vessel geometries
        for tracked_target in self.target_collection.values():
            tracked_target.vessel.update_geometry()
        
        return [tracked_target.vessel for tracked_target in self.target_collection.values()], info

    def __get__(self, states:np.ndarray, current:Current, wind:Wind, obstacles:List[Obstacle], target_vessels:List[IVessel],  control_commands: np.ndarray, *args, visibility: float = 1.0, illumination: float = 1.0, timestamp: Optional[datetime] = None, **kwargs) -> Tuple[Dict, Dict]:
        """
        target_vessels are does that are part of the simulation, i.e that we have control over.
        AIS is considered as a Sensor, and hence is and instance of ISensor.
        """
        # print(timestamp)
        wind_meas = self.measure_wind(wind or Wind(0, 0))
        current_meas = self.measure_current(current or Current(0, 0))

        states_meas = self.measure_states(states)
        states_estimation = self.state_estimator(control_commands, states_meas) # TODO: add wind, current measurements to state estimator

        if timestamp is not None:
            out = self.measure_nearby_vessels(states, states_estimation, timestamp, visibility=visibility, illumination=illumination)
            target_vessels: List[Vessel] = out[0]
            info = out[1]
        else:
            target_vessels, info = [], {}
        
        observation = {
            "eta": states_estimation[0:6],
            "nu": states_estimation[6:12],
            "states": states_estimation,
            "current": current_meas,
            "wind": wind_meas,
            "obstacles": obstacles,
            "target_vessels": target_vessels # Required for IGuidance
        }

        return observation, info
    
    def reset(self, states: npt.NDArray, seed: Optional[int] = None):
        self.prev = {"eta": states[0:6].copy(), "nu": states[6:12].copy(), "states": states.copy(), "current": None, "wind": None, "obstacles": None, "target_vessels": None, 'info': None}
        self.np_random, _ = gym.utils.seeding.np_random(seed) # type: ignore
        self.state_estimator.reset(states)
        
        # CRITICAL: Clear target collection to avoid state persistence across episodes
        self.target_collection.clear()
        
        # Reset AIS update timing
        self.last_ais_update_time = None

    def __plot__(self, ax:Axes, *args, verbose:int=0, **kwargs) -> Axes:
        if self.last_observation is None:
            return ax
        
        eta = self.last_observation["eta"]

        for target in self.target_collection.values():
            if target.vessel.geometry is not None:
                e, n = target.vessel.east, target.vessel.north
                # print(target.vessel.geometry.shape) (3, 8)
                ax.plot(target.vessel.geometry[1, :], target.vessel.geometry[0, :], c='red')
                ax.text(e+100, n+100, target.vessel.name or "Unknown", c='red')
                ax.text(e+100, n+50, f"SOG [m/s]: {target.vessel.sog:.2f}", c='red')

        if verbose >= 1:
            x, y = eta[1], eta[0]  # east, north
            
            # Plot the vessel position
            ax.scatter(x, y, c='purple', marker='x')
        
        if verbose >= 2:
            x, y, psi = eta[1], eta[0], eta[5]  # heading in radians
            # Plot an arrow showing the heading direction
            arrow_length = 10  # Adjust as needed for visualization
            dx = arrow_length * np.sin(psi)  # East component
            dy = arrow_length * np.cos(psi)  # North component
            
            ax.arrow(x, y, dx, dy, head_width=2, head_length=3, fc='purple', ec='purple')

        if verbose >= 3:
            if "ais" in self.sensors.keys():
                for vessel in self.prev['info']['raw_vessels_ais']:
                    if vessel.geometry is not None:
                        e, n = vessel.east, vessel.north
                        # print(target.vessel.geometry.shape) (3, 8)
                        ax.plot(vessel.geometry[1, :], vessel.geometry[0, :], c='green')
                        ax.text(e+100, n-100, f"new AIS data ({vessel.name})", c='green')

            if "camera" in self.sensors.keys():
                for vessel in self.prev["info"]['raw_vessels_camera']:
                    if vessel.geometry is not None:
                        e, n = vessel.east, vessel.north
                        # print(target.vessel.geometry.shape) (3, 8)
                        ax.plot(vessel.geometry[1, :], vessel.geometry[0, :], c='blue')
                        ax.text(e+100, n-50, f"new camera data ({vessel.name})", c='blue')
        
        return ax

    def __scatter__(self, ax:Axes, *args, **kwargs) -> Axes:
        if self.last_observation is None:
            return ax
        
        eta = self.last_observation["eta"]
        ax.scatter(eta[1], eta[0], c='purple')
        return ax

    def __fill__(self, ax:Axes, *args, **kwargs) -> Axes:
        return ax
    
if __name__ == "__main__":
    nav = NavigationAurora(np.array(20*[0]), 0.2)
    print(nav(np.array(20*[0.1]), None, None, [], [], timestamp=datetime.now(), control_commands=np.array(8*[1])))
    