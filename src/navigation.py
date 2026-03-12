from optparse import Option

from python_vehicle_simulator.lib.navigation import INavigation
from python_vehicle_simulator.lib.obstacle import Obstacle
from python_vehicle_simulator.lib.sensor import ISensor
from python_vehicle_simulator.lib.weather import Wind, Current
from python_vehicle_simulator.vehicles.vessel import IVessel
from python_vehicle_simulator.utils.unit_conversion import knot_to_m_per_sec, m_per_sec_to_knot

from src.camera import Camera
from src.ais import Vessel, AIS
from src.target_tracker import TargetTrackerSequentialEKF
from src.state_estimator import StateEstimatorEKF

from matplotlib.axes import Axes
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np, numpy.typing as npt
import gymnasium as gym

Q_AURORA = np.diag([0.3**2, 0.3**2, 0, 0, 0, (0.4*np.pi/180)**2, 0.02**2, 0.02**2, 0, 0, 0, 15*np.pi/180/3600, *np.array(4*[np.pi/100]), *np.array(4*[1.0])])
R_AURORA = np.diag([1e-2, 1e-2, 0.2*np.pi/180, 5e-2, 5e-2, 5e-2, *np.array(4*[np.pi/100])])

R_WIND = np.diag([np.pi/5, 0.2])    # direction [rad], speed [m/s]
R_CURRENT = np.diag([np.pi/20, 0.1]) # direction [rad], speed [m/s]

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
    
    def update_from_camera(self, measurement: np.ndarray) -> None:
        """Update tracker with camera measurement."""
        self.tracker.update_camera(measurement)
        # Update vessel with tracker state
        self.vessel.north, self.vessel.east, sog, cog = self.tracker.x.tolist()
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
            Q_tt: Optional[np.ndarray] = np.eye(4) * 1e-6,  # process noise     (target tracker)
            R_tt: Optional[np.ndarray] = np.eye(4),         # measurement noise (target tracker)
            P0_tt: np.ndarray = np.eye(4),                  # state covariance  (target tracker)
            Q_se: Optional[np.ndarray] = Q_AURORA,          # process noise     (state estimator)
            R_se: Optional[np.ndarray] = R_AURORA,          # measurement noise (state estimator)
            P0_se: np.ndarray = np.eye(20),                 # state covariance  (state estimator)
            sensors: Optional[Dict[str, ISensor]] = None,
            max_age_seconds: Optional[float] = None,
            seed: Optional[int] = None,
            **kwargs
    ):
        sensors = sensors if sensors is not None else {} # {'camera': Camera(), 'ais': AIS()}
            
        self.target_tracker_params = {
            'Q': Q_tt,
            'R': R_tt,
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
        
        super().__init__(states, sensors, *args, **kwargs)
        self.reset(states, seed=seed)

    def measure_states(self, states: npt.NDArray) -> npt.NDArray:
        noise = self.np_random.multivariate_normal(np.array(10*[0]), R_AURORA)
        noisy_states = np.array([states[0], states[1], states[5], states[6], states[7], states[11], *states[12:16]]) + noise
        return noisy_states

    def measure_wind(self, wind: Wind) -> Wind:
        beta, norm = self.np_random.multivariate_normal(np.array([wind.beta, wind.norm]), R_WIND).flatten().tolist()
        return Wind(beta, norm) # EXTREMELY IMPORTANT TO CREATE A NEW OBJECT -> OTHERWISE INITIAL OBJECT WILL BE AFFECTED (MEMORY IS SHARED) 
    
    def measure_current(self, current: Current) -> Current:
        beta, norm = self.np_random.multivariate_normal(np.array([current.beta, current.norm]), R_CURRENT).flatten().tolist()
        return Current(beta, norm) # EXTREMELY IMPORTANT TO CREATE A NEW OBJECT -> OTHERWISE INITIAL OBJECT WILL BE AFFECTED (MEMORY IS SHARED)
    
    def measure_target_vessels(self, timestamp: datetime) -> Tuple[List[Vessel], Dict]:
        ais = self.sensors['ais'](timestamp=timestamp, max_age_seconds=self.max_age_seconds)
        vessels_ais: List[Vessel] = ais[0] # Done in this way to specify type explicitely
        info_ais: Dict = ais[1]

        # Search for target vessels already existing in the target_tracker collection
        target_vessel_command = np.array([0, 0]) # Acceleration in North-East frame assumed for target vessels -> [0, 0] <-> constant speed

        # TODO: Filter out target ship that are too far away from us
        # Predict state for all target ships currently in our database
        for tracked_target in self.target_collection.values():
            tracked_target.predict(target_vessel_command)

        # AIS updates
        for vessel in vessels_ais:
            # Vessel is already being tracked -> Update
            if vessel.mmsi in self.target_collection:
                tracked_target = self.target_collection[vessel.mmsi]
                tracked_target.vessel = vessel  # Replace old vessel data
                measurement = np.array([vessel.north, vessel.east, knot_to_m_per_sec(vessel.sog), np.deg2rad(vessel.cog)])
                tracked_target.update_from_ais(measurement)
            
            # New vessel -> Create tracked target
            else:
                tracker = TargetTrackerSequentialEKF(
                    **self.target_tracker_params, 
                    x0=np.array([vessel.north, vessel.east, knot_to_m_per_sec(vessel.sog), np.deg2rad(vessel.cog)])
                )
                self.target_collection[vessel.mmsi] = TrackedTarget(vessel=vessel, tracker=tracker)
                
        # Camera updates would go here
        # for camera_detection in camera_detections:
        #     if camera_detection.mmsi in self.target_collection:
        #         tracked_target = self.target_collection[camera_detection.mmsi]
        #         measurement = np.array([...])  # camera measurement
        #         tracked_target.update_from_camera(measurement)

        info = {'raw_vessels_ais': vessels_ais}

        # Get updated vessels from tracked targets
        return [tracked_target.vessel for tracked_target in self.target_collection.values()], info
        
    def __get__(self, states:np.ndarray, current:Current, wind:Wind, obstacles:List[Obstacle], target_vessels:List[IVessel],  control_commands: np.ndarray, *args, timestamp: Optional[datetime] = None, **kwargs) -> Tuple[Dict, Dict]:
        """
        target_vessels are does that are part of the simulation, i.e that we have control over.
        AIS is considered as a Sensor, and hence is and instance of ISensor.
        """
        wind_meas = self.measure_wind(wind)
        current_meas = self.measure_current(current)

        states_meas = self.measure_states(states)
        states_estimation = self.state_estimator(control_commands, states_meas) # TODO: add wind, current measurements to state estimator

        if timestamp is not None and "ais" in self.sensors.keys():
            updated_vessels, info = self.measure_target_vessels(timestamp)
        else:
            updated_vessels, info = [], {}
        
        observation = {
            "eta": states_estimation[0:6],
            "nu": states_estimation[6:12],
            "states": states_estimation,
            "current": current_meas,
            "wind": wind_meas,
            "obstacles": obstacles,
            "vessels_ais": updated_vessels,  # Use filtered/tracked vessels instead of raw AIS
            "target_vessels": updated_vessels # Required for IGuidance
        }

        return observation, info
    
    def reset(self, states: npt.NDArray, seed: Optional[int] = None):
        self.prev = {"eta": states[0:6].copy(), "nu": states[6:12].copy(), "states": states.copy(), "current": None, "wind": None, "obstacles": None, "target_vessels": None, 'info': None}
        self.np_random, _ = gym.utils.seeding.np_random(seed) # type: ignore
        self.state_estimator.reset(states)
        
        # CRITICAL: Clear target collection to avoid state persistence across episodes
        self.target_collection.clear()

    def __plot__(self, ax:Axes, *args, verbose:int=0, **kwargs) -> Axes:
        if self.last_observation is None:
            return ax
        
        eta = self.last_observation["eta"]
        ax.scatter(eta[1], eta[0], c='purple')
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
    