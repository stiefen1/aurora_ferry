from python_vehicle_simulator.lib.navigation import INavigation
from src.target_tracker import TargetTrackerSequentialEKF
import numpy as np
from typing import List, Optional, Tuple, Dict
from src.ais import Vessel, AIS
from python_vehicle_simulator.lib.obstacle import Obstacle
from python_vehicle_simulator.lib.weather import Wind, Current
from python_vehicle_simulator.vehicles.vessel import IVessel
from src.camera import Camera
from matplotlib.axes import Axes
from datetime import datetime
from dataclasses import dataclass
from python_vehicle_simulator.utils.unit_conversion import knot_to_m_per_sec, m_per_sec_to_knot

Q_AURORA = None
R_AURORA = None

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
            eta: np.ndarray,
            nu: np.ndarray,
            dt: float,
            *args,
            Q: Optional[np.ndarray] = np.eye(4) * 1e-3, # Process noise
            R: Optional[np.ndarray] = np.eye(4), # measurement noise
            P0: np.ndarray = np.eye(4),
            sensors = {'camera': Camera(), 'ais': AIS()},
            max_age_seconds: Optional[float] = None,
            **kwargs
    ):
        
        super().__init__(eta, nu, sensors, *args, **kwargs)
        self.target_tracker_params = {
            'Q': Q,
            'R': R,
            'P0': P0,
            'dt': dt
        }

        # Clean type-safe collection of tracked targets
        self.target_collection: Dict[int, TrackedTarget] = {}
        self.max_age_seconds = max_age_seconds or dt
        
    def __get__(self, eta:np.ndarray, nu:np.ndarray, current:Current, wind:Wind, obstacles:List[Obstacle], target_vessels:List[IVessel], timestamp: datetime,  *args, tau_actuators: Optional[np.ndarray] = None, **kwargs) -> Tuple[Dict, Dict]:
        """
        target_vessels are does that are part of the simulation, i.e that we have control over.
        AIS is considered as a Sensor, and hence is and instance of ISensor.
        """

        ais = self.sensors['ais'](timestamp=timestamp, max_age_seconds=self.max_age_seconds)
        vessels_ais: List[Vessel] = ais[0] # Done in this way to specify type explicitely
        info_ais: Dict = ais[1]


        # Search for target vessels already existing in the target_tracker collection
        target_vessel_command = np.array([0, 0]) # Acceleration in North-East frame assumed for target vessels

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

        # Get updated vessels from tracked targets
        updated_vessels = [tracked_target.vessel for tracked_target in self.target_collection.values()]
        print(f"updated vessels: {len(updated_vessels)}")

        observation = {
            "eta": eta,
            "nu": nu,
            "current": current,
            "wind": wind,
            "obstacles": obstacles,
            "vessels_ais": updated_vessels,  # Use filtered/tracked vessels instead of raw AIS
        }

        info = {'raw_vessels_ais': vessels_ais, 'vessels_ais': updated_vessels}
        return observation, info
    
    def reset(self):
        pass

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