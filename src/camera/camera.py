from optparse import Option

from python_vehicle_simulator.lib.noise import INoise
from python_vehicle_simulator.lib.sensor import ISensor
from python_vehicle_simulator.utils.math_fn import ssa
import numpy as np, os, pandas as pd, numpy.typing as npt, yaml, pyproj, gymnasium as gym
from math import isnan
from typing import Tuple, Dict, List, Optional, Any
from datetime import datetime
from src.ais.ais import Vessel
# from src.camera.weather import is_target_detected
from dataclasses import dataclass


"""
Uncertainties will be parameterized by:
- weather type
- distance of the target vessel w.r.t ferry

First detection should be parameterized by distance, weather and target ship size

We probably assume the detector does not provide a good distance measure of target ship, or even no distance at all -> (AIS data are needed)

"""

DEFAULT_CAMERA_MAPPING_FILE = 'config.yaml'

@dataclass
class CameraParams:
    fov = 0

class Camera(ISensor):
    def __init__(
            self,
            csv_path: str,
            mapping_file: Optional[str] = None,
            t0: Optional[datetime] = None,
            tf: Optional[datetime] = None,
            mmsi_to_exclude: List[int] = [],
            update_every_sec: int = 1,
            params: CameraParams = CameraParams(),
            failure: Optional[Dict] = None,
            seed: Optional[int] = None,
            **kwargs
        ):
        """
        Initialize Camera sensor from CSV with flexible column mapping.
        
        Args:
            csv_path: Path to the camera CSV file  
            mapping_file: Path to JSON mapping file (optional)
            t0: Start time filter
            tf: End time filter
            mmsi_to_exclude: List of MMSIs to exclude
            update_every_sec: Update frequency
        """
        self.t0 = t0
        self.tf = tf
        self.mmsi_to_exclude = mmsi_to_exclude
        self.update_every_sec = update_every_sec
        self.params = params

        if failure is not None:
            self.failure_time = pd.to_datetime(failure["time"])
        else:
            self.failure_time = None

        super().__init__(**kwargs)

        # Load mapping configuration
        if mapping_file is None:
            mapping_file = os.path.join(os.path.dirname(__file__), DEFAULT_CAMERA_MAPPING_FILE)
        
        with open(mapping_file, 'r') as f:
            self.config = yaml.safe_load(f)

        # Load and process CSV data
        self.df = pd.read_csv(csv_path)
        self.column_mapping = self._create_column_mapping()
        
        # Apply filtering
        self._apply_mmsi_filter()
        self._apply_temporal_filter()
        
        # Setup coordinate transformer if needed
        utm_zone = self.config.get('coordinates', {}).get('utm_zone', 33)
        self.transformer = pyproj.Transformer.from_proj(
            '+proj=longlat +datum=WGS84',
            f'+proj=utm +zone={utm_zone} +datum=WGS84 +units=m +no_defs'
        )
        
        # Validate required columns exist
        self._validate_data()
        
        if len(self.df) == 0:
            raise ValueError(f"No camera data found in the specified time range or after filtering")
        
        self.reset(seed=seed)

    def failure(self, time: pd.Timestamp) -> bool:
        if self.failure_time is not None:
            return time >= self.failure_time
        return False

    def _create_column_mapping(self) -> Dict[str, str]:
        """Create mapping from Vessel attributes to actual CSV columns."""
        mapping = {}
        
        # Map mandatory fields
        for vessel_attr, possible_cols in self.config['mandatory'].items():
            for col in possible_cols:
                if col in self.df.columns:
                    mapping[vessel_attr] = col
                    break
        
        # Map optional fields
        for vessel_attr, field_config in self.config['optional'].items():
            for col in field_config['columns']:
                if col in self.df.columns:
                    mapping[vessel_attr] = col
                    break
                    
        return mapping

    def _validate_data(self):
        """Validate that all required data is available."""
        required_groups = [
            ['mmsi'],
            ['latitude', 'longitude', 'north', 'east'],  # Need either lat/lon OR north/east
            ['sog', 'cog', 'heading', 'timestamp']
        ]
        
        for group in required_groups:
            if group == ['latitude', 'longitude', 'north', 'east']:
                # Special case: need either lat/lon OR north/east
                has_latlon = 'latitude' in self.column_mapping and 'longitude' in self.column_mapping
                has_northeast = 'north' in self.column_mapping and 'east' in self.column_mapping
                if not (has_latlon or has_northeast):
                    raise ValueError("Missing position data: need either (latitude, longitude) OR (north, east)")
            else:
                for field in group:
                    if field not in self.column_mapping:
                        available = list(self.df.columns)
                        raise ValueError(f"Required field '{field}' not found. Available columns: {available}")

    def _apply_mmsi_filter(self):
        """Filter out vessels with excluded MMSIs."""
        if not self.mmsi_to_exclude:
            return  # No filtering needed
            
        # Get the MMSI column  
        mmsi_col = self.column_mapping.get('mmsi')
        if mmsi_col is None:
            return  # Can't filter without MMSI column
            
        # Create mask to exclude specified MMSIs
        mask = ~self.df[mmsi_col].isin(self.mmsi_to_exclude)
        
        # Apply the filter
        original_count = len(self.df)
        self.df = self.df[mask].reset_index(drop=True)
        filtered_count = len(self.df)
        
        if filtered_count < original_count:
            print(f"Camera: Excluded {original_count - filtered_count} records with MMSIs: {self.mmsi_to_exclude}")

    def _apply_temporal_filter(self):
        """Apply temporal filtering based on t0 and tf parameters."""
        if self.t0 is None and self.tf is None:
            return  # No filtering needed
            
        # Get the timestamp column
        timestamp_col = self.column_mapping.get('timestamp')
        if timestamp_col is None:
            return  # Can't filter without timestamp column
            
        # Convert timestamps to pandas datetime
        timestamps = pd.to_datetime(self.df[timestamp_col])
        
        # Create mask based on provided time bounds
        mask = pd.Series([True] * len(self.df), index=self.df.index)
        
        if self.t0 is not None:
            t0_pd = pd.to_datetime(self.t0)
            # Handle timezone compatibility
            if timestamps.dt.tz is not None and t0_pd.tz is None:
                t0_pd = t0_pd.tz_localize('UTC')
            elif timestamps.dt.tz is None and t0_pd.tz is not None:
                t0_pd = t0_pd.tz_localize(None)
            mask &= (timestamps >= t0_pd)
        
        if self.tf is not None:
            tf_pd = pd.to_datetime(self.tf)
            # Handle timezone compatibility
            if timestamps.dt.tz is not None and tf_pd.tz is None:
                tf_pd = tf_pd.tz_localize('UTC')
            elif timestamps.dt.tz is None and tf_pd.tz is not None:
                tf_pd = tf_pd.tz_localize(None)
            mask &= (timestamps <= tf_pd)
        
        # Apply the filter
        self.df = self.df[mask].reset_index(drop=True)

    def get_vessels_at_time(
        self, 
        target_time: datetime, 
        time_tolerance: float = 0.2
    ) -> List[Vessel]:
        """Get all vessels at a specific time with tolerance."""
        vessels = []
        
        # Convert target time to pandas datetime and handle timezone issues
        target_pd = pd.to_datetime(target_time)

        if self.failure(target_pd):
            return []
        
        # Get unique vessels
        mmsi_col = self.column_mapping['mmsi']
        timestamp_col = self.column_mapping['timestamp']
        
        for mmsi in self.df[mmsi_col].unique():
            vessel_data = self.df[self.df[mmsi_col] == mmsi].copy()
            
            # Parse timestamps
            vessel_timestamps = pd.to_datetime(vessel_data[timestamp_col])
            
            # Handle timezone mismatches
            if vessel_timestamps.dt.tz is not None and target_pd.tz is None:
                # Camera data is timezone-aware, target is naive - convert target to UTC
                target_pd = target_pd.tz_localize('UTC')
            elif vessel_timestamps.dt.tz is None and target_pd.tz is not None:
                # Camera data is naive, target is timezone-aware - remove target timezone
                target_pd = target_pd.tz_localize(None)
            
            # Find records within time window
            time_diff = (vessel_timestamps - target_pd).abs()
            valid_mask = time_diff <= pd.Timedelta(seconds=time_tolerance)
            
            if valid_mask.any():
                # Get closest record
                closest_idx = time_diff[valid_mask].idxmin()
                row = vessel_data.loc[closest_idx]
                
                # Create vessel
                vessel = self._create_vessel_from_row(row, vessel_timestamps.loc[closest_idx])
                vessels.append(vessel)
        
        return vessels

    def _create_vessel_from_row(self, row: pd.Series, timestamp: pd.Timestamp) -> Vessel:
        """Create a Vessel object from a DataFrame row."""
        # Get mandatory fields
        mmsi = int(row[self.column_mapping['mmsi']])
        sog = float(row[self.column_mapping['sog']])
        cog = float(row[self.column_mapping['cog']])
        heading = float(row[self.column_mapping['heading']])

        if isnan(heading): # handle nan heading values 
            heading = cog
        
        # Get position - prefer north/east, fallback to lat/lon
        if 'north' in self.column_mapping and 'east' in self.column_mapping:
            north = float(row[self.column_mapping['north']])
            east = float(row[self.column_mapping['east']])
            # If we also have lat/lon, use them
            if 'latitude' in self.column_mapping and 'longitude' in self.column_mapping:
                lat = float(row[self.column_mapping['latitude']])
                lon = float(row[self.column_mapping['longitude']])
            else:
                # Convert north/east back to lat/lon (approximate)
                lon, lat = self.transformer.transform(east, north, direction='INVERSE')
        else:
            # Only have lat/lon, convert to north/east
            lat = float(row[self.column_mapping['latitude']])
            lon = float(row[self.column_mapping['longitude']])
            east, north = self.transformer.transform(lon, lat)
        
        # Get optional fields with defaults
        name = self._get_optional_field(row, 'name')
        vessel_type = self._get_optional_field(row, 'vessel_type')
        nav_status = self._get_optional_field(row, 'nav_status')
        length = float(self._get_optional_field(row, 'length'))
        width = float(self._get_optional_field(row, 'width'))
        
        return Vessel(
            mmsi=mmsi,
            lat=lat,
            lon=lon,
            north=north,
            east=east,
            sog=sog,
            cog=cog,
            heading=heading,
            timestamp=timestamp.to_pydatetime(),
            name=name,
            vessel_type=vessel_type,
            nav_status=nav_status,
            length=length,
            width=width
        )

    def _get_optional_field(self, row: pd.Series, field: str) -> Any:
        """Get optional field value or return default."""
        if field in self.column_mapping:
            return row[self.column_mapping[field]]
        else:
            return self.config['optional'][field]['default']

    def get_rel_angles_and_distances(self, states: npt.NDArray, ts_enyaw: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
        # nv = len(all_vessels)
        os_en = np.array([[states[1], states[0]]])
        rel_en = ts_enyaw[:, 0:2] - os_en
        rel_angles = ssa(np.atan2(rel_en[:, 0], rel_en[:, 1]))
        rel_distances = np.linalg.norm(rel_en, axis=1)
        return rel_angles, rel_distances 

    def __get__(self, states: npt.NDArray, target_time: datetime, visibility: float, illumination: float, time_tolerance: float = 0.2) -> Tuple[List[Vessel], Dict]:
        """
        Get detected vessels based on camera simulation.
        
        Args:
            states: Own ship state vector [north, east, ...]
            t: Current simulation time
            visibility: Visibility factor (0-1)  
            illumination: Illumination factor (0-1)
            time_tolerance: Time tolerance for finding vessels (seconds)
            
        Returns:
            Tuple of (detected_vessels, info_dict)
        """
        # Retrieve all vessels at current timestep
        all_vessels = self.get_vessels_at_time(target_time, time_tolerance=time_tolerance)
        ts_states = np.array([[v.east, v.north, v.heading, v.length, v.width] for v in all_vessels]).reshape(-1, 5) # nv x 5
        ts_states[:, 2] = ssa(np.deg2rad(ts_states[:, 2])) # Convert heading to radians in (-pi, pi)

        # Compute relative angles and distances w.r.t to own ship
        rel_angles, rel_distances = self.get_rel_angles_and_distances(states, ts_states[:, 0:3])
        
        detected, info = self.is_target_detected(
                rel_angles,
                rel_distances,
                ts_states[:, 2],
                ts_states[:, 3], 
                ts_states[:, 4], 
                visibility, 
                illumination
            )
        
        detected_vessels: List[Vessel] = np.array(all_vessels)[detected == True].tolist()

        angles_std, distances_std = self.get_camera_std(rel_distances[detected == True], visibility, illumination)
        angles_cov, distances_cov = angles_std**2, distances_std**2
        if len(detected_vessels) > 0:
            rel_angles_wrt_psi = ssa(rel_angles - states[5])
            noisy_rel_angles = self.np_random.multivariate_normal(rel_angles_wrt_psi[detected == True], np.diag(angles_cov))
            noisy_rel_distances = self.np_random.multivariate_normal(rel_distances[detected == True], np.diag(distances_cov))
        else:
            noisy_rel_angles = np.array([])
            noisy_rel_distances = np.array([])

        info = {"noisy_rel_angles": noisy_rel_angles.tolist(), "noisy_rel_distances": noisy_rel_distances.tolist(), 'all_vessels_in_camera': all_vessels, "angles_cov": angles_cov, "distances_cov": distances_cov}
        return detected_vessels, info
    
    def get_camera_std(self, distance: float | npt.NDArray, visibility: float, illumination: float) -> Tuple[float | npt.NDArray, float | npt.NDArray]:
        """
        Return standard deviation of relative bearing angle (rad) and distance (m) depending on distance and weather.
        """
        sqrt_vis_ill = np.sqrt(visibility * illumination)
        a_gamma = np.deg2rad(1e-7) + np.deg2rad(3e-7) * (1 - sqrt_vis_ill) # To be provided as camera params
        a_dist = 1e-4 + 2e-4 * (1 - sqrt_vis_ill)

        c_gamma = np.deg2rad(0.1) + np.deg2rad(0.4) * (1 - sqrt_vis_ill)
        c_dist = 50 + 400 * (1 - sqrt_vis_ill)

        return a_gamma * distance**2 + c_gamma, a_dist * distance**2 + c_dist
    
    def get_detection_probability(
            self,
            rel_angles: npt.NDArray,
            rel_distances: npt.NDArray,
            yaw_ts: npt.NDArray,
            loa: npt.NDArray,
            beam: npt.NDArray,
            visibility: float,
            illumination: float,
        ) -> Tuple[npt.NDArray, Dict]:
        """
        Compute the probability of detection using a camera.

        os_ne: N-E position of the own ship with shape (2,) or (2, N)
        ts_neyaw: N-E-Yaw pose of the own ship with shape (3,) or (3, N)
        loa: Lenght-Over-All [m] of the target ship 
        beam: Beam [m] of the target ship 
        visibility: scalar ranging from 0 (dense fog) to 1 (clear)
        illumination: scalar randing from 0 (night) to 1 (daylight)
        
        """
        # target's FOV
        delta_angle_abs = np.abs(ssa(yaw_ts - rel_angles))
        corrected_size =  0.5 * (beam + loa) - 0.5 * np.cos(2*delta_angle_abs) * (loa - beam) # beam when 0 and loa when pi/2
        half_fov_rad = np.atan(corrected_size / 2 / rel_distances)
        fov = 2 * np.rad2deg(half_fov_rad)

        # effect of visibility & illumination
        sqrt_vis_ill = np.sqrt(visibility * illumination)
        scale = 0.7 - 0.35 * sqrt_vis_ill
        offset = 3 - 2 * sqrt_vis_ill

        # p -> 0 when FOV -> 0
        # p -> 1 when FOV -> 30
        # p -> 0 when sqrt_vis_ill -> 0
        return 1 / (1 + 1 * np.exp(-(fov-offset)/scale) ), {} #{"corrected_size": corrected_size, "rel_angle": rel_angle.item(), "rel_distance": rel_distance.item()} # "yaw_ts": yaw_ts, "rel_angle": rel_angle, "delta_angle_abs": delta_angle_abs, "b": beam, "l": loa}

    def is_target_detected(
            self,
            rel_angles: npt.NDArray,
            rel_distances: npt.NDArray,
            yaw_ts: npt.NDArray,
            loa: npt.NDArray,
            beam: npt.NDArray,
            visibility: float,
            illumination: float
        ) -> Tuple[np.bool, Dict]:
        p, info = self.get_detection_probability(rel_angles, rel_distances, yaw_ts, loa, beam, visibility, illumination)
        val = self.np_random.uniform(low=0, high=1, size=rel_angles.shape[0])
        return np.bool(val <= p), info
    
    def reset(self, seed: Optional[int] = None) -> None:
        self.np_random, _ = gym.utils.seeding.np_random(seed) # type: ignore

if __name__ == "__main__":
    """
    Modify the get_camera_std method (lines 365-376) of Camera class to change the behavior
    """
    import numpy as np, matplotlib.pyplot as plt, os
    from matplotlib import cm, colors
    from datetime import timedelta

    # Load camera data -> Actually useless here, but required to instantiate Camera object
    day_str = "2023-04-03"
    t0_str = "02:00:00.000Z" 
    t0 = pd.to_datetime(day_str + "T" + t0_str)
    duration = 1000
    time_window = (t0, t0 + timedelta(seconds=duration))
    camera = Camera(os.path.join('data', 'smooth_interp', day_str.replace('-', '_') + '_' + t0_str[0:5].replace(':', '_') + '.csv'), t0=time_window[0] - timedelta(hours=2), tf=time_window[1] - timedelta(hours=2))

    # Relative distances of target ships to be tested
    distances = np.linspace(0, 3000, 300)

    # Visibility and illumination values to be tested
    vis_and_illum = set([(1, 1), (0.7, 0.7), (0.4, 0.4), (0.1, 0.1)])

    # Figure and camp
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    norm = colors.Normalize(vmin=0.0, vmax=1.0)

    # Create plot
    for v, i in sorted(vis_and_illum):
        sqrt_vi = float(np.sqrt(v * i))
        color = cmap(norm(sqrt_vi))
        axs[0].plot(distances, np.rad2deg(camera.get_camera_std(distances, v, i)[0]), color=color, label=f'(v, i)={v, i}')
        axs[1].plot(distances, camera.get_camera_std(distances, v, i)[1], color=color, label=f'(v, i)={v, i}')
        
    axs[0].set_ylabel("Standard deviation $\\sigma_\\phi$ [°]")
    axs[0].set_xlabel("Distance to target [m]")
    axs[0].set_title(f"$\\phi$")
    axs[0].grid()
    axs[1].set_xlabel("Distance to target [m]")
    axs[1].set_ylabel("Standard deviation $\\sigma_d$ [m]")
    axs[1].set_title(f"$d$")
    axs[1].grid()
    fig.suptitle("Standard deviation for measurements of relative angle $\\phi$ and distance (depth) $d$ as a function of visibility (v) and illumination (i)")
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=axs, location="right", pad=0.02, label="sqrt(visibility*illumination)")
    plt.legend()
    plt.show()
    
