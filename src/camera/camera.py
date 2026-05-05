from python_vehicle_simulator.lib.noise import INoise
from python_vehicle_simulator.lib.sensor import ISensor
import numpy as np, os, pandas as pd, numpy.typing as npt, yaml, pyproj
from math import isnan
from typing import Tuple, Dict, List, Optional, Any
from datetime import datetime
from src.ais.ais import Vessel
from src.camera.weather import is_target_detected


"""
Uncertainties will be parameterized by:
- weather type
- distance of the target vessel w.r.t ferry

First detection should be parameterized by distance, weather and target ship size

We probably assume the detector does not provide a good distance measure of target ship, or even no distance at all -> (AIS data are needed)

"""

DEFAULT_CAMERA_MAPPING_FILE = 'config.yaml'


class Camera(ISensor):
    def __init__(
            self,
            csv_path: str,
            mapping_file: Optional[str] = None,
            t0: Optional[datetime] = None,
            tf: Optional[datetime] = None,
            mmsi_to_exclude: List[int] = [],
            update_every_sec: int = 1,
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
        detected_vessels: List[Vessel] = []
        all_vessels_in_camera = self.get_vessels_at_time(target_time, time_tolerance=time_tolerance)
        
        for vessel in all_vessels_in_camera:
            # Check if target is detected based on position, size, and weather conditions
            detected, info = is_target_detected(
                states[0:2], 
                np.array([vessel.north, vessel.east, np.deg2rad(vessel.heading)]), 
                vessel.length, 
                vessel.width, 
                visibility, 
                illumination
            )
            if detected:
                detected_vessels.append(vessel)
                
        
        info = {'all_vessels_in_camera': all_vessels_in_camera}
        return detected_vessels, info
    
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
    
