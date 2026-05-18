import os, yaml, pyproj, pandas as pd, numpy as np
from math import isnan
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from matplotlib.axes import Axes
from python_vehicle_simulator.utils.math_fn import Rzyx, ssa
from python_vehicle_simulator.vehicles.vessel import VESSEL_GEOMETRY
from python_vehicle_simulator.lib.sensor import ISensor


DEFAULT_AIS_DATA = '2023_04_06.csv' # '2023_04_15.csv'
DEFAULT_MAPPING_FILE = 'config.yaml'

@dataclass
class Vessel:
    """Simple vessel representation with essential data for collision avoidance."""
    mmsi: int
    lat: float
    lon: float
    north: float  
    east: float   
    sog: float    # Speed over ground (knots)
    cog: float    # Course over ground (degrees)
    heading: float  # True heading (degrees)
    timestamp: datetime
    name: str = "Unknown"
    vessel_type: str = "Unknown"
    nav_status: str = "Unknown"
    length: float = 22.0
    width: float = 10.0

    def __post_init__(self):
        """Validate and normalize vessel data."""
        self.sog = max(0.0, self.sog)
        self.cog = np.rad2deg(ssa(np.deg2rad(self.cog)))
        self.heading = np.rad2deg(ssa(np.deg2rad(self.heading)))
        self.update_geometry()

    def update_geometry(self) -> None:
        self.geometry = None if self.heading is None else Rzyx(0, 0, np.deg2rad(self.heading)) @ VESSEL_GEOMETRY(self.length, self.width) + np.array([self.north, self.east, 0]).reshape(3, 1)


class AIS(ISensor):
    """Simplified AIS data handler with YAML-based column mapping."""
    
    def __init__(
        self, 
        csv_path: str,
        mapping_file: Optional[str] = None,
        t0: Optional[datetime] = None,
        tf: Optional[datetime] = None,
        mmsi_to_exclude: List[int] = [],
        update_every_sec: int = 1,
    ):
        """
        Initialize AIS data from CSV with flexible column mapping.
        
        Args:
            csv_path: Path to the AIS CSV file
            mapping_file: Path to YAML mapping file (optional)
        """
        self.t0 = t0
        self.tf = tf
        self.mmsi_to_exclude = mmsi_to_exclude
        self.update_every_sec = update_every_sec
        # Load mapping configuration
        if mapping_file is None:
            mapping_file = os.path.join(os.path.dirname(__file__), DEFAULT_MAPPING_FILE)
        
        with open(mapping_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load and process CSV data
        self.df = pd.read_csv(csv_path)
        self.column_mapping = self._create_column_mapping()
        
        # Filter out excluded MMSIs
        self._apply_mmsi_filter()
        
        # Apply temporal filtering if t0 and/or tf are provided
        self._apply_temporal_filter()
        
        # Setup coordinate transformer if needed
        utm_zone = self.config.get('coordinates', {}).get('utm_zone', 33)
        self.transformer = pyproj.Transformer.from_proj(
            '+proj=longlat +datum=WGS84',
            f'+proj=utm +zone={utm_zone} +datum=WGS84 +units=m +no_defs'
        )
        
        # Validate required columns exist
        self._validate_data()
        
    def __get__(self, target_time: datetime, *args, time_tolerance: int = 30, **kwargs) -> Tuple[Any, Dict]:
        """
        Noiseless measurement
        """
        vessels = self.get_vessels_at_time(target_time, time_tolerance=time_tolerance)
        return vessels, {}

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
            print(f"Excluded {original_count - filtered_count} records with MMSIs: {self.mmsi_to_exclude}")
    
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
    
    def _apply_temporal_filter(self):
        """Apply temporal filtering based on t0 and tf parameters."""
        if self.t0 is None and self.tf is None:
            return  # No filtering needed
            
        # Get the timestamp column
        timestamp_col = self.column_mapping['timestamp']
        
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
        
        if len(self.df) == 0:
            raise ValueError(f"No data found in the specified time range: {self.t0} to {self.tf}")
    
    def get_vessels_at_time(
        self, 
        target_time: datetime, 
        time_tolerance: int = 30
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
                # AIS data is timezone-aware, target is naive - convert target to UTC
                target_pd = target_pd.tz_localize('UTC')
            elif vessel_timestamps.dt.tz is None and target_pd.tz is not None:
                # AIS data is naive, target is timezone-aware - remove target timezone
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
    
    def get_all_vessels(self) -> List[Vessel]:
        """Get all vessels from all timestamps."""
        vessels = []
        timestamp_col = self.column_mapping['timestamp']
        
        for _, row in self.df.iterrows():
            timestamp = pd.to_datetime(row[timestamp_col])
            vessel = self._create_vessel_from_row(row, timestamp)
            vessels.append(vessel)
            
        return vessels
    
    def get_time_range(self) -> tuple[datetime, datetime]:
        """Get the time range of available data."""
        timestamp_col = self.column_mapping['timestamp']
        timestamps = pd.to_datetime(self.df[timestamp_col])
        return timestamps.min().to_pydatetime(), timestamps.max().to_pydatetime()
    
    def get_vessel_count(self) -> int:
        """Get total number of unique vessels."""
        mmsi_col = self.column_mapping['mmsi']
        return len(self.df[mmsi_col].unique())
    
    def get_nearby_vessels(
        self,
        north: float,
        east: float,
        radius_meters: float,
        target_time: datetime,
        time_tolerance: int = 30
    ) -> List[Vessel]:
        """
        Get all vessels within a radius of specified coordinates at a specific time.
        
        Args:
            north: North UTM coordinate (meters)
            east: East UTM coordinate (meters)  
            radius_meters: Search radius in meters
            target_time: Target datetime
            time_tolerance: Maximum age tolerance for vessel data
            
        Returns:
            List of vessels within the specified radius
        """
        # Get all vessels at the target time
        all_vessels = self.get_vessels_at_time(target_time, time_tolerance)
        
        nearby_vessels = []
        
        for vessel in all_vessels:
            # Calculate Euclidean distance in UTM coordinates
            distance = np.sqrt((vessel.north - north)**2 + (vessel.east - east)**2)
            
            if distance <= radius_meters:
                nearby_vessels.append(vessel)
        
        return nearby_vessels
    
    def __plot__(self, target_time: datetime, ax:Axes, *args, north: Optional[float] = None, east: Optional[float] = None, radius_meters: Optional[float] = None, time_tolerance: int = 30, verbose: int = 0, **kwargs) -> Axes:
        if radius_meters is None or north is None or east is None:  
            vessels = self.get_vessels_at_time(target_time=target_time, time_tolerance=time_tolerance)
        else:
             vessels = self.get_nearby_vessels(north, east, radius_meters, target_time, time_tolerance)

        for vessel in vessels:
            if vessel.geometry is not None:
                ax.plot(vessel.geometry[1, :], vessel.geometry[0, :], label=vessel.name)
            else:
                ax.scatter(vessel.east, vessel.north, s=50, c='red')
                ax.text(vessel.east, vessel.north, f"Invalid geometry, heading={vessel.heading}")
        ax.set_aspect('equal')
        return ax
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Load default AIS data
    csv_path = os.path.join(os.path.dirname(__file__), '../../data/raw/', DEFAULT_AIS_DATA)
    ais = AIS(csv_path)
    all_vessels = ais.get_all_vessels()
    
    # Group by MMSI and plot trajectories
    fig, ax = plt.subplots(figsize=(14, 10))
    mmsi_groups = {}
    for vessel in all_vessels:
        if vessel.mmsi not in mmsi_groups:
            mmsi_groups[vessel.mmsi] = []
        mmsi_groups[vessel.mmsi].append(vessel)
    
    for mmsi, vessels in mmsi_groups.items():
        north_vals = [v.north for v in sorted(vessels, key=lambda v: v.timestamp)]
        east_vals = [v.east for v in sorted(vessels, key=lambda v: v.timestamp)]
        ax.plot(east_vals, north_vals, marker='.', alpha=0.7, label=f"MMSI {mmsi}")
    
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title("AIS Trajectories")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()
