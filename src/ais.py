import os
import pandas as pd
import pyproj
from datetime import datetime, timedelta
from typing import List, LiteralString, Optional, Tuple, Dict
from python_vehicle_simulator.visualizer.drawable import IDrawable
from python_vehicle_simulator.lib.sensor import ISensor
from python_vehicle_simulator.vehicles.vessel import VESSEL_GEOMETRY
from python_vehicle_simulator.utils.math_fn import Rzyx
from matplotlib.axes import Axes
from dataclasses import dataclass
import numpy as np

DEFAULT_LOA = 22
DEFAULT_BEAM = 10

@dataclass
class Vessel:
    """Data class representing a vessel with essential collision avoidance information."""
    mmsi: int
    name: str
    lat: float
    lon: float
    north: float  # Local coordinate system (or utm_north in the UTM version)
    east: float   # Local coordinate system (or utm_east in the UTM version)
    sog: float | None    # Speed over ground (knots)
    cog: float | None    # Course over ground (degrees)
    heading: float | None  # True heading (degrees)
    timestamp: datetime
    vessel_type: str
    nav_status: str
    length: float = DEFAULT_LOA
    width: float = DEFAULT_BEAM
    distance_to_own_vessel: float = 0.0
    
    
    def __post_init__(self):
        """Validate and normalize vessel data."""
        self.sog = max(0.0, self.sog) if not pd.isna(self.sog) else None
        self.cog = self.cog % 360 if not pd.isna(self.cog) else None
        self.heading = self.heading % 360 if not pd.isna(self.heading) else None
        self.geometry = None if self.heading is None else Rzyx(0, 0, self.heading*np.pi/180) @ VESSEL_GEOMETRY(self.length, self.width) + np.array([self.north, self.east, 0]).reshape(3, 1)

class AIS(IDrawable, ISensor, pd.DataFrame):
    def __init__(
            self,
            src: str = os.path.join('data', 'AIS.csv'),
            path_to_folder: Optional[str] = None,
            filename: Optional[str] = None,
            verbose_level: int = 0
    ):
        if path_to_folder is not None and filename is not None:
            src = os.path.join(path_to_folder, filename)
            
        pd.DataFrame.__init__(self, pd.read_csv(src))
        IDrawable.__init__(self, verbose_level)
        ISensor.__init__(self, noise=None)
        self.transformer = pyproj.Transformer.from_proj(
            '+proj=longlat +datum=WGS84',
            '+proj=utm +zone=33 +datum=WGS84 +units=m +no_defs'
        )
        self._convert_latlon_to_utm()

    def __get__(self, timestamp: Optional[datetime] = None, timestamp_sec: Optional[int] = None, max_age_seconds: int = 30, *args, **kwargs) -> Tuple[List[Vessel], Dict]:
        return self.get_vessels_at_time(timestamp=timestamp, timestamp_sec=timestamp_sec, max_age_seconds=max_age_seconds, *args, **kwargs), {}

    def __plot__(self, ax:Axes, *args, timestamp: Optional[datetime] = None, timestamp_sec: Optional[int] = None, distance: Optional[float] = None, max_age_seconds: int = 30, verbose: int = 0, **kwargs) -> Axes:
        if distance is None:  
            vessels = self.get_vessels_at_time(timestamp=timestamp, timestamp_sec=timestamp_sec, max_age_seconds=max_age_seconds)
        else:
             vessels = self.get_nearby_vessels(distance, timestamp=timestamp, timestamps_sec=timestamp_sec, max_age_seconds=max_age_seconds)

        for vessel in vessels:
            if vessel.geometry is not None:
                ax.plot(vessel.geometry[1, :], vessel.geometry[0, :], label=vessel.name)
            else:
                ax.scatter(vessel.east, vessel.north, s=50, c='red')
                ax.text(vessel.east, vessel.north, f"Invalid geometry, heading={vessel.heading}")
        ax.set_aspect('equal')
        return ax

    def _convert_latlon_to_utm(self) -> None:
        north, east = [], []
        for lat, lon in zip(self['lat'], self['lon']):
            e, n = self.transformer.transform(lon, lat)
            north.append(n)
            east.append(e)
        self.insert(0, "east", east)
        self.insert(0, "north", north)

    def get_vessels_at_time(self, timestamp: Optional[datetime] = None, timestamp_sec: Optional[int] = None, max_age_seconds: int = 30) -> List[Vessel]:
        """Get all vessels at a specific timestamp with a time tolerance."""
        
        # Convert timestamp to pandas datetime if needed
        if timestamp is not None:
            target_time = pd.to_datetime(timestamp)
        elif timestamp_sec is not None:
            target_time = pd.to_datetime(timestamp_sec, unit='s')
        else:
            raise TypeError(f"Either timestamp or timestamp_sec must be != None")
        
        # Define time window
        time_tolerance = timedelta(seconds=max_age_seconds)
        start_time = target_time - time_tolerance
        end_time = target_time + time_tolerance
        vessels = []
        
        for mmsi in self.get_all_mmsi():
            vessel_data = self.get_single_vessel_data(mmsi)
            
            # Get timestamps for this vessel
            if 'timestamp' in vessel_data.columns:
                vessel_timestamps = pd.to_datetime(vessel_data['timestamp'])
            else:
                vessel_timestamps = pd.to_datetime(vessel_data['timestamp_sec'], unit='s')
            
            # Find records within time window
            time_mask = (vessel_timestamps >= start_time) & (vessel_timestamps <= end_time)
            valid_records = vessel_data[time_mask]
            
            if len(valid_records) > 0:
                # Get the record closest to target time
                closest_idx = (vessel_timestamps[time_mask] - target_time).abs().idxmin()
                row = valid_records.loc[closest_idx]
                
                # Create Vessel object
                vessel = Vessel(
                    mmsi=int(row['mmsi']),
                    name=str(row['name']),
                    lat=float(row['lat']),
                    lon=float(row['lon']),
                    north=float(row['north']),
                    east=float(row['east']),
                    sog=float(row['sog']),
                    cog=float(row['cog']),
                    heading=float(row['heading']),
                    timestamp=vessel_timestamps.loc[closest_idx],
                    vessel_type=str(row['ship_type']),
                    length=float(row['length']),
                    width=float(row['width']),
                    nav_status=str(row['nav_status']),
                    distance_to_own_vessel=float(row['distance_m'])
                )
                vessels.append(vessel)
        
        return vessels
    
    def get_nearby_vessels(self, distance: float, timestamp: Optional[datetime] = None, timestamps_sec: Optional[int] = None, max_age_seconds: int = 30) -> List[Vessel]:
        vessels = []
        for vessel in self.get_vessels_at_time(timestamp=timestamp, timestamp_sec=timestamps_sec, max_age_seconds=max_age_seconds):
            if vessel.distance_to_own_vessel <= distance:
                vessels.append(vessel)
        return vessels

    def get_single_vessel_data(self, mmsi:int) -> pd.DataFrame:
        return self[self['mmsi']==mmsi]
    
    def get_all_mmsi(self) -> pd.DataFrame:
        return self['mmsi'].drop_duplicates(inplace=False)
    
    @property
    def n_vessels(self) -> int:
        return len(self.get_all_mmsi())
    
    
    def __scatter__(self, ax:Axes, *args, **kwargs) -> Axes:
        return ax

    def __fill__(self, ax:Axes, *args, **kwargs) -> Axes:
        return ax
    

        

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    ais = AIS()
    print("MMSI list:")
    print(ais.get_all_mmsi())
    print(f"\nNumber of vessels: {ais.n_vessels}")
    print(f"\nColumns: {list(ais.columns)}")
    
    # Test get_vessels_at_time function
    print(f"\n=== Testing get_vessels_at_time ===")
    
    # Get a timestamp from the data
    sample_timestamp = pd.to_datetime(ais['timestamp'].iloc[0])
    print(f"Sample timestamp: {sample_timestamp}")
    
    vessels = ais.get_nearby_vessels(300, sample_timestamp, max_age_seconds=60)
    print(f"Found {len(vessels)} vessels at time {sample_timestamp}")
    
    # Show first few vessels
    for i, vessel in enumerate(vessels):
        print(f"  {i+1}. {vessel.name} (MMSI: {vessel.mmsi})")
        print(f"     Position [N-E]: {vessel.north:.4f}, {vessel.east:.4f}")
        print(f"     Distance: {vessel.distance_to_own_vessel:.4f} m")
        print(f"     Speed: {vessel.sog:.1f} kts, Course: {vessel.cog:.0f}°")
        print(f"     Timestamp: {vessel.timestamp}")
        print(f"     Dim: {vessel.length}, {vessel.width}")

    ais.plot(timestamp=sample_timestamp)
    plt.show()
