from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional
import numpy as np, yaml, os, json
from pathlib import Path

DEFAULT_ODM_YAML = 'odm_training.yaml'

@dataclass  
class ODM:
    ferry: Dict = field(default_factory=dict)
    wind: Dict = field(default_factory=dict)
    current: Dict = field(default_factory=dict)
    sensors: Dict = field(default_factory=dict)
    src: Optional[str] = field(default=None, init=True, repr=False)

    def __post_init__(self):
        """Load ODM parameters after initialization"""
        if self.src is None:
            self.src = str(Path(__file__).parent / DEFAULT_ODM_YAML)
        
        if Path(self.src).exists():
            # Load from specified path
            match self.src.split('.')[-1]:
                case "json":
                    self.from_json(self.src)
                case "yaml":
                    self.from_yaml(self.src)
                case _:
                    raise ValueError(f"src must have the .yaml or .json extension, got {self.src}")

    def from_yaml(self, src: str) -> None:
        """Load ODM parameters from YAML file"""
        if not os.path.exists(src):
            raise FileNotFoundError(f"ODM YAML file not found: {src}")
        
        with open(src, 'r') as f:
            data = yaml.safe_load(f)["scenario_generation"]
            
        
        # Map JSON keys to dataclass attributes and convert to numpy arrays
        self.ferry = self._convert_from_numpy(data["operational_domain"].get('ferry', {})) # type:ignore
        self.wind = self._convert_to_numpy(data["operational_domain"].get('wind', {})) # type:ignore
        self.current = self._convert_to_numpy(data["operational_domain"].get('current', {})) # type:ignore
        self.sensors = self._convert_to_numpy(data.get('sensors', {})) # type:ignore

    def from_json(self, src: str) -> None:
        """Load ODM parameters from JSON file"""
        if not os.path.exists(src):
            raise FileNotFoundError(f"ODM JSON file not found: {src}")
        
        with open(src, 'r') as f:
            data = json.load(f)
        
        # Map JSON keys to dataclass attributes and convert to numpy arrays
        self.ferry = self._convert_from_numpy(data.get('ferry', {})) # type:ignore
        self.wind = self._convert_to_numpy(data.get('wind', {})) # type:ignore
        self.current = self._convert_to_numpy(data.get('current', {})) # type:ignore
        self.sensors = self._convert_to_numpy(data.get('sensors', {})) # type:ignore

    def _convert_to_numpy(self, obj):
        """Recursively convert all values in nested dict/list structure to numpy arrays"""
        if isinstance(obj, dict):
            return {key: self._convert_to_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            try:
                return np.array(obj, dtype=np.float64)
            except (ValueError, TypeError):
                return np.array(obj)
        elif isinstance(obj, (int, float)):
            return np.array(obj)
        else:
            return obj

    def to_json(self, src: str) -> None:
        """Save ODM parameters to JSON file"""
        # Create data dictionary with proper JSON structure
        data = {}
        
        if self.ferry:
            data['ferry'] = self._convert_from_numpy(self.ferry)
        if self.wind:
            data['wind'] = self._convert_from_numpy(self.wind)
        if self.current:
            data['current'] = self._convert_from_numpy(self.current)
        if self.sensors:
            data['sensors'] = self._convert_from_numpy(self.sensors)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(src)), exist_ok=True)
        
        # Write JSON file with proper formatting
        with open(src, 'w') as f:
            json.dump(data, f, indent=4)

    def _convert_from_numpy(self, obj):
        """Recursively convert numpy arrays back to Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self._convert_from_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, np.ndarray):
            # Convert numpy arrays back to lists or scalars
            if obj.ndim == 0:  # scalar
                return obj.item()
            else:  # array
                return obj.tolist()
        else:
            return obj

    @property
    def sensor_noise_covariance(self) -> Optional[Dict]:
        """Access sensor noise covariance from sensors dict"""
        return self.sensors["states"].get('noise_covariance') if self.sensors else None
    
    @property 
    def camera_params(self) -> Optional[Dict]:
        """Access camera parameters from sensors dict"""
        return self.sensors.get('camera') if self.sensors else None

if __name__ == "__main__":
    # Example 1: Load default ODM configuration (from odm_training.json)
    print("Loading default ODM configuration...")
    odm_default = ODM()
    print(f"Wind parameters: {odm_default.wind}")
    print(f"  - Angle range: {odm_default.wind['angle'] if odm_default.wind else None}")
    print(f"    - Min (numpy): {odm_default.wind['angle']['min']} (type: {type(odm_default.wind['angle']['min'])})")
    print(f"    - Max (numpy): {odm_default.wind['angle']['max']} (type: {type(odm_default.wind['angle']['max'])})")
    print(f"  - Speed range: {odm_default.wind['speed'] if odm_default.wind else None}")
    print(f"Current parameters: {odm_default.current}")
    print(f"Sensors: {odm_default.sensors}")
    print(f"  - Sensor noise covariance: {odm_default.sensor_noise_covariance}")
    if odm_default.sensor_noise_covariance:
        print(f"    - Min array shape: {odm_default.sensor_noise_covariance['min'].shape}")
        print(f"    - Min array type: {type(odm_default.sensor_noise_covariance['min'])}")
        print(f"    - Max array shape: {odm_default.sensor_noise_covariance['max'].shape}")
    print(f"  - Camera parameters: {odm_default.camera_params}")
    print()
    