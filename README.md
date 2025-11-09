# Aurora Ferry - Maritime Navigation and Simulation System

A comprehensive maritime simulation and navigation system for autonomous ferry operations, featuring AIS data processing, real-time vessel tracking, and advanced navigation algorithms.

## Features

- **AIS Data Processing**: Parse and visualize Automatic Identification System data
- **Interactive Mapping**: Real-time vessel visualization with Helsingborg ferry routes
- **Target Tracking**: Kalman filter-based vessel tracking and prediction  
- **Navigation System**: Advanced navigation with obstacle avoidance
- **Ferry Simulation**: Complete Aurora ferry dynamics simulation

## Installation

### Method 1: Install from PyPI (Recommended)
```bash
pip install aurora-ferry
```

### Method 2: Development Installation

1. **Clone with submodules:**
```bash
git clone --recurse-submodules https://github.com/stiefen1/aurora_ferry.git
cd aurora_ferry
```

2. **Create conda environment:**
```bash
conda env create -f env.yml
conda activate aurora-ferry
```

3. **Install the package in development mode:**
```bash
pip install -e .
```

4. **Install the PythonVehicleSimulator submodule:**
```bash
pip install -e ./submodules/PythonVehicleSimulator/
```

## Quick Start

```python
from aurora_ferry import AIS, HelsingborgMap, NavigationAurora

# Load AIS data and map
ais = AIS('data/AIS.csv')
map_env = HelsingborgMap()

# Create navigation system
nav = NavigationAurora(
    eta=np.array([0, 0, 0, 0, 0, 0]),
    nu=np.array([0, 0, 0, 0, 0, 0]),
    dt=0.1
)
```

## Usage Examples

### Interactive AIS Visualization
```bash
python test/interactive_map_with_ais.py
```

### Run Ferry Control Test
```bash
python test/revolt_control.py
```

## Testing

Run the revolt_control.py to verify installation was successful:
```bash
python test/revolt_control.py
```

A new gif should be generated in the `videos/` folder.

## Development

### Optional Dependencies

Install development dependencies:
```bash
pip install -e ".[dev]"
```

Install Jupyter dependencies for notebooks:
```bash
pip install -e ".[jupyter]"
```

### Running Tests
```bash
pytest
```

### Code Formatting
```bash
black src/ test/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Stephen Monnet** - [stiefen1](https://github.com/stiefen1)
- Email: stephen.monnet@outlook.com

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request