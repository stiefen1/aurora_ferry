# Aurora Ferry - Timespace and RL for collision avoidance

## Installation
1. **Clone with submodules:**
```bash
git clone --recurse-submodules https://github.com/stiefen1/aurora_ferry.git
cd aurora_ferry
```

2. **Install the PythonVehicleSimulator submodule:**
```bash
cd ./submodules/PythonVehicleSimulator/
git switch rk4-model-aurora-ferry
pip install -e .
cd ../..
```

3. **Install the Timespace-COLAV submodule:**
```bash
pip install -e ./submodules/timespace-colav/
```

4. **Install the package in development mode:**
```bash
pip install -e .
```

## Quick Start

### Setup configuration file
Go to ```src/scenarios``` and open ```test.yaml```, which is an example configuration file for simulations generation. You can copy this file to a new folder and edit the configuration based on your own requirements. 

### Setup AIS data
Create a folder ```data/raw``` in the root folder and place all the AIS data available inside this folder. Make sure that you specify the path to each .csv file in your configuration file (field scenario_generation->ais_data_paths).

### Setup RL weights
Place the NN weights of the RL agent inside a folder and provide the path to this folder inside your configuration file (field scenario_generation->control->path_to_weights). You must also specify which algorithm ('sac' or 'ppo') the RL agent is based on (field scenario_generation->control->algorithm). 

### Launch simulations
The main.py file (root folder) allows to launch the simulations as described in your configuration file. For this purpose, edit the ```FOLDER``` and ```CONFIG_FILENAME``` to match the path to your configuration file. 

For example, if your configuration file is named "config1.yaml" and placed in /sim/tests, then ```FOLDER="sim/tests"``` AND ```CONFIG_FILENAME=config1.yaml```.

### Results
Figures and a short report will be generated in the same folder as your configuration file. 

### Weather Forecasts
- [DMI](https://www.dmi.dk/)
- [FCOO](https://app.fcoo.dk/ifm-maps/denmark/)

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

