import pathlib, os, glob
from typing import List
from src.scenarios.odm_generator import ODMGenerator
from src.scenarios.sim_launcher import SimLauncher
from src.scenarios.sim_analyzer import SimAnalyzer
from src.utils.interpolate_ais import load_ais_csv, interpolate_ais_data

FOLDER = os.path.join("sim_data", "test")
CONFIG_FILENAME = "test.yaml"
PATH_TO_CONFIG = os.path.join(FOLDER, CONFIG_FILENAME)

# Generate configuration files
odm_gen = ODMGenerator(PATH_TO_CONFIG)
odm_gen() # Generate configuration files in same directory
sim_config_files = glob.glob(os.path.join(os.path.dirname(PATH_TO_CONFIG), "scenarios", "*.json"))

# Interpolate AIS data
list_of_config_files: List[str] = odm_gen.config["scenario_generation"]["ais_data_paths"]
for f in list_of_config_files:
    path_to_interp = pathlib.Path(f.replace("raw", "smooth_interp"))
    os.makedirs(path_to_interp.parent, exist_ok=True)
    df = load_ais_csv(f)
    df_smooth_interp = interpolate_ais_data(df, dt=1.0, smooth=True, sigma=10.0, exclude_ships=odm_gen.config["scenario_generation"]["mmsi_to_exclude"], remove_stationary=False)
    df_smooth_interp.to_csv(path_to_interp)
    print(f"Processed {os.path.basename(f)}")

# Launch simulations
launcher = SimLauncher()
for i, config_file in enumerate(sim_config_files):
    launcher.run_single_sim(config_file, render=False)

# Analyze results
analyzer = SimAnalyzer(FOLDER)
analyzer()



