from traj_tracking_env import TrajTrackingEnv

import gymnasium as gym
from stable_baselines3.ppo import PPO
from stable_baselines3.sac import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

from datetime import datetime
import os, pathlib, sys
from src.ferry.aurora import AuroraFerry

# Add src directory to Python path to enable local imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

root_dir = pathlib.Path(__file__).parent.parent.parent # rl-afd directory
today_and_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
name_prefix = "aurora"
alg = "sac"


dt = 0.5
env = TrajTrackingEnv(
    dt,
    n_wpts=2,
    action_repeat=10,
    wpts_space_multiplicator=25
)

env = gym.wrappers.FlattenObservation(env) # Needed for Dict observation space

# save NN weights at a given frequency
checkpoints_path = os.path.join(root_dir, 'checkpoints', alg, today_and_now)
checkpoint_callback = CheckpointCallback(
    save_freq=100_000,
    save_path=checkpoints_path,
    name_prefix=name_prefix
)

# Train NN using Proximal Policy Optimization (PPO)
tensorboard_path = os.path.join(root_dir, 'tensorboard_logs') # You can check the learning curve by opening a new terminal and typing tensorboard --logdir=tensorboard_logs
match alg:
    case "sac":
        model = SAC( # Or SAC # PPO
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=tensorboard_path
        )
    case "ppo":
        model = PPO( # Or SAC # PPO
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=tensorboard_path
        )
    case _:
        raise ValueError(f"Selected algorithm invalid")


model.learn(
    total_timesteps=3_000_000,
    tb_log_name=name_prefix,
    callback=checkpoint_callback
)

# Save NN weights
models_path = os.path.join(root_dir, 'models', alg, today_and_now, "_" + name_prefix)
model.save(models_path)
