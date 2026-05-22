from traj_tracking_env import TrajTrackingEnv

import gymnasium as gym
from stable_baselines3.ppo import PPO
from stable_baselines3.sac import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv # Explicit import for clarity

from datetime import datetime
import os, pathlib

root_dir = pathlib.Path(__file__).parent.parent.parent # rl-afd directory
today_and_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
name_prefix = "aurora"
alg = "sac"

dt = 0.2
N_WPTS = 2
WPTS_SPACE_MULTIPLICATOR = 7

def make_env():
    env = TrajTrackingEnv(
        dt,
        n_wpts=N_WPTS,
        wpts_space_multiplicator=WPTS_SPACE_MULTIPLICATOR
    )
    return gym.wrappers.FlattenObservation(env) # Needed for Dict observation space


if __name__ == '__main__':
    n_envs = 4
    vec_env = make_vec_env(make_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)


    # save NN weights at a given frequency
    checkpoints_path = os.path.join(root_dir, 'checkpoints', alg, today_and_now)
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000 // n_envs,
        save_path=checkpoints_path,
        name_prefix=name_prefix
    )

    # Train NN using Proximal Policy Optimization (PPO) or Soft Actor-Critic (SAC)
    tensorboard_path = os.path.join(root_dir, 'tensorboard_logs') # You can check the learning curve by opening a new terminal and typing tensorboard --logdir=tensorboard_logs
    match alg:
        case "sac":
            model = SAC( # Or SAC # PPO
                "MlpPolicy",
                vec_env,
                verbose=1,
                tensorboard_log=tensorboard_path
            )
        case "ppo":
            model = PPO( # Or SAC # PPO
                "MlpPolicy",
                vec_env,
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
    models_path = os.path.join(root_dir, 'models', alg, today_and_now, name_prefix)
    model.save(models_path)
    TrajTrackingEnv(dt, n_wpts=N_WPTS, wpts_space_multiplicator=WPTS_SPACE_MULTIPLICATOR).export_observation_space_ranges_to(os.path.join(root_dir, 'models', alg, today_and_now, "observation_space_ranges.json"))
