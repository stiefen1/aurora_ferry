from env import GymNavEnv
from traj_tracking_env import TrajTrackingEnv
from src.navigation import NavigationAurora

import gymnasium as gym
from stable_baselines3.ppo import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv # Explicit import for clarity

from datetime import datetime
import os, pathlib
from src.aurora import AuroraFerry

root_dir = pathlib.Path(__file__).parent.parent.parent # rl-afd directory
today_and_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
name_prefix = "aurora"

dt = 0.2

def make_env():
    revolt = AuroraFerry(
        dt=dt,
    )
    navigation=NavigationAurora(
        revolt.states,
        dt=dt
    )

    revolt.navigation = navigation
    env = TrajTrackingEnv(
        own_vessel=revolt,
        n_wpts=3,
        wpts_space_multiplicator=20
    )
    return gym.wrappers.FlattenObservation(env) # Needed for Dict observation space


if __name__ == '__main__':
    n_envs = 8
    vec_env = make_vec_env(make_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)


    # save NN weights at a given frequency
    checkpoints_path = os.path.join(root_dir, 'checkpoints', 'ppo', today_and_now)
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000 // n_envs,
        save_path=checkpoints_path,
        name_prefix=name_prefix
    )

    # Train NN using Proximal Policy Optimization (PPO)
    tensorboard_path = os.path.join(root_dir, 'tensorboard_logs') # You can check the learning curve by opening a new terminal and typing tensorboard --logdir=tensorboard_logs
    model = PPO( # Or SAC 
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=tensorboard_path
    )
    model.learn(
        total_timesteps=3_000_000,
        tb_log_name=name_prefix,
        callback=checkpoint_callback
    )

    # Save NN weights
    models_path = os.path.join(root_dir, 'models', 'ppo', today_and_now, "_" + name_prefix)
    model.save(models_path)