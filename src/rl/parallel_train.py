from traj_tracking_env import TrajTrackingEnv

import gymnasium as gym
from stable_baselines3.ppo import PPO
from stable_baselines3.sac import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv # Explicit import for clarity
import torch as th

from datetime import datetime
import os, pathlib
import json

root_dir = pathlib.Path(__file__).parent.parent.parent # rl-afd directory
today_and_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
name_prefix = os.getenv("RUN_PREFIX", "aurora")
alg = os.getenv("ALG", "sac").lower()
n_envs = int(os.getenv("N_ENVS", "16"))
total_timesteps = int(os.getenv("TOTAL_TIMESTEPS", "5000000"))
requested_device = os.getenv("RL_DEVICE", "cuda")
net_arch_str = os.getenv("NET_ARCH", '{"qf": [256, 256], "pi": [256, 256]}')
try:
    net_arch = json.loads(net_arch_str)
except json.JSONDecodeError:
    net_arch = {"qf": [256, 256], "pi": [256, 256]}
    print(f"Warning: Could not parse NET_ARCH={net_arch_str}, using default")

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
    run_name = f"{name_prefix}_{alg}_nenvs{n_envs}_arch{net_arch['qf'][0] if net_arch['qf'] else 'none'}_{net_arch['pi'][0] if net_arch['pi'] else 'none'}"
    if requested_device == "cuda" and not th.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"
    else:
        device = requested_device

    print(f"Training device: {device}")
    vec_env = make_vec_env(make_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)


    # save NN weights at a given frequency
    checkpoints_path = os.path.join(root_dir, 'checkpoints', alg, today_and_now)
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000 // n_envs,
        save_path=checkpoints_path,
        name_prefix=run_name
    )

    # Train NN using Proximal Policy Optimization (PPO) or Soft Actor-Critic (SAC)
    tensorboard_path = os.path.join(root_dir, 'tensorboard_logs') # You can check the learning curve by opening a new terminal and typing tensorboard --logdir=tensorboard_logs
    match alg:
        case "sac":
            model = SAC( # Or SAC # PPO
                "MlpPolicy",
                vec_env,
                verbose=1,
                tensorboard_log=tensorboard_path,
                gradient_steps=n_envs // 4, # baseline was 1 for 8 envs
                device=device,
                policy_kwargs={'net_arch': net_arch} # Architecture from NET_ARCH env var
            )
        case "ppo":
            model = PPO( # Or SAC # PPO
                "MlpPolicy",
                vec_env,
                verbose=1,
                tensorboard_log=tensorboard_path,
                device=device,
            )
        case _:
            raise ValueError(f"Selected algorithm invalid")
        
    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name=run_name,
        callback=checkpoint_callback,
    )

    # Save NN weights
    models_path = os.path.join(root_dir, 'models', alg, today_and_now, run_name)
    model.save(models_path)
    TrajTrackingEnv(dt, n_wpts=N_WPTS, wpts_space_multiplicator=WPTS_SPACE_MULTIPLICATOR).export_observation_space_ranges_to(os.path.join(root_dir, 'models', alg, today_and_now, "observation_space_ranges.json"))
