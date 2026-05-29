import gymnasium as gym
from stable_baselines3.ppo import PPO
from stable_baselines3.sac import SAC
import os, time

from src.ferry.aurora import AuroraFerry
from src.rl.traj_tracking_env import TrajTrackingEnv

alg = "sac"
date_and_time = "2026_05_27_20_35_02" # "hpc"
weights_name = "aurora_4500000_steps.zip"
# Z:\dev\aurora_ferry\models\sac\2026_05_27_20_35_02


dt = 0.2
env = TrajTrackingEnv(
    dt,
    render_mode="human",
    n_wpts=2,
    action_repeat=10,
    wpts_space_multiplicator=7,
    # path_to_obs_ranges=os.path.join("models", alg, '2026_04_30_22_20_25', 'observation_space_ranges.json')
)

env.max_steps = 100

env = gym.wrappers.FlattenObservation(env)

# Load trained PPO model
# model = SAC.load(os.path.join("models", alg, date_and_time, 'aurora.zip'))
# model = SAC.load(os.path.join("checkpoints", alg, date_and_time, weights_name))
# model = PPO.load(os.path.join("models", alg, date_and_time, weights_name))
# model = SAC.load(os.path.join("Z:\\dev", "aurora_ferry", "models", alg, date_and_time, 'aurora.zip'))
model = SAC.load(os.path.join("Z:\\dev", "aurora_ferry", "checkpoints", alg, date_and_time, weights_name))



# Run episode using trained model
obs, info = env.reset()

for _ in range(1000):
    action, states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render() 
    # time.sleep(dt)
    if terminated or truncated:
        obs, info = env.reset()