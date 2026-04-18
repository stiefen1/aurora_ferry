import gymnasium as gym
from stable_baselines3.ppo import PPO
import os, time

from src.aurora import AuroraFerry
from src.rl.traj_tracking_env import TrajTrackingEnv

alg = "ppo"
data_and_time = "2026_04_17_13_08_57" # "hpc"
weights_name = "aurora_200000_steps"


dt = 0.2
env = TrajTrackingEnv(
    dt,
    render_mode="human",
    n_wpts=2,
)

# env.max_steps = 100

env = gym.wrappers.FlattenObservation(env)

# Load trained PPO model
model = PPO.load(os.path.join("checkpoints", alg, data_and_time, weights_name))
# model = PPO.load(os.path.join("models", alg, data_and_time, weights_name))

# Run episode using trained model
obs, info = env.reset()
for _ in range(1000):
    action, states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render() 
    time.sleep(dt)
    if terminated or truncated:
        obs, info = env.reset()