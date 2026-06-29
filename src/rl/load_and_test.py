import gymnasium as gym
from stable_baselines3.ppo import PPO
from stable_baselines3.sac import SAC
import os, time

from src.ferry.aurora import AuroraFerry
from src.rl.traj_tracking_env import TrajTrackingEnv

alg = "sac"
date_and_time = "2026_06_24_15_47_22" # "hpc"
weights_name = "aurora_sac_nenvs16_arch_256x256_64x32_1500000_steps.zip"
# Z:\dev\aurora_ferry\models\sac\2026_05_27_20_35_02


dt = 0.2
env = TrajTrackingEnv(
    dt,
    render_mode="human",
    n_wpts=2,
    action_repeat=10,
    # dist_between_target_wpts=70.0
    wpts_space_multiplicator=7,
    # path_to_obs_ranges=os.path.join("models", alg, '2026_04_30_22_20_25', 'observation_space_ranges.json')
)

# env.max_steps = 200

env = gym.wrappers.FlattenObservation(env)

# Load trained PPO model
# model = SAC.load(os.path.join("models", alg, date_and_time, 'aurora.zip'))
# model = SAC.load(os.path.join("checkpoints", alg, date_and_time, weights_name))
# model = PPO.load(os.path.join("models", alg, date_and_time, weights_name))
# model = SAC.load(os.path.join("Z:\\dev", "aurora_ferry", "models", alg, date_and_time, weights_name))
# model = SAC.load(os.path.join("Z:\\dev", "aurora_ferry", "models", alg, date_and_time, 'aurora_sac_nenvs32.zip'))
model = SAC.load(os.path.join("Z:\\dev", "aurora_ferry", "checkpoints", alg, date_and_time, weights_name))

# Observation space is 19-D
no = model.observation_space.shape
print("Observation space: ", no)
# Action space is 16-D
na = model.action_space.shape
print("Action space: ", na)
# NN architecture
net_arch = model.policy.net_arch
print("Architecture: ", net_arch) 

assert isinstance(net_arch, dict), f"net_arch must be a dict"
assert no is not None and na is not None, f"no or na is None"

### Verify number of NN parameters
n_pi, n_qf = 0, 0

# Number of parameters in policy
layers = [no[0]] + net_arch["pi"] + [2*na[0]]
for i in range(1, len(layers)):
    n_pi += layers[i-1] * layers[i] # weights
n_pi += sum(layers[1::]) # bias

# Number of parameters in single critic
layers = [no[0] + na[0]] + net_arch["qf"] + [1]
for i in range(1, len(layers)):
    n_qf += layers[i-1] * layers[i] # weights
n_qf += sum(layers[1::]) # bias

print(f"Expected number of parameters: {n_pi+4*n_qf:,} (n_pi={n_pi:,})")

total_params = sum(p.numel() for p in model.policy.parameters())
print(f"Actual number of weights/parameters: {total_params:,}")

# Run episode using trained model
obs, info = env.reset()

for _ in range(1000):
    action, states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render() 
    # time.sleep(dt)
    if terminated or truncated:
        obs, info = env.reset()