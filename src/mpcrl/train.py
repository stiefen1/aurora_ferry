from src.mpcrl.trajectory_tracker import ParametricTrajectoryTracker
from stable_baselines3 import TD3
from stable_baselines3.td3.nmpc_td3 import TD3NMPCPolicy
from src.mpcrl.traj_tracking_env import MPCRLTrajTrackingEnv
import numpy as np, os

mpc_horizon = 10
dt = 1

env = MPCRLTrajTrackingEnv(
    dt=dt,
    mpc_horizon=mpc_horizon
)
nmpc = ParametricTrajectoryTracker(mpc_horizon, dt)
# model = TD3(TD3NMPCPolicy(nmpc, {"Q": np.array([1e3, 10, 10])}, 1e-3, net_arch={"qf": [256, 256], "pi": [0, 0]}), env)

model = TD3(
    "NMPC",
    env,
    policy_kwargs={
        "nmpc": nmpc,
        "initial_value_learnable_params": {"Q": np.array([1e3, 10, 10])},
        "net_arch": {"qf": [24, 24], "pi": [0, 0]}
    },
    batch_size=20,
    stats_window_size=1,
)

model.learn(100_000, log_interval=1)

# "wpts": np.array([mpc_horizon*[0, 0, 0]]).T, "nu_des": np.array([env.u_des, 0, 0])
#, policy_kwargs={
    # "nmpc": ParametricTrajectoryTracker(
    #     mpc_horizon, dt
    # ),
    # "initial_value_learnable_params": {
    #     "wpts": np.array([mpc_horizon*[0, 0, 0]]).T,
    #     "nu_des": np.array([env.u_des, 0, 0])
    # }
# })