"""Simple RL weight evaluation.

Runs N episodes, sums each metric from env.eval() per episode,
then computes mean/std across episodes and saves the result
in the same folder as the weight .zip file.
"""
from typing import Tuple
import random

from pathlib import Path
import json

import gymnasium as gym
import numpy as np
from stable_baselines3.ppo import PPO
from stable_baselines3.sac import SAC

from src.rl.traj_tracking_env import TrajTrackingEnv


# Edit only these values when evaluating another checkpoint.
ALG = "sac"
DATE_AND_TIME = "2026_06_24_15_47_22"
WEIGHTS_NAME = "aurora_sac_nenvs16_arch_256x256_64x32_1500000_steps.zip"
N_EPISODES = 100
SEED = 42

DT = 0.2
N_WPTS = 2
ACTION_REPEAT = 10
WPTS_SPACE_MULTIPLICATOR = 7


def set_global_seeds(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	try:
		import torch as th

		th.manual_seed(seed)
		if th.cuda.is_available():
			th.cuda.manual_seed_all(seed)
		# Best-effort deterministic behavior for inference-time ops.
		th.backends.cudnn.deterministic = True
		th.backends.cudnn.benchmark = False
	except Exception:
		pass


def load_model(alg: str, weights_path: Path):
	if alg == "sac":
		return SAC.load(str(weights_path))
	if alg == "ppo":
		return PPO.load(str(weights_path))
	raise ValueError(f"Unsupported algorithm: {alg}")


def run_episode(env: gym.Env, model, episode_seed: int, save_from_t_sec: float = 200, dt: float = 2.0) -> Tuple[dict[str, float], dict[str, float], list[float]]:
	obs, _ = env.reset(seed=episode_seed)
	sums: dict[str, float] = {}
	worst: dict[str, float] = {}
	delay: list[float] = []

	t = 0.0
	while True:
		action, _ = model.predict(obs, deterministic=True)
		obs, _, terminated, truncated, _ = env.step(action)
		# if int(episode_seed - SEED) in [83, 97]:
		# 	env.render()
		
		t += dt # dt = action_repeat * sim_dt = 10 * 0.2

		if t >= save_from_t_sec:
			metrics = env.unwrapped.eval()  # type: ignore[attr-defined]
			V = np.linalg.norm(env.unwrapped.own_vessel.states[6:8]).astype(float)
			delay.append(float(((env.unwrapped.V_des[0]/V) - 1) * dt))
			for key, value in metrics.items():
				sums[key] = sums.get(key, 0.0) + float(value)
				worst[key] = max(worst.get(key, 0.0), float(value))
			if worst["distance"] >= 25:
				print(t, worst["distance"], env.unwrapped.V_des[0])

		if terminated or truncated:
			return sums, worst, delay


if __name__ == "__main__":
	import os, matplotlib.pyplot as plt
	# root_dir = Path(__file__).resolve().parents[2]
    # weights_path = root_dir / "checkpoints" / ALG / DATE_AND_TIME / WEIGHTS_NAME

	weights_path = Path(os.path.join("Z:\\dev", "aurora_ferry", "checkpoints", ALG, DATE_AND_TIME, WEIGHTS_NAME)).resolve()

	if not weights_path.exists():
		raise FileNotFoundError(f"Weights not found: {weights_path}")

	env = TrajTrackingEnv(
		DT,
		n_wpts=N_WPTS,
		action_repeat=ACTION_REPEAT,
		wpts_space_multiplicator=WPTS_SPACE_MULTIPLICATOR,
		max_steps=600,
		render_mode='human',
		simple_path=True,
		path_to_config=os.path.join("src", "rl", "eval.yaml")
	)
	env = gym.wrappers.FlattenObservation(env)

	set_global_seeds(SEED)
	env.action_space.seed(SEED)
	env.observation_space.seed(SEED)

	model = load_model(ALG, weights_path)

	per_episode: list[dict[str, float]] = []
	per_episode_worst: list[dict[str, float]] = []
	per_episode_delay: list[list[float]] = []
	for episode_idx in range(N_EPISODES):
		episode_sums, episode_worst, episode_delay = run_episode(env, model, episode_seed=SEED + episode_idx)
		per_episode.append(episode_sums)
		per_episode_worst.append(episode_worst)
		per_episode_delay.append(episode_delay)
		print(f"Episode {episode_idx + 1}/{N_EPISODES}: {episode_worst}")

	metric_names = per_episode[0].keys()
	summary: dict[str, dict[str, float]] = {}
	for name in metric_names:
		values = np.array([episode[name] for episode in per_episode], dtype=float)
		summary[name] = {
			"mean": float(np.mean(values)),
			"std": float(np.std(values)),
		}

	results = {
		"algorithm": ALG,
		"weights": str(weights_path),
		"n_episodes": N_EPISODES,
		"seed": SEED,
		"summary": summary,
	}

	output_path = weights_path.parent / "eval_results.json"
	with output_path.open("w", encoding="utf-8") as f:
		json.dump(results, f, indent=2)

	print("\nSummary:")
	print(json.dumps(summary, indent=2))
	print(f"\nSaved: {output_path}")

	plt.figure()
	plt.hist([worst["distance"] for worst in per_episode_worst])
	plt.show()

	plt.figure()
	plt.plot(np.linspace(0, 1000, 501), np.array(per_episode_delay).T)
	plt.show()

	plt.figure()
	for i in range(len(per_episode_delay)):
		plt.plot(np.linspace(0, 1000, 501), np.cumsum(per_episode_delay[i]))
	plt.show()