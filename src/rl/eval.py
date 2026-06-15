"""Simple RL weight evaluation.

Runs N episodes, sums each metric from env.eval() per episode,
then computes mean/std across episodes and saves the result
in the same folder as the weight .zip file.
"""

from pathlib import Path
import json

import gymnasium as gym
import numpy as np
from stable_baselines3.ppo import PPO
from stable_baselines3.sac import SAC

from src.rl.traj_tracking_env import TrajTrackingEnv


# Edit only these values when evaluating another checkpoint.
ALG = "sac"
DATE_AND_TIME = "2026_06_12_11_39_46"
WEIGHTS_NAME = "aurora_4800000_steps.zip"
N_EPISODES = 20

DT = 0.2
N_WPTS = 2
ACTION_REPEAT = 10
WPTS_SPACE_MULTIPLICATOR = 7


def load_model(alg: str, weights_path: Path):
	if alg == "sac":
		return SAC.load(str(weights_path))
	if alg == "ppo":
		return PPO.load(str(weights_path))
	raise ValueError(f"Unsupported algorithm: {alg}")


def run_episode(env: gym.Env, model) -> dict[str, float]:
	obs, _ = env.reset()
	sums: dict[str, float] = {}

	while True:
		action, _ = model.predict(obs, deterministic=True)
		obs, _, terminated, truncated, _ = env.step(action)

		metrics = env.unwrapped.eval()  # type: ignore[attr-defined]
		for key, value in metrics.items():
			sums[key] = sums.get(key, 0.0) + float(value)

		if terminated or truncated:
			return sums


if __name__ == "__main__":
	import os
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
	)
	env = gym.wrappers.FlattenObservation(env)

	model = load_model(ALG, weights_path)

	per_episode: list[dict[str, float]] = []
	for episode_idx in range(N_EPISODES):
		episode_sums = run_episode(env, model)
		per_episode.append(episode_sums)
		print(f"Episode {episode_idx + 1}/{N_EPISODES}: {episode_sums}")

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
		"summary": summary,
	}

	output_path = weights_path.parent / "eval_results.json"
	with output_path.open("w", encoding="utf-8") as f:
		json.dump(results, f, indent=2)

	print("\nSummary:")
	print(json.dumps(summary, indent=2))
	print(f"\nSaved: {output_path}")