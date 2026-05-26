"""
Run N scenario simulations across M worker processes.

Each worker loads heavy shared resources once (map, ferry route) via the pool
initializer, then handles multiple simulations sequentially.

Usage:
    python -m src.scenarios.parallel_sim_launcher sim_data/test/scenarios --workers 8
"""
import os, glob, argparse, multiprocessing
from src.scenarios.sim_launcher import SimLauncher

# Process-local launcher (loaded once per worker via _init_worker)
_launcher: SimLauncher | None = None


def _init_worker() -> None:
    global _launcher
    _launcher = SimLauncher()  # loads map, ferry route, and NN weights once per worker


def _run_sim(path_to_config: str) -> tuple[str, str]:
    try:
        assert _launcher is not None
        _launcher.run_single_sim(path_to_config, render=False, use_tqdm=False)
        return "ok", path_to_config
    except Exception as e:
        return "fail", f"{path_to_config}: {e}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario_dir", nargs="?", default=os.path.join("sim_data", "test", "scenarios"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    configs = sorted(glob.glob(os.path.join(args.scenario_dir, "*.json")))
    print(f"Found {len(configs)} scenarios, launching with {args.workers} workers")

    with multiprocessing.Pool(processes=args.workers, initializer=_init_worker) as pool:
        for i, (status, msg) in enumerate(pool.imap_unordered(_run_sim, configs), 1):
            print(f"[{i}/{len(configs)}] {status.upper()}: {msg}")
