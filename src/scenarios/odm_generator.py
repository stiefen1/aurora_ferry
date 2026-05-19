"""
Randomly sample an operational domain based on the admissible ranges provided in a yaml file and save it (json)
"""
import pathlib
from typing import LiteralString, Optional, Any

import os, json, csv, hashlib, numpy as np, yaml
from datetime import datetime, timezone

DEFAULT_PATH_TO_CONFIG = os.path.join("sim_data", "test", "test.yaml")

class ODMGenerator:
    def __init__(
            self,
            path_to_config: LiteralString,
    ):
        self.path_to_config = path_to_config
        with open(self.path_to_config, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.seed = self.config["scenario_generation"]["seed"]
        self.rng = np.random.default_rng(self.seed)
        

    def __call__(self, folder: str = "scenarios"):
        for i in range(self.config["scenario_generation"]["number_of_scenarios"]):
            path_to_config = pathlib.Path(self.path_to_config)
            self.sample_single(os.path.join(
                path_to_config.parent,
                folder,
                path_to_config.name.rsplit('.')[0] + f"_{i}.json"
                )
            )

    @staticmethod
    def _is_distribution_node(node: Any) -> bool:
        return isinstance(node, dict) and "min" in node and "max" in node

    def _sample_range(self, value_min: Any, value_max: Any) -> Any:
        if isinstance(value_min, list) and isinstance(value_max, list):
            return [self._sample_range(vmin, vmax) for vmin, vmax in zip(value_min, value_max)]

        value_min_f: float = float(value_min)  # type: ignore[arg-type]
        value_max_f: float = float(value_max)  # type: ignore[arg-type]
        if value_min_f > value_max_f:
            value_min_f, value_max_f = value_max_f, value_min_f

        if isinstance(value_min, int) and isinstance(value_max, int):
            return int(self.rng.integers(int(value_min_f), int(value_max_f) + 1))

        return float(self.rng.uniform(value_min_f, value_max_f))

    def _sample_distribution_node(self, node: dict[str, Any]) -> Any:
        lower = node["min"]
        upper = node["max"]

        if "mean" in node and "std" in node:
            sample = self.rng.normal(float(node["mean"]), float(node["std"]))
            lower_f = float(min(lower, upper))
            upper_f = float(max(lower, upper))
            sampled_value: Any = float(np.clip(sample, lower_f, upper_f))
        else:
            sampled_value = self._sample_range(lower, upper)

        extra_keys = [
            key for key in node.keys()
            if key not in {"min", "max", "mean", "std"}
        ]

        if not extra_keys:
            return sampled_value

        out: dict[str, Any] = {"value": sampled_value}
        for key in extra_keys:
            out[key] = self._sample_node(node[key])
        return out

    @staticmethod
    def _is_failure_node(node: Any) -> bool:
        return isinstance(node, dict) and "prob" in node and "time" in node

    def _sample_failure_node(self, node: dict[str, Any]) -> dict[str, Any]:
        prob = float(node["prob"])
        failure_occurs = float(self.rng.uniform(0.0, 1.0)) < prob
        if failure_occurs:
            time_fraction = float(self._unwrap_scalar(self._sample_node(node["time"])))
        else:
            time_fraction = None
        return {"time": time_fraction}

    def _sample_node(self, node: Any) -> Any:
        if self._is_distribution_node(node):
            return self._sample_distribution_node(node)

        if self._is_failure_node(node):
            return self._sample_failure_node(node)

        if isinstance(node, dict):
            sampled: dict[str, Any] = {}
            for key, value in node.items():
                if key == "info":
                    sampled[key] = value
                elif key == "ais_data_paths" and isinstance(value, list) and value:
                    sampled[key] = value[int(self.rng.integers(0, len(value)))]
                elif key == "start" and isinstance(value, list) and value:
                    sampled[key] = self._sample_node(value[int(self.rng.integers(0, len(value)))])
                else:
                    sampled[key] = self._sample_node(value)
            return sampled

        if isinstance(node, list):
            return [self._sample_node(value) for value in node]

        return node

    @staticmethod
    def _unwrap_scalar(value: Any) -> Any:
        if isinstance(value, dict) and "value" in value:
            return value["value"]
        return value

    def _resolve_data_path(self, data_path: str) -> str:
        if os.path.isabs(data_path):
            return data_path

        candidates = [
            os.path.join(os.getcwd(), data_path),
            os.path.join(os.path.dirname(self.path_to_config), data_path),
            os.path.join(os.path.dirname(self.path_to_config), "..", "..", data_path),
        ]

        for candidate in candidates:
            candidate_abs = os.path.abspath(candidate)
            if os.path.exists(candidate_abs):
                return candidate_abs

        return os.path.abspath(candidates[0])

    def _load_time_window_from_csv(self, csv_path: str, excluded_mmsi: set[int]) -> tuple[float, float]:
        min_t: Optional[float] = None
        max_t: Optional[float] = None

        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if excluded_mmsi:
                    mmsi_raw = (row.get("mmsi") or "").strip()
                    if mmsi_raw:
                        try:
                            if int(float(mmsi_raw)) in excluded_mmsi:
                                continue
                        except ValueError:
                            pass

                raw = (row.get("timestamp") or "").strip()
                if not raw:
                    continue
                try:
                    dt = datetime.fromisoformat(raw)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    else:
                        dt = dt.astimezone(timezone.utc)
                    t = dt.timestamp()
                except ValueError:
                    continue

                if min_t is None or t < min_t:
                    min_t = t
                if max_t is None or t > max_t:
                    max_t = t

        if min_t is None or max_t is None:
            if excluded_mmsi:
                raise ValueError(
                    f"No valid timestamp values found in CSV after applying mmsi_to_exclude: {csv_path}"
                )
            raise ValueError(f"No valid timestamp values found in CSV: {csv_path}")

        return min_t, max_t

    def _attach_simulation_start_time(self, sampled: dict[str, Any]) -> tuple[float, float]:
        simulation_cfg = sampled.get("simulation")
        if not isinstance(simulation_cfg, dict):
            raise ValueError("Missing required 'scenario_generation.simulation' section in config")

        duration_raw = simulation_cfg.get("duration_sec")
        if duration_raw is None:
            raise ValueError("Missing required 'scenario_generation.simulation.duration_sec' in config")

        duration_sec = float(self._unwrap_scalar(duration_raw))
        if duration_sec <= 0:
            raise ValueError("simulation.duration_sec must be > 0")

        ais_path_raw = sampled.get("ais_data_paths")
        if not isinstance(ais_path_raw, str) or not ais_path_raw:
            raise ValueError("Sampled scenario must contain one selected 'ais_data_paths' string")

        excluded_raw = sampled.get("mmsi_to_exclude", [])
        if excluded_raw is None:
            excluded_raw = []
        if not isinstance(excluded_raw, list):
            raise ValueError("scenario_generation.mmsi_to_exclude must be a list of integers")
        excluded_mmsi: set[int] = set()
        for value in excluded_raw:
            try:
                excluded_mmsi.add(int(value))
            except (TypeError, ValueError):
                raise ValueError("scenario_generation.mmsi_to_exclude must contain only integer values")

        csv_path = self._resolve_data_path(ais_path_raw)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Selected AIS CSV does not exist: {csv_path}")

        t_min, t_max = self._load_time_window_from_csv(csv_path, excluded_mmsi)
        available_duration = t_max - t_min
        if duration_sec > available_duration:
            raise ValueError(
                f"Requested simulation duration ({duration_sec:.1f}s) exceeds data coverage "
                f"({available_duration:.1f}s) for {ais_path_raw}"
            )

        start_sec = float(self.rng.uniform(t_min, t_max - duration_sec))
        start_iso = datetime.fromtimestamp(start_sec, tz=timezone.utc).isoformat()

        simulation_cfg["start_time_iso_utc"] = start_iso

        return start_sec, duration_sec

    def _resolve_failure_times(self, node: Any, start_sec: float, duration_sec: float) -> None:
        """Walk sampled tree and convert failure time fractions (0-1) to ISO datetimes."""
        if not isinstance(node, dict):
            return
        if set(node.keys()) == {"time"}:
            frac = node["time"]
            if frac is not None:
                t = start_sec + float(frac) * duration_sec
                node["time"] = datetime.fromtimestamp(t, tz=timezone.utc).isoformat()
            return
        for value in node.values():
            self._resolve_failure_times(value, start_sec, duration_sec)

    def sample_single(self, save_path: Optional[str] = None) -> None:
        save_path = save_path or (self.path_to_config.rsplit(".", 1)[0] + ".json")
        scenario_generation = self.config.get("scenario_generation", self.config)
        sampled = self._sample_node(scenario_generation)
        if isinstance(sampled, dict):
            start_sec, duration_sec = self._attach_simulation_start_time(sampled)
            self._resolve_failure_times(sampled, start_sec, duration_sec)
            sampled.pop("number_of_scenarios", None)

        payload = {
            "seed": self.seed,
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "source_config": self.path_to_config,
            "config_hash": self._config_hash(),
            "scenario_generation": sampled,
        }

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print(f"Saved sampled ODM scenario to {save_path}")

    def _config_hash(self) -> str:
        with open(self.path_to_config, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        return digest[:8]

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        

if __name__ == "__main__":
    odm_gen = ODMGenerator(DEFAULT_PATH_TO_CONFIG, seed=None)
    odm_gen()

    # config.yaml
    ### sim1
    ##### config1.json
    ##### sim_results1
    ######## data.npz
    ######## data.jsonl