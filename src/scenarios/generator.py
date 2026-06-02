"""
Randomly sample an operational domain based on the admissible ranges provided in a yaml file and save it (json)
"""
import pathlib
from typing import Dict, LiteralString, Optional, Any
from python_vehicle_simulator.lib.obstacle import Obstacle
from src.environment.map import HelsingborgMap

import os, json, csv, hashlib, numpy as np, yaml, pyproj, shapely
import glob
from shapely.ops import unary_union
from datetime import datetime, timezone

DEFAULT_PATH_TO_CONFIG = os.path.join("sim_data", "cos_sin_obs", "cos_sin_obs.yaml")

class ScenarioGenerator:
    _seed: Optional[int] = None
    _shore_geom: Any = None
    def __init__(
            self,
            path_to_config: LiteralString,
    ):
        self.path_to_config = path_to_config
        with open(self.path_to_config, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.reset(seed=self.config["scenario_generation"]["seed"])
        
    @property
    def seed(self) -> int | None:
        return self._seed
    
    @seed.setter
    def seed(self, val: int | None) -> None:
        self._seed = val
        self.rng = np.random.default_rng(self._seed)

    def reset(self, seed: Optional[int] = None) -> None:
        self.seed = seed

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
                elif key == "ais_data_paths" and isinstance(value, str) and value:
                    folder_path = self._resolve_data_path(value)
                    csv_paths = glob.glob(os.path.join(folder_path, "*.csv"))
                    if not csv_paths:
                        raise ValueError(f"No CSV files found in ais_data_paths folder: {value}")
                    selected_csv = csv_paths[int(self.rng.integers(0, len(csv_paths)))]
                    selected_rel = os.path.relpath(selected_csv, os.getcwd()).replace("\\", "/")
                    sampled[key] = f"/{selected_rel}"
                elif key == "start":
                    pass  # Deferred: sampled collision-free in sample_single
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
        if os.path.isabs(data_path) and os.path.exists(data_path):
            return data_path

        # Allow portable repo-root style paths such as /data/raw/file.csv.
        normalized_data_path = data_path.lstrip("/\\")

        candidates = [
            os.path.join(os.getcwd(), normalized_data_path),
            os.path.join(os.path.dirname(self.path_to_config), normalized_data_path),
            os.path.join(os.path.dirname(self.path_to_config), "..", "..", normalized_data_path),
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

    def _get_ts_positions_at_time(self, csv_path: str, excluded_mmsi: set[int], start_sec: float, tolerance_sec: float = 30.0, lookahead_sec: float = 30.0) -> list[tuple[float, float, float]]:
        """Return (north, east, envelope_radius_m) of TS in [start_sec - tolerance_sec, start_sec + lookahead_sec]."""
        transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)
        positions: list[tuple[float, float, float]] = []
        t_lo = start_sec - tolerance_sec
        t_hi = start_sec + lookahead_sec
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    mmsi = int(float((row.get("mmsi") or "").strip()))
                except (ValueError, AttributeError):
                    continue
                if mmsi in excluded_mmsi:
                    continue
                try:
                    t = float((row.get("timestamp_sec") or "").strip())
                except (ValueError, AttributeError):
                    continue
                if not (t_lo <= t <= t_hi):
                    continue
                try:
                    east, north = transformer.transform(float(row["lon"]), float(row["lat"]))
                except (ValueError, KeyError):
                    continue
                length_m = 0.0
                width_m = 0.0
                try:
                    length_m = float((row.get("length") or "").strip() or 0.0)
                except (ValueError, AttributeError):
                    pass
                try:
                    width_m = float((row.get("width") or "").strip() or 0.0)
                except (ValueError, AttributeError):
                    pass
                envelope_radius_m = float(np.hypot(0.5 * length_m, 0.5 * width_m))
                positions.append((north, east, envelope_radius_m))
        return positions

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

    def sample_single(self, save_path: Optional[str] = None, save: bool = True) -> Dict:
        save_path = save_path or (self.path_to_config.rsplit(".", 1)[0] + ".json")
        scenario_generation = self.config.get("scenario_generation", self.config)
        sampled = self._sample_node(scenario_generation)
        if isinstance(sampled, dict):
            start_sec, duration_sec = self._attach_simulation_start_time(sampled)
            # Sample a collision-free start position now that start_sec is known
            safety_dist = float(scenario_generation.get("safety_distance_at_spawn", 200)) # float(guidance_cfg.get("buffer_target_ships", 100.0)) + float(guidance_cfg.get("corridor_width", 50.0)) / 2.0
            csv_path = self._resolve_data_path(sampled["ais_data_paths"]).replace('raw', 'smooth_interp')
            excluded_mmsi = {int(v) for v in (sampled.get("mmsi_to_exclude") or [])}
            ts_positions = self._get_ts_positions_at_time(csv_path, excluded_mmsi, start_sec, lookahead_sec=30.0)
            start_cfg = self.config["scenario_generation"]["start"]
            if self._shore_geom is None:
                _map = HelsingborgMap()
                _obs = [Obstacle(geometry=list(zip(*poly.exterior.coords.xy[::-1]))) for poly in _map.polygons]
                self._shore_geom = unary_union([shapely.Polygon(obs.geometry.T) for obs in _obs])
            locations = {k: v for k, v in start_cfg.items() if k != "info"}
            loc_name = list(locations.keys())[int(self.rng.integers(0, len(locations)))]
            ranges = locations[loc_name]
            pos = None
            for _ in range(100):
                pos = {
                    "north": self._sample_range(ranges[0][0], ranges[0][1]),
                    "east": self._sample_range(ranges[1][0], ranges[1][1]),
                }
                if (all(np.hypot(pos["north"] - tn, pos["east"] - te) >= (safety_dist + ts_radius)
                    for tn, te, ts_radius in ts_positions)
                        and float(shapely.distance(shapely.Point(pos["north"], pos["east"]), self._shore_geom)) >= safety_dist):
                    break
            assert pos is not None, f"Failed to find a valid start position ({start_cfg}, {start_sec})"
            sampled["start"] = pos
            self._resolve_failure_times(sampled, start_sec, duration_sec)
            sampled.pop("number_of_scenarios", None)

        payload = {
            "seed": self.seed,
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "source_config": self.path_to_config,
            "config_hash": self._config_hash(),
            "scenario_generation": sampled,
        }

        if save:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

            print(f"Saved sampled ODM scenario to {save_path}")
        
        return payload

    def _config_hash(self) -> str:
        with open(self.path_to_config, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        return digest[:8]
        

if __name__ == "__main__":
    odm_gen = ScenarioGenerator(DEFAULT_PATH_TO_CONFIG)
    odm_gen()

    # config.yaml
    ### sim1
    ##### config1.json
    ##### sim_results1
    ######## data.npz
    ######## data.jsonl