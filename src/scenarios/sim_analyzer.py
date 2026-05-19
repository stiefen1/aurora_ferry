from typing import List, Dict, Tuple
from pathlib import Path

from python_vehicle_simulator.lib.obstacle import Obstacle
from python_vehicle_simulator.utils.math_fn import ssa
from src.environment.map import HelsingborgMap
from src.ais.ais import AIS, Vessel
import glob, os, numpy as np, json, pandas as pd, datetime, matplotlib.pyplot as plt, numpy.typing as npt
import shapely
from tqdm import tqdm
from shapely.ops import unary_union


class SimAnalyzer:
    def __init__(
            self,
            path_to_dir: str
    ):
        self.path_to_dir = path_to_dir
        self.path_to_json = glob.glob(os.path.join(path_to_dir, "scenarios", "*.json"))
        self.sim: List[Dict] = []
        self.config: List[Dict] = []
        self.load(self.path_to_json)

    def load(self, path_to_json: List[str]) -> None:
        for path in path_to_json:
            candidate_path_to_sim_folder = path.replace('scenarios', 'simulations').replace('.json', '')
            if Path(candidate_path_to_sim_folder).exists():
                self.sim.append(self.load_sim(candidate_path_to_sim_folder))
                self.config.append(self.load_config(path))
            else:
                print(f"simulation folder <<{candidate_path_to_sim_folder}>> matching configuration file <<{path}>> not found")

    def __call__(self) -> None:
        ## Load shore
        helsingborg = HelsingborgMap()
        ferry_route = helsingborg.get_ferry_routes()['Helsingør (DK) - Helsingborg (SE)']
        obstacles = [Obstacle(geometry=list(zip(*poly.exterior.coords.xy[::-1]))) for poly in helsingborg.polygons] # HelsingborgMap().get_shore_as_obstacles()
        shore_geom = unary_union([shapely.Polygon(obs.geometry.T) for obs in obstacles])
        print(f"{len(obstacles)} obstacles loaded from Helsingborg map.")

        # Low-level graphs
        fig_dist2ts, ax_dist2ts = plt.subplots()
        fig_os_traj, ax_os_traj = plt.subplots()
        fig_dist2shore, ax_dist2shore = plt.subplots()
        fig_travel_dist, ax_travel_dist = plt.subplots()
        fig_pos_track_error, ax_pos_track_error = plt.subplots()
        fig_speed_track_error, ax_speed_track_error = plt.subplots()
        fig_pos_est_error, ax_pos_est_error = plt.subplots()
        fig_tt, ax_tt = plt.subplots()
        fig_power, ax_power = plt.subplots()

        # High-level graphs
        fig_ts_dist_vs_tt, ax_ts_dist_vs_tt = plt.subplots()
        fig_ts_tt_vs_visill, ax_ts_tt_vs_visill = plt.subplots()
        fig_ts_dist_vs_yaw_tt, ax_ts_dist_vs_yaw_tt = plt.subplots()

        run_summaries: List[Dict] = []

        print("Analyzing results..")
        for i, (config, sim) in enumerate(tqdm(zip(self.config, self.sim), total=len(self.config))):
            scenario_generation = config["scenario_generation"]
            simulation = scenario_generation["simulation"]
            t0 = pd.to_datetime(simulation["start_time_iso_utc"])
            duration = simulation["duration_sec"]
            time_window = (t0, t0 + datetime.timedelta(seconds=duration))

            match scenario_generation["start"]:
                case "Helsingborg":
                    color = 'green'
                case "Helsingor":
                    color = 'red'
                case _:
                    raise ValueError(f"Invalid start name <<{scenario_generation['start']}>>")

            # Compute metrics for each simulation
            ## Distance to TS
            sim["distance_to_ts"] = self.distance_to_ts(
                sim["x"],
                scenario_generation["ais_data_paths"].replace('raw', 'smooth_interp'),
                time_window,
                dt=simulation["dt"],
                mmsi_to_exclude=scenario_generation["mmsi_to_exclude"]
            )
            
            
            for mmsi in sim["distance_to_ts"].keys():
                dist = np.asarray(sim["distance_to_ts"][mmsi], dtype=object)
                t = np.asarray(sim["x"]["time"], dtype=float)
                valid = np.array([v is not None for v in dist])
                ax_dist2ts.semilogy(t[valid], np.asarray(dist[valid], dtype=float), color=color)

            ax_os_traj.plot(sim["x"]["data"][:, 1], sim["x"]["data"][:, 0], color=color)

            ## Distance to shore
            sim["distance_to_shore"] = self.distance_to_shore(
                sim["x"],
                shore_geom
            )

            ax_dist2shore.plot(sim["x"]["time"], sim["distance_to_shore"], color=color)

            ## Target reached
            sim["target_reached"] = self.target_reached(sim["x"], ferry_route.waypoints, scenario_generation["start"], scenario_generation["guidance"]["term_dist"])

            ## Travel distance
            sim["travel_distance"] = self.travel_distance(
                sim["x"]
            )
            ax_travel_dist.plot(sim["x"]["time"], sim["travel_distance"], color=color)

            ## Trajectory tracking accuracy
            sim["pos_error"], sim["speed_error"] = self.trajectory_tracking_error(
                sim["x"], sim["x_des"]
            )
            ax_pos_track_error.plot(sim["x"]["time"], sim["pos_error"], color=color)
            ax_speed_track_error.plot(sim["x"]["time"], sim["speed_error"], color=color)
            
            ## Target tracking accuracy
            sim["target_tracking_accuracy"] = self.target_tracking_accuracy(
                sim["ts_est"],
                scenario_generation["ais_data_paths"].replace('raw', 'smooth_interp'),
                time_window,
                dt=simulation["dt"],
                mmsi_to_exclude=scenario_generation["mmsi_to_exclude"])

            t_acc = sim["target_tracking_accuracy"]
            if len(t_acc) > 0:
                for mmsi, data in t_acc.items():
                    ax_tt.plot(data["time"], data["error"], label=str(mmsi), color=color)
                    ax_ts_tt_vs_visill.scatter(np.sqrt(scenario_generation["operational_domain"]["illumination"] * scenario_generation["operational_domain"]["visibility"]), np.mean(data["error"]), color=color)

            smallest_dist = {}
            t_acc_mean = {}
            for mmsi in sim["distance_to_ts"].keys():
                if mmsi in t_acc.keys():
                    dist = np.asarray(sim["distance_to_ts"][mmsi], dtype=object)
                    valid = np.array([v is not None for v in dist])
                    smallest_dist[mmsi] = np.min(dist[valid])
                    t_acc_mean[mmsi] = np.mean(t_acc[mmsi]["error"])
                    ax_ts_dist_vs_tt.scatter(t_acc_mean[mmsi], smallest_dist[mmsi], color=color)
                    ax_ts_dist_vs_yaw_tt.scatter(np.mean(t_acc[mmsi]["motion_error"]["cog"]), smallest_dist[mmsi], color=color)
          

            ## Pose estimation accuracy
            sim["pos_est_error"] = self.pose_estimation_error(
                sim["x"], sim["x_est"]
            )
            ax_pos_est_error.plot(sim["x"]["time"], np.hypot(sim["pos_est_error"][:, 0], sim["pos_est_error"][:, 1]), color=color)

            ## Power consumption
            sim["power_cons"] = {}
            sim["power_cons"]["azimuth"], sim["power_cons"]["thrust"] = self.power_cons(
                sim["u"], sim["x"]
            )
            ax_power.plot(sim["x"]["time"], sim["power_cons"]["azimuth"] + sim["power_cons"]["thrust"], color=color)

            # Collect mission-level summary for report.
            buffer_dist = float(scenario_generation["guidance"]["buffer_target_ships"])
            min_dist_shore = float(np.min(sim["distance_to_shore"]))

            min_dist_ts, ts_collision_events = self._summarize_target_ship_distances(
                distance_to_ts=sim["distance_to_ts"],
                time=sim["x"]["time"],
                buffer_dist=buffer_dist,
            )

            target_reached_ok = bool(sim["target_reached"][0])
            ts_ok = bool(min_dist_ts >= buffer_dist)
            shore_ok = bool(min_dist_shore >= buffer_dist)
            mission_success = target_reached_ok and ts_ok and shore_ok

            failure_causes: List[str] = []
            if not ts_ok:
                failure_causes.append("collision with TS")
            if not shore_ok:
                failure_causes.append("collision with shore")
            if not target_reached_ok:
                failure_causes.append("target not reached")

            run_summaries.append({
                "json_file": os.path.basename(self.path_to_json[i]),
                "success": mission_success,
                "failure_causes": failure_causes,
                "ts_collisions": ts_collision_events,
                "min_dist_ts": float(min_dist_ts) if np.isfinite(min_dist_ts) else None,
                "min_dist_shore": min_dist_shore,
                "target_reached": target_reached_ok,
                "travel_distance": float(sim["travel_distance"][-1]),
                "mean_pos_error": float(np.mean(sim["pos_error"])),
                "mean_speed_error": float(np.mean(np.abs(sim["speed_error"]))),
                "mean_power": float(np.mean(sim["power_cons"]["azimuth"] + sim["power_cons"]["thrust"])),
            })

        ax_tt.set_xlabel("time [s]")
        ax_tt.set_ylabel("NE error norm [m]")
        ax_os_traj.set_aspect('equal')


        path_to_figures = os.path.join(self.path_to_dir, "figures")
        os.makedirs(path_to_figures, exist_ok=True)
        fig_dist2ts.savefig(os.path.join(path_to_figures, "dist2ts.png"))
        fig_os_traj.savefig(os.path.join(path_to_figures, "os_traj.png"))
        fig_dist2shore.savefig(os.path.join(path_to_figures, "dist2shore.png"))
        fig_travel_dist.savefig(os.path.join(path_to_figures, "travel_dist.png"))
        fig_pos_track_error.savefig(os.path.join(path_to_figures, "pos_error.png"))
        fig_speed_track_error.savefig(os.path.join(path_to_figures, "speed_error.png"))
        fig_pos_est_error.savefig(os.path.join(path_to_figures, "pos_est_error.png"))
        fig_tt.savefig(os.path.join(path_to_figures, "target_tracking.png"))
        fig_power.savefig(os.path.join(path_to_figures, "power.png"))
        fig_ts_dist_vs_tt.savefig(os.path.join(path_to_figures, "ts_dist_vs_tt.png"))
        fig_ts_tt_vs_visill.savefig(os.path.join(path_to_figures, "ts_tt_vs_visill.png"))
        fig_ts_dist_vs_yaw_tt.savefig(os.path.join(path_to_figures, "ts_dist_vs_yaw_tt.png"))
        print("Figures saved!")

        # Write simple textual report.
        total_runs = len(run_summaries)
        success_runs = sum(1 for r in run_summaries if r["success"])
        success_rate = 100.0 * success_runs / total_runs if total_runs > 0 else 0.0

        fail_ts = sum(1 for r in run_summaries if "collision with TS" in r["failure_causes"])
        fail_shore = sum(1 for r in run_summaries if "collision with shore" in r["failure_causes"])
        fail_target = sum(1 for r in run_summaries if "target not reached" in r["failure_causes"])

        avg_travel_distance = float(np.mean([r["travel_distance"] for r in run_summaries])) if run_summaries else float("nan")
        avg_min_dist_ts = float(np.mean([r["min_dist_ts"] for r in run_summaries if r["min_dist_ts"] is not None])) if any(r["min_dist_ts"] is not None for r in run_summaries) else float("nan")
        avg_min_dist_shore = float(np.mean([r["min_dist_shore"] for r in run_summaries])) if run_summaries else float("nan")

        report_lines = [
            "Simulation Analysis Report",
            "",
            f"Total missions: {total_runs}",
            f"Mission success rate: {success_runs}/{total_runs} ({success_rate:.1f}%)",
            "Success criterion: target reached AND min distance to target ships >= guidance.buffer_target_ships AND min distance to shore >= guidance.buffer_target_ships",
            "",
            "Failure summary:",
            f"- collision with TS: {fail_ts}",
            f"- collision with shore: {fail_shore}",
            f"- target not reached: {fail_target}",
            "",
            "Failed missions:",
        ]

        failed = [r for r in run_summaries if not r["success"]]
        if not failed:
            report_lines.append("- None")
        else:
            for r in failed:
                causes = ", ".join(r["failure_causes"]) if r["failure_causes"] else "unknown"
                details = ""
                if "collision with TS" in r["failure_causes"] and r.get("ts_collisions"):
                    ts_details = "; ".join(
                        [f"MMSI {e['mmsi']} at t={e['time_s']:.1f}s" for e in r["ts_collisions"]]
                    )
                    details = f" | TS details: {ts_details}"
                report_lines.append(f"- {r['json_file']}: {causes}{details}")

        report_lines.extend([
            "",
            "Additional metrics:",
            f"- average travel distance [m]: {avg_travel_distance:.2f}",
            f"- average minimum distance to TS [m]: {avg_min_dist_ts:.2f}",
            f"- average minimum distance to shore [m]: {avg_min_dist_shore:.2f}",
        ])

        report_path = os.path.join(self.path_to_dir, "report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines) + "\n")

        print("Report saved!")

    def _summarize_target_ship_distances(self, distance_to_ts: Dict, time: npt.NDArray, buffer_dist: float) -> Tuple[float, List[Dict]]:
        """Compute minimum TS distance and first collision events (if any) in one pass."""
        min_dist_ts = np.inf
        time_arr = np.asarray(time, dtype=float)
        ts_collision_events: List[Dict] = []

        for mmsi, distances in distance_to_ts.items():
            dist_arr_obj = np.asarray(distances, dtype=object)
            valid = np.array([v is not None for v in dist_arr_obj])
            if not np.any(valid):
                continue

            dist_arr = np.full(dist_arr_obj.shape, np.nan, dtype=float)
            dist_arr[valid] = np.asarray(dist_arr_obj[valid], dtype=float)

            local_min = float(np.min(dist_arr[np.isfinite(dist_arr)]))
            if local_min < min_dist_ts:
                min_dist_ts = local_min

            collided = np.where(np.isfinite(dist_arr) & (dist_arr < buffer_dist))[0]
            if collided.size > 0:
                idx0 = int(collided[0])
                ts_collision_events.append({
                    "mmsi": int(mmsi),
                    "time_s": float(time_arr[idx0]),
                    "distance_m": float(dist_arr[idx0]),
                })

        ts_collision_events.sort(key=lambda e: e["time_s"])
        return min_dist_ts, ts_collision_events
            

    def target_tracking_accuracy(self, ts_est: List[Dict], path_to_csv: str, time_window: Tuple[datetime.datetime, datetime.datetime], dt: float, mmsi_to_exclude: List[int]) -> Dict:
        """
        ts_est: List of dictionnary with keys "time" (float) and "vessels" (List[List[float]]). Each element of vessels
        is a list containing (mmsi, north, east, cog, sog).

        returns: Dictionnary keyed by mmsi, each value is a dict with:
        - "time": timestamps where an estimate was available
        - "error": NE position error norm
        - "motion_error": dict with "sog" and "cog" tracking errors
        """
        time_tolerance = dt / 2
        out: Dict = {}

        ais = AIS(path_to_csv, t0=time_window[0], tf=time_window[1], mmsi_to_exclude=mmsi_to_exclude)
        mmsi_col = ais.column_mapping["mmsi"]
        ts_col = ais.column_mapping["timestamp"]

        ais_df = ais.df.copy()
        ais_df["timestamp"] = pd.to_datetime(ais_df[ts_col], utc=True, errors="coerce")

        if "north" in ais.column_mapping and "east" in ais.column_mapping:
            ais_df["north"] = ais_df[ais.column_mapping["north"]].astype(float)
            ais_df["east"] = ais_df[ais.column_mapping["east"]].astype(float)
        else:
            lon = ais_df[ais.column_mapping["longitude"]].to_numpy(dtype=float)
            lat = ais_df[ais.column_mapping["latitude"]].to_numpy(dtype=float)
            east, north = ais.transformer.transform(lon, lat)
            ais_df["north"] = north
            ais_df["east"] = east

        ais_df["sog"] = ais_df[ais.column_mapping["sog"]].astype(float)
        ais_df["cog"] = ais_df[ais.column_mapping["cog"]].astype(float)
        ais_df = ais_df[[mmsi_col, "timestamp", "north", "east", "sog", "cog"]].dropna(subset=["north", "east"]).sort_values("timestamp")

        t0_utc = pd.to_datetime(time_window[0], utc=True)
        est_rows = []
        for i, row in enumerate(ts_est):
            timestamp = t0_utc + pd.to_timedelta(float(row.get("time", 0.0)), unit="s")
            for vessel in row.get("vessels", []):
                est_rows.append({
                    "mmsi": int(vessel[0]),
                    "timestamp": timestamp,
                    "north_est": float(vessel[1]),
                    "east_est": float(vessel[2]),
                    "cog_est": float(vessel[3]),
                    "sog_est": float(vessel[4]),
                    "t_idx": i,
                })

        if len(est_rows) == 0:
            return out

        est_df = pd.DataFrame(est_rows).sort_values("timestamp")
        tracked_mmsi = sorted(est_df["mmsi"].unique().tolist())
        tolerance = pd.Timedelta(seconds=time_tolerance)
        times_sec = np.asarray([row.get("time", 0.0) for row in ts_est], dtype=float)

        for mmsi in tracked_mmsi:
            est_m = est_df[est_df["mmsi"] == mmsi][["timestamp", "north_est", "east_est", "sog_est", "cog_est", "t_idx"]].sort_values("timestamp")
            gt_m = ais_df[ais_df[mmsi_col] == mmsi][["timestamp", "north", "east", "sog", "cog"]].sort_values("timestamp")
            if len(gt_m) == 0:
                continue

            aligned = pd.merge_asof(
                est_m,
                gt_m,
                on="timestamp",
                direction="nearest",
                tolerance=tolerance,
            ).dropna(subset=["north", "east", "sog", "cog"])

            if len(aligned) == 0:
                continue

            pos_error = np.hypot(
                aligned["north_est"].to_numpy() - aligned["north"].to_numpy(),
                aligned["east_est"].to_numpy() - aligned["east"].to_numpy(),
            )
            sog_error = np.abs(aligned["sog_est"].to_numpy() - aligned["sog"].to_numpy())
            cog_error = np.abs(np.rad2deg(ssa(np.deg2rad(aligned["cog_est"].to_numpy() - aligned["cog"].to_numpy()))))

            t_idx = aligned["t_idx"].to_numpy(dtype=int)
            valid = np.isfinite(pos_error) & np.isfinite(sog_error) & np.isfinite(cog_error)

            out[mmsi] = {
                "time": times_sec[t_idx[valid]].tolist(),
                "error": pos_error[valid].tolist(),
                "motion_error": {
                    "sog": sog_error[valid].tolist(),
                    "cog": cog_error[valid].tolist(),
                },
            }

        return out

    def target_reached(self, states: Dict, ferry_route: npt.NDArray, start: str, threshold: float) -> Tuple[bool, float, float]:
        match start:
            case "Helsingborg":
                target = ferry_route[0]
            case "Helsingor":
                target = ferry_route[-1]
            case _:
                raise ValueError(f"Invalid start name <<{start}>>")
            
        dist_to_target = np.linalg.norm(states["data"][:, 0:2] - target[None, :], axis=1)
        idx_min_dist_to_target = np.argmin(dist_to_target) 
        min_dist_to_target = float(dist_to_target[idx_min_dist_to_target])
        time_at_min_dist_to_target = float(states["time"][idx_min_dist_to_target])
        return min_dist_to_target <= threshold + 1, min_dist_to_target, time_at_min_dist_to_target # just adding a 1m margin to avoid numerical issues

    def pose_estimation_error(self, states: Dict, states_est: Dict) -> npt.NDArray:
        return states["data"] - states_est["data"][1::]
        # print(states["data"].shape, states["time"].shape, states_est["data"].shape, states_est["time"].shape)

    def distance_to_shore(self, states: Dict, shore_geom) -> npt.NDArray:
        north = np.asarray(states["data"][:, 0], dtype=np.float64)
        east = np.asarray(states["data"][:, 1], dtype=np.float64)
        points = shapely.points(north, east)
        return np.asarray(shapely.distance(points, shore_geom), dtype=np.float64)


    def power_cons(self, commands: Dict, states: Dict, mu: float = 1e-3) -> Tuple[npt.NDArray, npt.NDArray]:
        azimuth_commands = commands["data"][1::, 0:4]
        azimuth_angle = states["data"][:, 12:16]
        thruster_speed = states["data"][:, 16:20]

        delta_azimuth = np.abs(ssa(azimuth_commands - azimuth_angle))
        return np.sum(delta_azimuth**2, axis=1), mu * np.sum(thruster_speed**2, axis=1)


    def trajectory_tracking_error(self, states: Dict, traj_data: Dict) -> Tuple[npt.NDArray, npt.NDArray]:
        pos_error = np.hypot(states["data"][:, 0] - traj_data["data"][:, 0], states["data"][:, 1] - traj_data["data"][:, 1])
        speed_error = np.sqrt(states["data"][:, 6]**2 + states["data"][:, 7]**2) - traj_data["data"][:, 6]
        return pos_error, speed_error

    def travel_distance(self, states: Dict) -> npt.NDArray:
        ne = np.asarray(states["data"][:, 0:2], dtype=np.float64)
        d_ne = np.diff(ne, axis=0)
        step_distance = np.hypot(d_ne[:, 0], d_ne[:, 1])
        cumulative_distance = np.concatenate(([0.0], np.cumsum(step_distance)))
        return cumulative_distance


    def distance_to_ts(self, states: Dict, path_to_csv: str, time_window: Tuple[datetime.datetime, datetime.datetime], dt: float, mmsi_to_exclude: List[int]) -> Dict:
        """
        Compute distance w.r.t each target ship in ts_data at each timestep and output a dictionnary
        of trajectories indexed by mmsi. If no data is available for a specific timestamp, set distance to None.

        states: dict with keys "time" and "data" with shapes (N,) and (N, 20)
        ts_data: panda dataframe containing AIS-like data 

        out = {
            "mmsi1": [d(0), d(1), ..., d(T)],
            "mmsi2": [d(0), d(1), ..., d(tau), None, ..., None],
            ...
            "mmsiN": [None, ..., None, d(tau), d(tau+1), ..., d(T)]
        }
        """
        time_tolerance = dt / 2
        ais = AIS(path_to_csv, t0=time_window[0], tf=time_window[1], mmsi_to_exclude=mmsi_to_exclude)

        # Build own-ship dataframe once (vectorized time and position extraction).
        # Use simulation t0 + i*dt for robust timestamp alignment.
        os_states = np.asarray(states["data"], dtype=np.float64)
        os_time = pd.to_datetime(time_window[0], utc=True) + pd.to_timedelta(np.arange(len(os_states)) * dt, unit="s")
        os_df = pd.DataFrame(
            {
                "timestamp": os_time,
                "os_north": os_states[:, 0],
                "os_east": os_states[:, 1],
            }
        ).sort_values("timestamp")

        # Use AIS raw dataframe directly instead of per-timestep sensor calls.
        mmsi_col = ais.column_mapping["mmsi"]
        ts_col = ais.column_mapping["timestamp"]

        ais_df = ais.df.copy()
        ais_df["timestamp"] = pd.to_datetime(ais_df[ts_col], utc=True, errors="coerce")

        if "north" in ais.column_mapping and "east" in ais.column_mapping:
            ais_df["north"] = ais_df[ais.column_mapping["north"]].astype(np.float64)
            ais_df["east"] = ais_df[ais.column_mapping["east"]].astype(np.float64)
        else:
            lon = ais_df[ais.column_mapping["longitude"]].to_numpy(dtype=np.float64)
            lat = ais_df[ais.column_mapping["latitude"]].to_numpy(dtype=np.float64)
            east, north = ais.transformer.transform(lon, lat)
            ais_df["north"] = north
            ais_df["east"] = east

        ais_df = ais_df[[mmsi_col, "timestamp", "north", "east"]].dropna().sort_values("timestamp")

        distances: Dict[int, List[float]] = {}
        tolerance = pd.Timedelta(seconds=time_tolerance)

        for mmsi, vessel_df in ais_df.groupby(mmsi_col, sort=False):
            vessel_df = vessel_df[["timestamp", "north", "east"]].sort_values("timestamp")
            aligned = pd.merge_asof(
                os_df,
                vessel_df,
                on="timestamp",
                direction="nearest",
                tolerance=tolerance,
            )

            d = np.hypot(aligned["os_north"].to_numpy() - aligned["north"].to_numpy(), aligned["os_east"].to_numpy() - aligned["east"].to_numpy())
            valid = np.isfinite(d)

            if np.any(valid):
                distances[int(mmsi)] = [float(di) if vi else None for di, vi in zip(d, valid)]

        return distances

    def load_config(self, path: str) -> Dict:
        with open(path, 'r') as f:
            config = json.load(f)
        return config
    
    def load_sim(self, folder: str) -> Dict:
        with open(os.path.join(folder, "navigation_target_vessels.jsonl"), 'r') as f:
            ts_est = [json.loads(line) for line in f]  

        control_u_npz = np.load(os.path.join(folder, "control_u.npz"), allow_pickle=True)
        navigation_actual_states_npz = np.load(os.path.join(folder, "navigation_actual_states.npz"), allow_pickle=True)
        navigation_states_npz = np.load(os.path.join(folder, "navigation_states.npz"), allow_pickle=True)
        guidance_states_des_npz = np.load(os.path.join(folder, "guidance_states_des.npz"), allow_pickle=True)

        data = {
            "u": {k:control_u_npz[k] for k in control_u_npz.files},
            "x": {k:navigation_actual_states_npz[k] for k in navigation_actual_states_npz.files},
            "x_est": {k:navigation_states_npz[k] for k in navigation_states_npz.files},
            "x_des": {k:guidance_states_des_npz[k] for k in guidance_states_des_npz.files},
            "ts_est": ts_est
        }

        control_u_npz.close()
        navigation_actual_states_npz.close()
        navigation_states_npz.close()
        guidance_states_des_npz.close()


        return data


if __name__ == "__main__":
    import os
    analyzer = SimAnalyzer(os.path.join("sim_data", "test"))
    analyzer()