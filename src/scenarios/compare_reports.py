import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set


@dataclass
class ReportData:
	name: str
	path: Path
	total_missions: int
	success_num: int
	success_den: int
	success_rate_pct: float
	fail_collision_ts: int
	fail_collision_shore: int
	fail_target_not_reached: int
	fail_corridor_exceeded: int
	avg_travel_distance: float
	avg_min_dist_ts: float
	avg_min_dist_shore: float
	failed_scenario_ids: Set[int]


def _extract_int(pattern: str, text: str, label: str) -> int:
	match = re.search(pattern, text, flags=re.MULTILINE)
	if not match:
		raise ValueError(f"Could not parse {label}")
	return int(match.group(1))


def _extract_float(pattern: str, text: str, label: str) -> float:
	match = re.search(pattern, text, flags=re.MULTILINE)
	if not match:
		raise ValueError(f"Could not parse {label}")
	return float(match.group(1))


def parse_report(path: Path) -> ReportData:
	text = path.read_text(encoding="utf-8")

	total_missions = _extract_int(r"^Total missions:\s*(\d+)\s*$", text, "total missions")

	success_match = re.search(
		r"^Mission success rate:\s*(\d+)/(\d+)\s*\(([\d.]+)%\)\s*$",
		text,
		flags=re.MULTILINE,
	)
	if not success_match:
		raise ValueError("Could not parse mission success rate")
	success_num = int(success_match.group(1))
	success_den = int(success_match.group(2))
	success_rate_pct = float(success_match.group(3))

	fail_collision_ts = _extract_int(r"^-\s*collision with TS:\s*(\d+)\s*$", text, "collision with TS")
	fail_collision_shore = _extract_int(r"^-\s*collision with shore:\s*(\d+)\s*$", text, "collision with shore")
	fail_target_not_reached = _extract_int(r"^-\s*target not reached:\s*(\d+)\s*$", text, "target not reached")
	fail_corridor_exceeded = _extract_int(r"^-\s*corridor exceeded:\s*(\d+)\s*$", text, "corridor exceeded")

	avg_travel_distance = _extract_float(
		r"^-\s*average travel distance \[m\]:\s*([-+]?\d*\.?\d+)\s*$",
		text,
		"average travel distance",
	)
	avg_min_dist_ts = _extract_float(
		r"^-\s*average minimum distance to TS \[m\]:\s*([-+]?\d*\.?\d+)\s*$",
		text,
		"average minimum distance to TS",
	)
	avg_min_dist_shore = _extract_float(
		r"^-\s*average minimum distance to shore \[m\]:\s*([-+]?\d*\.?\d+)\s*$",
		text,
		"average minimum distance to shore",
	)

	# Capture scenario ids from failed mission filenames like: <name>_123.json
	failed_scenario_ids: Set[int] = set()
	for scenario_id in re.findall(r"_(\d+)\.json", text):
		failed_scenario_ids.add(int(scenario_id))

	name = path.parent.name if path.name == "report.txt" else path.stem

	return ReportData(
		name=name,
		path=path,
		total_missions=total_missions,
		success_num=success_num,
		success_den=success_den,
		success_rate_pct=success_rate_pct,
		fail_collision_ts=fail_collision_ts,
		fail_collision_shore=fail_collision_shore,
		fail_target_not_reached=fail_target_not_reached,
		fail_corridor_exceeded=fail_corridor_exceeded,
		avg_travel_distance=avg_travel_distance,
		avg_min_dist_ts=avg_min_dist_ts,
		avg_min_dist_shore=avg_min_dist_shore,
		failed_scenario_ids=failed_scenario_ids,
	)


def resolve_report_path(raw_path: str) -> Path:
	p = Path(raw_path)
	if p.is_dir():
		candidate = p / "report.txt"
		if not candidate.exists():
			raise FileNotFoundError(f"No report.txt found in directory: {p}")
		return candidate
	if not p.exists():
		raise FileNotFoundError(f"Path does not exist: {p}")
	return p


def print_scenario_to_config_map(reports: List[ReportData]) -> None:
	print("\n=== Failed Scenarios Across Configurations ===")

	scenario_to_reports: Dict[int, List[str]] = {}
	for report in reports:
		for scenario_id in report.failed_scenario_ids:
			scenario_to_reports.setdefault(scenario_id, []).append(report.name)

	if not scenario_to_reports:
		print("No failed scenarios found in the provided reports.")
		return

	for scenario_id in sorted(scenario_to_reports):
		configs = sorted(scenario_to_reports[scenario_id])
		print(f"scenario {scenario_id}: {', '.join(configs)}")


def main() -> None:
	reports = [
		"Z:\\dev\\aurora_ferry\\sim_data\\01_07_26_move_p_f_after_8_iter\\report.txt",
		"Z:\\dev\\aurora_ferry\\sim_data\\01_07_26_dchi_7_5\\report.txt",
		"Z:\\dev\\aurora_ferry\\sim_data\\01_07_26_colregs_2\\report.txt",
		"Z:\\dev\\aurora_ferry\\sim_data\\01_07_26_colregs_3\\report.txt",
		"Z:\\dev\\aurora_ferry\\sim_data\\01_07_26_v_min_0_5\\report.txt",
		"Z:\\dev\\aurora_ferry\\sim_data\\30_06_26_1\\report.txt",
    ]
	report_paths = [resolve_report_path(p) for p in reports]
	parsed_reports = [parse_report(p) for p in report_paths]

	print_scenario_to_config_map(parsed_reports)


if __name__ == "__main__":
	main()
