"""
Convert a huge csv file (~400MB) which contains AIS data between 2023-04-03 and 2023-04-13 in format:

timestamp	type_of_mobile	mmsi	lat	lon	nav_status	rot	sog	cog	heading	imo	callsign	name	ship_type	cargo_type	length	width	type_pos_device	draught	dest	eta	data_source	a	b	c	d	distance_m	timestamp_sec

into csv chunks of 30-minutes long with only the mandatory and optional columns mentioned in src/ais/config.yaml.

The method should take as input the path to the source csv and the path to the target folder for the output chunks (by default equal to the folder that contains the source csv file)
"""

import os
from typing import Any, Optional, cast

import pandas as pd
import yaml


def _first_existing_column(columns: list[str], available_columns: list[str]) -> Optional[str]:
	for column in columns:
		if column in available_columns:
			return column
	return None


def split_ais_csv(source_csv_path: str, target_folder: Optional[str] = None) -> None:
	"""
	Split AIS CSV into 30-minute chunks keeping only columns defined in src/ais/config.yaml.
	"""
	config_path = os.path.join(os.path.dirname(__file__), "..", "ais", "config.yaml")
	with open(config_path, "r", encoding="utf-8") as f:
		config = yaml.safe_load(f)

	source_csv_path = os.path.abspath(source_csv_path)
	if target_folder is None:
		target_folder = os.path.dirname(source_csv_path)
	target_folder = os.path.abspath(target_folder)
	os.makedirs(target_folder, exist_ok=True)

	header_columns = pd.read_csv(source_csv_path, nrows=0).columns.tolist()
	timestamp_sec_col = "timestamp_sec" if "timestamp_sec" in header_columns else None

	mandatory_cfg = config["mandatory"]
	optional_cfg = config["optional"]

	mandatory_map: dict[str, str] = {}
	for out_col, aliases in mandatory_cfg.items():
		selected = _first_existing_column(aliases, header_columns)
		if selected is not None:
			mandatory_map[out_col] = selected

	# Validate mandatory fields with simple position rule: lat/lon OR north/east.
	required_non_position = ["mmsi", "sog", "cog", "heading", "timestamp"]
	missing_required = [col for col in required_non_position if col not in mandatory_map]
	if missing_required:
		raise ValueError(f"Missing mandatory columns in source CSV: {missing_required}")

	has_latlon = "latitude" in mandatory_map and "longitude" in mandatory_map
	has_ne = "north" in mandatory_map and "east" in mandatory_map
	if not (has_latlon or has_ne):
		raise ValueError("Missing mandatory position columns: need latitude/longitude or north/east")

	optional_map: dict[str, tuple[Optional[str], Any]] = {}
	for out_col, spec in optional_cfg.items():
		selected = _first_existing_column(spec["columns"], header_columns)
		optional_map[out_col] = (selected, spec["default"])

	usecols = sorted(set(mandatory_map.values()) | {c for c, _ in optional_map.values() if c is not None})
	if timestamp_sec_col is not None:
		usecols.append(timestamp_sec_col)
	written_files: set[str] = set()

	for chunk in pd.read_csv(source_csv_path, usecols=usecols, chunksize=200_000):
		out = pd.DataFrame()

		for out_col, in_col in mandatory_map.items():
			if out_col == "latitude":
				out["lat"] = chunk[in_col]
			elif out_col == "longitude":
				out["lon"] = chunk[in_col]
			else:
				out[out_col] = chunk[in_col]

		for out_col, (in_col, default_value) in optional_map.items():
			if in_col is not None:
				out[out_col] = chunk[in_col].fillna(default_value)
			else:
				out[out_col] = [default_value] * len(out)

		if timestamp_sec_col is not None:
			out["timestamp_sec"] = pd.to_numeric(chunk[timestamp_sec_col], errors="coerce")

		out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
		out = out.dropna(subset=["timestamp"])
		if out.empty:
			continue

		out["_window_start"] = out["timestamp"].dt.floor("30min")

		for window_start, group in out.groupby("_window_start"):
			window_ts = pd.Timestamp(cast(Any, window_start))
			filename = window_ts.strftime("%Y_%m_%d_%H_%M") + ".csv"
			file_path = os.path.join(target_folder, filename)
			group = group.drop(columns=["_window_start"])

			write_header = file_path not in written_files and not os.path.exists(file_path)
			group.to_csv(file_path, mode="a", index=False, header=write_header)
			written_files.add(file_path)


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Split AIS CSV into 30-minute chunks")
	parser.add_argument("source_csv_path", type=str)
	parser.add_argument("target_folder", nargs="?", default=None)
	args = parser.parse_args()

	split_ais_csv(args.source_csv_path, args.target_folder)