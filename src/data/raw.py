# Raw flight CSV parsing (formerly py_modules/Step1_Data_Preprocessing.py + common_utils).
import re
from pathlib import Path

import pandas as pd

# Filenames look like F05-01244.csv -> plane "F05", flight 1244.
FLIGHT_RE = re.compile(r"^(?P<plane>[A-Za-z0-9]+)-(?P<flight>\d{5})$")


def parse_flight_id(csv_path) -> tuple[str, int]:
    stem = Path(csv_path).stem
    m = FLIGHT_RE.match(stem)
    if not m:
        raise ValueError(f"Cannot parse plane/flight number from filename: {stem}")
    return m.group("plane"), int(m.group("flight"))


def flights_from_folder(folder_path) -> list[tuple[str, int, Path]]:
    """List flight CSVs in a folder as (plane, flight, path), sorted by flight number."""
    folder = Path(folder_path)
    items = []
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() not in (".csv", ".txt"):
            continue
        plane, flight = parse_flight_id(p)
        items.append((plane, flight, p))
    items.sort(key=lambda x: x[1])
    return items


def read_rawdata(csv_path):
    """Read one flight CSV -> (TO_Y, TO_Z, LD_Y, LD_Z) series.

    Row 0 holds the takeoff/landing segment lengths; data rows follow with
    columns Seq, T, C (airspeed), Y, Z. Segments are filtered to airspeed > 48 kt.
    """
    csv_path = Path(csv_path)
    plane_num, flight_num = parse_flight_id(csv_path)
    tag = f"{plane_num}-{flight_num:05d}"

    try:
        head = pd.read_csv(csv_path, header=None, usecols=[0, 1], nrows=1)
        TO_index = int(head.iloc[0, 0])
        LD_index = int(head.iloc[0, 1])
    except Exception as e:
        raise Exception(f"{tag}: error obtaining T/O & L/D data index, {e}")

    try:
        flight_df = pd.read_csv(csv_path, header=1)
    except Exception as e:
        raise Exception(f"{tag}: error reading rawdata, {e}")

    flight_TO_df = flight_df[: TO_index + 1]
    flight_LD_df = flight_df[-LD_index:] if LD_index > 0 else flight_df.iloc[0:0]

    try:
        flight_TO_df = flight_TO_df[flight_TO_df.C > 48]
        flight_LD_df = flight_LD_df[flight_LD_df.C > 48]
    except Exception as e:
        raise Exception(f"{tag}: error selecting data by airspeed, {e}")

    segments = []
    for seg in (flight_TO_df.Y, flight_TO_df.Z, flight_LD_df.Y, flight_LD_df.Z):
        seg = seg.reset_index(drop=True)
        seg.name = tag
        segments.append(seg)
    return tuple(segments)
