# Statistical feature extraction (formerly py_modules/Step2_Feature_Extraction.py + common_utils).
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from ..config import FEATURES_DIR, TESTING_DIR, TRAINING_FAULTY_DIR, TRAINING_HEALTHY_DIR
from .raw import flights_from_folder, read_rawdata

SEGMENTS = ["TO-Y", "TO-Z", "LD-Y", "LD-Z"]
SEGMENT_FEATURES = [
    "rms", "mean", "std", "peak2peak", "kurtosis", "skewness",
    "crest_indicator", "clearance_indicator", "shape_indicator", "impulse_indicator",
]

# 40 columns: 10 features x 4 segments, numbered to match the original CSV headers.
FEATURE_NAMES = [
    f"{i * 10 + j + 1}-{seg}-{feat}"
    for i, seg in enumerate(SEGMENTS)
    for j, feat in enumerate(SEGMENT_FEATURES)
]


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """10 statistical features per column. Each column is one signal (one flight segment).

    NOTE: crest/clearance/impulse use abs(data.max()) rather than max(|data|),
    matching the formulas the deployed HI pipeline was trained with; changing
    them requires retraining the HI model.
    """
    features = np.zeros((df.shape[1], len(SEGMENT_FEATURES)))

    for i, col in enumerate(df.columns):
        data = df[col].dropna().to_numpy(dtype=float)
        if len(data) == 0:
            continue

        rms = np.sqrt(np.mean(data ** 2))
        features[i, 0] = rms
        features[i, 1] = np.mean(data)
        features[i, 2] = np.std(data)
        features[i, 3] = data.max() - data.min()
        features[i, 4] = kurtosis(data, fisher=False)
        features[i, 5] = skew(data)
        features[i, 6] = abs(data.max()) / rms
        features[i, 7] = abs(data.max()) / (np.mean(np.sqrt(np.abs(data))) ** 2)
        features[i, 8] = rms / np.mean(np.abs(data))
        features[i, 9] = abs(data.max()) / np.mean(np.abs(data))

    return pd.DataFrame(features, index=df.columns, columns=SEGMENT_FEATURES)


def extract_flight_features(csv_path, flight_tag: str | None = None) -> pd.DataFrame:
    """One flight CSV -> a (1, 40) feature row."""
    segments = read_rawdata(csv_path)
    feats = []
    for seg in segments:
        colname = seg.name if getattr(seg, "name", None) else (flight_tag or "flight")
        seg_df = pd.DataFrame({colname: seg})
        feats.append(extract_features(seg_df).reset_index(drop=True))
    row = pd.concat(feats, axis="columns")
    row.columns = FEATURE_NAMES
    return row


def build_feature_table(folder_path, with_id: bool = False) -> pd.DataFrame:
    """All flights in a folder -> (n_flights, 40) table indexed by flight number.

    with_id=True prepends `plane` and `flight` columns so downstream code can
    split by aircraft (leave-one-aircraft-out). The 40 feature columns keep
    their order either way, so FEAT_IDXS still refers to the right features
    once the id columns are dropped.
    """
    rows, planes, flights = [], [], []
    for plane, flight, fp in flights_from_folder(folder_path):
        rows.append(extract_flight_features(fp))
        planes.append(plane)
        flights.append(flight)
    table = pd.concat(rows, axis="rows").reset_index(drop=True)
    if with_id:
        table.insert(0, "flight", flights)
        table.insert(0, "plane", planes)
    table.index = pd.Index(flights, name="flight")
    return table


def prepare_feature_csvs() -> None:
    """Raw training/testing folders -> myfeature/*.csv (replaces Step1 + Step2).

    Tables carry `plane`/`flight` id columns; leave-one-aircraft-out training
    and the F18-F20 validation split both need to know which flight is which.
    """
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [
        (TRAINING_HEALTHY_DIR, "healthyAllFeatures.csv"),
        (TRAINING_FAULTY_DIR, "faultyAllFeatures.csv"),
        (TESTING_DIR, "testingAllFeatures.csv"),
    ]
    for folder, out_name in jobs:
        table = build_feature_table(folder, with_id=True)
        out = FEATURES_DIR / out_name
        table.reset_index(drop=True).to_csv(out, index=False)
        n_ava = table["plane"].nunique()
        print(f"[features] {folder.name}: {table.shape[0]} flights / {n_ava} aircraft -> {out}")
