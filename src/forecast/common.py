import numpy as np
import pandas as pd

from ..config import (
    AVA, MODE_SUFFIX, N_CONTEXT, N_HORIZON, SCALE_DEFAULT, SCALES, ma_column, pred_dir,
)


def load_hi_full(ava: str = AVA) -> pd.DataFrame:
    path = pred_dir(ava) / f"01_{ava}_HI_full.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run `python main.py build-hi` first.")
    return pd.read_csv(path, index_col=0).sort_index()


def context_and_future(df: pd.DataFrame, ma: str, scale: str = SCALE_DEFAULT):
    """(context values on `scale`, the flight ids the horizon covers).

    `ma` is "raw" or an MA name; `scale` picks the CV or logit family of columns.
    """
    col = ma_column(ma, scale)
    if col not in df.columns:
        raise KeyError(
            f"{col!r} missing from the HI series — it has {list(df.columns)}. "
            f"Re-run `python main.py build-hi` (logit columns were added in the phase 2 rebuild)."
        )
    y_all = df[col].astype(float).values
    y_ctx = y_all[:N_CONTEXT]
    future_flights = df.index[N_CONTEXT:N_CONTEXT + N_HORIZON]
    return y_ctx, future_flights


def to_cv(values, scale: str = SCALE_DEFAULT) -> np.ndarray:
    """Bring a forecast back to the probability scale for reporting.

    Modelling happens on the logit scale but every saved prediction is a CV value,
    so results stay comparable across scales and with the pre-2026-09 CV-scale runs.
    The sigmoid also guarantees [0, 1]; a CV-scale forecaster can overshoot the
    bounds, which is one more reason the logit path is the default.
    """
    arr = np.asarray(values, dtype=float)
    if scale == "cv":
        return arr
    if scale == "logit":
        return 1.0 / (1.0 + np.exp(-arr))
    raise ValueError(f"Unknown scale {scale!r} (expected one of {SCALES})")


def mode_suffix(mode: str) -> str:
    try:
        return MODE_SUFFIX[mode]
    except KeyError:
        raise ValueError(f"Unknown forecast mode: {mode!r} (expected {list(MODE_SUFFIX)})")


def scale_suffix(scale: str = SCALE_DEFAULT) -> str:
    """CV keeps the bare legacy filename; logit runs are tagged so both can coexist."""
    if scale not in SCALES:
        raise ValueError(f"Unknown scale {scale!r} (expected one of {SCALES})")
    return "" if scale == "cv" else f"_{scale}"


def save_pred(ava, file_id, model_name, ma, mode, future_flights, pred,
              scale: str = SCALE_DEFAULT):
    """Write one forecast. `pred` is on `scale`; the file is always CV."""
    out = pred_dir(ava) / (
        f"{file_id}_{ava}_{model_name}_{ma}_pred{N_HORIZON}"
        f"{scale_suffix(scale)}{mode_suffix(mode)}.csv"
    )
    cv = to_cv(pred, scale)
    frame = {"flight": future_flights, f"{model_name}_pred": cv}
    if scale != "cv":
        # keep the raw model output so the back-transform stays auditable
        frame[f"{model_name}_pred_{scale}"] = np.asarray(pred, dtype=float)
    pd.DataFrame(frame).to_csv(out, index=False)
    print(f"[{model_name}] Saved: {out.name}")
