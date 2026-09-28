import pandas as pd

from ..config import N_HORIZON, SCALE_DEFAULT
from ..forecast.common import mode_suffix, scale_suffix

# "TTMs" (09) is retired but kept so pre-2026-09 outputs stay readable.
MODEL_FILE_ID = {
    "TimesFM": "07", "Chronos2": "08", "TTMs": "09", "PatchTSTFM": "14", "MOIRAI2": "15",
}
# The models this study reports, in the order tables should list them.
ACTIVE_MODELS = ["TimesFM", "Chronos2", "PatchTSTFM", "MOIRAI2"]


def read_classic(out_dir, ava, ma, scale: str = SCALE_DEFAULT):
    path = out_dir / (
        f"03_{ava}_traditional_{ma}_pred{N_HORIZON}{scale_suffix(scale)}_all.csv"
    )
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0)


def read_pretrained(out_dir, ava, name, ma, mode="zero-shot", scale: str = SCALE_DEFAULT):
    """The CV-scale forecast for one model, or None if that run is missing.

    Every forecaster writes its prediction back-transformed to CV (see
    forecast.common.save_pred), so the first column is directly comparable across
    models and across modelling scales.
    """
    path = out_dir / (
        f"{MODEL_FILE_ID[name]}_{ava}_{name}_{ma}_pred{N_HORIZON}"
        f"{scale_suffix(scale)}{mode_suffix(mode)}.csv"
    )
    if not path.exists():
        return None
    df = pd.read_csv(path).set_index("flight")
    return df.iloc[:, 0].values
