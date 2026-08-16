import pandas as pd

from ..config import N_HORIZON
from ..forecast.common import mode_suffix

MODEL_FILE_ID = {"TimesFM": "07", "Chronos": "08", "TTMs": "09"}


def read_classic(out_dir, ava, ma):
    path = out_dir / f"03_{ava}_traditional_{ma}_pred{N_HORIZON}_all.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0)


def read_pretrained(out_dir, ava, name, ma, mode="zero-shot"):
    path = out_dir / (
        f"{MODEL_FILE_ID[name]}_{ava}_{name}_{ma}_pred{N_HORIZON}{mode_suffix(mode)}.csv"
    )
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df.set_index("flight").iloc[:, 0].values
