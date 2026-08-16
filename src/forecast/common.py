import pandas as pd

from ..config import AVA, MODE_SUFFIX, N_CONTEXT, N_HORIZON, pred_dir


def load_hi_full(ava: str = AVA) -> pd.DataFrame:
    path = pred_dir(ava) / f"01_{ava}_HI_full.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run `python main.py build-hi` first.")
    return pd.read_csv(path, index_col=0).sort_index()


def context_and_future(df: pd.DataFrame, ma: str):
    y_all = df[ma].astype(float).values
    y_ctx = y_all[:N_CONTEXT]
    future_flights = df.index[N_CONTEXT:N_CONTEXT + N_HORIZON]
    return y_ctx, future_flights


def mode_suffix(mode: str) -> str:
    try:
        return MODE_SUFFIX[mode]
    except KeyError:
        raise ValueError(f"Unknown forecast mode: {mode!r} (expected {list(MODE_SUFFIX)})")


def save_pred(ava, file_id, model_name, ma, mode, future_flights, pred):
    out = pred_dir(ava) / (
        f"{file_id}_{ava}_{model_name}_{ma}_pred{N_HORIZON}{mode_suffix(mode)}.csv"
    )
    pd.DataFrame({"flight": future_flights, f"{model_name}_pred": pred}).to_csv(out, index=False)
    print(f"[{model_name}] Saved: {out.name}")
