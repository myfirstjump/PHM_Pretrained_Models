# IBM Granite TinyTimeMixer zero-shot forecast (formerly 05_forecast_ttm_F05.py).
#
# The pipeline must receive the RAW context dataframe: with a feature_extractor
# attached it scales inputs itself and inverse-scales outputs. Feeding it
# tsp.preprocess(df) (as the original script did) scales twice and returns
# z-space values; the old manual mu/sigma rescale compensated, and TTM's own
# per-instance normalization absorbed the rest, so old numbers were ~correct —
# but this direct path is the version-stable one.
import numpy as np
import pandas as pd

from ..config import AVA, MA_LIST, N_CONTEXT, N_HORIZON
from .common import context_and_future, load_hi_full, save_pred


def _make_ctx_df(y_ctx, series_id):
    return pd.DataFrame({
        "timestamp": pd.date_range("2000-01-01", periods=N_CONTEXT, freq="h"),
        "id": series_id,
        "y": y_ctx.astype(float),
    })


def run(ava: str = AVA, ma_list=None, mode: str = "zero-shot", checkpoint: str | None = None) -> None:
    import torch
    from tsfm_public import TimeSeriesForecastingPipeline, TinyTimeMixerForPrediction
    from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor

    df = load_hi_full(ava)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if checkpoint:
        model = TinyTimeMixerForPrediction.from_pretrained(checkpoint)
    else:
        model = TinyTimeMixerForPrediction.from_pretrained(
            "ibm-granite/granite-timeseries-ttm-r2", revision="512-192-r2"
        )

    for ma in (ma_list or MA_LIST):
        y_ctx, future_flights = context_and_future(df, ma)
        df_ctx = _make_ctx_df(y_ctx, ava)

        tsp = TimeSeriesPreprocessor(
            timestamp_column="timestamp", id_columns=["id"], target_columns=["y"],
            context_length=N_CONTEXT, prediction_length=N_HORIZON, scaling=True,
        ).train(df_ctx)

        pipe = TimeSeriesForecastingPipeline(
            model=model, device=device, feature_extractor=tsp, batch_size=1
        )
        pred_df = pipe.predict(df_ctx)
        y_pred = pd.Series(pred_df["y_prediction"].iloc[0]).to_numpy(dtype=float)[:N_HORIZON]

        ctx_mu = float(np.mean(y_ctx))
        ctx_sd = float(np.std(y_ctx, ddof=0)) or 1.0
        if abs(np.mean(y_pred) - ctx_mu) > 5 * ctx_sd:
            print(f"[TTMs][WARN] {ma}: prediction scale far from context "
                  f"(pred mean {np.mean(y_pred):.4f} vs ctx mean {ctx_mu:.4f}).")

        save_pred(ava, "09", "TTMs", ma, mode, future_flights, y_pred)
