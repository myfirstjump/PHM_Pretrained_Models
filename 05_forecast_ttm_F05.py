import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tsfm_public import TinyTimeMixerForPrediction, TimeSeriesForecastingPipeline
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor

root = os.getcwd()
pred_dir = Path(f"{root}\\prediction\\F05")
pred_dir.mkdir(parents=True, exist_ok=True)

CSV_INPUT = pred_dir / "01_F05_HI_full.csv"
N_CONTEXT, N_HORIZON = 65, 16
MA_LIST = ["MA05", "MA10", "MA20", "MA30", "MA40", "MA50"]
SERIES_ID = "F05"

df_full = pd.read_csv(CSV_INPUT, index_col=0).sort_index()
future_flights = df_full.index.values[N_CONTEXT:N_CONTEXT + N_HORIZON]

device = "cuda" if torch.cuda.is_available() else "cpu"
model = TinyTimeMixerForPrediction.from_pretrained(
    "ibm-granite/granite-timeseries-ttm-r2", revision="512-192-r2"
)

def make_ctx_df(y_ctx):
    return pd.DataFrame({
        "timestamp": pd.date_range("2000-01-01", periods=N_CONTEXT, freq="h"),
        "id": SERIES_ID,
        "y": y_ctx.astype(float),
    })

def extract_first_window(pred_df, target, horizon):
    col = f"{target}_prediction"
    full = pd.Series(pred_df[col].iloc[0]).to_numpy(dtype=float)
    return full[:horizon]

for MA in MA_LIST:
    y_all = df_full[MA].astype(float).values
    y_ctx = y_all[:N_CONTEXT]
    df_ctx = make_ctx_df(y_ctx)

    tsp = TimeSeriesPreprocessor(
        timestamp_column="timestamp", id_columns=["id"], target_columns=["y"],
        context_length=N_CONTEXT, prediction_length=N_HORIZON, scaling=True
    ).train(df_ctx)

    ds = tsp.preprocess(df_ctx)
    pipe = TimeSeriesForecastingPipeline(model=model, device=device, feature_extractor=tsp, batch_size=1)
    pred_df = pipe.predict(ds)
    y_pred = extract_first_window(pred_df, "y", N_HORIZON)

    mu, sigma = float(np.mean(y_ctx)), float(np.std(y_ctx, ddof=0)) or 1.0
    y_pred = y_pred * sigma + mu

    out = pd.DataFrame({"flight": future_flights, "TTMs_pred": y_pred})
    out.to_csv(pred_dir / f"09_F05_TTMs_{MA}_pred{N_HORIZON}.csv", index=False)
    print(f"[TTMs] Saved: 09_F05_TTMs_{MA}_pred{N_HORIZON}.csv")
