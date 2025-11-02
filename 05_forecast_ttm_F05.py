# 05_forecast_ttm_F05.py
# 用 IBM Granite TTM 對 F05 的 MA20/30/40/50 進行 first-window 多步預測
# context=42, horizon=6；輸出檔名對齊你現有的 TimesFM/Chronos 腳本

import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from pandas.tseries.frequencies import to_offset
from tsfm_public import TinyTimeMixerForPrediction, TimeSeriesForecastingPipeline
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor

# === 路徑與參數 ===
root = os.getcwd()
pred_dir = Path(f"{root}\\prediction\\F05")
pred_dir.mkdir(parents=True, exist_ok=True)

CSV_INPUT = pred_dir / "01_F05_HI_full.csv"     # 與 TimesFM/Chronos 相同的輸入
N_CONTEXT, N_HORIZON = 65, 16                 # 與 TimesFM/Chronos 相同
MA_LIST = ["MA30"]   # 逐一輸出
SERIES_ID = "F05"

# === 輔助：從 TTM pipeline 的 list 預測欄位取「第一段 H 步」 ===
def _infer_freq_from_ts(ts: pd.Series) -> pd.DateOffset:
    ts = pd.to_datetime(ts)
    freq = pd.infer_freq(ts)
    if freq is not None:
        from pandas.tseries.frequencies import to_offset
        return to_offset(freq)
    if len(ts) >= 2:
        return to_offset(ts.iloc[1] - ts.iloc[0])
    return to_offset("h")

def extract_first_window(pred_df: pd.DataFrame, target: str, horizon: int) -> np.ndarray:
    pred_col = f"{target}_prediction"
    if pred_col not in pred_df.columns:
        raise KeyError(f"TTM 輸出中找不到欄位: {pred_col}")
    full = pd.Series(pred_df[pred_col].iloc[0]).to_numpy(dtype=float)  # 通常長度=192
    return full[:horizon]  # 只取前 horizon=6


# === 載入資料 ===
df_full = pd.read_csv(CSV_INPUT, index_col=0).sort_index()
# 之後會用同一組 future_flights 當輸出 index
future_flights = df_full.index.values[N_CONTEXT:N_CONTEXT + N_HORIZON]

# === 準備 TTM 模型與推論管線（一次載入，重複用於 4 條 MA 序列） ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model = TinyTimeMixerForPrediction.from_pretrained(
    "ibm-granite/granite-timeseries-ttm-r2",
    revision="512-192-r2",   # 支援更長 context/horizon；此處實際只用 42/6
)

# 注意：TTM 需要 timestamp；以「等間隔虛擬時間」餵入（不影響相對位置）
def make_ctx_df(y_ctx: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": pd.date_range("2000-01-01", periods=N_CONTEXT, freq="h"),
        "id": SERIES_ID,
        "y": y_ctx.astype(float),
    })

# === 逐一輸出 MA20 / MA30 / MA40 / MA50 ===
for MA in MA_LIST:
    # 取前 42 點作為模型輸入
    y_all = df_full[MA].astype(float).values
    y_ctx = y_all[:N_CONTEXT]

    df_ctx = make_ctx_df(y_ctx)

    # 建立 Preprocessor（單通道 y），只用 context 訓練 scaler
    tsp = TimeSeriesPreprocessor(
        timestamp_column="timestamp",
        id_columns=["id"],
        target_columns=["y"],
        control_columns=[],
        context_length=N_CONTEXT,
        prediction_length=N_HORIZON,
        scaling=True,
        encode_categorical=False,
        scaler_type="standard",
    )
    tsp = tsp.train(df_ctx)          # 只以歷史段訓練 scaler
    ds = tsp.preprocess(df_ctx)      # 產生一筆樣本（first-window）

    pipe = TimeSeriesForecastingPipeline(
        model=model,
        device=device,
        feature_extractor=tsp,
        batch_size=1,
    )

    # 推論
    pred_df = pipe.predict(ds)   # 各欄位包含 'y_prediction'（list 長度=H）
    y_pred = extract_first_window(pred_df, target="y", horizon=N_HORIZON)

    # 反標準化：用 context 的均值/標準差還原
    mu = float(np.mean(y_ctx))
    sigma = float(np.std(y_ctx, ddof=0)) or 1.0
    y_pred = y_pred * sigma + mu

    # 資料長度安全檢查
    assert len(future_flights) == len(y_pred) == N_HORIZON

    # 存檔（依 TimesFM/Chronos 的命名習慣）
    print(future_flights, len(future_flights))
    print(y_pred, len(y_pred))
    out = pd.DataFrame({"flight": future_flights, "TTMs_pred": y_pred})
    out_name = pred_dir / f"09_F05_TTMs_{MA}_pred{N_HORIZON}.csv"  # 與你現有腳本相同的檔名風格
    out.to_csv(out_name, index=False)
    print(f"[TTMs] Saved: {out_name}")
