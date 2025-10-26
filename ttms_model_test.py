# ttm_model_test.py
import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
import torch
import matplotlib.pyplot as plt

from tsfm_public import TinyTimeMixerForPrediction, TimeSeriesForecastingPipeline
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor

# ==== 1) 造一段「多變量」序列（TTM 是多變量友善）====
np.random.seed(42)
L = 512                # context length，需與所選模型規格一致（512/1024/1536）
H = 96                 # horizon（<= 720，視模型 revision）
t = np.arange(L + H)

# 兩個通道：sin / cos + 雜訊
ch1 = 2.0*np.sin(2*np.pi*t/48) + 0.05*t + 0.3*np.random.randn(L + H)
ch2 = 1.0*np.cos(2*np.pi*t/24) + 0.02*t + 0.3*np.random.randn(L + H)

df = pd.DataFrame({
    "timestamp": pd.date_range("2024-01-01", periods=L+H, freq="h"),
    "id": "demo",
    "y1": ch1,
    "y2": ch2
})

# ==== 2) 建立 Preprocessor：每個 channel 做 z-score ====
tsp = TimeSeriesPreprocessor(
    timestamp_column="timestamp",
    id_columns=["id"],
    target_columns=["y1", "y2"],     # 多通道
    control_columns=[],              # 此例不加外生變數
    context_length=L,
    prediction_length=H,
    scaling=False,
    encode_categorical=False,
    scaler_type="standard",
)
# 只用歷史段訓練 scaler
tsp = tsp.train(df.iloc[:L])

# 預處理（產生 Dataset 給 Pipeline 用）
ds = tsp.preprocess(df)

# ==== 3) 載入 TTM（選你預先下載的本地資料夾或 HF Hub 上的路徑）====
device = "cuda" if torch.cuda.is_available() else "cpu"
model = TinyTimeMixerForPrediction.from_pretrained(
    "ibm-granite/granite-timeseries-ttm-r2",
    revision="512-192-r2",  # 要與 L/H 相容（可改成你的目標 revision）
)

pipe = TimeSeriesForecastingPipeline(
    model=model,
    device=device,
    feature_extractor=tsp,
    batch_size=32,
)

# ==== 4) 產生預測 ====
pred_df = pipe.predict(ds)   # 回傳展平的 DataFrame，含每通道的未來預測


def _infer_freq_from_ts(ts: pd.Series) -> pd.DateOffset:
    """根據 pred_df['timestamp'] 推斷頻率，永遠回傳 DateOffset。"""
    ts = pd.to_datetime(ts)
    # 先嘗試由規則頻率推斷
    freq = pd.infer_freq(ts)
    if freq is not None:
        return to_offset(freq)  # 轉成 DateOffset
    # 推不出就取第一個差值
    if len(ts) >= 2:
        delta = ts.iloc[1] - ts.iloc[0]
        return to_offset(delta)
    # 實在推不出：預設 1 小時
    return to_offset("h")

def extract_forecast_from_listdf(
    pred_df: pd.DataFrame,
    target: str = "y1",
    mode: str = "first_window",   # 'first_window' 或 'rolling_1step'
    dropna: bool = True
):
    """
    pred_df 結構需包含：
      - 'timestamp'：各 rolling 視窗的預測起點（forecast origin）
      - f'{target}_prediction'：list（長度約 = horizon），每列一串未來步預測
    回傳：DataFrame[['timestamp','y_pred']]
    """
    pred_col = f"{target}_prediction"
    assert pred_col in pred_df.columns, f"找不到欄位: {pred_col}"

    # 轉成 2D：shape [n_rows, horizon]（不足處以 NaN 補）
    M = pd.DataFrame(pred_df[pred_col].tolist())
    ts = pd.to_datetime(pred_df["timestamp"])
    off = _infer_freq_from_ts(ts)

    if mode == "first_window":
        # 取第一列整段多步預測
        first_row = M.iloc[0].to_numpy(dtype=float)
        # 依 horizon 產生未來 H 個時間點：從「第一列的 forecast origin + 1 step」開始
        H = len(first_row)
        start = ts.iloc[0] + off
        fut_ts = pd.date_range(start=start, periods=H, freq=off)
        yhat = first_row

    elif mode == "rolling_1step":
        # 每列的一步預測（第 0 個）
        yhat = M.iloc[:, 0].to_numpy(dtype=float)
        # 將一步預測對齊到「origin + 1*freq」
        fut_ts = ts + off
    else:
        raise ValueError("mode 僅支援 'first_window' 或 'rolling_1step'")

    # 組裝輸出（先確保長度一致）
    assert len(fut_ts) == len(yhat), f"長度不齊：len(ts)={len(fut_ts)}, len(yhat)={len(yhat)}"
    out = pd.DataFrame({"timestamp": fut_ts, "y_pred": yhat})

    if dropna:
        out = out.dropna(subset=["y_pred"])

    return out.sort_values("timestamp").reset_index(drop=True)

# === 把真值併回預測（用 timestamp 對齊） ===
def attach_ground_truth(pred_out: pd.DataFrame, raw_df: pd.DataFrame, target: str = "y1"):
    gt = raw_df.loc[:, ["timestamp", target]].copy()
    out = pred_out.merge(gt, on="timestamp", how="left")
    return out.rename(columns={target: "y_true"})

# 取得「第一段 H 步」預測
pred_y1_seg = extract_forecast_from_listdf(pred_df, target="y1", mode="first_window")
pred_y1_seg = attach_ground_truth(pred_y1_seg, df, target="y1")

# 逐時的一步預測（rolling one-step ahead）
pred_y1_1s = extract_forecast_from_listdf(pred_df, target="y1", mode="rolling_1step")
pred_y1_1s = attach_ground_truth(pred_y1_1s, df, target="y1")


# ==== 5) 視覺化（以 y1 為例）====
import matplotlib.pyplot as plt

# 情境 A：第 1 段 H-step 預測 vs 真值
plt.figure(figsize=(10,4))
plt.plot(pred_y1_seg["timestamp"], pred_y1_seg["y_true"], label="Ground Truth y1", linestyle="--")
plt.plot(pred_y1_seg["timestamp"], pred_y1_seg["y_pred"], label="TTM Forecast (first window)")
plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

# 情境 B：rolling one-step 預測
plt.figure(figsize=(10,4))
plt.plot(pred_y1_1s["timestamp"], pred_y1_1s["y_true"], label="Ground Truth y1", linestyle="--")
plt.plot(pred_y1_1s["timestamp"], pred_y1_1s["y_pred"], label="TTM 1-step-ahead (rolling)")
plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()
