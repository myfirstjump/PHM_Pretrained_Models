# 03_forecast_timesfm_F05.py
import os, numpy as np, pandas as pd, timesfm
from pathlib import Path

root = os.getcwd()
pred_dir = Path(f"{root}\\prediction\\F05")
os.makedirs(pred_dir, exist_ok=True)

# === 讀完整 HI (CV, MA50)，切分前48後16 ===
df = pd.read_csv(pred_dir / "F05_HI_full.csv", index_col=0)
df = df.sort_index()
y_all = df["MA50"].astype(float).values
N_CONTEXT, N_HORIZON = 48, 16
y_ctx = y_all[:N_CONTEXT]

# === TimesFM ===
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
cfg = timesfm.ForecastConfig(
    max_context=1024, max_horizon=256,
    normalize_inputs=True,
    use_continuous_quantile_head=True,
    force_flip_invariance=True,
    infer_is_positive=False,
    fix_quantile_crossing=True,
)
model.compile(cfg)

pred, _ = model.forecast(horizon=N_HORIZON, inputs=[y_ctx])
pred = pred[0]

# === 存檔 ===
future_flights = df.index[N_CONTEXT:N_CONTEXT+N_HORIZON]
out = pd.DataFrame({"flight": future_flights, "TimesFM_pred": pred})
out.to_csv(pred_dir / "F05_TimesFM_pred16.csv", index=False)
print(f"[03] Saved: {pred_dir / 'F05_TimesFM_pred16.csv'}")
