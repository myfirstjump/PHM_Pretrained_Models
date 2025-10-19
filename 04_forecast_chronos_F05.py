# 04_forecast_chronos_F05.py
import os, numpy as np, pandas as pd, torch
from pathlib import Path
from chronos import ChronosPipeline

root = os.getcwd()
pred_dir = Path(f"{root}\\prediction\\F05")
os.makedirs(pred_dir, exist_ok=True)

# === 讀完整 HI (CV, MA50)，切分前48後16 ===
df = pd.read_csv(pred_dir / "F05_HI_full.csv", index_col=0)
df = df.sort_index()
y_all = df["MA50"].astype(float).values
N_CONTEXT, N_HORIZON = 48, 16
y_ctx = y_all[:N_CONTEXT]

device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype  = torch.bfloat16 if (device.startswith("cuda") and torch.cuda.is_bf16_supported()) else (
         torch.float16 if device.startswith("cuda") else torch.float32)

pipe = ChronosPipeline.from_pretrained("amazon/chronos-t5-small", device_map=device, dtype=dtype)
ctx_t = torch.tensor(y_ctx, dtype=torch.float32).unsqueeze(0)
samples = pipe.predict(context=ctx_t, prediction_length=N_HORIZON, num_samples=200)
if samples.ndim == 3:
    samples = samples[:, 0, :]
pred_p50 = torch.quantile(samples, 0.5, dim=0).cpu().numpy()

# === 存檔 ===
future_flights = df.index[N_CONTEXT:N_CONTEXT+N_HORIZON]
out = pd.DataFrame({"flight": future_flights, "Chronos_pred": pred_p50})
out.to_csv(pred_dir / "F05_Chronos_pred16.csv", index=False)
print(f"[04] Saved: {pred_dir / 'F05_Chronos_pred16.csv'}")
