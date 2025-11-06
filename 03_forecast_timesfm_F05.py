import os, numpy as np, pandas as pd, timesfm
from pathlib import Path

root = os.getcwd()
pred_dir = Path(f"{root}\\prediction\\F05")
os.makedirs(pred_dir, exist_ok=True)

df = pd.read_csv(pred_dir / "01_F05_HI_full.csv", index_col=0).sort_index()
N_CONTEXT, N_HORIZON = 65, 16
MA_LIST = ["MA05", "MA10", "MA20", "MA30", "MA40", "MA50"]

# 初始化模型一次即可重複使用
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

for MA in MA_LIST:
    y_all = df[MA].astype(float).values
    y_ctx = y_all[:N_CONTEXT]
    pred, _ = model.forecast(horizon=N_HORIZON, inputs=[y_ctx])
    pred = pred[0]
    future_flights = df.index[N_CONTEXT:N_CONTEXT+N_HORIZON]
    out = pd.DataFrame({"flight": future_flights, "TimesFM_pred": pred})
    out.to_csv(pred_dir / f"07_F05_TimesFM_{MA}_pred{N_HORIZON}.csv", index=False)
    print(f"[TimesFM] Saved: 07_F05_TimesFM_{MA}_pred{N_HORIZON}.csv")
