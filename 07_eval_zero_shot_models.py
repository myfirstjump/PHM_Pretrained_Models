# 07_eval_zero_shot_models.py
# 目的：
# - 對 MA05~MA50 每一條都做一次「context 65 → horizon 16」的 TSF 評估
# - 讀你已經算好的 03_ 傳統模型、07/08/09 三家預訓練模型
# - 每一條 MA 都輸出自己的 metrics + 圖

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

root = os.getcwd()
pred_dir = Path(f"{root}\\prediction\\F05")

hi_df = pd.read_csv(pred_dir / "01_F05_HI_full.csv", index_col=0).sort_index()

N_CONTEXT, N_HORIZON = 65, 16
MA_LIST = ["MA05", "MA10", "MA20", "MA30", "MA40", "MA50"]

def mae(a, b): return np.mean(np.abs(a - b))
def rmse(a, b): return np.sqrt(np.mean((a - b) ** 2))
def smape(a, b, eps=1e-8): return 100 * np.mean(2 * np.abs(a - b) / (np.abs(a) + np.abs(b) + eps))

def read_pretrained(pred_dir, name, MA, H):
    model_id = {"TimesFM": "07", "Chronos": "08", "TTMs": "09"}
    p = pred_dir / f"{model_id[name]}_F05_{name}_{MA}_pred{H}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    # 一樣是 flight + 預測欄
    return df.set_index("flight").iloc[:, 0].values

for MA in MA_LIST:
    if MA not in hi_df.columns:
        print(f"[WARN] {MA} not found in 01_F05_HI_full.csv, skip.")
        continue

    # 真值
    y_true = hi_df[MA].values[N_CONTEXT:N_CONTEXT+N_HORIZON]
    x_full = hi_df.index.values
    future_x = x_full[N_CONTEXT:N_CONTEXT+N_HORIZON]

    # 傳統模型的預測
    classic_path = pred_dir / f"03_F05_traditional_{MA}_pred{N_HORIZON}_all.csv"
    if not classic_path.exists():
        print(f"[WARN] {classic_path} not found, skip {MA}.")
        continue
    classic = pd.read_csv(classic_path, index_col=0)

    pred_ar    = classic["AR"].values
    pred_gpr   = classic["GPR"].values
    pred_arima = classic["ARIMA"].values

    # 三家預訓練
    pred_tfm = read_pretrained(pred_dir, "TimesFM", MA, N_HORIZON)
    pred_chr = read_pretrained(pred_dir, "Chronos", MA, N_HORIZON)
    pred_ttm = read_pretrained(pred_dir, "TTMs", MA, N_HORIZON)

    # === 評估表 ===
    metrics = []
    def add_metrics(name, pred):
        metrics.append({
            "MA": MA,
            "model": name,
            "MAE": mae(y_true, pred),
            "RMSE": rmse(y_true, pred),
            "sMAPE(%)": smape(y_true, pred),
        })

    add_metrics("AR",     pred_ar)
    add_metrics("GPR",    pred_gpr)
    add_metrics("ARIMA",  pred_arima)
    if pred_tfm is not None: add_metrics("TimesFM", pred_tfm)
    if pred_chr is not None: add_metrics("Chronos", pred_chr)
    if pred_ttm is not None: add_metrics("TTMs",    pred_ttm)

    metrics_df = pd.DataFrame(metrics).sort_values("MAE").reset_index(drop=True)
    out_csv = pred_dir / f"12_F05_eval_metrics_{MA}.csv"
    metrics_df.to_csv(out_csv, index=False)
    print(f"[07][{MA}] metrics saved → {out_csv}")
    print(metrics_df)

    # === 視覺化：全長 65+16 ===
    plt.figure(figsize=(10, 5))
    plt.plot(x_full, hi_df[MA].values, label="Ground Truth", linewidth=3, color="black")
    plt.axvline(x_full[N_CONTEXT-1], color="gray", linestyle=":", alpha=0.6)

    plt.plot(future_x, pred_ar,    "--", label="AR")
    plt.plot(future_x, pred_gpr,   "--", label="GPR")
    plt.plot(future_x, pred_arima, "--", label="ARIMA")
    if pred_tfm is not None: plt.plot(future_x, pred_tfm, "--", label="TimesFM")
    if pred_chr is not None: plt.plot(future_x, pred_chr, "--", label="Chronos")
    if pred_ttm is not None: plt.plot(future_x, pred_ttm, "--", label="TTMs")

    plt.title(f"Flight00 | {N_CONTEXT}-context + {N_HORIZON}-forecast")
    plt.xlabel("Flight")
    plt.ylabel(f"HI")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_png = pred_dir / f"13_F05_eval_full{N_CONTEXT}_{MA}.png"
    plt.savefig(out_png, dpi=160)
    plt.close()
    print(f"[07][{MA}] plot saved → {out_png}")
