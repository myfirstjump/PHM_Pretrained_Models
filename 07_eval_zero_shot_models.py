# 05_eval_zero_shot_models.py
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

root = os.getcwd()
pred_dir = Path(f"{root}\\prediction\\F05")

# === 讀取資料 ===
hi_df = pd.read_csv(pred_dir / "F05_HI_full.csv", index_col=0).sort_index()
N_CONTEXT, N_HORIZON = 65, 16
MA = 'MA50'

# Ground truth: 後16筆
y_true = hi_df[MA].values[N_CONTEXT:N_CONTEXT+N_HORIZON]
x_full = hi_df.index.values

# === 讀取傳統模型結果 ===
classic = pd.read_csv(pred_dir / f"F05_traditional_{MA}_pred16_all.csv", index_col=0)
pred_ar    = classic["AR"].values
pred_gpr   = classic["GPR"].values
pred_arima = classic["ARIMA"].values

# === 讀取 TimesFM / Chronos ===
def read_pred(name, MA):
    p = pred_dir / f"F05_{name}_{MA}_pred16.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    return df.set_index("flight").iloc[:, 0].values

pred_tfm = read_pred("TimesFM", MA)
pred_chr = read_pred("Chronos", MA)
pred_ttm = read_pred('TTMs', MA)

# === 評估 ===
def mae(a,b): return np.mean(np.abs(a-b))
def rmse(a,b): return np.sqrt(np.mean((a-b)**2))
def smape(a,b,eps=1e-8): return 100*np.mean(2*np.abs(a-b)/(np.abs(a)+np.abs(b)+eps))

metrics = []
def add_metrics(name, pred):
    metrics.append({
        "model": name,
        "MAE": mae(y_true, pred),
        "RMSE": rmse(y_true, pred),
        "sMAPE(%)": smape(y_true, pred),
    })

add_metrics("AR", pred_ar)
add_metrics("GPR", pred_gpr)
add_metrics("ARIMA", pred_arima)
if pred_tfm is not None: add_metrics("TimesFM", pred_tfm)
if pred_chr is not None: add_metrics("Chronos", pred_chr)
if pred_ttm is not None: add_metrics("TTMs", pred_chr)

metrics_df = pd.DataFrame(metrics).sort_values("MAE").reset_index(drop=True)
metrics_df.to_csv(pred_dir / "F05_eval_metrics.csv", index=False)
print(metrics_df)

# === 視覺化：全長64點 ===
plt.figure(figsize=(10,5))
plt.plot(x_full, hi_df[MA].values, label="Ground Truth", linewidth=3, color="black")

future_x = hi_df.index.values[N_CONTEXT:N_CONTEXT+N_HORIZON]
plt.plot(future_x, pred_ar, "--", label="AR")
plt.plot(future_x, pred_gpr, "--", label="GPR")
plt.plot(future_x, pred_arima, "--", label="ARIMA")
if pred_tfm is not None: plt.plot(future_x, pred_tfm, "--", label="TimesFM")
if pred_chr is not None: plt.plot(future_x, pred_chr, "--", label="Chronos")
if pred_ttm is not None: plt.plot(future_x, pred_ttm, "--", label="TTMs")

plt.axvline(x_full[N_CONTEXT-1], color="gray", linestyle=":", alpha=0.6)
plt.title(f"F05 | {N_CONTEXT}-context + {N_HORIZON}-forecast")
plt.xlabel("Flight")
plt.ylabel(f"CV ({MA})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(pred_dir / "F05_eval_full64.png", dpi=160)
plt.show()
print(f"[05] Saved results and plot to {pred_dir}")
