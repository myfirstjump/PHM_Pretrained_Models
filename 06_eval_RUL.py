# === RUL 評估：誰先觸碰到 HI=0.6 ===
import numpy as np, pandas as pd, matplotlib.pyplot as plt, os
from pathlib import Path

root = os.getcwd()
pred_dir = Path(f"{root}\\prediction\\F05")

# 讀 64 點 HI（MA50）
hi_df = pd.read_csv(pred_dir / "F05_HI_full.csv", index_col=0).sort_index()
y_all = hi_df["MA50"].values
fl_all = hi_df.index.values
N_CONTEXT, H = 42, 6

ctx = y_all[:N_CONTEXT]
y_true = y_all[N_CONTEXT:N_CONTEXT+H]
fl_ctx = fl_all[:N_CONTEXT]
fl_fut = fl_all[N_CONTEXT:N_CONTEXT+H]

# 讀傳統模型 16 點預測
# classic = pd.read_csv(pred_dir / "F05_traditional_pred16_all.csv", index_col=0)
MA = 'MA20'
classic = pd.read_csv(pred_dir / f"F05_traditional_{MA}_pred16_all.csv", index_col=0)
preds = {
    "AR": classic["AR"].values,
    "GPR": classic["GPR"].values,
    "ARIMA": classic["ARIMA"].values,
}

# 讀 TimesFM / Chronos（若存在）
def read_pred(name, MA):
    p = pred_dir / f"F05_{name}_{MA}_pred16.csv"
    if p.exists():
        df = pd.read_csv(p)
        return df["pred" if "TimesFM" not in name and "Chronos" not in name and "TTMs" not in name else df.columns[-1]].values
    return None

# 若你用了我上一版檔名：F05_TimesFM_pred16.csv / F05_Chronos_pred16.csv
# if (pred_dir / "F05_TimesFM_pred16.csv").exists():
#     preds["TimesFM"] = pd.read_csv(pred_dir / "F05_TimesFM_pred16.csv")["TimesFM_pred"].values
# if (pred_dir / "F05_Chronos_pred16.csv").exists():
#     preds["Chronos"] = pd.read_csv(pred_dir / "F05_Chronos_pred16.csv")["Chronos_pred"].values
preds["TimesFM"] = read_pred('TimesFM', MA)
preds["Chronos"] = read_pred('Chronos', MA)
preds['TTMs'] = read_pred('TTMs', MA)

# --- 閾值與 helper ---
THR = 0.75
THR_STR = str(THR).replace('.', '_')

def first_cross_idx(arr, thr=THR):
    """回傳第一次 <= thr 的索引（0-based），若無則回 None"""
    idx = np.where(arr <= thr)[0]
    return int(idx[0]) if len(idx) > 0 else None

# 真值觸發點（在後 16 範圍內）
true_idx = first_cross_idx(y_true, THR)  # 0..15 or None
true_row = {
    "true_cross_idx": true_idx,
    "true_cross_flight": (fl_fut[true_idx] if true_idx is not None else None),
}

# 各模型 RUL 誤差（步數與 flight）
rows=[]
for m, phat in preds.items():
    pred_idx = first_cross_idx(phat, THR)
    pred_flt = fl_fut[pred_idx] if pred_idx is not None else None
    # 誤差定義：
    # 1) steps：|pred_idx - true_idx|
    # 2) flights：以 flight 編號差值（若連號，與 steps 等價）
    if (true_idx is not None) and (pred_idx is not None):
        err_steps = abs(pred_idx - true_idx)
        err_flt   = abs(int(pred_flt) - int(true_row["true_cross_flight"]))
    else:
        err_steps = np.nan
        err_flt   = np.nan

    rows.append({
        "model": m,
        "pred_cross_idx": pred_idx,
        "pred_cross_flight": pred_flt,
        "AbsErr_steps": err_steps,
        "AbsErr_flight": err_flt,
    })

rul_df = pd.DataFrame(rows).sort_values("AbsErr_steps", na_position="last").reset_index(drop=True)
rul_df.insert(0, "true_cross_idx", true_row["true_cross_idx"])
rul_df.insert(1, "true_cross_flight", true_row["true_cross_flight"])
rul_df.to_csv(pred_dir / f"F05_eval_RUL_thr0p70.csv", index=False)
print(f"\n[RUL @ HI<={THR}]")
print(rul_df)

# --- 視覺化：畫 64 點 + 閾值 + 各模型觸發點 ---
plt.figure(figsize=(10,5))
plt.plot(fl_all, y_all, color="black", linewidth=3, label=f"Ground Truth ({MA})")
plt.axvline(fl_ctx[-1], color="gray", linestyle=":", alpha=0.6)
plt.axhline(THR, color="red", linestyle="--", alpha=0.7, label=f"Threshold {THR:.2f}")

# 真值觸發點
if true_row["true_cross_flight"] is not None:
    plt.scatter([true_row["true_cross_flight"]], [THR], s=70, c="red", marker="x", label="True crossing")

# 各模型觸發點（在後 16）
for m, phat in preds.items():
    idx = first_cross_idx(phat, THR)
    if idx is not None:
        plt.scatter([fl_fut[idx]], [THR], s=60, marker="o", label=f"{m} crossing")

plt.title(f"F05 | RUL based on HI<={THR} (context={N_CONTEXT}, horizon={H})")
plt.xlabel("Flight")
plt.ylabel(f"CV ({MA})")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(pred_dir / "F05_eval_full64_RUL.png", dpi=160)
plt.show()
