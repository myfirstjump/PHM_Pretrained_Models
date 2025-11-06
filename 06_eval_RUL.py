# 06_eval_RUL.py
# 目的：
# 1. 對 MA05~MA50 每一條都做 RUL 評估
# 2. 保留原本「誰先碰到 THR → 算水平距離」的誤差
# 3. 新增「GT 到達 THR 那一刻，各模型預測值距離 THR 的誤差」
# 4. 檔名加上 MA，避免互相覆蓋

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

root = os.getcwd()
pred_dir = Path(f"{root}\\prediction\\F05")

# 你在 02_landing_gear_HI_TSF.py 就是這 6 條
MA_LIST = ["MA05", "MA10", "MA20", "MA30", "MA40", "MA50"]

THR = 0.6                # 你原來的門檻是 0.6
THR_STR = str(THR).replace('.', '_')
N_CONTEXT, H = 65, 16    # 你的 F05 案子固定就是 65 + 16

def first_cross_idx(arr, thr):
    """回傳第一次 <= thr 的 index，若沒有就回 None"""
    if arr is None:
        return None
    idx = np.where(arr <= thr)[0]
    return int(idx[0]) if len(idx) > 0 else None

def read_model_pred(pred_dir, name, MA, H):
    """讀 07/08/09 那些模型的預測，按你的檔名規則"""
    model_id = {"TimesFM": "07", "Chronos": "08", "TTMs": "09"}
    p = pred_dir / f"{model_id[name]}_F05_{name}_{MA}_pred{H}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    # 你之前的檔案都是 flight + 預測值一欄，所以拿最後一欄最穩
    return df.iloc[:, -1].values

for MA in MA_LIST:
    # 讀完整 HI
    hi_df = pd.read_csv(pred_dir / "01_F05_HI_full.csv", index_col=0).sort_index()
    if MA not in hi_df.columns:
        print(f"[WARN] {MA} not found in 01_F05_HI_full.csv, skip.")
        continue

    y_all = hi_df[MA].values
    fl_all = hi_df.index.values

    ctx = y_all[:N_CONTEXT]
    y_true = y_all[N_CONTEXT:N_CONTEXT+H]
    fl_ctx = fl_all[:N_CONTEXT]
    fl_fut = fl_all[N_CONTEXT:N_CONTEXT+H]

    # 傳統模型（你之前就有產生）
    classic_path = pred_dir / f"03_F05_traditional_{MA}_pred{H}_all.csv"
    if not classic_path.exists():
        print(f"[WARN] {classic_path} not found, skip {MA}.")
        continue
    classic = pd.read_csv(classic_path, index_col=0)

    preds = {
        "AR": classic["AR"].values,
        "GPR": classic["GPR"].values,
        "ARIMA": classic["ARIMA"].values,
    }

    # TimesFM / Chronos / TTMs
    preds["TimesFM"] = read_model_pred(pred_dir, "TimesFM", MA, H)
    preds["Chronos"] = read_model_pred(pred_dir, "Chronos", MA, H)
    preds["TTMs"]    = read_model_pred(pred_dir, "TTMs", MA, H)

    # --- 真值的觸發點 ---
    true_idx = first_cross_idx(y_true, THR)
    true_row = {
        "true_cross_idx": true_idx,
        "true_cross_flight": (fl_fut[true_idx] if true_idx is not None else None),
    }

    # --- 各模型計算兩種誤差 ---
    rows = []
    for m, phat in preds.items():
        # ① 原本的 RUL 誤差：看模型自己何時掉到 THR
        pred_idx = first_cross_idx(phat, THR)
        pred_flt = fl_fut[pred_idx] if pred_idx is not None else None

        if (true_idx is not None) and (pred_idx is not None):
            err_steps = abs(pred_idx - true_idx)
            err_flt   = abs(int(pred_flt) - int(true_row["true_cross_flight"]))
        else:
            err_steps = np.nan
            err_flt   = np.nan

        # ② 新的：GT 到 THR 那一刻，模型的值離 THR 多遠
        if true_idx is not None and phat is not None and true_idx < len(phat):
            pred_val_at_true = phat[true_idx]
            abs_val_err = abs(pred_val_at_true - THR)
            signed_err  = pred_val_at_true - THR
        else:
            abs_val_err = np.nan
            signed_err  = np.nan

        rows.append({
            "model": m,
            "pred_cross_idx": pred_idx,
            "pred_cross_flight": pred_flt,
            "AbsErr_steps": err_steps,
            "AbsErr_flight": err_flt,
            "AbsErr_value_at_true_thr": abs_val_err,
            "SignedErr_value_at_true_thr": signed_err,
        })

    rul_df = pd.DataFrame(rows)
    rul_df.insert(0, "MA", MA)
    rul_df.insert(1, "true_cross_idx", true_row["true_cross_idx"])
    rul_df.insert(2, "true_cross_flight", true_row["true_cross_flight"])

    # 排序一下：先看「時間軸 RUL 誤差」、再看「同時刻值誤差」
    rul_df = rul_df.sort_values(["AbsErr_steps", "AbsErr_value_at_true_thr"], na_position="last").reset_index(drop=True)

    out_csv = pred_dir / f"10_F05_eval_RUL_{MA}_thr{THR_STR}.csv"
    rul_df.to_csv(out_csv, index=False)
    print(f"[06][{MA}] RUL evaluation saved → {out_csv}")

    # --- 畫圖：跟你原本一樣，只是檔名加 MA ---
    plt.figure(figsize=(10, 5))
    plt.plot(fl_all, y_all, color="black", linewidth=3, label=f"Ground Truth")
    plt.axvline(fl_ctx[-1], color="gray", linestyle=":", alpha=0.6)
    plt.axhline(THR, color="red", linestyle="--", alpha=0.7, label=f"Threshold {THR:.2f}")

    if true_row["true_cross_flight"] is not None:
        plt.scatter([true_row["true_cross_flight"]], [THR], s=70, c="red", marker="x", label="True crossing")

    # 模型觸發點 + noise
    for m, phat in preds.items():
        if phat is None:
            continue
        idx = first_cross_idx(phat, THR)
        if idx is not None:
            jitter = np.random.uniform(-0.02, 0.02)   # 在 THR 附近加擾動
            plt.scatter([fl_fut[idx]], [THR + jitter], s=70, marker="o", label=f"{m} crossing")

    plt.title(f"Flight00 | RUL based on HI<={THR} context={N_CONTEXT}, horizon={H}")
    plt.xlabel("Flight")
    plt.ylabel("HI")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_png = pred_dir / f"11_F05_eval_full{N_CONTEXT+H}_RUL_{MA}.png"
    plt.savefig(out_png, dpi=160)
    plt.close()
    print(f"[06][{MA}] plot saved → {out_png}")
