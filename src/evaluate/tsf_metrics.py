# Forecast accuracy evaluation: MAE / RMSE / sMAPE (formerly 07_eval_zero_shot_models.py).
import numpy as np
import pandas as pd

from ..config import AVA, MA_LIST, N_CONTEXT, N_HORIZON, pred_dir
from ..plotting import plt
from .readers import read_classic, read_pretrained


def mae(a, b):
    return np.mean(np.abs(a - b))


def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


def smape(a, b, eps=1e-8):
    return 100 * np.mean(2 * np.abs(a - b) / (np.abs(a) + np.abs(b) + eps))


def run(ava: str = AVA, ma_list=None, mode: str = "zero-shot") -> None:
    out_dir = pred_dir(ava)
    hi_df = pd.read_csv(out_dir / f"01_{ava}_HI_full.csv", index_col=0).sort_index()
    suffix_tag = "" if mode == "zero-shot" else "_ft"

    for ma in (ma_list or MA_LIST):
        if ma not in hi_df.columns:
            print(f"[WARN] {ma} not found in 01_{ava}_HI_full.csv, skip.")
            continue

        y_true = hi_df[ma].values[N_CONTEXT:N_CONTEXT + N_HORIZON]
        x_full = hi_df.index.values
        future_x = x_full[N_CONTEXT:N_CONTEXT + N_HORIZON]

        classic = read_classic(out_dir, ava, ma)
        if classic is None:
            print(f"[WARN] traditional predictions for {ma} not found, skip.")
            continue

        preds = {
            "AR": classic["AR"].values,
            "GPR": classic["GPR"].values,
            "ARIMA": classic["ARIMA"].values,
        }
        for name in ("TimesFM", "Chronos", "TTMs"):
            p = read_pretrained(out_dir, ava, name, ma, mode)
            if p is not None:
                preds[name] = p

        metrics = [
            {"MA": ma, "model": name, "MAE": mae(y_true, p),
             "RMSE": rmse(y_true, p), "sMAPE(%)": smape(y_true, p)}
            for name, p in preds.items()
        ]
        metrics_df = pd.DataFrame(metrics).sort_values("MAE").reset_index(drop=True)
        out_csv = out_dir / f"12_{ava}_eval_metrics_{ma}{suffix_tag}.csv"
        metrics_df.to_csv(out_csv, index=False)
        print(f"[eval-tsf][{ma}] metrics saved -> {out_csv}")
        print(metrics_df)

        plt.figure(figsize=(10, 5))
        plt.plot(x_full, hi_df[ma].values, label="Ground Truth", linewidth=3, color="black")
        plt.axvline(x_full[N_CONTEXT - 1], color="gray", linestyle=":", alpha=0.6)
        for name, p in preds.items():
            plt.plot(future_x, p, "--", label=name)
        plt.title(f"Flight00 | {N_CONTEXT}-context + {N_HORIZON}-forecast ({mode})")
        plt.xlabel("Flight")
        plt.ylabel("HI")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_png = out_dir / f"13_{ava}_eval_full{N_CONTEXT}_{ma}{suffix_tag}.png"
        plt.savefig(out_png, dpi=160)
        plt.close()
        print(f"[eval-tsf][{ma}] plot saved -> {out_png}")
