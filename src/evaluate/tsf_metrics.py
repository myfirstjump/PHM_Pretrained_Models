# Forecast accuracy evaluation: MAE / RMSE / sMAPE (formerly 07_eval_zero_shot_models.py).
import numpy as np
import pandas as pd

from ..config import (
    AVA, MA_LIST, N_CONTEXT, N_HORIZON, SCALE_DEFAULT, ma_column, pred_dir,
)
from ..plotting import plt
from .readers import ACTIVE_MODELS, read_classic, read_pretrained


def mae(a, b):
    return np.mean(np.abs(a - b))


def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


def smape(a, b, eps=1e-8):
    return 100 * np.mean(2 * np.abs(a - b) / (np.abs(a) + np.abs(b) + eps))


def run(ava: str = AVA, ma_list=None, mode: str = "zero-shot",
        scale: str = SCALE_DEFAULT) -> None:
    """Score the forecasts produced at `scale`.

    Ground truth is always read on the CV scale, because every forecaster saves its
    prediction back-transformed to CV. So "which scale was the model fitted on" is
    the only thing `scale` changes here — it selects which run to score, not which
    units the error is in. That keeps logit-scale and CV-scale runs directly
    comparable, and comparable with the pre-2026-09 numbers.
    """
    out_dir = pred_dir(ava)
    hi_df = pd.read_csv(out_dir / f"01_{ava}_HI_full.csv", index_col=0).sort_index()
    suffix_tag = "" if mode == "zero-shot" else "_ft"

    for ma in (ma_list or MA_LIST):
        truth_col = ma_column(ma, "cv")
        if truth_col not in hi_df.columns:
            print(f"[WARN] {truth_col} not found in 01_{ava}_HI_full.csv, skip.")
            continue

        y_true = hi_df[truth_col].values[N_CONTEXT:N_CONTEXT + N_HORIZON]
        x_full = hi_df.index.values
        future_x = x_full[N_CONTEXT:N_CONTEXT + N_HORIZON]

        classic = read_classic(out_dir, ava, ma, scale)
        if classic is None:
            print(f"[WARN] traditional predictions for {ma} ({scale}) not found, skip.")
            continue

        preds = {
            "AR": classic["AR"].values,
            "GPR": classic["GPR"].values,
            "ARIMA": classic["ARIMA"].values,
        }
        for name in ACTIVE_MODELS:
            p = read_pretrained(out_dir, ava, name, ma, mode, scale)
            if p is not None:
                preds[name] = p

        metrics = [
            {"MA": ma, "scale": scale, "model": name, "MAE": mae(y_true, p),
             "RMSE": rmse(y_true, p), "sMAPE(%)": smape(y_true, p)}
            for name, p in preds.items()
        ]
        metrics_df = pd.DataFrame(metrics).sort_values("MAE").reset_index(drop=True)
        out_csv = out_dir / f"12_{ava}_eval_metrics_{ma}_{scale}{suffix_tag}.csv"
        metrics_df.to_csv(out_csv, index=False)
        print(f"[eval-tsf][{ma}] metrics saved -> {out_csv}")
        print(metrics_df)

        plt.figure(figsize=(10, 5))
        plt.plot(x_full, hi_df[truth_col].values, label="Ground Truth", linewidth=3, color="black")
        plt.axvline(x_full[N_CONTEXT - 1], color="gray", linestyle=":", alpha=0.6)
        for name, p in preds.items():
            plt.plot(future_x, p, "--", label=name)
        plt.title(f"{ava} {ma} | {N_CONTEXT}-context + {N_HORIZON}-forecast "
                  f"(fitted on {scale}, reported on CV, {mode})")
        plt.xlabel("Flight")
        plt.ylabel("HI")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_png = out_dir / f"13_{ava}_eval_full{N_CONTEXT}_{ma}_{scale}{suffix_tag}.png"
        plt.savefig(out_png, dpi=160)
        plt.close()
        print(f"[eval-tsf][{ma}] plot saved -> {out_png}")
