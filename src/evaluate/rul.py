# RUL evaluation via HI-threshold crossing (formerly 06_eval_RUL.py).
import numpy as np
import pandas as pd

from ..config import AVA, MA_LIST, N_CONTEXT, N_HORIZON, THR, pred_dir
from ..plotting import plt
from .readers import read_classic, read_pretrained


def first_cross_idx(arr, thr):
    """Index of the first value <= thr, or None."""
    if arr is None:
        return None
    idx = np.where(arr <= thr)[0]
    return int(idx[0]) if len(idx) > 0 else None


def run(ava: str = AVA, ma_list=None, mode: str = "zero-shot", thr: float = THR) -> None:
    out_dir = pred_dir(ava)
    thr_str = str(thr).replace(".", "_")
    suffix_tag = "" if mode == "zero-shot" else "_ft"

    hi_df = pd.read_csv(out_dir / f"01_{ava}_HI_full.csv", index_col=0).sort_index()

    for ma in (ma_list or MA_LIST):
        if ma not in hi_df.columns:
            print(f"[WARN] {ma} not found in 01_{ava}_HI_full.csv, skip.")
            continue

        y_all = hi_df[ma].values
        fl_all = hi_df.index.values
        y_true = y_all[N_CONTEXT:N_CONTEXT + N_HORIZON]
        fl_ctx = fl_all[:N_CONTEXT]
        fl_fut = fl_all[N_CONTEXT:N_CONTEXT + N_HORIZON]

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
            preds[name] = read_pretrained(out_dir, ava, name, ma, mode)

        true_idx = first_cross_idx(y_true, thr)
        true_flight = fl_fut[true_idx] if true_idx is not None else None

        rows = []
        for m, phat in preds.items():
            # (1) horizontal RUL error: when does the model itself cross THR
            pred_idx = first_cross_idx(phat, thr)
            pred_flt = fl_fut[pred_idx] if pred_idx is not None else None
            if (true_idx is not None) and (pred_idx is not None):
                err_steps = abs(pred_idx - true_idx)
                err_flt = abs(int(pred_flt) - int(true_flight))
            else:
                err_steps = np.nan
                err_flt = np.nan

            # (2) vertical error: model value vs THR at the true crossing moment
            if true_idx is not None and phat is not None and true_idx < len(phat):
                signed_err = phat[true_idx] - thr
                abs_val_err = abs(signed_err)
            else:
                abs_val_err = np.nan
                signed_err = np.nan

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
        rul_df.insert(0, "MA", ma)
        rul_df.insert(1, "true_cross_idx", true_idx)
        rul_df.insert(2, "true_cross_flight", true_flight)
        rul_df = rul_df.sort_values(
            ["AbsErr_steps", "AbsErr_value_at_true_thr"], na_position="last"
        ).reset_index(drop=True)

        out_csv = out_dir / f"10_{ava}_eval_RUL_{ma}_thr{thr_str}{suffix_tag}.csv"
        rul_df.to_csv(out_csv, index=False)
        print(f"[eval-rul][{ma}] saved -> {out_csv}")

        plt.figure(figsize=(10, 5))
        plt.plot(fl_all, y_all, color="black", linewidth=3, label="Ground Truth")
        plt.axvline(fl_ctx[-1], color="gray", linestyle=":", alpha=0.6)
        plt.axhline(thr, color="red", linestyle="--", alpha=0.7, label=f"Threshold {thr:.2f}")
        if true_flight is not None:
            plt.scatter([true_flight], [thr], s=70, c="red", marker="x", label="True crossing")
        rng = np.random.default_rng(0)
        for m, phat in preds.items():
            idx = first_cross_idx(phat, thr)
            if idx is not None:
                jitter = rng.uniform(-0.02, 0.02)  # visual separation only
                plt.scatter([fl_fut[idx]], [thr + jitter], s=70, marker="o", label=f"{m} crossing")
        plt.title(f"Flight00 | RUL based on HI<={thr} context={N_CONTEXT}, horizon={N_HORIZON} ({mode})")
        plt.xlabel("Flight")
        plt.ylabel("HI")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        out_png = out_dir / f"11_{ava}_eval_full{N_CONTEXT + N_HORIZON}_RUL_{ma}{suffix_tag}.png"
        plt.savefig(out_png, dpi=160)
        plt.close()
        print(f"[eval-rul][{ma}] plot saved -> {out_png}")
