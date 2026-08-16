# Deriving the two HI thresholds from data instead of picking one by hand.
#
# thr_fail  — "this flight looks faulty". Read off the F18-F20 validation labels.
#             Youden-J normally picks it; when the two classes separate perfectly
#             (which they do here) every cut inside the gap scores identically, so
#             we take the max-margin point: the midpoint of the gap in logit space.
#
# thr_alert — "this aircraft has left its healthy baseline", i.e. the first
#             predicting time (FPT) marker. Follows the 3-sigma rule that Li, Lei,
#             Lin & Ding (2015, IEEE TIE) introduced for adaptive FPT selection:
#             build [mu - k*sigma, mu + k*sigma] from a non-degraded reference
#             period and call FPT once the HI leaves it. Two departures from the
#             paper, both recorded in the output file:
#               * the reference period is the aircraft's labelled post-maintenance
#                 healthy pool, not the head of the series being monitored (with
#                 only 11 flights the head is too tight and fires immediately);
#               * ALERT_RULE="mad" swaps mean/std for median/MAD so a single
#                 outlier in a 14-flight pool cannot move the threshold.
import json

import numpy as np
import pandas as pd

from ..config import (
    ALERT_BASELINE_AVA, ALERT_K, ALERT_RULE, FPT_CONSECUTIVE, HI_THRESHOLDS_PATH,
    MODEL_DIR, THR_LEGACY,
)

_MAD_TO_SIGMA = 1.4826  # makes MAD a consistent estimator of sigma for normal data


def _logit(p, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def _expit(z) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def baseline_threshold(hi_baseline, k: float = ALERT_K, rule: str = ALERT_RULE) -> dict:
    """Lower edge of the healthy interval: centre - k * spread."""
    hi = np.asarray(hi_baseline, dtype=float)
    if rule == "std":
        centre, spread = float(np.mean(hi)), float(np.std(hi, ddof=1))
    elif rule == "mad":
        centre = float(np.median(hi))
        spread = float(_MAD_TO_SIGMA * np.median(np.abs(hi - centre)))
    else:
        raise ValueError(f"Unknown ALERT_RULE {rule!r} (expected 'std' or 'mad')")
    return {
        "rule": f"{rule}: centre - {k}*spread",
        "n_baseline": int(len(hi)),
        "centre": centre,
        "spread": spread,
        "threshold": float(np.clip(centre - k * spread, 0.0, 1.0)),
    }


def fault_threshold(y_true, hi) -> dict:
    """thr_fail from labelled data. Max-margin cut if separable, else Youden-J."""
    from sklearn.metrics import roc_curve

    y_true = np.asarray(y_true)
    hi = np.asarray(hi, dtype=float)
    hi_faulty, hi_healthy = hi[y_true == 1], hi[y_true == 0]
    gap_lo, gap_hi = float(hi_faulty.max()), float(hi_healthy.min())

    fpr, tpr, thr_pf = roc_curve(y_true, 1.0 - hi)
    j = int(np.argmax(tpr - fpr))
    youden = float(1.0 - thr_pf[j])
    separable = gap_lo < gap_hi

    if separable:
        # Every cut in (gap_lo, gap_hi) is Youden-optimal; centre it in logit space,
        # where the HI is not squashed against 0 and 1.
        threshold = float(_expit(0.5 * (_logit(gap_lo) + _logit(gap_hi))))
        rule = "max-margin midpoint of the separating gap (logit space)"
    else:
        threshold = youden
        rule = "Youden-J"

    return {
        "rule": rule,
        "separable": bool(separable),
        "gap_lo_max_faulty_hi": gap_lo,
        "gap_hi_min_healthy_hi": gap_hi,
        "youden_threshold": youden,
        "youden_tpr": float(tpr[j]),
        "youden_fpr": float(fpr[j]),
        "threshold": threshold,
    }


def detect_fpt(series: pd.Series, thr_alert: float, n_consecutive: int = FPT_CONSECUTIVE):
    """First predicting time: index of the n-th of n consecutive points below thr_alert.

    Requiring a run (Li et al. use 2) stops one noisy flight from declaring onset.
    Returns None when the series never sustains a departure.
    """
    below = (series < thr_alert).to_numpy()
    run = 0
    for i, b in enumerate(below):
        run = run + 1 if b else 0
        if run >= n_consecutive:
            return series.index[i]
    return None


def derive_thresholds(pipe, save: bool = True) -> dict:
    """Compute thr_alert / thr_fail from the fitted HI pipeline and persist them."""
    from .hi import _load_training_pools, _load_validation_set, hi_of

    Xv, yv, _, _ = _load_validation_set()
    fail = fault_threshold(yv, hi_of(pipe, Xv))

    X, y, plane, flight = _load_training_pools()
    m = (plane == ALERT_BASELINE_AVA) & (y == 0)
    if not m.any():
        raise ValueError(f"No labelled healthy flights for {ALERT_BASELINE_AVA}")
    alert = baseline_threshold(hi_of(pipe, X[m]))
    alert["baseline_ava"] = ALERT_BASELINE_AVA
    alert["baseline_sorties"] = [int(flight[m].min()), int(flight[m].max())]

    out = {
        "thr_alert": alert["threshold"],
        "thr_fail": fail["threshold"],
        "thr_legacy": THR_LEGACY,
        "fpt_consecutive": FPT_CONSECUTIVE,
        "alert_detail": alert,
        "fail_detail": fail,
    }

    print(f"[thresholds] thr_alert = {out['thr_alert']:.4f}  "
          f"(logit {_logit(out['thr_alert']):+.3f})  <- {ALERT_BASELINE_AVA} healthy pool "
          f"{alert['baseline_sorties'][0]}-{alert['baseline_sorties'][1]}, n={alert['n_baseline']}, "
          f"{alert['rule']}")
    print(f"[thresholds] thr_fail  = {out['thr_fail']:.4f}  "
          f"(logit {_logit(out['thr_fail']):+.3f})  <- {fail['rule']}; "
          f"gap [{fail['gap_lo_max_faulty_hi']:.4f}, {fail['gap_hi_min_healthy_hi']:.4f}], "
          f"Youden TPR={fail['youden_tpr']:.3f} FPR={fail['youden_fpr']:.3f}")
    print(f"[thresholds] thr_legacy= {THR_LEGACY:.4f}  (hand-picked, kept for comparison)")

    if save:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        HI_THRESHOLDS_PATH.write_text(json.dumps(out, indent=2))
        print(f"[thresholds] saved: {HI_THRESHOLDS_PATH}")
    return out
