# Health Indicator: LR pipeline training (formerly 01_) and HI series building (02_ part 1).
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import (
    AVA, CONTEXT_FOLDER, F05_SERIES_FOLDER, FEAT_IDXS, FEATURES_DIR, GT_FOLDER,
    HI_HOLDOUT_AVAS, HI_PIPELINE_LOAO_PATH, HI_PIPELINE_PATH, ID_COLS, MA_WINDOWS,
    MODEL_DIR, N_CONTEXT, THR_LEGACY, pred_dir,
)
from ..plotting import label, plt
from .features import build_feature_table
from .labels import label_testing_flights
from .thresholds import _logit, derive_thresholds, detect_fpt


def _read_features(name: str):
    """myfeature/<name> -> (40-column feature frame, plane Series, flight Series)."""
    df = pd.read_csv(FEATURES_DIR / name)
    missing = [c for c in ID_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name} has no {missing} column(s) — regenerate it with "
            f"`python main.py prepare-features` (the id columns were added in the LOAO rework)."
        )
    return df.drop(columns=ID_COLS), df["plane"], df["flight"]


def _fit_pipeline(X: pd.DataFrame, y) -> Pipeline:
    """select-3-features -> StandardScaler -> LogisticRegression."""
    sel = ColumnTransformer(
        transformers=[("pick", "passthrough", FEAT_IDXS)], remainder="drop"
    )
    pipe = Pipeline(steps=[
        ("sel", sel),
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(random_state=0, solver="lbfgs")),
    ])
    pipe.fit(X, y)
    return pipe


def _healthy_index(pipe) -> int:
    return int(np.where(pipe.named_steps["lr"].classes_ == 0)[0][0])


def hi_of(pipe, X) -> np.ndarray:
    """HI = P(healthy). Accepts a 40-column frame or array."""
    values = X.values if isinstance(X, pd.DataFrame) else X
    return pipe.predict_proba(values)[:, _healthy_index(pipe)]


def train_hi_pipeline() -> None:
    """Fit select-3-features -> StandardScaler -> LogisticRegression and save the pipeline.

    PCA was dropped in the original experiments because its components polarized
    the HI poorly; three hand-picked TO-Y features are used instead.
    """
    healthy = _read_features("healthyAllFeatures.csv")[0]
    faulty = _read_features("faultyAllFeatures.csv")[0]

    all_features = pd.concat([healthy, faulty], axis="rows", ignore_index=True)
    labels = np.array([0] * healthy.shape[0] + [1] * faulty.shape[0])

    pipe = _fit_pipeline(all_features, labels)

    i_healthy = int(np.where(pipe.named_steps["lr"].classes_ == 0)[0][0])
    hi_healthy = pipe.predict_proba(healthy.values)[:, i_healthy]
    hi_faulty = pipe.predict_proba(faulty.values)[:, i_healthy]
    print(f"[train-hi] samples: healthy={healthy.shape[0]}, faulty={faulty.shape[0]}")
    print(f"[train-hi] Healthy HI median: {np.median(hi_healthy):.4f}")
    print(f"[train-hi] Faulty  HI median: {np.median(hi_faulty):.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, HI_PIPELINE_PATH)
    print(f"[train-hi] Saved: {HI_PIPELINE_PATH}")

    _plot_lr_curve(pipe, all_features, labels)


def _plot_lr_curve(pipe, all_features, labels) -> None:
    X_sel = pipe.named_steps["sel"].transform(all_features)
    X_norm = pipe.named_steps["sc"].transform(X_sel)
    logit_all = pipe.named_steps["lr"].decision_function(X_norm)
    p_all = 1.0 / (1.0 + np.exp(-logit_all))

    x_curve = np.linspace(logit_all.min() - 0.5, logit_all.max() + 0.5, 600)
    y_curve = 1.0 / (1.0 + np.exp(-x_curve))

    # Jitter is purely cosmetic so samples appear "around" the curve.
    rng = np.random.default_rng(0)
    y_scatter = np.clip(p_all + rng.normal(0.0, 0.05, size=p_all.shape), 0.0, 1.0)

    plt.figure(figsize=(10, 6))
    plt.plot(x_curve, y_curve, lw=3, label="LR curve (sigmoid of score)")
    plt.scatter(logit_all[labels == 0], y_scatter[labels == 0],
                s=45, c="#2E86AB", edgecolor="k", linewidth=0.6, alpha=0.85, label="Healthy (0)")
    plt.scatter(logit_all[labels == 1], y_scatter[labels == 1],
                s=45, c="#E74C3C", edgecolor="k", linewidth=0.6, alpha=0.85, label="Faulty (1)")
    plt.ylim(-0.02, 1.02)
    plt.yticks(np.linspace(0, 1, 6))
    plt.xlabel(label("Logistic Regression 決策函數值", "Logistic Regression decision function"))
    plt.ylabel(label("Logistic Regression 臨界故障預測值", "Logistic Regression P(faulty)"))
    plt.title("Logistic Regression Health Indicator")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()
    out = MODEL_DIR / "LR_curve_with_samples.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[train-hi] plot saved: {out}")


def make_hi_series(folder_path, pipe_path=HI_PIPELINE_PATH) -> pd.DataFrame:
    """All flights in a folder -> HI series indexed by flight.

    Two columns for the same quantity on different scales:
      CV    = P(healthy), the historical HI, bounded in [0, 1]
      logit = log(CV / (1-CV)), unbounded and monotone in CV
    The sigmoid saturates — on the F05 series ~30% of flights sit above 0.95 or
    below 0.05, where the probability scale compresses real differences to
    nothing. The logit keeps them, so it is the better target to forecast.
    """
    pipe = joblib.load(pipe_path)
    feats = build_feature_table(folder_path)

    cv = pd.Series(hi_of(pipe, feats), index=feats.index).sort_index()
    return pd.DataFrame({"CV": cv.values, "logit": _logit(cv.values)}, index=cv.index)


def build_hi_full(ava: str = AVA) -> pd.DataFrame:
    """Context + ground-truth folders -> full HI series with causal MAs, saved to prediction/."""
    out_dir = pred_dir(ava)

    print(f"[build-hi] context: {CONTEXT_FOLDER}")
    ctx_df = make_hi_series(CONTEXT_FOLDER)
    print(f"[build-hi] ground truth: {GT_FOLDER}")
    gt_df = make_hi_series(GT_FOLDER)

    hi_df = pd.concat([ctx_df, gt_df]).sort_index()
    for name, window in MA_WINDOWS.items():
        hi_df[name] = hi_df["CV"].rolling(window=window, min_periods=1).mean()
        hi_df[f"logit_{name}"] = hi_df["logit"].rolling(window=window, min_periods=1).mean()

    out_csv = out_dir / f"01_{ava}_HI_full.csv"
    hi_df.to_csv(out_csv)
    print(f"[build-hi] saved full HI series ({len(hi_df)} flights) -> {out_csv}")

    ctx = hi_df.iloc[:N_CONTEXT]
    ctx[list(MA_WINDOWS)].to_csv(out_dir / f"02_{ava}_train_context_MA05-50.csv")

    plt.figure(figsize=(9, 4.5))
    plt.scatter(ctx.index.values, ctx["CV"].astype(float).values, s=26, alpha=0.9)
    plt.title(f"Flight00 context HI — first {N_CONTEXT} flights")
    plt.xlabel("Flight")
    plt.ylabel("HI")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_png = out_dir / f"06_{ava}_ctx_CV_scatter_{N_CONTEXT}.png"
    plt.savefig(out_png, dpi=160)
    plt.close()
    print(f"[build-hi] scatter saved: {out_png}")

    return hi_df


# --------------------------------------------------------------------------
# Leave-one-aircraft-out HI training + F18-F20 validation
# --------------------------------------------------------------------------
# The legacy pipeline above fits all 554 training flights with no hold-out, so
# every reported number is in-sample and the F05/F08 flights that later feed
# the forecasting experiment were seen during HI training. The functions below
# fit the same model on the remaining aircraft only, and score it on the
# F18-F20 split that testingLabel.xlsx labels.

def _load_training_pools():
    """training/Healthy + training/Faulty -> (X40, y, plane, flight)."""
    healthy, h_plane, h_flight = _read_features("healthyAllFeatures.csv")
    faulty, f_plane, f_flight = _read_features("faultyAllFeatures.csv")
    X = pd.concat([healthy, faulty], axis="rows", ignore_index=True)
    y = np.array([0] * len(healthy) + [1] * len(faulty))
    plane = pd.concat([h_plane, f_plane], ignore_index=True).to_numpy()
    flight = pd.concat([h_flight, f_flight], ignore_index=True).to_numpy()
    return X, y, plane, flight


def _load_validation_set():
    """testing/ (F18-F20) -> (X40, y, plane, flight), labels from testingLabel.xlsx."""
    X, plane, flight = _read_features("testingAllFeatures.csv")
    y = label_testing_flights(plane, flight).to_numpy()
    return X, y, plane.to_numpy(), flight.to_numpy()


def _score(pipe, X, y) -> dict:
    """AUC / accuracy / HI medians for one evaluation set. Score = P(faulty) = 1 - HI."""
    from sklearn.metrics import accuracy_score, roc_auc_score

    hi = hi_of(pipe, X)
    p_faulty = 1.0 - hi
    both = len(np.unique(y)) > 1
    return {
        "n": int(len(y)),
        "n_healthy": int((y == 0).sum()),
        "n_faulty": int((y == 1).sum()),
        "auc": float(roc_auc_score(y, p_faulty)) if both else float("nan"),
        "acc": float(accuracy_score(y, (p_faulty >= 0.5).astype(int))),
        "hi_med_healthy": float(np.median(hi[y == 0])) if (y == 0).any() else float("nan"),
        "hi_med_faulty": float(np.median(hi[y == 1])) if (y == 1).any() else float("nan"),
    }


def _loao_cv(X, y, plane, avas):
    """Leave-one-aircraft-out CV over `avas`. Returns (per-aircraft rows, pooled OOF score)."""
    from sklearn.metrics import roc_auc_score

    rows = []
    oof = np.full(len(y), np.nan)
    for ava in avas:
        te = plane == ava
        tr = ~te & np.isin(plane, avas)
        fold = _fit_pipeline(X[tr], y[tr])
        oof[te] = 1.0 - hi_of(fold, X[te])
        rows.append({"set": f"loao-cv/{ava}", **_score(fold, X[te], y[te])})

    mask = np.isin(plane, avas)
    pooled = {
        "set": "loao-cv/pooled-oof",
        "n": int(mask.sum()),
        "n_healthy": int((y[mask] == 0).sum()),
        "n_faulty": int((y[mask] == 1).sum()),
        "auc": float(roc_auc_score(y[mask], oof[mask])),
        "acc": float(((oof[mask] >= 0.5).astype(int) == y[mask]).mean()),
        "hi_med_healthy": float(np.median(1.0 - oof[mask][y[mask] == 0])),
        "hi_med_faulty": float(np.median(1.0 - oof[mask][y[mask] == 1])),
    }
    return rows, pooled


def train_hi_loao(holdout=None, run_cv: bool = True) -> None:
    """Fit the HI pipeline with `holdout` aircraft excluded; validate on F18-F20."""
    holdout = list(holdout if holdout is not None else HI_HOLDOUT_AVAS)

    X, y, plane, flight = _load_training_pools()
    fit_mask = ~np.isin(plane, holdout)
    train_avas = sorted(set(plane[fit_mask]))

    print(f"[train-hi-loao] holdout aircraft : {holdout}")
    print(f"[train-hi-loao] fitted on        : {len(train_avas)} aircraft, "
          f"{int(fit_mask.sum())} flights "
          f"(healthy={int((y[fit_mask] == 0).sum())}, faulty={int((y[fit_mask] == 1).sum())})")

    pipe = _fit_pipeline(X[fit_mask], y[fit_mask])

    Xv, yv, plane_v, _ = _load_validation_set()

    rows = [{"set": "train (in-sample)", **_score(pipe, X[fit_mask], y[fit_mask])}]
    for ava in holdout:
        m = plane == ava
        rows.append({"set": f"holdout/{ava}", **_score(pipe, X[m], y[m])})
    rows.append({"set": "validation F18-F20", **_score(pipe, Xv, yv)})
    for ava in sorted(set(plane_v)):
        m = plane_v == ava
        rows.append({"set": f"validation/{ava}", **_score(pipe, Xv[m], yv[m])})

    if run_cv:
        cv_rows, pooled = _loao_cv(X, y, plane, train_avas)
        rows.append(pooled)
        rows.extend(cv_rows)

    metrics = pd.DataFrame(rows)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = MODEL_DIR / "hi_loao_metrics.csv"
    metrics.to_csv(out_csv, index=False)

    with pd.option_context("display.width", 120, "display.max_columns", 20):
        print(metrics.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"[train-hi-loao] metrics saved: {out_csv}")

    joblib.dump(pipe, HI_PIPELINE_LOAO_PATH)
    print(f"[train-hi-loao] saved: {HI_PIPELINE_LOAO_PATH}")

    eval_sets = [
        (label(f"訓練集 ({len(train_avas)} 架, 排除 {'+'.join(holdout)})",
               f"Train ({len(train_avas)} aircraft, {'+'.join(holdout)} excluded)"),
         X[fit_mask], y[fit_mask]),
        (label(f"保留機體 {'+'.join(holdout)} (unseen)", f"Held-out {'+'.join(holdout)} (unseen)"),
         X[np.isin(plane, holdout)], y[np.isin(plane, holdout)]),
        (label("驗證集 F18-F20 (unseen)", "Validation F18-F20 (unseen)"), Xv, yv),
    ]
    _plot_loao_curves(pipe, eval_sets)
    _plot_loao_roc(pipe, eval_sets)

    thr = derive_thresholds(pipe)
    _plot_hi_vs_logit(pipe, thr)


def _plot_loao_curves(pipe, eval_sets) -> None:
    """Original LR-curve-with-samples plot, one panel per evaluation set."""
    dec = [pipe.decision_function(X.values) for _, X, _ in eval_sets]
    lo = min(d.min() for d in dec) - 0.5
    hi = max(d.max() for d in dec) + 0.5
    x_curve = np.linspace(lo, hi, 600)
    y_curve = 1.0 / (1.0 + np.exp(-x_curve))

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, len(eval_sets), figsize=(6 * len(eval_sets), 5.2), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, d, (title, X, y_true) in zip(axes, dec, eval_sets):
        p = 1.0 / (1.0 + np.exp(-d))
        # Jitter is purely cosmetic so samples appear "around" the curve.
        y_scatter = np.clip(p + rng.normal(0.0, 0.05, size=p.shape), 0.0, 1.0)
        ax.plot(x_curve, y_curve, lw=3, color="#444444", label="LR curve (sigmoid of score)")
        ax.scatter(d[y_true == 0], y_scatter[y_true == 0], s=45, c="#2E86AB",
                   edgecolor="k", linewidth=0.6, alpha=0.85, label="Healthy (0)")
        ax.scatter(d[y_true == 1], y_scatter[y_true == 1], s=45, c="#E74C3C",
                   edgecolor="k", linewidth=0.6, alpha=0.85, label="Faulty (1)")
        ax.axvline(0.0, color="gray", ls="--", lw=1, alpha=0.7)
        s = _score(pipe, X, y_true)
        ax.set_title(f"{title}\nn={s['n']}  AUC={s['auc']:.3f}  Acc={s['acc']:.3f}", fontsize=11)
        ax.set_xlabel(label("Logistic Regression 決策函數值", "Logistic Regression decision function"))
        ax.set_ylim(-0.02, 1.02)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", frameon=True, fontsize=9)

    axes[0].set_ylabel(label("Logistic Regression 臨界故障預測值", "Logistic Regression P(faulty)"))
    fig.suptitle("Logistic Regression Health Indicator — leave-one-aircraft-out", fontsize=13)
    fig.tight_layout()
    out = MODEL_DIR / "LR_curve_LOAO_samples.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[train-hi-loao] plot saved: {out}")


def _plot_loao_roc(pipe, eval_sets) -> None:
    from sklearn.metrics import roc_auc_score, roc_curve

    plt.figure(figsize=(6.2, 6))
    for title, X, y_true in eval_sets:
        if len(np.unique(y_true)) < 2:
            continue
        p_faulty = 1.0 - hi_of(pipe, X)
        fpr, tpr, _ = roc_curve(y_true, p_faulty)
        plt.plot(fpr, tpr, lw=2, label=f"{title}  AUC={roc_auc_score(y_true, p_faulty):.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(label("HI 分類器 ROC（故障為正類）", "HI classifier ROC (faulty = positive class)"))
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    out = MODEL_DIR / "LR_ROC_LOAO.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[train-hi-loao] plot saved: {out}")


def _plot_hi_vs_logit(pipe, thr) -> None:
    """The real F05 series on both scales, with the derived thresholds marked.

    Left panel is the historical HI (probability); right is its logit. Same
    ordering, same information -- but the probability scale flattens everything
    near 0 and 1, which is where a third of this series lives.
    """
    feats = build_feature_table(F05_SERIES_FOLDER)
    cv = pd.Series(hi_of(pipe, feats), index=feats.index).sort_index()
    lg = pd.Series(_logit(cv.values), index=cv.index)

    lines = [
        ("thr_alert", thr["thr_alert"], "#E67E22"),
        ("thr_fail", thr["thr_fail"], "#C0392B"),
        (label("0.6 (自訂)", "0.6 (hand-picked)"), THR_LEGACY, "#7F8C8D"),
    ]
    fpt = detect_fpt(cv, thr["thr_alert"], thr["fpt_consecutive"])

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4))
    for ax, series, scale in ((axes[0], cv, "HI = P(healthy)"), (axes[1], lg, "logit(HI)")):
        ax.plot(series.index, series.values, "-o", ms=4, lw=1.2,
                color="#2E86AB", alpha=0.9, label=scale)
        for name, value, colour in lines:
            y = value if scale.startswith("HI") else float(_logit(value))
            ax.axhline(y, color=colour, ls="--", lw=1.4,
                       label=f"{name} = {y:.3f}")
        if fpt is not None:
            ax.axvline(fpt, color="#16A085", ls=":", lw=1.8, label=f"FPT @ sortie {fpt}")
        ax.set_xlabel(label("架次 (sortie)", "Sortie"))
        ax.set_ylabel(scale)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left", fontsize=8, ncol=2)

    sat = int(((cv < 0.05) | (cv > 0.95)).sum())
    fig.suptitle(label(
        f"F05 真實序列 sortie {cv.index.min()}-{cv.index.max()} ({len(cv)} 班)"
        f" — 機率尺度 vs logit 尺度（{sat}/{len(cv)} 班落在飽和區 HI<0.05 或 >0.95）",
        f"F05 real series, sorties {cv.index.min()}-{cv.index.max()} ({len(cv)} flights)"
        f" — probability vs logit scale ({sat}/{len(cv)} in the saturated region)"),
        fontsize=12)
    fig.tight_layout()
    out = MODEL_DIR / "HI_vs_logit_F05_series.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[thresholds] plot saved: {out}")
