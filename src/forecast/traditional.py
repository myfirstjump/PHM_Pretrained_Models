# Classical baselines: AR / GPR / ARIMA (formerly 02_landing_gear_HI_TSF.py part 2).
import warnings

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from statsmodels.tsa.ar_model import ar_select_order
from statsmodels.tsa.arima.model import ARIMA

from ..config import (
    AVA, MA_LIST, N_CONTEXT, N_HORIZON, SCALE_DEFAULT, ma_column, pred_dir,
)
from ..plotting import plt
from .common import load_hi_full, scale_suffix, to_cv

warnings.filterwarnings("ignore", category=FutureWarning)


def _forecast_ar(y_ctx):
    try:
        ar_order = ar_select_order(y_ctx, maxlag=10, glob=True, trend="ct")
        print(f"[AR] selected lags: {ar_order.ar_lags}")
        ar_res = ar_order.model.fit()
        pred = ar_res.predict(start=len(y_ctx), end=len(y_ctx) + N_HORIZON - 1)
        return np.asarray(pred).ravel()
    except Exception as e:
        print("[AR] fallback:", e)
        return np.full(N_HORIZON, np.nan)


def _forecast_gpr(y_ctx):
    # The periodic (ExpSineSquared) term was dropped on 2026-09-28. It was tuned for
    # the retired 65-point context; with N_CONTEXT = 32 a periodicity of 40-80 spans
    # less than one full cycle, so it is not identifiable from the data. The fit
    # confirmed it: `periodicity` railed at its upper bound 80 while the component's
    # length_scale collapsed to ~0.03, i.e. the optimiser switched the term off after
    # paying for it. `noise_level` also sat on its lower bound 0.01 on both scales,
    # so those bounds were too tight. Removing the term and widening the bounds cut
    # MAE from 0.2697 to 0.2484 over the 25 rolling origins.
    #
    # ConstantKernel carries the signal amplitude (normalize_y standardises y, so the
    # remaining freedom is the signal-to-noise split); the RBF length scale is bounded
    # to 2-60 points, i.e. from "just above measurement noise" to "longer than the
    # context", which is the full range a 32-point window can speak to.
    kernel = (
        ConstantKernel(1.0, constant_value_bounds=(1e-2, 1e2))
        * RBF(length_scale=10.0, length_scale_bounds=(2.0, 60.0))
        + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-3, 1.0))
    )
    X_train = np.arange(len(y_ctx)).reshape(-1, 1)
    gpr = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=5, normalize_y=True, random_state=0
    )
    gpr.fit(X_train, y_ctx.reshape(-1, 1))
    X_future = np.arange(len(y_ctx), len(y_ctx) + N_HORIZON).reshape(-1, 1)
    pred, _ = gpr.predict(X_future, return_std=True)
    return pred.ravel()


def _forecast_arima(y_ctx, ma):
    # AIC grid search; the series is short and pre-smoothed, so a small grid suffices.
    best_aic, best_order, best_model = float("inf"), None, None
    for p in range(0, 6):
        for d in range(0, 2):
            for q in range(0, 4):
                try:
                    model = ARIMA(y_ctx, order=(p, d, q)).fit()
                    if model.aic < best_aic:
                        best_aic, best_order, best_model = model.aic, (p, d, q), model
                except Exception:
                    continue

    if best_model is None:
        print(f"[ARIMA-grid] {ma} no valid model found, fill NaN")
        return np.full(N_HORIZON, np.nan)
    print(f"[ARIMA-grid] {ma} best order (p,d,q) = {best_order}, AIC = {best_aic:.2f}")
    return np.asarray(best_model.forecast(steps=N_HORIZON)).ravel()


def run(ava: str = AVA, ma_list=None, scale: str = SCALE_DEFAULT) -> None:
    """AR / GPR / ARIMA on `scale`, saved back-transformed to the CV scale.

    The classical baselines get the same treatment as the foundation models so the
    only thing that differs between the two families is the extrapolator itself.
    """
    out_dir = pred_dir(ava)
    hi_df = load_hi_full(ava)
    ctx_df = hi_df.iloc[:N_CONTEXT]
    sfx = scale_suffix(scale)

    col = ma_column(ma_list[0] if ma_list else "raw", scale)
    if col not in hi_df.columns:
        raise KeyError(f"{col!r} missing — re-run `python main.py build-hi`.")

    last_flt = int(ctx_df.index[-1])
    future_index = np.arange(last_flt + 1, last_flt + 1 + N_HORIZON)

    for ma in (ma_list or MA_LIST):
        col = ma_column(ma, scale)
        y_ctx = ctx_df[col].values.astype(float)

        raw = {
            "AR": _forecast_ar(y_ctx),
            "GPR": _forecast_gpr(y_ctx),
            "ARIMA": _forecast_arima(y_ctx, ma),
        }
        pred_df = pd.DataFrame(
            {"flight": future_index, **{k: to_cv(v, scale) for k, v in raw.items()}}
        ).set_index("flight")
        pred_df.to_csv(out_dir / f"03_{ava}_traditional_{ma}_pred{N_HORIZON}{sfx}_all.csv")

        # Plot on the modelling scale — that is where the forecast actually happened.
        ctx_plot = ctx_df[[col]].rename(columns={col: ma})
        plot_df = pd.concat(
            [ctx_plot,
             pd.DataFrame({"flight": future_index, **{f"{ma}_{k}": v for k, v in raw.items()}})
               .set_index("flight")],
            axis="columns",
        )
        plot_df.to_csv(
            out_dir / f"04_{ava}_{ma}_context{N_CONTEXT}_and_pred{N_HORIZON}{sfx}_for_plot.csv")

        plt.figure(figsize=(8, 5))
        plt.plot(ctx_df.index.values, y_ctx, label="Context", linewidth=2)
        for name in ("AR", "GPR", "ARIMA"):
            plt.plot(future_index, raw[name], "--", label=f"{name} pred", linewidth=2)
        plt.title(f"{ava} {ma} | context={N_CONTEXT} → forecast {N_HORIZON} "
                  f"({scale} scale, AR/GPR/ARIMA)")
        plt.xlabel("Flight")
        plt.ylabel("logit(HI)" if scale == "logit" else "HI")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            out_dir / f"05_{ava}_{ma}_context{N_CONTEXT}_pred{N_HORIZON}{sfx}.png", dpi=160)
        plt.close()
        print(f"[traditional] {ma} ({scale}) done")
