# Classical baselines: AR / GPR / ARIMA (formerly 02_landing_gear_HI_TSF.py part 2).
import warnings

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, WhiteKernel
from statsmodels.tsa.ar_model import ar_select_order
from statsmodels.tsa.arima.model import ARIMA

from ..config import AVA, MA_LIST, N_CONTEXT, N_HORIZON, pred_dir
from ..plotting import plt
from .common import load_hi_full

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
    kernel = (
        WhiteKernel(noise_level=0.3 ** 2, noise_level_bounds=(0.1 ** 2, 0.5 ** 2))
        + ExpSineSquared(length_scale=1.0, periodicity=40, periodicity_bounds=(20, 80))
        + RBF(length_scale=1e2, length_scale_bounds=(1, 1e3))
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


def run(ava: str = AVA, ma_list=None) -> None:
    out_dir = pred_dir(ava)
    hi_df = load_hi_full(ava)
    ctx_df = hi_df.iloc[:N_CONTEXT]

    last_flt = int(ctx_df.index[-1])
    future_index = np.arange(last_flt + 1, last_flt + 1 + N_HORIZON)

    for ma in (ma_list or MA_LIST):
        y_ctx = ctx_df[ma].values.astype(float)

        pred_df = pd.DataFrame({
            "flight": future_index,
            "AR": _forecast_ar(y_ctx),
            "GPR": _forecast_gpr(y_ctx),
            "ARIMA": _forecast_arima(y_ctx, ma),
        }).set_index("flight")
        pred_df.to_csv(out_dir / f"03_{ava}_traditional_{ma}_pred{N_HORIZON}_all.csv")

        plot_df = pd.concat(
            [ctx_df[[ma]],
             pred_df.rename(columns={c: f"{ma}_{c}" for c in ("AR", "GPR", "ARIMA")})],
            axis="columns",
        )
        plot_df.to_csv(out_dir / f"04_{ava}_{ma}_context{N_CONTEXT}_and_pred{N_HORIZON}_for_plot.csv")

        plt.figure(figsize=(8, 5))
        plt.plot(ctx_df.index.values, ctx_df[ma].values, label="Context", linewidth=2)
        plt.plot(future_index, pred_df["AR"].values, "--", label="AR pred", linewidth=2)
        plt.plot(future_index, pred_df["GPR"].values, "--", label="GPR pred", linewidth=2)
        plt.plot(future_index, pred_df["ARIMA"].values, "--", label="ARIMA pred", linewidth=2)
        plt.title(f"Flight00 | context={N_CONTEXT} → forecast {N_HORIZON} (AR/GPR/ARIMA)")
        plt.xlabel("Flight")
        plt.ylabel("HI")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"05_{ava}_{ma}_context{N_CONTEXT}_pred{N_HORIZON}.png", dpi=160)
        plt.close()
        print(f"[traditional] {ma} done")
