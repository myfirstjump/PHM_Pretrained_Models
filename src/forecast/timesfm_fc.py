# Google TimesFM 2.5 zero-shot forecast (formerly 03_forecast_timesfm_F05.py).
from ..config import AVA, MA_LIST, N_HORIZON
from .common import context_and_future, load_hi_full, save_pred


def run(ava: str = AVA, ma_list=None, mode: str = "zero-shot", checkpoint: str | None = None) -> None:
    import timesfm  # heavy import, keep lazy

    df = load_hi_full(ava)

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        checkpoint or "google/timesfm-2.5-200m-pytorch"
    )
    model.compile(timesfm.ForecastConfig(
        max_context=1024,
        max_horizon=256,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=False,
        fix_quantile_crossing=True,
    ))

    for ma in (ma_list or MA_LIST):
        y_ctx, future_flights = context_and_future(df, ma)
        pred, _ = model.forecast(horizon=N_HORIZON, inputs=[y_ctx])
        save_pred(ava, "07", "TimesFM", ma, mode, future_flights, pred[0])
