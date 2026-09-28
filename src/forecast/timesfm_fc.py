# Google TimesFM 3.0 zero-shot forecast (formerly 03_forecast_timesfm_F05.py).
#
# TimesFM 3.0 replaced 2.5 here on 2026-09-19. The API is not backwards
# compatible: 2.5 was `TimesFM_2p5_200M_torch.from_pretrained(...)` + `.compile(
# ForecastConfig(...))` + `.forecast(horizon, inputs=[...])`; 3.0 builds a
# forecaster from a ModelConfig and takes `predict(context=<2-D array>, horizon=...)`,
# returning a `ForecastOutput(ts_id, forecast, quantiles)`.
#
# Licence note (belongs in the paper, not just here): the 3.0 weights are under the
# TimesFM Non-Commercial License v1.0 — academic research is permitted, commercial
# or production use is not. TimesFM 2.5 was Apache 2.0; replacing it means no
# TimesFM result in this project may enter a deliverable. See docs/tsfm_landscape.md §3.3.
from ..config import AVA, MA_LIST, N_HORIZON, SCALE_DEFAULT
from .common import context_and_future, load_hi_full, save_pred

CHECKPOINT = "google/timesfm-3.0-pytorch"


def run(ava: str = AVA, ma_list=None, mode: str = "zero-shot", checkpoint: str | None = None,
        scale: str = SCALE_DEFAULT) -> None:
    import torch  # heavy imports, keep lazy
    from timesfm3.timesfm3_forecaster import ModelConfig, TimesFM3Forecaster

    df = load_hi_full(ava)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fc = TimesFM3Forecaster(ModelConfig(
        checkpoint_path=checkpoint or CHECKPOINT, device=device,
    ))

    for ma in (ma_list or MA_LIST):
        y_ctx, future_flights = context_and_future(df, ma, scale)
        # predict() wants a 2-D (n_series, context) array, not 2.5's list of 1-D inputs.
        out = fc.predict(context=y_ctx.reshape(1, -1).astype("float32"), horizon=N_HORIZON)
        pred = out.forecast.reshape(-1)[:N_HORIZON]
        save_pred(ava, "07", "TimesFM", ma, mode, future_flights, pred, scale)
