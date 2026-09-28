# Amazon Chronos-2 zero-shot forecast (formerly 04_forecast_chronos_F05.py).
#
# Upgraded from chronos-t5-small to Chronos-2 on 2026-09-20. Chronos-2 (120M,
# Apache 2.0) is smaller than the Bolt/T5 checkpoints it replaces and tops both
# GIFT-Eval and TIME on the overall metric. The class also changed:
# `ChronosPipeline` tokenised values into discrete bins, `Chronos2Pipeline` takes
# `inputs=` and returns a list of per-series quantile tensors.
from ..config import AVA, MA_LIST, N_HORIZON, SCALE_DEFAULT
from .common import context_and_future, load_hi_full, save_pred

CHECKPOINT = "amazon/chronos-2"


def run(ava: str = AVA, ma_list=None, mode: str = "zero-shot", checkpoint: str | None = None,
        scale: str = SCALE_DEFAULT) -> None:
    import torch
    from chronos import Chronos2Pipeline

    df = load_hi_full(ava)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pipe = Chronos2Pipeline.from_pretrained(checkpoint or CHECKPOINT, device_map=device)

    for ma in (ma_list or MA_LIST):
        y_ctx, future_flights = context_and_future(df, ma, scale)
        out = pipe.predict(inputs=[torch.tensor(y_ctx, dtype=torch.float32)],
                           prediction_length=N_HORIZON)
        # (1, n_quantiles, horizon) — take the median level, the middle row.
        q = out[0].squeeze(0) if out[0].ndim == 3 else out[0]
        median = q[q.shape[0] // 2] if q.ndim == 2 else q
        pred = median.detach().cpu().numpy().reshape(-1)[:N_HORIZON]
        save_pred(ava, "08", "Chronos2", ma, mode, future_flights, pred, scale)
