# Amazon Chronos zero-shot forecast (formerly 04_forecast_chronos_F05.py).
from ..config import AVA, MA_LIST, N_HORIZON
from .common import context_and_future, load_hi_full, save_pred


def run(ava: str = AVA, ma_list=None, mode: str = "zero-shot", checkpoint: str | None = None) -> None:
    import torch
    from chronos import ChronosPipeline

    df = load_hi_full(ava)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    pipe = ChronosPipeline.from_pretrained(
        checkpoint or "amazon/chronos-t5-small", device_map=device, dtype=dtype
    )

    for ma in (ma_list or MA_LIST):
        y_ctx, future_flights = context_and_future(df, ma)
        ctx_t = torch.tensor(y_ctx, dtype=torch.float32).unsqueeze(0)
        # chronos-forecasting >= 2.0 renamed the `context` kwarg to `inputs`
        samples = pipe.predict(ctx_t, prediction_length=N_HORIZON, num_samples=200)
        if samples.ndim == 3:
            samples = samples[:, 0, :]
        pred_p50 = torch.quantile(samples, 0.5, dim=0).cpu().numpy()
        save_pred(ava, "08", "Chronos", ma, mode, future_flights, pred_p50)
