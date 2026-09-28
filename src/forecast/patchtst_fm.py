# IBM Granite PatchTST-FM-r2 zero-shot forecast (added 2026-09-19).
#
# Not to be confused with the original PatchTST (ICLR 2023), which is a supervised
# architecture trained per dataset — `transformers.PatchTSTForPrediction` builds a
# randomly initialised model and would need training on the target series. This is
# IBM's *pretrained foundation model* built on that architecture (~385M params,
# released 2026-09-09), used zero-shot like TimesFM / Chronos / TTM.
#
# Why it matters here: unlike TTM's `512-192-r2` revision, which the model card pins
# at a 512-point minimum context (the roadmap phase 4 mismatch), PatchTST-FM accepts
# our 65-point context directly — verified at context 65 / 32 / 16. It is also
# Apache 2.0 + OpenMDW 1.0, so unlike TimesFM 3.0 it carries no commercial
# restriction. See docs/tsfm_landscape.md §3.2.
import numpy as np

from ..config import AVA, MA_LIST, N_CONTEXT, N_HORIZON, SCALE_DEFAULT
from .common import context_and_future, load_hi_full, save_pred

CHECKPOINT = "ibm-granite/granite-timeseries-patchtst-fm-r2"


def run(ava: str = AVA, ma_list=None, mode: str = "zero-shot", checkpoint: str | None = None,
        scale: str = SCALE_DEFAULT) -> None:
    import torch  # heavy imports, keep lazy
    from tsfm_public import PatchTSTFMForPrediction

    df = load_hi_full(ava)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PatchTSTFMForPrediction.from_pretrained(checkpoint or CHECKPOINT).to(device).eval()

    for ma in (ma_list or MA_LIST):
        y_ctx, future_flights = context_and_future(df, ma, scale)
        # (batch, context, channels) — univariate, so one channel.
        x = torch.tensor(np.asarray(y_ctx, dtype="float32")).reshape(1, N_CONTEXT, 1).to(device)
        with torch.no_grad():
            out = model(past_values=x, prediction_length=N_HORIZON)
        pred = out.prediction_outputs.detach().cpu().numpy().reshape(-1)[:N_HORIZON]
        # 07-09 is the foundation-model band and is full (TimesFM/Chronos/TTMs);
        # 10-13 already belong to eval outputs, so this model takes 14.
        save_pred(ava, "14", "PatchTSTFM", ma, mode, future_flights, pred, scale)
