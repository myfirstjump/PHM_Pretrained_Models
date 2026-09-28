# Salesforce MOIRAI 2.0 zero-shot forecast (added 2026-09-20).
#
# Licence: MOIRAI 2.0 weights are CC-BY-NC-4.0, "for research purposes only". This
# project is academic, so the model is in scope, but no MOIRAI result may enter a
# deliverable system. State it in the paper. See docs/tsfm_landscape.md §3.3.
#
# Install note: `uni2ts` declares torch<2.5 / numpy~=1.26 / gluonts~=0.14.3. The
# torch and numpy pins are over-conservative — the model loads and runs fine on
# torch 2.11+cu128 and numpy 2.5 — but the gluonts pin is real: 0.17 changed
# `make_predictions` and the predictor raises "not enough values to unpack". So the
# environment carries uni2ts and gluonts==0.14.4 installed with `--no-deps`, which
# keeps the CUDA build the RTX 5060 (sm_120) needs. See requirements-models.txt.
import numpy as np

from ..config import AVA, MA_LIST, N_CONTEXT, N_HORIZON, SCALE_DEFAULT
from .common import context_and_future, load_hi_full, save_pred

CHECKPOINT = "Salesforce/moirai-2.0-R-small"


def run(ava: str = AVA, ma_list=None, mode: str = "zero-shot", checkpoint: str | None = None,
        scale: str = SCALE_DEFAULT) -> None:
    import pandas as pd  # heavy imports, keep lazy
    from gluonts.dataset.pandas import PandasDataset
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

    df = load_hi_full(ava)
    module = Moirai2Module.from_pretrained(checkpoint or CHECKPOINT)

    for ma in (ma_list or MA_LIST):
        y_ctx, future_flights = context_and_future(df, ma, scale)

        model = Moirai2Forecast(
            module=module, prediction_length=N_HORIZON, context_length=N_CONTEXT,
            target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0,
        )
        # gluonts wants a dated index; the HI series is indexed by sortie number and
        # the spacing is irregular anyway, so the frequency here is a placeholder.
        ctx = pd.DataFrame(
            {"target": np.asarray(y_ctx, dtype="float32")},
            index=pd.period_range("2000-01-01", periods=N_CONTEXT, freq="h"),
        )
        ds = PandasDataset(dict(s=ctx["target"]))

        forecast = next(iter(model.create_predictor(batch_size=1).predict(ds)))
        # MOIRAI 2.0 emits quantiles directly (QuantileForecast), not samples.
        pred = np.asarray(forecast.quantile(0.5), dtype=float).reshape(-1)[:N_HORIZON]
        save_pred(ava, "15", "MOIRAI2", ma, mode, future_flights, pred, scale)
