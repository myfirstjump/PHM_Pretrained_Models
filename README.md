# PHM_Pretrained_Models

A test lab for PHM tasks using pretrained time-series foundation models.

Case study: landing-gear Health Indicator (HI) forecasting — 65-flight context,
16-flight horizon — comparing classical models (AR / GPR / ARIMA) against
zero-shot and fine-tuned foundation models (TimesFM 2.5, Chronos, IBM TTM).

## Pipeline

```
raw flight CSVs ──prepare-features──> myfeature/*.csv
                 ──train-hi────────> LR health-indicator pipeline (.pkl)
                 ──build-hi────────> prediction/F05/01_F05_HI_full.csv (+ MA05..MA50)
                 ──forecast────────> prediction/F05/{03,07,08,09}_*.csv
                 ──eval────────────> prediction/F05/{10,11,12,13}_*  (RUL + accuracy)
```

## Usage

```bash
python main.py prepare-features                # raw -> feature CSVs (Step1+Step2)
python main.py train-hi                        # LOAO fit + F18-F20 validation
python main.py train-hi --scheme legacy        # old no-hold-out fit (reproducibility only)
python main.py build-hi                        # HI series + moving averages
python main.py forecast --model traditional    # AR / GPR / ARIMA
python main.py forecast --model timesfm        # or chronos / ttm / all
python main.py eval --task all                 # TSF metrics + RUL (thr 0.6)
```

`forecast` and `eval` accept `--ma MA50` (default: all six MAs) and
`--mode finetuned` (reads/writes `_ft`-suffixed files). `forecast --checkpoint`
points a model at a local (e.g. fine-tuned) checkpoint.

## Environment

Core stack (HI + classical models) only needs `requirements.txt`.
Foundation models additionally need `requirements-models.txt` (CPU torch):

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt -r requirements-models.txt
```

Or use Docker (installs Noto CJK fonts for the Chinese figure labels):

```bash
docker compose build
docker compose run --rm phm build-hi
docker compose run --rm phm forecast --model all
docker compose run --rm phm eval --task all
```

`data/` and `models/` are not in git; mount or copy them in (see
`docker-compose.yml`).

## HI model validation

**What the labels mean.** Every aircraft contributes a few dozen sorties flown
*before* its depot visit and a few dozen flown *after*. `Healthy` / `Faulty` are
legacy folder names for post-depot / pre-depot — no failure event is recorded
anywhere in this dataset. The HI is therefore P(post-depot), a proxy for
maintenance need, not a failure probability.

`train-hi` defaults to `--scheme loao`: the LR pipeline is fitted on 16 aircraft
with **F05 held out** (its series drives the forecasting experiment), then scored
on the **F18-F20** split labelled by `data/.../testingLabel.xlsx`. Outputs land in
`data/.../model/`:

| file | contents |
|---|---|
| `hi_lr_pipeline_loao.pkl` | the LOAO-fitted pipeline |
| `hi_loao_metrics.csv` | AUC / accuracy / HI medians per set and per aircraft |
| `hi_thresholds.json` | derived `thr_alert` / `thr_fail` plus their provenance |
| `LR_curve_LOAO_samples.png` | LR sigmoid + samples, one panel per set |
| `LR_ROC_LOAO.png` | ROC for train / held-out / validation |
| `HI_vs_logit_F05_series.png` | the real F05 series on both scales, thresholds marked |

Headline: validation AUC 1.000 (acc 0.954), held-out F05 AUC 1.000,
leave-one-aircraft-out pooled out-of-fold AUC 0.987.

### The two thresholds

The old `THR = 0.6` was hand-picked and lands in the empty valley between the HI's
two modes — only 7 of 554 labelled flights fall in [0.5, 0.6). `train-hi` now derives
two thresholds and writes them to `hi_thresholds.json`; `config.THR_ALERT` /
`config.THR_FAIL` read them back.

- **`thr_alert` = 0.845** (logit +1.69) — degradation onset / first predicting time.
  `median - 3*MAD` over F05's own labelled post-maintenance healthy pool (sorties
  872-913, n=14). This is the 3-sigma FPT rule of Li, Lei, Lin & Ding (2015, IEEE
  TIE), with median/MAD substituted for mean/std so one outlier in 14 flights cannot
  move it. FPT is declared only after `FPT_CONSECUTIVE = 2` points stay below.
- **`thr_fail` = 0.144** (logit −1.78) — legacy name; it is the *depot threshold*
  ("looks like it needs a depot visit"), not a failure level, from the F18-F20 labels. The
  classes separate perfectly there (Youden TPR 1.000 / FPR 0.000), so every cut in
  the gap [0.044, 0.380] scores the same; we take the max-margin midpoint in logit
  space rather than an arbitrary endpoint.

`THR = THR_LEGACY = 0.6` is still what `eval --thr` defaults to — rewiring the RUL
evaluation onto the alert/fail pair is phase 5.

### HI is bimodal: forecast the logit, not the probability

88.6% of labelled flights sit outside HI 0.10-0.80, but **40.6% of the F05 series
lives inside that band** — we are forecasting the region where the HI has the least
supporting data. Worse, the sigmoid saturates: 19 of the 64 F05 flights are above
0.95 or below 0.05, where real differences compress to nothing.

`make_hi_series` therefore emits both `CV` (= P(healthy), the historical HI) and
`logit`, and `build-hi` adds `logit_MA05..logit_MA50` next to the existing MA
columns. The logit is a monotone transform (identical Spearman) with ~7x the spread,
and is the better forecasting target. Compare `HI_vs_logit_F05_series.png`.

Caveat — **ranking is near-perfect, absolute HI level is not**. The healthy-group
HI median swings from 0.51 (F03) to 1.00 (F13/F16) across aircraft, and all three
validation errors are F18 healthy flights sitting at HI 0.37-0.41. Any single
absolute HI threshold therefore means a different degradation state on each
aircraft. `thr_alert` sidesteps this by being derived per aircraft; `thr_fail` does
not, so cross-aircraft fault-level comparisons still need calibration work.

`--scheme legacy` reproduces the original fit (all 554 flights, no hold-out, every
metric in-sample) and writes the pre-existing
`hi_lr_pipeline_1141019_self_features.pkl`. Verified bit-identical to the
committed pickle after the id-column rework.

## Data notes (F05 case)

- `data/LandingGear_relatives/data/F05_custom` — the 65-flight context. These
  are real F05 flights but hand-picked and **re-sequenced** (original flight
  numbers 01244–01405, non-chronological) to shape a clearer degradation trend,
  then renumbered 00001–00065.
- `data/.../F05_prediction_gt` — the 16 "future ground truth" flights are
  actually **F08 pre-depot flights** (originally F08-00690..00705, renumbered
  00066–00081). The same files appear in `training/Faulty` (a legacy folder name —
  it means pre-depot, not failed), i.e. the HI model
  saw them during training. Fine for comparing forecasters on the same HI
  curve; not a clean generalization claim.
