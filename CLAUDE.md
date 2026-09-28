# PHM_Pretrained_Models — working context

Forecasting a landing-gear Health Indicator with TSF foundation models
(TimesFM 3.0 / Chronos-2 / PatchTST-FM-r2 / MOIRAI 2.0) against classical baselines
(AR / GPR / ARIMA).

Reply to the user in **zh-TW**. Structure work as `main.py` + `argparse`
subcommands over `src/`. Plan phase by phase and confirm before moving on.

## Roadmap

Redesigned 2026-08-09. **Currently on phase 3.**

1. **HI rebuild** — LOAO training + F18-F20 validation, report AUC. ✅ done
   Also delivered: two derived thresholds, logit output, dual-scale figure.
2. **HI curve recompute** — F05 sorties 1244-1431 in true order. ✅ done 2026-09-20
   `build-hi` now reads `F05_SERIES_FOLDER` + `HI_PIPELINE_LOAO_PATH` and emits
   `CV`, `logit`, `MA05..50` and `logit_MA05..50` (64 flights × 14 columns).
   `F05_custom` / `F05_prediction_gt` are off the code path entirely (files kept).
3. **Backtest framework** — rolling-origin; get naive + ARIMA working first. ← **here**
   The forecast path still cuts a single fixed origin at index 32; 25 are available.
4. **Foundation models** — ✅ attached (see "Model lineup"). TTM dropped, not fixed.
5. **Stats + censored RUL** — Diebold-Mariano with HAC (lag >= h-1), censored evaluation.

## Constraints that shape every decision

**No failure data in the subset we hold.** Every one of the 17 training aircraft holds a
few dozen sorties flown *before* it went into the depot (orange) and a few dozen flown
*after* (green). The label is "which side of the depot visit", never "did this flight
fail". What the LR actually learns is pre-depot vs post-depot separability, i.e. a proxy
for maintenance need, not failure probability. Within an aircraft the series therefore
trends *upward* (maintenance restores HI). `training/Faulty` / `training/Healthy` are
legacy folder names — read them as pre-depot / post-depot. `thr_fail` is likewise legacy:
it is the **進廠門檻**, "looks like it needs a depot visit", not a failure threshold. Any
RUL derived here means "sorties until the depot threshold is crossed" — hence censored.

**Be precise about whose data lacks what.** The source papers (Hsu 2022, Chang 2023 —
see "Provenance") *did* hold maintenance records and 20 fault-item labels. We hold a
20-aircraft / 619-flight subset without them. Write it as "the subset obtained for this
study carries no fault labels", never "this dataset cannot define failure" — a reviewer
who reads the source papers will catch the difference.

**No synthetic data.** `F05_custom` and `F05_prediction_gt` were hand-picked and
re-sequenced for a report deadline (the "ground truth" was F08 pre-depot flights that
were also in HI training; `F05_custom`'s 65th file is an F08 flight too). They stay on
disk as a documented fallback. Do not reintroduce them.

**F05 sorties 1244-1431 (64 flights) is the only genuine degradation series.** No
maintenance event inside it. Do NOT splice the 30 F05 training flights (845-913) back
on — 331-sortie gap plus a maintenance jump.

**Sampling is not uniform.** 88.9% (56/63) of adjacent gaps are 1 sortie; there are five
gaps of 18-30. The real structure is 6 bursts of consecutive sorties (11/17/9/8/10/9)
separated by 5 jumps, and the degradation happens *between* bursts — within-burst net
change is mostly under ±0.04. Cause: the old FDR overwrote 15-25% of sorties before
download (Hsu 2022 §4.2), so the gaps are a real acquisition limit, not curation. A
horizon of h points therefore spans a sortie count that depends on where it lands.

**Hardware:** two different machines. The dev box (drive A:) has an **RTX 5060
(Blackwell, sm_120)** and needs torch cu128 — cu121 has no kernel for it. The CPU-only
box described in earlier notes (4 cores, ~7 GB) is a different machine. Everything runs
in the single conda env `phm-env` (Python 3.12, torch 2.11.0+cu128), which the user
intends to pack and move — keep it to one env.

## Model lineup

| `--model` | model | file id | licence |
|---|---|---|---|
| `timesfm` | TimesFM 3.0 (330M) | 07 | **Non-Commercial v1.0** |
| `chronos` | Chronos-2 (120M) | 08 | Apache 2.0 |
| `patchtst` | PatchTST-FM-r2 (385M) | 14 | Apache 2.0 + OpenMDW |
| `moirai` | MOIRAI 2.0-R-small (11.4M) | 15 | **CC-BY-NC-4.0** |

TimesFM 3.0 and MOIRAI 2.0 are research-only: fine for the thesis, must be declared, and
no result from them may enter a deliverable. `ttm_fc.py` stays on disk for the §3.2
discussion but is out of `FORECAST_MODELS` — its 512-point minimum context cannot take a
32-point window.

**`uni2ts` must be installed with `--no-deps`.** Its `torch<2.5` / `numpy~=1.26` pins are
over-conservative (MOIRAI runs fine on torch 2.11 / numpy 2.5), but letting pip resolve
them drops torch to 2.4.1+cpu — killing CUDA on Blackwell *and* breaking granite-tsfm
(`torch>=2.10`). Only the `gluonts~=0.14.3` pin is real: 0.17 changed `make_predictions`
and the predictor raises "not enough values to unpack". See `requirements-models.txt`.

## Traps found the hard way

- **MA targets restate the context — this is NOT leakage.** The MA itself is causal
  (pandas `.rolling()` is trailing; verified to 2e-16 against a manual causal MA). The
  problem is the *target*: of the w terms in MA_w(T+h), w−h of them are values already
  observed at forecast time — 84% for MA50 at h=8. A predictor that does no forecasting
  at all (assume CV stays flat) scores MAE 0.0193 on MA50 vs 0.2686 on raw, 14x
  "better". So the metric stops measuring forecasting skill, and **MAE is not comparable
  across different w**. Headline results use raw / MA05 (0% at h>=8); MA10-MA50 go in an
  appendix with the known fraction stated.
- **n=1 proves nothing, and it is also optimistic.** The forecast path cuts one fixed
  origin. Over the 25 available origins the classical models score MAE 0.24-0.27; the
  single origin gave 0.13 — roughly 2x optimistic.
- **Threshold and smoothing interact.** FPT moves from sortie 1284 (raw) to 1366 (MA40/50).
  Raw is also far more robust to the 3-sigma multiplier: k=1..5 moves raw FPT only
  1284→1285, but MA50 1286→1423.
- **The 3-sigma FPT rule fails in two opposite ways, both present in this fleet.** F03's
  MAD is large enough that median−3·spread goes negative and `np.clip` pins thr_alert at
  0.0 — that aircraft can never declare an FPT. F07/F16 have MAD ~0.0008, giving
  thr_alert 0.99+, which any normal flight trips. Four aircraft have <10 healthy flights,
  where MAD itself is unstable. Treat "no FPT" as a legitimate right-censored
  observation, not an error to be relaxed away.
- **Forecast on the logit, report on CV.** ~30% of the F05 series sits in the sigmoid's
  saturated region, where the average step is 0.39x the non-saturated one on the CV scale
  but 1.02x on logit. `--scale` defaults to `logit`; `save_pred` back-transforms so every
  saved prediction is CV and stays comparable. Note `logit_MAw` is **mean(logit)**, not
  `logit(mean(CV))` — Jensen makes them different series (0.34-0.84 logit units apart).
  Thresholds are unaffected: monotone transform, crossing times identical.
- **GPR kernel parameters are context-length dependent.** The old `ExpSineSquared`
  (periodicity 40, bounds 20-80) was tuned for a 65-point context; at 32 points that is
  under one cycle and unidentifiable — the fit railed periodicity at 80 and collapsed its
  length_scale to 0.03. Removed 2026-09-28; MAE 0.2697 → 0.2477. AR's `maxlag=10` and
  ARIMA's grid were also suspected but A/B tests did not support changing them.
- `myfeature/*.csv` carry `plane`/`flight` id columns; drop `config.ID_COLS` before use or
  `FEAT_IDXS` selects the wrong features. `F05AllFeatures.csv` is a stale orphan with no
  id columns — no code reads it.
- `testingLabel.xlsx` lists 67 flights, only 65 CSVs exist. Labels are assigned by
  sortie-number range, so missing files just warn. Reading it needs `openpyxl`.
- No CJK font on this machine (Docker has Noto). Use `plotting.label(zh, en)`.
- LibreOffice is installed at `C:\Program Files\LibreOffice\program\soffice.exe` but is
  **not on PATH**. Use it to render .docx → PDF; `pdftoppm` is absent, use `pymupdf`.
- Don't write scratch scripts into `/tmp` with module-like names — a stray `/tmp/h2.py`
  shadowed the `h2` package `httpcore` imports and produced baffling failures. Use the
  session scratchpad.

## Positioning

Yan, Koç & Lee (2004, *Production Planning & Control* 15(8)) built an LR degradation
index and extrapolated it with ARMA. This project swaps the extrapolator for TSF
foundation models. Frame the work as a **HI-forecasting benchmark with censored RUL**,
not as RUL point prediction.

**Provenance.** The data came from Prof. 張淵仁 (Chang, Y.-J.), Feng Chia University,
co-author of both source papers — Hsu et al. (2022) *Aerospace* 9(8) 462 and Chang et al.
(2023) *Aerospace* 10(11) 963, PDFs in `docs/literatures/`. Our feature table, feature
selection (TO-Y rms/std/peak2peak) and LR HI all come from them; AR/GPR/ARIMA is their
comparison set. **The user has decided not to frame their MA50 choice as a target-leakage
flaw** — keep that analysis as justification for our own raw/MA05 headline, not as a
critique.

Verified against the PDF in `docs/` (2026-08-16) — three traps:
- **"confidence value"/`CV` is NOT Yan's term.** The paper says *performance index* /
  *probability of failure*; CV comes from Lee's later IMS/Watchdog-Agent line. `CV` is
  fine as this code's own symbol, never as a quote from Yan.
- **Their index is P(failure), rising**; ours is its complement, falling. Threshold
  crossing runs the opposite way (they cross a "failure line" upward).
- **Their §2.2** fits the LR from technician-assigned failure probabilities
  (normal 0.01 / acceptable 0.25 / unacceptable 0.5) when no failure history exists.
  Our data is §2.1 in form, §2.2 in meaning — cite this as precedent for the
  pre-depot/post-depot labels. Hsu (2022) does the same thing with 0.95 / 0.05.
Jamshidi, Kim & Arif (2025, arXiv:2506.20090) is CC BY 4.0 (checked on the arXiv abs
page); its Fig. 4 prognostics tree has five branches, and data-driven still splits only
into ML / DL — that gap is the paper's opening.

**HI quality is a known weakness, and that is the point.** By Coble & Hines (2009),
monotonicity is 0.0476 raw and never reaches 0.500 (their "poor parameter" benchmark) at
any smoothing. Trendability and prognosability are population metrics and degenerate on a
single path. The existing quality criteria and ISO 13381-1 all presuppose run-to-failure
data; real fleet maintenance rarely has it. Frame the research question as "how well can
a TSFM extrapolate an HI that the established criteria call unfit?" — not as a claim that
the HI is fit.

## Documents

`docs/` is gitignored (paper drafts, licensed PDFs). Living documents:
- `docs/literature_review.html` — research narrative, HI quality criteria, execution plan
- `docs/tsfm_landscape.html` / `.md` — model survey and selection
- `docs/implementation_status.html` — what the code currently does, env install order
- `docs/Thesis/115-09-28_...docx` — thesis; 壹/貳/參 written, 肆/伍 pending
