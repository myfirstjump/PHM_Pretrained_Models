# PHM_Pretrained_Models — working context

Forecasting a landing-gear Health Indicator with TSF foundation models
(TimesFM 2.5 / Chronos / IBM TTM) against classical baselines (AR / GPR / ARIMA).

Reply to the user in **zh-TW**. Structure work as `main.py` + `argparse`
subcommands over `src/`. Plan phase by phase and confirm before moving on.

## Roadmap

Redesigned 2026-08-09. **Currently on phase 1 (done, awaiting confirmation).**

1. **HI rebuild** — LOAO training + F18-F20 validation, report AUC. ✅ done
   Also delivered: two derived thresholds, logit output, dual-scale figure.
2. **HI curve recompute** — F05 sorties 1244-1431 in true order.
   Point `build-hi` at `config.F05_SERIES_FOLDER` and `HI_PIPELINE_LOAO_PATH`;
   drop `F05_custom` / `F05_prediction_gt` from the default path (**keep the files**).
3. **Backtest framework** — rolling-origin; get naive + ARIMA working first.
4. **Foundation models** — attach, and fix the TTM context-length mismatch
   (`ttm_fc.py` loads the `512-192-r2` revision but feeds 65 points).
5. **Stats + censored RUL** — Diebold-Mariano with HAC (lag >= h-1), censored evaluation.

## Constraints that shape every decision

**No failure data at all — not merely no run-to-failure trajectories.** Every one of
the 17 training aircraft holds a few dozen sorties flown *before* it went into the
depot (orange) and a few dozen flown *after* (green). The label is "which side of the
depot visit", never "did this flight fail" — no failure event is recorded anywhere in
the dataset. Depot entry follows a fixed cycle set from prior reliability knowledge,
so the harsher framing is: the failure time is unobserved *and* "failure" itself is
undefinable from this data. What the LR actually learns is pre-depot vs post-depot
separability, i.e. a proxy for maintenance need, not failure probability. Within an
aircraft the series therefore trends *upward* (maintenance restores HI).
`training/Faulty` / `training/Healthy` are legacy folder names — read them as
pre-depot / post-depot. `thr_fail` is likewise legacy: it is the **進廠門檻**, "looks
like it needs a depot visit", not a failure threshold. Any RUL derived here means
"sorties until the depot threshold is crossed" — hence the censored framing.

**No synthetic data.** `F05_custom` and `F05_prediction_gt` were hand-picked and
re-sequenced for a report deadline (the "ground truth" was F08 faulty flights that
were also in HI training). They stay on disk as a fallback and are documented in
README "Data notes", but they are not research inputs. Do not reintroduce them.

**F05 sorties 1244-1431 (64 flights) is the only genuine degradation series.** No
maintenance event inside it; treat as sampled roughly every 25 sorties. Do NOT splice
the 30 F05 training flights (845-913) back on — 331-sortie gap plus a maintenance jump.

**Hardware:** no GPU, 4 cores, ~7 GB RAM. Favour TTM / Chronos-Bolt tiny for any
fine-tuning; TimesFM 2.5 (200M) is likely too heavy to fine-tune on CPU.

## Traps found the hard way

- **MA smoothing leaks the target.** Forecasting MA(w) at step h, a fraction (w-h)/w
  of the answer is already observed — MA50/h=8 is 84% known. The old MA50 results
  largely measure carry-forward. Headline results should use raw CV / MA05 (0% leakage
  at h>=8); MA10-MA50 belong in an appendix with the leakage fraction stated.
- **n=1 proves nothing.** The old design produced a single forecast window. Track A
  (F05, context 32 / horizon 8) gives 25 rolling windows; track B (fleet, context 16 /
  horizon 4, segments split at the maintenance event) gives 9 usable segments / 52
  windows. Track B cannot support RUL claims — its HI trends upward.
- **Threshold and smoothing interact.** At thr 0.6 the first crossing moves from
  sortie 1285 (raw) to 1403 (MA05). Fix the (threshold, MA) pair *before* looking at
  any forecast, and report a sensitivity sweep.
- `myfeature/*.csv` carry `plane`/`flight` id columns; drop `config.ID_COLS` before
  use or `FEAT_IDXS` selects the wrong features.
- `testingLabel.xlsx` lists 67 flights, only 65 CSVs exist. Labels are assigned by
  sortie-number range, so missing files just warn.
- No CJK font on this machine (Docker has Noto). Use `plotting.label(zh, en)`.

## Positioning

Yan, Koç & Lee (2004, *Production Planning & Control* 15(8)) built an LR degradation
index and extrapolated it with ARMA. This project swaps the extrapolator for TSF
foundation models. Frame the work as a **HI-forecasting benchmark with censored RUL**,
not as RUL point prediction.

Verified against the PDF in `docs/` (2026-08-16) — three traps:
- **"confidence value"/`CV` is NOT Yan's term.** The paper says *performance index* /
  *probability of failure*; CV comes from Lee's later IMS/Watchdog-Agent line. `CV` is
  fine as this code's own symbol, never as a quote from Yan.
- **Their index is P(failure), rising**; ours is its complement, falling. Threshold
  crossing runs the opposite way (they cross a "failure line" upward).
- **Their §2.2** fits the LR from technician-assigned failure probabilities
  (normal 0.01 / acceptable 0.25 / unacceptable 0.5) when no failure history exists.
  Our data is §2.1 in form, §2.2 in meaning — cite this as precedent for the
  pre-depot/post-depot labels.
Jamshidi, Kim & Arif (2025, arXiv:2506.20090) is CC BY 4.0 (checked on the arXiv abs
page); its Fig. 4 prognostics tree has five branches, and data-driven still splits only
into ML / DL — that gap is the paper's opening.
