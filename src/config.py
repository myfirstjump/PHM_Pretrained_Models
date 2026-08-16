# Central configuration: paths and experiment constants.
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# --- Landing gear dataset layout ---
LG_DIR       = ROOT / "data" / "LandingGear_relatives"
RAW_DIR      = LG_DIR / "data"
CSV_DIR      = LG_DIR / "csv"
FEATURES_DIR = LG_DIR / "myfeature"
MODEL_DIR    = LG_DIR / "model"

TRAINING_HEALTHY_DIR = RAW_DIR / "training" / "Healthy"
TRAINING_FAULTY_DIR  = RAW_DIR / "training" / "Faulty"
TESTING_DIR          = RAW_DIR / "testing"
TESTING_LABEL_XLSX   = RAW_DIR / "testingLabel.xlsx"

PREDICTION_DIR = ROOT / "prediction"

# --- HI model ---
# Legacy pipeline: fitted on all 554 training flights, no hold-out. Kept so the
# pre-2026-08 outputs stay reproducible; not used by the LOAO workflow.
HI_PIPELINE_PATH = MODEL_DIR / "hi_lr_pipeline_1141019_self_features.pkl"
# LOAO pipeline: F05/F08 held out of training so the F05 forecast series and the
# F08 flights are unseen data at HI inference time.
HI_PIPELINE_LOAO_PATH = MODEL_DIR / "hi_lr_pipeline_loao.pkl"
# F08 was held out only because the abandoned F05_prediction_gt reused F08 training
# flights; that reason is gone. F05 stays out so "F05 never trained the HI model" is
# true without qualification (costs 30/554 samples and shifts F05 HI by <=0.017).
HI_HOLDOUT_AVAS = ["F05"]
# Column indices of the hand-picked features: 1-TO-Y-rms, 3-TO-Y-std, 4-TO-Y-peak2peak
FEAT_IDXS = [0, 2, 3]
# Identifier columns prepended to the myfeature/*.csv tables.
ID_COLS = ["plane", "flight"]

# --- Forecast experiment (F05 case) ---
AVA = "F05"
CONTEXT_FOLDER = RAW_DIR / "F05_custom"
GT_FOLDER      = RAW_DIR / "F05_prediction_gt"
# The real, unedited F05 series: sorties 1244-1431 in true chronological order,
# no maintenance event inside it. Replaces CONTEXT_FOLDER/GT_FOLDER in phase 2.
F05_SERIES_FOLDER = RAW_DIR / "F05_reserved"
N_CONTEXT = 65
N_HORIZON = 16
MA_WINDOWS = {"MA05": 5, "MA10": 10, "MA20": 20, "MA30": 30, "MA40": 40, "MA50": 50}
MA_LIST = list(MA_WINDOWS)

# --- RUL evaluation ---
# Two derived thresholds replace the single hand-picked 0.6 (see thresholds.py):
#   thr_alert — degradation onset (FPT), from the target aircraft's own healthy pool
#   thr_fail  — fault level, from the F18-F20 labels
# Both are written to HI_THRESHOLDS_PATH by `main.py train-hi` and read back here.
HI_THRESHOLDS_PATH = MODEL_DIR / "hi_thresholds.json"
ALERT_BASELINE_AVA = "F05"   # whose post-maintenance healthy pool defines the baseline
ALERT_K = 3.0                # k in "baseline centre - k * spread" (the 3-sigma rule)
ALERT_RULE = "mad"           # "mad" = robust (median/MAD), "std" = literature 3-sigma
FPT_CONSECUTIVE = 2          # points that must stay below thr_alert before FPT is called

THR_LEGACY = 0.6             # the original hand-picked value, kept for back-comparison


def _load_thresholds() -> dict:
    import json
    try:
        return json.loads(HI_THRESHOLDS_PATH.read_text())
    except (OSError, ValueError):
        return {}


_THR = _load_thresholds()
THR_ALERT = _THR.get("thr_alert")
THR_FAIL = _THR.get("thr_fail")

# `eval --thr` default. Still the legacy value: rewiring the RUL evaluation onto the
# alert/fail pair is phase 5, so results stay comparable until then.
THR = THR_LEGACY

# Forecast mode -> filename suffix. Zero-shot keeps the legacy names so results
# stay diff-able against previously committed outputs.
MODE_SUFFIX = {"zero-shot": "", "finetuned": "_ft"}


def pred_dir(ava: str = AVA) -> Path:
    d = PREDICTION_DIR / ava
    d.mkdir(parents=True, exist_ok=True)
    return d
