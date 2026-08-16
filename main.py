#!/usr/bin/env python3
"""PHM pretrained-model test lab — single entry point.

Examples:
    python main.py prepare-features
    python main.py train-hi
    python main.py build-hi
    python main.py forecast --model traditional
    python main.py forecast --model timesfm --ma MA50
    python main.py forecast --model all
    python main.py eval --task tsf
    python main.py eval --task rul --thr 0.6
"""
import argparse

from src.config import AVA, HI_HOLDOUT_AVAS, MA_LIST, THR

FORECAST_MODELS = ["traditional", "timesfm", "chronos", "ttm"]


def _ma_args(args):
    return None if args.ma == "all" else [args.ma]


def cmd_prepare_features(args):
    from src.data.features import prepare_feature_csvs
    prepare_feature_csvs()


def cmd_train_hi(args):
    if args.scheme == "legacy":
        from src.data.hi import train_hi_pipeline
        train_hi_pipeline()
    else:
        from src.data.hi import train_hi_loao
        train_hi_loao(holdout=args.holdout, run_cv=not args.no_cv)


def cmd_build_hi(args):
    from src.data.hi import build_hi_full
    build_hi_full(args.ava)


def cmd_forecast(args):
    models = FORECAST_MODELS if args.model == "all" else [args.model]
    for model in models:
        if model == "traditional":
            from src.forecast import traditional
            traditional.run(args.ava, _ma_args(args))
        elif model == "timesfm":
            from src.forecast import timesfm_fc
            timesfm_fc.run(args.ava, _ma_args(args), mode=args.mode, checkpoint=args.checkpoint)
        elif model == "chronos":
            from src.forecast import chronos_fc
            chronos_fc.run(args.ava, _ma_args(args), mode=args.mode, checkpoint=args.checkpoint)
        elif model == "ttm":
            from src.forecast import ttm_fc
            ttm_fc.run(args.ava, _ma_args(args), mode=args.mode, checkpoint=args.checkpoint)


def cmd_eval(args):
    if args.task in ("tsf", "all"):
        from src.evaluate import tsf_metrics
        tsf_metrics.run(args.ava, _ma_args(args), mode=args.mode)
    if args.task in ("rul", "all"):
        from src.evaluate import rul
        rul.run(args.ava, _ma_args(args), mode=args.mode, thr=args.thr)


def build_parser():
    p = argparse.ArgumentParser(prog="main.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp, ava=True, ma=False, mode=False):
        if ava:
            sp.add_argument("--ava", default=AVA, help=f"aircraft/case id (default: {AVA})")
        if ma:
            sp.add_argument("--ma", default="all", choices=MA_LIST + ["all"],
                            help="which smoothed HI series to use (default: all)")
        if mode:
            sp.add_argument("--mode", default="zero-shot", choices=["zero-shot", "finetuned"],
                            help="forecast mode; affects output filenames (default: zero-shot)")

    sp = sub.add_parser("prepare-features", help="raw training/testing CSVs -> myfeature/*.csv")
    sp.set_defaults(func=cmd_prepare_features)

    sp = sub.add_parser("train-hi", help="train the LR health-indicator pipeline")
    sp.add_argument("--scheme", default="loao", choices=["loao", "legacy"],
                    help="loao: hold out aircraft + validate on F18-F20 (default); "
                         "legacy: fit all 554 training flights, no hold-out")
    sp.add_argument("--holdout", nargs="*", default=HI_HOLDOUT_AVAS, metavar="AVA",
                    help=f"aircraft excluded from LOAO training (default: {' '.join(HI_HOLDOUT_AVAS)})")
    sp.add_argument("--no-cv", action="store_true",
                    help="skip the leave-one-aircraft-out CV diagnostic")
    sp.set_defaults(func=cmd_train_hi)

    sp = sub.add_parser("build-hi", help="build the full HI series (context + ground truth)")
    add_common(sp)
    sp.set_defaults(func=cmd_build_hi)

    sp = sub.add_parser("forecast", help="run a forecasting model on the HI series")
    sp.add_argument("--model", required=True, choices=FORECAST_MODELS + ["all"])
    sp.add_argument("--checkpoint", default=None,
                    help="override model checkpoint (e.g. a fine-tuned local path)")
    add_common(sp, ma=True, mode=True)
    sp.set_defaults(func=cmd_forecast)

    sp = sub.add_parser("eval", help="evaluate forecasts (tsf metrics and/or RUL)")
    sp.add_argument("--task", default="all", choices=["tsf", "rul", "all"])
    sp.add_argument("--thr", type=float, default=THR, help=f"RUL HI threshold (default: {THR})")
    add_common(sp, ma=True, mode=True)
    sp.set_defaults(func=cmd_eval)

    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
