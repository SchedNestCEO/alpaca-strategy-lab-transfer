from __future__ import annotations

import argparse
import json
import os
import sys

import uvicorn

from bot.config import Config


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phone-first adaptive Alpaca PAPER trading research app")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("web", help="Run the authenticated mobile dashboard")
    sub.add_parser("scan", help="Run one market scan")
    sub.add_parser("cycle", help="Run scan + outcome labeling + weekly challenger maintenance")
    sub.add_parser("status", help="Print PAPER account / safety status")
    sub.add_parser("label", help="Label due candidate outcomes")
    sub.add_parser("train", help="Train a challenger model if enough labeled observations exist")
    sub.add_parser("init-db", help="Create database tables and default safety settings")

    backtest = sub.add_parser("backtest", help="Run an isolated historical daily-bar backtest")
    backtest.add_argument("--years", type=int, help="Number of calendar years ending today")
    backtest.add_argument("--start", help="Inclusive start date in YYYY-MM-DD format")
    backtest.add_argument("--end", help="Inclusive end date in YYYY-MM-DD format")
    backtest.add_argument("--capital", default="100", help="Starting capital, default: 100")
    backtest.add_argument("--slippage", default="0.001", help="Adverse slippage, default: 0.001")

    compare = sub.add_parser("compare", help="Run the research-only multi-strategy comparison lab")
    compare.add_argument("--start", required=True, help="Inclusive start date in YYYY-MM-DD format")
    compare.add_argument("--end", required=True, help="Inclusive end date in YYYY-MM-DD format")
    compare.add_argument("--capital", default="100", help="Starting capital, default: 100")
    compare.add_argument("--slippage", default="0.001", help="Adverse slippage, default: 0.001")
    compare.add_argument("--holdout-start", required=True, help="Start date for untouched holdout evaluation")

    flat = sub.add_parser("flatten", help="Cancel orders and close all PAPER positions")
    flat.add_argument("--confirm", required=True)

    promote = sub.add_parser("promote", help="Promote a challenger model to production ranking status")
    promote.add_argument("model_id", type=int)
    promote.add_argument("--confirm", required=True)
    return p


def main() -> int:
    args = parser().parse_args()
    try:
        cfg = Config.load()
        if args.command == "init-db":
            from bot.db import Database

            Database(cfg.database_url).init()
            print("Database initialized. Paper execution defaults to DISARMED.")
            return 0
        if args.command == "status":
            from bot.main import status

            print(json.dumps(status(cfg), indent=2, sort_keys=True))
            return 0
        if args.command == "scan":
            from bot.main import run_scan

            return run_scan(cfg)
        if args.command == "cycle":
            from bot.main import run_cycle

            print(json.dumps(run_cycle(cfg), indent=2, sort_keys=True))
            return 0
        if args.command == "label":
            from bot.main import label_outcomes

            print(json.dumps(label_outcomes(cfg), indent=2, sort_keys=True))
            return 0
        if args.command == "train":
            from bot.main import train_model

            print(json.dumps(train_model(cfg), indent=2, sort_keys=True))
            return 0
        if args.command == "backtest":
            from bot.backtest import run_backtest

            return run_backtest(
                cfg,
                years=args.years,
                start_text=args.start,
                end_text=args.end,
                capital=args.capital,
                slippage=args.slippage,
            )
        if args.command == "compare":
            from bot.strategy_lab import run_comparison

            return run_comparison(
                cfg,
                start_text=args.start,
                end_text=args.end,
                capital=args.capital,
                slippage=args.slippage,
                holdout_start_text=args.holdout_start,
            )
        if args.command == "promote":
            from bot.main import promote_model

            ok = promote_model(cfg, args.model_id, args.confirm)
            print("promoted" if ok else "model_not_found")
            return 0 if ok else 2
        if args.command == "flatten":
            from bot.main import flatten

            return flatten(cfg, args.confirm)
        if args.command == "web":
            from bot.dashboard import create_app

            cfg.require_dashboard_security()
            port = int(os.getenv("PORT", "5000"))
            uvicorn.run(create_app(cfg), host="0.0.0.0", port=port, log_level="info")
            return 0
        return 2
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
