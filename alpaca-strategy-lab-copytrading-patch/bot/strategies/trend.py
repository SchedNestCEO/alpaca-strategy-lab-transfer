from __future__ import annotations

import pandas as pd

from .common import StrategySignal, current_row, long_term_uptrend, stop_pct


def ma_20_100_signal(history: pd.DataFrame, regime: str) -> StrategySignal | None:
    prepared = current_row(history)
    if prepared is None or regime == "RED":
        return None
    _, row = prepared
    if not (
        row["sma20"] > row["sma100"]
        and row["close"] > row["sma200"]
        and bool(row["sma200_rising"])
        and row["momentum20"] > 0
        and row["momentum60"] > 0
    ):
        return None
    risk_stop = stop_pct(row)
    return StrategySignal(risk_stop, score=float(row["momentum20"])) if risk_stop else None


def ma_50_200_signal(history: pd.DataFrame, regime: str) -> StrategySignal | None:
    prepared = current_row(history)
    if prepared is None or regime == "RED":
        return None
    _, row = prepared
    if not (
        row["sma50"] > row["sma200"]
        and row["close"] > row["sma200"]
        and bool(row["sma200_rising"])
        and row["momentum20"] > 0
        and row["momentum60"] > 0
    ):
        return None
    risk_stop = stop_pct(row)
    return StrategySignal(risk_stop, score=float(row["momentum60"])) if risk_stop else None


def ma_20_exit(history: pd.DataFrame) -> bool:
    prepared = current_row(history)
    return prepared is not None and bool(prepared[1]["close"] < prepared[1]["sma20"])


def ma_50_exit(history: pd.DataFrame) -> bool:
    prepared = current_row(history)
    return prepared is not None and bool(prepared[1]["close"] < prepared[1]["sma50"])