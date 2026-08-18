from __future__ import annotations

from decimal import Decimal

import pandas as pd

from .common import (
    StrategySignal,
    bearish_engulfing_pattern,
    current_row,
    long_term_uptrend,
    stop_pct,
    three_down_pattern,
    three_line_strike_pattern,
)


def three_down_signal(
    history: pd.DataFrame,
    regime: str,
    *,
    yellow_rsi_max: float | None = None,
    require_green: bool = True,
) -> StrategySignal | None:
    prepared = current_row(history)
    if prepared is None or (require_green and regime != "GREEN") or (not require_green and regime == "RED"):
        return None
    _, row = prepared
    rsi_limit = yellow_rsi_max if yellow_rsi_max is not None else 40.0
    if not (
        long_term_uptrend(row)
        and three_down_pattern(history)
        and row["rsi14"] <= rsi_limit
        and row["close"] < row["sma20"]
    ):
        return None
    risk_stop = stop_pct(row, maximum=Decimal("0.04") if yellow_rsi_max is not None else Decimal("0.05"))
    return StrategySignal(risk_stop, score=float(40 - row["rsi14"])) if risk_stop else None


def bearish_engulfing_signal(
    history: pd.DataFrame,
    regime: str,
    *,
    yellow_rsi_max: float | None = None,
) -> StrategySignal | None:
    prepared = current_row(history)
    if prepared is None or regime == "RED":
        return None
    if yellow_rsi_max is None and regime != "GREEN":
        return None
    _, row = prepared
    rsi_limit = yellow_rsi_max if yellow_rsi_max is not None else 40.0
    if not (long_term_uptrend(row) and bearish_engulfing_pattern(history) and row["rsi14"] <= rsi_limit):
        return None
    risk_stop = stop_pct(row, maximum=Decimal("0.04") if yellow_rsi_max is not None else Decimal("0.05"))
    return StrategySignal(risk_stop, score=float(40 - row["rsi14"])) if risk_stop else None


def three_line_strike_signal(
    history: pd.DataFrame,
    regime: str,
    *,
    yellow_rsi_max: float | None = None,
) -> StrategySignal | None:
    prepared = current_row(history)
    if prepared is None or regime == "RED":
        return None
    work, row = prepared
    day_three = work.iloc[-2]
    short_term_retracement = pd.notna(day_three["sma20"]) and day_three["close"] < day_three["sma20"]
    if not (
        three_line_strike_pattern(history)
        and short_term_retracement
        and long_term_uptrend(row)
        and (yellow_rsi_max is None or row["rsi14"] <= yellow_rsi_max)
    ):
        return None
    risk_stop = stop_pct(row, maximum=Decimal("0.04") if yellow_rsi_max is not None else Decimal("0.05"))
    return StrategySignal(risk_stop, target_multiple=Decimal("2"), score=1.0) if risk_stop else None


def mean_reversion_exit(history: pd.DataFrame) -> bool:
    prepared = current_row(history)
    return prepared is not None and bool(prepared[1]["close"] >= prepared[1]["sma20"])