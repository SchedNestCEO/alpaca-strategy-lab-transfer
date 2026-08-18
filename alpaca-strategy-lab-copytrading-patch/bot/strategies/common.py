from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import pandas as pd

from ..indicators import atr_wilder, rsi_wilder


@dataclass(frozen=True)
class StrategySignal:
    stop_pct: Decimal
    target_multiple: Decimal | None = None
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


def indicators(history: pd.DataFrame) -> pd.DataFrame:
    """Calculate only information available through the last row of history."""
    work = history.copy().sort_index()
    close = work["close"]
    work["sma20"] = close.rolling(20).mean()
    work["sma50"] = close.rolling(50).mean()
    work["sma100"] = close.rolling(100).mean()
    work["sma200"] = close.rolling(200).mean()
    work["rsi14"] = rsi_wilder(close, 14)
    work["atr14"] = atr_wilder(work, 14)
    work["momentum20"] = close.pct_change(20)
    work["momentum60"] = close.pct_change(60)
    work["sma200_rising"] = work["sma200"] > work["sma200"].shift(20)
    work["prior55_high"] = work["high"].rolling(55).max().shift(1)
    work["prior20_low"] = work["low"].rolling(20).min().shift(1)
    return work


def current_row(history: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series] | None:
    required = (
        "close",
        "sma20",
        "sma50",
        "sma100",
        "sma200",
        "rsi14",
        "atr14",
        "momentum20",
        "momentum60",
    )
    # Strategy Lab precomputes backward-looking indicators once per symbol.
    # Reuse them when present; otherwise preserve the standalone pure-function behavior.
    work = history if all(column in history.columns for column in required) else indicators(history)
    if work.empty:
        return None
    row = work.iloc[-1]
    if any(pd.isna(row[column]) for column in required):
        return None
    return work, row


def stop_pct(row: pd.Series, *, maximum: Decimal = Decimal("0.05")) -> Decimal | None:
    atr_pct = float(row["atr14"] / row["close"])
    if not pd.notna(atr_pct):
        return None
    result = max(Decimal("0.02"), Decimal(str(2 * atr_pct)))
    return result if result <= maximum else None


def long_term_uptrend(row: pd.Series) -> bool:
    return (
        row["close"] > row["sma200"]
        and row["sma50"] > row["sma200"]
        and bool(row["sma200_rising"])
    )


def three_down_pattern(history: pd.DataFrame) -> bool:
    if len(history) < 4:
        return False
    closes = history["close"].iloc[-4:]
    return bool(
        closes.iloc[-3] < closes.iloc[-4]
        and closes.iloc[-2] < closes.iloc[-3]
        and closes.iloc[-1] < closes.iloc[-2]
    )


def bearish_engulfing_pattern(history: pd.DataFrame) -> bool:
    if len(history) < 2:
        return False
    previous = history.iloc[-2]
    current = history.iloc[-1]
    return bool(
        previous["close"] > previous["open"]
        and current["close"] < current["open"]
        and current["open"] >= previous["close"]
        and current["close"] <= previous["open"]
    )


def three_line_strike_pattern(history: pd.DataFrame) -> bool:
    if len(history) < 4:
        return False
    candles = history.iloc[-4:]
    first, second, third, fourth = (candles.iloc[index] for index in range(4))
    first_three_bearish = all(
        candle["close"] < candle["open"] for candle in (first, second, third)
    )
    lower_lows = second["low"] < first["low"] and third["low"] < second["low"]
    declining_closes = (
        second["close"] < first["close"] and third["close"] < second["close"]
    )
    fourth_bullish = fourth["close"] > fourth["open"]
    reversal = fourth["open"] < third["close"] and fourth["close"] > first["open"]
    return bool(first_three_bearish and lower_lows and declining_closes and fourth_bullish and reversal)