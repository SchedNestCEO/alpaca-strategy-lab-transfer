from __future__ import annotations

import pandas as pd

from .common import StrategySignal, current_row, stop_pct


def donchian_55_signal(history: pd.DataFrame, regime: str) -> StrategySignal | None:
    prepared = current_row(history)
    if prepared is None or regime == "RED":
        return None
    _, row = prepared
    if pd.isna(row["prior55_high"]) or not (
        row["close"] > row["prior55_high"]
        and row["close"] > row["sma200"]
        and bool(row["sma200_rising"])
    ):
        return None
    risk_stop = stop_pct(row)
    return StrategySignal(risk_stop, score=float(row["close"] / row["prior55_high"] - 1)) if risk_stop else None


def donchian_20_exit(history: pd.DataFrame) -> bool:
    prepared = current_row(history)
    return (
        prepared is not None
        and pd.notna(prepared[1]["prior20_low"])
        and bool(prepared[1]["close"] < prepared[1]["prior20_low"])
    )