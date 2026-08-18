from __future__ import annotations

import pandas as pd

from .breakout import donchian_55_signal
from .common import StrategySignal
from .mean_reversion import (
    bearish_engulfing_signal,
    three_down_signal,
    three_line_strike_signal,
)
from .trend import ma_20_100_signal, ma_50_200_signal


def regime_adaptive_signal(
    history: pd.DataFrame,
    regime: str,
    spy_history: pd.DataFrame | None,
) -> StrategySignal | None:
    if spy_history is None or spy_history.empty:
        return None
    spy_close = spy_history["close"]
    if len(spy_close) < 21:
        return None
    spy_momentum20 = float(spy_close.iloc[-1] / spy_close.iloc[-21] - 1)

    if regime == "GREEN" and spy_momentum20 >= 0.02:
        for route, signal_fn in (
            ("donchian_55", donchian_55_signal),
            ("ma_20_100", ma_20_100_signal),
            ("ma_50_200", ma_50_200_signal),
        ):
            signal = signal_fn(history, regime)
            if signal:
                return StrategySignal(
                    signal.stop_pct,
                    signal.target_multiple,
                    signal.score,
                    {**signal.metadata, "adaptive_route": route},
                )
        # Fixed fallback: only consider reversal setups if no trend setup qualifies.
        for route, signal_fn in (
            ("three_down_mean_reversion", three_down_signal),
            ("bearish_engulfing_mean_reversion", bearish_engulfing_signal),
            ("three_line_strike", three_line_strike_signal),
        ):
            signal = signal_fn(history, regime)
            if signal:
                return StrategySignal(
                    signal.stop_pct,
                    signal.target_multiple,
                    signal.score,
                    {**signal.metadata, "adaptive_route": route},
                )
        return None

    if regime == "GREEN" and spy_momentum20 < 0.02:
        for route, signal_fn in (
            ("three_down_mean_reversion", three_down_signal),
            ("bearish_engulfing_mean_reversion", bearish_engulfing_signal),
            ("three_line_strike", three_line_strike_signal),
        ):
            signal = signal_fn(history, regime)
            if signal:
                return StrategySignal(
                    signal.stop_pct,
                    signal.target_multiple,
                    signal.score,
                    {**signal.metadata, "adaptive_route": route},
                )
        return None

    if regime == "YELLOW":
        for route, signal_fn in (
            ("three_down_mean_reversion", three_down_signal),
            ("bearish_engulfing_mean_reversion", bearish_engulfing_signal),
            ("three_line_strike", three_line_strike_signal),
        ):
            if signal_fn is three_down_signal:
                signal = signal_fn(history, regime, yellow_rsi_max=35, require_green=False)
            else:
                signal = signal_fn(history, regime, yellow_rsi_max=35)
            if signal:
                return StrategySignal(
                    signal.stop_pct,
                    signal.target_multiple,
                    signal.score,
                    {**signal.metadata, "adaptive_route": route},
                )
    return None