from __future__ import annotations

import inspect
from datetime import date
from decimal import Decimal

import pandas as pd

from bot.backtest import BacktestSettings, position_notional_cap
from bot.indicators import RegimeAssessment
from bot.strategies.breakout import donchian_20_exit
from bot.strategies.common import (
    StrategySignal,
    bearish_engulfing_pattern,
    indicators,
    three_down_pattern,
    three_line_strike_pattern,
)
from bot.strategies.mean_reversion import three_down_signal
from bot.strategies.trend import ma_20_100_signal, ma_50_200_signal
from bot.strategy_lab import (
    LabPosition,
    StrategyDefinition,
    StrategySimulator,
    _adverse_entry,
    _adverse_long_exit,
    resolve_stop_target_exit,
)


def frame_from_closes(closes, *, start="2023-01-02"):
    dates = pd.bdate_range(start, periods=len(closes)).date
    return pd.DataFrame(
        {
            "open": closes,
            "high": [float(c) * 1.003 for c in closes],
            "low": [float(c) * 0.997 for c in closes],
            "close": closes,
            "volume": [1_000_000.0] * len(closes),
        },
        index=dates,
    )


def flat_bars(dates, *, open_price=100.0, high=101.0, low=99.0, close=100.0):
    return pd.DataFrame(
        {
            "open": [open_price] * len(dates),
            "high": [high] * len(dates),
            "low": [low] * len(dates),
            "close": [close] * len(dates),
            "volume": [1_000_000.0] * len(dates),
        },
        index=dates,
    )


def green_regime(history, fast, slow):
    del history, fast, slow
    return RegimeAssessment("GREEN", 100, 100, 100, 100, 0.01, 0.10)


def red_regime(history, fast, slow):
    del history, fast, slow
    return RegimeAssessment("RED", 0, 100, 100, 100, -0.01, 0.50)


def base_settings(dates, **overrides):
    values = dict(
        start=dates[0],
        end=dates[-1],
        initial_capital=Decimal("100"),
        watchlist=("AAA",),
        trend_slow_sma=1,
        max_daily_drawdown_pct=Decimal("1"),
        max_high_water_drawdown_pct=Decimal("1"),
    )
    values.update(overrides)
    return BacktestSettings(**values)


def test_lab_signal_on_t_enters_t_plus_one():
    dates = list(pd.bdate_range("2025-01-02", periods=35).date)
    signal_date = dates[26]

    def signal(history, regime, spy_history):
        del regime, spy_history
        if history.index[-1] == signal_date:
            return StrategySignal(Decimal("0.05"), score=1.0)
        return None

    definition = StrategyDefinition("test", "Test", signal)
    bars = {"AAA": flat_bars(dates), "SPY": flat_bars(dates)}
    result = StrategySimulator(definition, base_settings(dates), bars, regime_fn=green_regime).run()
    assert result.trades
    assert result.trades[0]["signal_date"] == signal_date.isoformat()
    assert result.trades[0]["entry_date"] == dates[27].isoformat()


def test_lab_indicator_exit_on_t_executes_t_plus_one_open():
    dates = list(pd.bdate_range("2025-01-02", periods=36).date)
    signal_date = dates[26]
    exit_signal_date = dates[28]

    def signal(history, regime, spy_history):
        del regime, spy_history
        return StrategySignal(Decimal("0.05"), score=1.0) if history.index[-1] == signal_date else None

    def exit_signal(history, position):
        del position
        return history.index[-1] == exit_signal_date

    definition = StrategyDefinition("test", "Test", signal, exit_signal)
    bars = {"AAA": flat_bars(dates), "SPY": flat_bars(dates)}
    result = StrategySimulator(definition, base_settings(dates), bars, regime_fn=green_regime).run()
    trade = result.trades[0]
    assert trade["entry_date"] == dates[27].isoformat()
    assert trade["exit_date"] == dates[29].isoformat()
    assert trade["exit_reason"] == "signal_exit"


def test_donchian_55_threshold_excludes_todays_high():
    closes = [100 + i * 0.1 for i in range(60)]
    frame = frame_from_closes(closes)
    frame.iloc[-1, frame.columns.get_loc("high")] = 999.0
    work = indicators(frame)
    expected = frame["high"].iloc[-56:-1].max()
    assert work.iloc[-1]["prior55_high"] == expected
    assert work.iloc[-1]["prior55_high"] != 999.0


def test_donchian_20_exit_threshold_excludes_todays_low():
    closes = [100 + i * 0.1 for i in range(230)]
    frame = frame_from_closes(closes)
    previous_20_low = frame["low"].iloc[-21:-1].min()
    frame.iloc[-1, frame.columns.get_loc("low")] = 1.0
    frame.iloc[-1, frame.columns.get_loc("close")] = previous_20_low + 1.0
    work = indicators(frame)
    assert work.iloc[-1]["prior20_low"] == previous_20_low
    assert not donchian_20_exit(frame)


def test_ma_20_100_detection_on_clean_uptrend():
    closes = [100 * (1.001 ** i) for i in range(240)]
    frame = frame_from_closes(closes)
    assert ma_20_100_signal(frame, "GREEN") is not None


def test_ma_50_200_detection_on_clean_uptrend():
    closes = [100 * (1.001 ** i) for i in range(240)]
    frame = frame_from_closes(closes)
    assert ma_50_200_signal(frame, "GREEN") is not None


def test_three_consecutive_down_close_detection():
    frame = frame_from_closes([100, 101, 100, 99, 98])
    assert three_down_pattern(frame)


def test_bearish_engulfing_detection():
    dates = list(pd.bdate_range("2025-01-02", periods=2).date)
    frame = pd.DataFrame(
        {
            "open": [100, 103],
            "high": [103, 104],
            "low": [99, 98],
            "close": [102, 99],
            "volume": [1_000_000, 1_000_000],
        },
        index=dates,
    )
    assert bearish_engulfing_pattern(frame)


def test_exact_three_line_strike_detection():
    dates = list(pd.bdate_range("2025-01-02", periods=4).date)
    frame = pd.DataFrame(
        {
            "open": [110, 108, 105, 99],
            "high": [111, 109, 106, 112],
            "low": [105, 102, 99, 98],
            "close": [106, 103, 100, 111],
            "volume": [1_000_000] * 4,
        },
        index=dates,
    )
    assert three_line_strike_pattern(frame)


def test_near_three_line_strike_violation_is_rejected():
    dates = list(pd.bdate_range("2025-01-02", periods=4).date)
    frame = pd.DataFrame(
        {
            "open": [110, 108, 105, 99],
            "high": [111, 109, 106, 109],
            "low": [105, 102, 99, 98],
            "close": [106, 103, 100, 109],  # Does not close above day-1 open (110).
            "volume": [1_000_000] * 4,
        },
        index=dates,
    )
    assert not three_line_strike_pattern(frame)


def test_red_regime_opens_zero_positions():
    dates = list(pd.bdate_range("2025-01-02", periods=35).date)

    def signal(history, regime, spy_history):
        del history, regime, spy_history
        return StrategySignal(Decimal("0.05"), score=1.0)

    definition = StrategyDefinition("test", "Test", signal)
    bars = {"AAA": flat_bars(dates), "SPY": flat_bars(dates)}
    result = StrategySimulator(definition, base_settings(dates), bars, regime_fn=red_regime).run()
    assert result.trades == []
    assert max(row["positions_open"] for row in result.equity_curve) == 0


def qualifying_three_down_history():
    prices = []
    value = 100.0
    for _ in range(237):
        value *= 1.0015
        prices.append(value)
    for multiplier in (0.97, 0.96, 0.95):
        value *= multiplier
        prices.append(value)
    return frame_from_closes(prices)


def test_mean_reversion_only_trades_in_permitted_green_regime():
    history = qualifying_three_down_history()
    assert three_down_signal(history, "GREEN") is not None
    assert three_down_signal(history, "YELLOW") is None
    assert three_down_signal(history, "RED") is None


def test_risk_sizing_stays_at_or_below_half_percent_planned_risk():
    risk_capital = Decimal("100")
    stop = Decimal("0.05")
    notional = position_notional_cap(risk_capital, Decimal("100"), stop)
    planned_loss = notional * stop
    assert planned_loss <= risk_capital * Decimal("0.005")


def test_compounding_up_and_down_in_strategy_lab_position_cap():
    stop = Decimal("0.05")
    base = position_notional_cap(Decimal("100"), Decimal("100"), stop)
    up = position_notional_cap(Decimal("110"), Decimal("110"), stop)
    down = position_notional_cap(Decimal("90"), Decimal("90"), stop)
    assert up > base > down


def test_five_percent_account_drawdown_halts_future_entries():
    dates = list(pd.bdate_range("2025-01-02", periods=36).date)

    def signal(history, regime, spy_history):
        del history, regime, spy_history
        return StrategySignal(Decimal("0.05"), score=1.0)

    definition = StrategyDefinition("test", "Test", signal)
    aaa = flat_bars(dates, open_price=100.0, high=101.0, low=94.0, close=100.0)
    bars = {"AAA": aaa, "SPY": flat_bars(dates)}
    settings = base_settings(
        dates,
        position_alloc_pct=Decimal("1"),
        risk_per_trade_pct=Decimal("0.05"),
        max_positions=1,
        max_new_trades_per_day=1,
        max_high_water_drawdown_pct=Decimal("0.05"),
        slippage=Decimal("0"),
    )
    result = StrategySimulator(definition, settings, bars, regime_fn=green_regime).run()
    assert len(result.trades) == 1
    assert result.summary["portfolio_drawdown_halt_date"] == dates[26].isoformat()


def test_slippage_is_adverse_on_entry_and_exit():
    slip = Decimal("0.001")
    price = Decimal("100")
    assert _adverse_entry(price, slip) > price
    assert _adverse_long_exit(price, slip) < price
    decision = resolve_stop_target_exit(
        open_price=Decimal("100"),
        high=Decimal("110"),
        low=Decimal("99"),
        stop_price=Decimal("95"),
        target_price=Decimal("105"),
        slippage=slip,
    )
    assert decision is not None
    assert decision.price < Decimal("105")


def test_holdout_simulator_contains_no_pre_holdout_trades():
    all_dates = list(pd.bdate_range("2024-11-01", periods=45).date)
    holdout_start = all_dates[30]

    def signal(history, regime, spy_history):
        del regime, spy_history
        if history.index[-1] >= holdout_start:
            return StrategySignal(Decimal("0.05"), score=1.0)
        return None

    definition = StrategyDefinition("test", "Test", signal)
    bars = {"AAA": flat_bars(all_dates), "SPY": flat_bars(all_dates)}
    settings = BacktestSettings(
        start=holdout_start,
        end=all_dates[-1],
        initial_capital=Decimal("100"),
        watchlist=("AAA",),
        trend_slow_sma=1,
        max_daily_drawdown_pct=Decimal("1"),
        max_high_water_drawdown_pct=Decimal("1"),
    )
    result = StrategySimulator(definition, settings, bars, regime_fn=green_regime).run()
    assert result.trades
    assert all(date.fromisoformat(trade["entry_date"]) >= holdout_start for trade in result.trades)


def test_strategy_lab_has_no_broker_order_submission_path():
    import bot.strategy_lab as strategy_lab

    source = inspect.getsource(strategy_lab)
    assert "submit_order" not in source
    assert "TradingClient" not in source


def test_compare_smoke_runs_all_strategies_without_broker(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import bot.strategy_lab as lab

    all_dates = list(pd.bdate_range("2024-01-02", "2025-03-31").date)

    def synthetic_frame(multiplier: float):
        closes = [100 * multiplier * (1.0008 ** i) for i in range(len(all_dates))]
        return pd.DataFrame(
            {
                "open": closes,
                "high": [value * 1.004 for value in closes],
                "low": [value * 0.996 for value in closes],
                "close": closes,
                "volume": [1_000_000.0] * len(closes),
            },
            index=all_dates,
        )

    def fake_fetch(api_key, secret_key, symbols, *, start, end, feed_name):
        del api_key, secret_key, start, end, feed_name
        return {
            symbol: synthetic_frame(1 + index * 0.01)
            for index, symbol in enumerate(dict.fromkeys(symbols))
        }

    monkeypatch.setattr(lab, "fetch_historical_bars", fake_fetch)
    cfg = SimpleNamespace(
        watchlist=("SPY", "QQQ", "AAA"),
        position_alloc_pct=Decimal("0.10"),
        risk_per_trade_pct=Decimal("0.005"),
        max_positions=3,
        yellow_max_positions=1,
        max_new_trades_per_day=3,
        min_stop_pct=Decimal("0.02"),
        max_stop_pct=Decimal("0.05"),
        atr_stop_multiple=Decimal("2"),
        reward_risk_multiple=Decimal("2"),
        max_daily_drawdown_pct=Decimal("0.01"),
        max_high_water_drawdown_pct=Decimal("0.05"),
        cooldown_hours=72,
        trend_fast_sma=50,
        trend_slow_sma=200,
        pullback_sma=20,
        atr_period=14,
        rsi_period=14,
        max_atr_pct=0.025,
        min_technical_score=75.0,
        yellow_min_score=85.0,
        data_feed="iex",
        alpaca_api_key="paper-key",
        alpaca_secret_key="paper-secret",
        project_root=tmp_path,
    )

    rc = lab.run_comparison(
        cfg,
        start_text="2025-01-02",
        end_text="2025-03-31",
        capital="100",
        slippage="0.001",
        holdout_start_text="2025-02-17",
    )
    assert rc == 0
    roots = list((tmp_path / "backtests" / "comparisons").iterdir())
    assert len(roots) == 1
    root = roots[0]
    assert (root / "comparison.csv").exists()
    assert (root / "comparison_holdout.csv").exists()
    assert (root / "summary.json").exists()
    for key in lab.STRATEGY_KEYS:
        assert (root / key / "trades.csv").exists()
        assert (root / key / "equity_curve.csv").exists()
        assert (root / key / "summary.json").exists()


def test_vectorized_regime_matches_production_regime_logic():
    from bot.indicators import market_regime
    from bot.strategy_lab import precompute_market_regimes

    closes = [100 * (1.0007 ** i) * (1 + 0.002 * ((i % 10) - 5) / 5) for i in range(260)]
    frame = frame_from_closes(closes)
    mapping = precompute_market_regimes(frame, fast=50, slow=200)
    for current_date in list(mapping)[-10:]:
        expected = market_regime(frame.loc[:current_date], 50, 200)
        actual = mapping[current_date]
        assert actual.regime == expected.regime
        assert abs(actual.score - expected.score) < 1e-12
        assert abs(actual.sma200_slope20 - expected.sma200_slope20) < 1e-12
        assert abs(actual.realized_vol20 - expected.realized_vol20) < 1e-12
