from __future__ import annotations

import inspect
from datetime import date
from decimal import Decimal
from types import SimpleNamespace

import pandas as pd

import bot.robustness_lab as robust


def synthetic_frame(dates, *, growth=1.001):
    closes = [100 * (growth ** i) for i in range(len(dates))]
    return pd.DataFrame(
        {
            "open": closes,
            "high": [value * 1.004 for value in closes],
            "low": [value * 0.996 for value in closes],
            "close": closes,
            "volume": [1_000_000.0] * len(closes),
        },
        index=dates,
    )


def test_rolling_windows_are_fixed_length_and_step_without_partial_tail():
    windows = robust.rolling_windows(
        date(2022, 1, 15),
        date(2024, 7, 14),
        months=12,
        step_months=6,
    )
    assert [(w.start, w.end) for w in windows] == [
        (date(2022, 1, 15), date(2023, 1, 14)),
        (date(2022, 7, 15), date(2023, 7, 14)),
        (date(2023, 1, 15), date(2024, 1, 14)),
        (date(2023, 7, 15), date(2024, 7, 14)),
    ]


def test_calendar_year_windows_include_only_complete_years():
    windows = robust.calendar_year_windows(date(2021, 8, 18), date(2026, 8, 17))
    assert [window.label for window in windows] == ["2022", "2023", "2024", "2025"]


def test_exposure_matched_benchmark_uses_thirty_percent_capital():
    dates = list(pd.bdate_range("2025-01-02", periods=20).date)
    spy = synthetic_frame(dates, growth=1.01)
    qqq = synthetic_frame(dates, growth=1.0)
    rows = robust.exposure_matched_benchmarks(
        {"SPY": spy, "QQQ": qqq},
        start=dates[0],
        end=dates[-1],
        capital=Decimal("100"),
        slippage=Decimal("0"),
    )
    spy30 = next(row for row in rows if row["benchmark"] == "spy_30_cash_70")
    full_spy_return = float(spy.iloc[-1]["close"] / spy.iloc[0]["open"] - 1)
    assert abs(spy30["total_return_pct"] - full_spy_return * 0.30) < 1e-10


def test_window_stats_tracks_halts_and_medians():
    stats = robust._window_stats(
        [
            {"total_return_pct": 0.10, "profit_factor": 1.5, "maximum_drawdown_pct": 0.03, "portfolio_drawdown_halt_date": None},
            {"total_return_pct": -0.02, "profit_factor": 0.8, "maximum_drawdown_pct": 0.06, "portfolio_drawdown_halt_date": "2024-01-05"},
            {"total_return_pct": 0.04, "profit_factor": 1.2, "maximum_drawdown_pct": 0.04, "portfolio_drawdown_halt_date": None},
        ]
    )
    assert stats["window_count"] == 3
    assert stats["positive_rate"] == 2 / 3
    assert stats["halt_rate"] == 1 / 3
    assert stats["median_return"] == 0.04
    assert stats["median_profit_factor"] == 1.2
    assert stats["worst_drawdown"] == 0.06


def test_robustness_lab_has_no_broker_submission_path():
    source = inspect.getsource(robust)
    assert "submit_order" not in source
    assert "TradingClient" not in source


def test_robustness_smoke_writes_outputs(monkeypatch, tmp_path):
    all_dates = list(pd.bdate_range("2023-01-02", "2025-03-31").date)

    def fake_fetch(api_key, secret_key, symbols, *, start, end, feed_name):
        del api_key, secret_key, start, end, feed_name
        return {
            symbol: synthetic_frame(all_dates, growth=1.0008 + index * 0.00001)
            for index, symbol in enumerate(dict.fromkeys(symbols))
        }

    monkeypatch.setattr(robust, "fetch_historical_bars", fake_fetch)
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

    rc = robust.run_robustness(
        cfg,
        start_text="2024-01-02",
        end_text="2025-03-31",
        holdout_start_text="2024-09-02",
        capital="100",
        strategies_text="ma_20_100",
        slippages_text="0.001,0.0025,0.005",
        window_slippages_text="0.001,0.0025",
        rolling_months_text="6",
        window_step_months=12,
    )
    assert rc == 0
    roots = list((tmp_path / "backtests" / "robustness").iterdir())
    assert len(roots) == 1
    root = roots[0]
    for name in (
        "summary.csv",
        "summary.json",
        "slippage_stress.csv",
        "rolling_and_calendar_windows.csv",
        "shadow_continuation.csv",
        "exposure_matched_benchmarks.csv",
    ):
        assert (root / name).exists()
