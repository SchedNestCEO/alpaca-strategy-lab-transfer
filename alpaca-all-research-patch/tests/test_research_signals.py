from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace

import pandas as pd
import pytest

from bot.backtest import BacktestSettings
from bot.research_signals import (
    CongressDisclosure,
    HistoricalNewsAssessment,
    congress_score_at,
    load_congress_signals_csv,
    load_news_assessments_csv,
)
from bot.strategies.common import StrategySignal
from bot.strategy_lab import HistoricalOverlaySimulator, StrategyDefinition

UTC = timezone.utc


def _bars(periods: int = 290) -> pd.DataFrame:
    dates = list(pd.bdate_range("2024-01-02", periods=periods).date)
    closes = [100 * (1.0008 ** i) for i in range(periods)]
    return pd.DataFrame(
        {
            "open": closes,
            "high": [v * 1.01 for v in closes],
            "low": [v * 0.99 for v in closes],
            "close": closes,
            "volume": [1_000_000.0] * periods,
        },
        index=dates,
    )


def _settings(frame: pd.DataFrame) -> BacktestSettings:
    return BacktestSettings(
        start=frame.index[230],
        end=frame.index[250],
        initial_capital=Decimal("100"),
        watchlist=("AAA",),
        trend_fast_sma=50,
        trend_slow_sma=200,
        max_daily_drawdown_pct=Decimal("1"),
        max_high_water_drawdown_pct=Decimal("0.05"),
        slippage=Decimal("0"),
    )


def _green(history, fast, slow):
    del history, fast, slow
    return SimpleNamespace(regime="GREEN", score=100.0)


def _always_signal(history, regime, spy_history):
    del history, regime, spy_history
    return StrategySignal(stop_pct=Decimal("0.02"), target_multiple=Decimal("1"), score=50.0)


def test_congress_loader_rejects_disclosure_before_transaction(tmp_path):
    path = tmp_path / "congress.csv"
    path.write_text(
        "member_id,symbol,side,transaction_time,disclosure_time\n"
        "m1,AAA,BUY,2024-02-02T10:00:00+00:00,2024-02-01T10:00:00+00:00\n"
    )
    with pytest.raises(ValueError, match="cannot precede"):
        load_congress_signals_csv(path)


def test_congress_score_uses_disclosure_not_transaction_and_never_same_day():
    signal_day = date(2024, 3, 1)
    early_transaction_late_disclosure = CongressDisclosure(
        member_id="m1",
        symbol="AAA",
        side="BUY",
        transaction_time=datetime(2024, 1, 1, tzinfo=UTC),
        disclosure_time=datetime(2024, 3, 1, 12, tzinfo=UTC),
    )
    prior_public = CongressDisclosure(
        member_id="m2",
        symbol="AAA",
        side="BUY",
        transaction_time=datetime(2024, 1, 2, tzinfo=UTC),
        disclosure_time=datetime(2024, 2, 15, 12, tzinfo=UTC),
    )
    assert congress_score_at([early_transaction_late_disclosure, prior_public], symbol="AAA", signal_date=signal_day) == 25.0


def test_news_loader_requires_exact_signal_date(tmp_path):
    path = tmp_path / "news.csv"
    path.write_text(
        "symbol,signal_date,assessment_time,verdict\n"
        "AAA,2024-02-01,2024-02-02T01:00:00+00:00,SAFE\n"
    )
    with pytest.raises(ValueError, match="must match signal_date"):
        load_news_assessments_csv(path)


def test_news_overlay_fails_closed_when_assessment_missing():
    frame = _bars()
    definition = StrategyDefinition(
        "ma_20_100_news",
        "test",
        _always_signal,
        None,
        max_hold_bars=1,
    )
    result = HistoricalOverlaySimulator(
        definition,
        _settings(frame),
        {"AAA": frame, "SPY": frame},
        news_assessments=[],
        regime_fn=_green,
    ).run()
    assert result.trades == []
    assert result.summary["news_missing_candidates"] > 0


def test_news_risky_veto_blocks_candidate():
    frame = _bars()
    start = _settings(frame).start
    assessment = HistoricalNewsAssessment(
        symbol="AAA",
        signal_date=start,
        assessment_time=datetime.combine(start, datetime.min.time(), tzinfo=UTC) + timedelta(hours=20),
        verdict="RISKY",
        reason="test_risk",
    )
    definition = StrategyDefinition("ma_20_100_news", "test", _always_signal, None, max_hold_bars=1)
    result = HistoricalOverlaySimulator(
        definition,
        _settings(frame),
        {"AAA": frame, "SPY": frame},
        news_assessments=[assessment],
        regime_fn=_green,
    ).run()
    assert result.trades == []
    assert result.summary["news_vetoed_candidates"] == 1


def test_congress_overlay_cannot_create_trade_without_technical_signal():
    frame = _bars()
    start = _settings(frame).start
    disclosure = CongressDisclosure(
        member_id="m1",
        symbol="AAA",
        side="BUY",
        transaction_time=datetime.combine(start - timedelta(days=20), datetime.min.time(), tzinfo=UTC),
        disclosure_time=datetime.combine(start - timedelta(days=10), datetime.min.time(), tzinfo=UTC),
    )
    definition = StrategyDefinition(
        "ma_20_100_congress",
        "test",
        lambda history, regime, spy: None,
        None,
    )
    result = HistoricalOverlaySimulator(
        definition,
        _settings(frame),
        {"AAA": frame, "SPY": frame},
        congress_signals=[disclosure],
        regime_fn=_green,
    ).run()
    assert result.trades == []


def test_safe_news_and_congress_metadata_reach_trade():
    frame = _bars()
    cfg = _settings(frame)
    assessments = []
    for day in frame.index:
        if cfg.start <= day <= cfg.end:
            assessments.append(
                HistoricalNewsAssessment(
                    symbol="AAA",
                    signal_date=day,
                    assessment_time=datetime.combine(day, datetime.min.time(), tzinfo=UTC) + timedelta(hours=20),
                    verdict="SAFE",
                    reason="safe",
                )
            )
    disclosure = CongressDisclosure(
        member_id="m1",
        symbol="AAA",
        side="BUY",
        transaction_time=datetime.combine(cfg.start - timedelta(days=20), datetime.min.time(), tzinfo=UTC),
        disclosure_time=datetime.combine(cfg.start - timedelta(days=10), datetime.min.time(), tzinfo=UTC),
    )
    definition = StrategyDefinition(
        "ma_20_100_news_congress",
        "test",
        _always_signal,
        None,
        max_hold_bars=1,
    )
    result = HistoricalOverlaySimulator(
        definition,
        cfg,
        {"AAA": frame, "SPY": frame},
        congress_signals=[disclosure],
        news_assessments=assessments,
        regime_fn=_green,
    ).run()
    assert result.trades
    assert result.trades[0]["news_verdict"] == "SAFE"
    assert result.trades[0]["congress_score"] == 25.0
