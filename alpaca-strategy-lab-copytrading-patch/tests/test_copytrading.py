from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace

import pandas as pd
import pytest

from bot.backtest import BacktestSettings
from bot.copy_signals import CopySignal, load_copy_signals_csv
from bot.strategies.copytrading import (
    COPY_CONSENSUS,
    COPY_FILTERED,
    COPY_RAW,
    COPY_STRATEGY_KEYS,
    DEFAULT_COPY_POLICY,
    consensus_count_at,
    evaluate_copy_signal,
    leader_quality_at,
    pair_closed_leader_trades,
)
from bot.strategy_lab import CopyStrategySimulator

UTC = timezone.utc


def signal(
    leader: str,
    side: str,
    when: datetime,
    *,
    leader_price: str = "100",
    follower_price: str = "100",
    symbol: str = "AAA",
    trade_id: str | None = None,
    verified: bool | None = True,
) -> CopySignal:
    return CopySignal(
        leader_id=leader,
        symbol=symbol,
        side=side,
        leader_time=when - timedelta(seconds=60),
        leader_price=Decimal(leader_price),
        signal_time=when,
        follower_price=Decimal(follower_price),
        source_trade_id=trade_id,
        verified=verified,
    )


def bars(periods: int = 280) -> pd.DataFrame:
    dates = list(pd.bdate_range("2024-01-02", periods=periods).date)
    closes = [100 * (1.0006 ** i) for i in range(periods)]
    return pd.DataFrame(
        {
            "open": closes,
            "high": [v * 1.004 for v in closes],
            "low": [v * 0.996 for v in closes],
            "close": closes,
            "volume": [1_000_000.0] * periods,
        },
        index=dates,
    )


def settings(frame: pd.DataFrame) -> BacktestSettings:
    return BacktestSettings(
        start=frame.index[230],
        end=frame.index[-1],
        initial_capital=Decimal("100"),
        watchlist=("AAA",),
        trend_fast_sma=50,
        trend_slow_sma=200,
        max_daily_drawdown_pct=Decimal("1"),
        max_high_water_drawdown_pct=Decimal("0.05"),
        slippage=Decimal("0"),
    )


def green_regime(history, fast, slow):
    del history, fast, slow
    return SimpleNamespace(regime="GREEN", score=100.0)


def test_loader_requires_timezone_aware_timestamps(tmp_path):
    path = tmp_path / "signals.csv"
    path.write_text(
        "leader_id,symbol,side,leader_time,leader_price,signal_time,follower_price\n"
        "x,AAA,BUY,2025-01-02T10:00:00,100,2025-01-02T10:01:00+00:00,101\n"
    )
    with pytest.raises(ValueError, match="timezone-aware"):
        load_copy_signals_csv(path)


def test_loader_rejects_unknown_columns(tmp_path):
    path = tmp_path / "signals.csv"
    path.write_text(
        "leader_id,symbol,side,leader_time,leader_price,signal_time,follower_price,surprise\n"
        "x,AAA,BUY,2025-01-02T10:00:00+00:00,100,2025-01-02T10:01:00+00:00,101,x\n"
    )
    with pytest.raises(ValueError, match="unknown columns"):
        load_copy_signals_csv(path)


def test_leader_quality_uses_only_trades_closed_strictly_before_candidate():
    base = datetime(2025, 1, 1, 15, tzinfo=UTC)
    signals = []
    for i in range(5):
        buy = signal("leader", "BUY", base + timedelta(days=i * 2), trade_id=f"t{i}")
        sell = signal(
            "leader",
            "SELL",
            base + timedelta(days=i * 2 + 1),
            leader_price="110",
            follower_price="109",
            trade_id=f"t{i}",
        )
        signals.extend([buy, sell])
    candidate_time = base + timedelta(days=12)
    candidate = signal("leader", "BUY", candidate_time, trade_id="candidate")
    # This loss closes exactly at the candidate timestamp and must not enter the gate.
    signals.extend(
        [
            signal("leader", "BUY", base + timedelta(days=11), leader_price="100", trade_id="late"),
            signal("leader", "SELL", candidate_time, leader_price="50", follower_price="50", trade_id="late"),
            candidate,
        ]
    )
    quality = leader_quality_at(candidate, pair_closed_leader_trades(signals))
    assert quality.closed_trades == 5
    assert quality.qualified


def test_consensus_never_counts_future_signal():
    now = datetime(2025, 4, 1, 15, tzinfo=UTC)
    candidate = signal("a", "BUY", now)
    future = signal("b", "BUY", now + timedelta(minutes=5))
    assert consensus_count_at(
        candidate,
        [candidate, future],
        closed_trades=[],
        qualified_only=False,
    ) == 1


def test_filtered_copy_rejects_fomo_disadvantage():
    frame = bars()
    now = datetime.combine(frame.index[-1], datetime.min.time(), tzinfo=UTC) + timedelta(hours=15)
    history = frame.loc[: frame.index[-1]]
    candidate = signal("new", "BUY", now, leader_price="100", follower_price="103")
    decision = evaluate_copy_signal(
        COPY_FILTERED,
        candidate,
        all_signals=[candidate],
        closed_trades=[],
        history=history,
        regime="GREEN",
        settings=settings(frame),
    )
    assert not decision.accepted
    assert decision.entry_disadvantage_pct > DEFAULT_COPY_POLICY.max_entry_disadvantage_pct


def test_all_eight_copy_strategies_are_registered():
    assert len(COPY_STRATEGY_KEYS) == 8
    assert COPY_RAW in COPY_STRATEGY_KEYS
    assert COPY_CONSENSUS in COPY_STRATEGY_KEYS


def test_copy_simulator_never_uses_leader_entry_price_as_follower_fill():
    frame = bars()
    start_day = frame.index[230]
    buy_time = datetime.combine(start_day, datetime.min.time(), tzinfo=UTC) + timedelta(hours=15)
    sell_day = frame.index[235]
    sell_time = datetime.combine(sell_day, datetime.min.time(), tzinfo=UTC) + timedelta(hours=15)
    signals = [
        signal("leader", "BUY", buy_time, leader_price="50", follower_price="150", trade_id="x"),
        signal("leader", "SELL", sell_time, leader_price="55", follower_price="151", trade_id="x"),
    ]
    sim = CopyStrategySimulator(
        COPY_RAW,
        settings(frame),
        {"AAA": frame, "SPY": frame},
        signals,
        regime_fn=green_regime,
    )
    result = sim.run()
    assert result.trades
    trade = result.trades[0]
    assert trade["entry_date"] > start_day.isoformat()
    assert Decimal(str(trade["entry_price"])) >= Decimal("150")
    assert Decimal(str(trade["entry_price"])) != Decimal("50")
    assert trade["leader_id"] == "leader"
    assert trade["copy_latency_seconds"] == 60.0


def test_pre_holdout_copy_signal_cannot_create_holdout_position():
    frame = bars()
    holdout_start = frame.index[250]
    pre_time = datetime.combine(frame.index[249], datetime.min.time(), tzinfo=UTC) + timedelta(hours=15)
    sig = signal("leader", "BUY", pre_time, follower_price="100")
    cfg = settings(frame)
    cfg = BacktestSettings(**{**cfg.__dict__, "start": holdout_start})
    result = CopyStrategySimulator(COPY_RAW, cfg, {"AAA": frame, "SPY": frame}, [sig], regime_fn=green_regime).run()
    assert result.trades == []
