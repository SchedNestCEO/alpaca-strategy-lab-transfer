from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal
from typing import Iterable

import pandas as pd

from ..copy_signals import CopySignal
from ..indicators import analyze_symbol
from .common import current_row

COPY_RAW = "copy_raw"
COPY_FILTERED = "copy_filtered"
COPY_CONSENSUS = "copy_consensus"
COPY_ELITE = "copy_elite"
COPY_ADAPTIVE = "copy_adaptive"
COPY_V2_HYBRID = "copy_v2_hybrid"
KAY_STYLE_RAW = "kay_style_raw"
KAY_STYLE_FILTERED = "kay_style_filtered"

COPY_STRATEGY_KEYS = (
    COPY_RAW,
    COPY_FILTERED,
    COPY_CONSENSUS,
    COPY_ELITE,
    COPY_ADAPTIVE,
    COPY_V2_HYBRID,
    KAY_STYLE_RAW,
    KAY_STYLE_FILTERED,
)


@dataclass(frozen=True)
class CopyPolicy:
    # INTEGRATION.md names the gates but does not prescribe numerical thresholds.
    # These are conservative, fixed research defaults -- not optimized parameters.
    rolling_trades: int = 20
    min_closed_trades: int = 5
    min_win_rate: float = 0.50
    min_profit_factor: float = 1.00
    min_average_return: float = 0.0
    max_entry_disadvantage_pct: float = 0.02
    consensus_window_hours: int = 24
    consensus_min_leaders: int = 2
    elite_min_closed_trades: int = 8
    elite_min_win_rate: float = 0.60
    elite_min_profit_factor: float = 1.50
    elite_min_average_return: float = 0.005
    elite_max_entry_disadvantage_pct: float = 0.01


DEFAULT_COPY_POLICY = CopyPolicy()


@dataclass(frozen=True)
class LeaderClosedTrade:
    leader_id: str
    symbol: str
    entry: CopySignal
    exit: CopySignal
    return_pct: float


@dataclass(frozen=True)
class LeaderQuality:
    qualified: bool
    elite: bool
    closed_trades: int
    win_rate: float
    profit_factor: float | None
    average_return: float


@dataclass(frozen=True)
class CopyDecision:
    accepted: bool
    score: float
    leader_quality: LeaderQuality
    consensus_count: int
    entry_disadvantage_pct: float
    copy_latency_seconds: float
    reason: str
    stop_pct: Decimal | None = None
    technical_score: float = 0.0


def pair_closed_leader_trades(signals: Iterable[CopySignal]) -> list[LeaderClosedTrade]:
    ordered = sorted(signals, key=lambda item: (item.signal_time, item.leader_id, item.symbol))
    exact_open: dict[tuple[str, str, str], deque[CopySignal]] = defaultdict(deque)
    generic_open: dict[tuple[str, str], deque[CopySignal]] = defaultdict(deque)
    closed: list[LeaderClosedTrade] = []

    for signal in ordered:
        if signal.side == "BUY":
            if signal.source_trade_id:
                exact_open[(signal.leader_id, signal.symbol, signal.source_trade_id)].append(signal)
            else:
                generic_open[(signal.leader_id, signal.symbol)].append(signal)
            continue

        entry: CopySignal | None = None
        if signal.source_trade_id:
            queue = exact_open.get((signal.leader_id, signal.symbol, signal.source_trade_id))
            if queue:
                entry = queue.popleft()
        if entry is None:
            queue = generic_open.get((signal.leader_id, signal.symbol))
            if queue:
                entry = queue.popleft()
        if entry is None or signal.signal_time <= entry.signal_time:
            continue
        closed.append(
            LeaderClosedTrade(
                leader_id=signal.leader_id,
                symbol=signal.symbol,
                entry=entry,
                exit=signal,
                return_pct=float(signal.leader_price / entry.leader_price - Decimal("1")),
            )
        )
    return closed


def _quality_from_returns(returns: list[float], policy: CopyPolicy) -> LeaderQuality:
    returns = returns[-policy.rolling_trades :]
    wins = [value for value in returns if value > 0]
    losses = [value for value in returns if value <= 0]
    gross_wins = sum(wins)
    gross_losses = abs(sum(losses))
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else (float("inf") if gross_wins > 0 else None)
    win_rate = len(wins) / len(returns) if returns else 0.0
    average_return = sum(returns) / len(returns) if returns else 0.0
    qualified = (
        len(returns) >= policy.min_closed_trades
        and win_rate >= policy.min_win_rate
        and average_return > policy.min_average_return
        and profit_factor is not None
        and profit_factor >= policy.min_profit_factor
    )
    elite = (
        len(returns) >= policy.elite_min_closed_trades
        and win_rate >= policy.elite_min_win_rate
        and average_return >= policy.elite_min_average_return
        and profit_factor is not None
        and profit_factor >= policy.elite_min_profit_factor
    )
    return LeaderQuality(qualified, elite, len(returns), win_rate, profit_factor, average_return)


def leader_quality_at(
    candidate: CopySignal,
    closed_trades: Iterable[LeaderClosedTrade],
    policy: CopyPolicy = DEFAULT_COPY_POLICY,
) -> LeaderQuality:
    # Strictly earlier close timestamps prevent the candidate or future trades from
    # leaking into the leader gate.
    returns = [
        trade.return_pct
        for trade in closed_trades
        if trade.leader_id == candidate.leader_id and trade.exit.signal_time < candidate.signal_time
    ]
    return _quality_from_returns(returns, policy)


def consensus_count_at(
    candidate: CopySignal,
    signals: Iterable[CopySignal],
    *,
    closed_trades: Iterable[LeaderClosedTrade],
    policy: CopyPolicy = DEFAULT_COPY_POLICY,
    qualified_only: bool = True,
) -> int:
    cutoff = candidate.signal_time - timedelta(hours=policy.consensus_window_hours)
    leaders: set[str] = set()
    for signal in signals:
        if signal.side != "BUY" or signal.symbol != candidate.symbol:
            continue
        if not (cutoff <= signal.signal_time <= candidate.signal_time):
            continue
        if signal.verified is False:
            continue
        if qualified_only and not leader_quality_at(signal, closed_trades, policy).qualified:
            continue
        leaders.add(signal.leader_id)
    return len(leaders)


def _technical_hybrid_ok(history: pd.DataFrame, regime: str, settings) -> tuple[bool, float]:
    if history.empty:
        return False, 0.0
    min_score = settings.min_technical_score if regime == "GREEN" else settings.yellow_min_score
    technical = analyze_symbol(
        history,
        rsi_period=settings.rsi_period,
        atr_period=settings.atr_period,
        pullback_sma=settings.pullback_sma,
        fast_sma=settings.trend_fast_sma,
        slow_sma=settings.trend_slow_sma,
        max_atr_pct=settings.max_atr_pct,
        min_score=min_score,
    )
    return technical.eligible, technical.score


def evaluate_copy_signal(
    strategy_key: str,
    candidate: CopySignal,
    *,
    all_signals: Iterable[CopySignal],
    closed_trades: Iterable[LeaderClosedTrade],
    history: pd.DataFrame,
    regime: str,
    settings,
    policy: CopyPolicy = DEFAULT_COPY_POLICY,
) -> CopyDecision:
    empty_quality = LeaderQuality(False, False, 0, 0.0, None, 0.0)
    if candidate.side != "BUY":
        return CopyDecision(False, 0.0, empty_quality, 0, candidate.entry_disadvantage_pct, candidate.latency_seconds, "not_long_entry")
    if regime == "RED":
        return CopyDecision(False, 0.0, empty_quality, 0, candidate.entry_disadvantage_pct, candidate.latency_seconds, "red_regime")

    prepared = current_row(history)
    if prepared is None:
        return CopyDecision(False, 0.0, empty_quality, 0, candidate.entry_disadvantage_pct, candidate.latency_seconds, "insufficient_history")
    _, row = prepared
    atr_pct = Decimal(str(float(row["atr14"] / row["close"])))
    dynamic_stop = max(settings.min_stop_pct, settings.atr_stop_multiple * atr_pct)
    if dynamic_stop > settings.max_stop_pct:
        return CopyDecision(False, 0.0, empty_quality, 0, candidate.entry_disadvantage_pct, candidate.latency_seconds, "stop_rejected")

    quality = leader_quality_at(candidate, closed_trades, policy)
    disadvantage = candidate.entry_disadvantage_pct
    consensus = consensus_count_at(
        candidate,
        all_signals,
        closed_trades=closed_trades,
        policy=policy,
        qualified_only=True,
    )
    verified_ok = candidate.verified is not False
    filtered_ok = verified_ok and quality.qualified and disadvantage <= policy.max_entry_disadvantage_pct
    elite_ok = verified_ok and quality.elite and disadvantage <= policy.elite_max_entry_disadvantage_pct
    consensus_ok = filtered_ok and consensus >= policy.consensus_min_leaders
    technical_score = 0.0

    if strategy_key in {COPY_RAW, KAY_STYLE_RAW}:
        accepted, reason = True, "raw"
    elif strategy_key in {COPY_FILTERED, KAY_STYLE_FILTERED}:
        accepted, reason = filtered_ok, "filtered"
    elif strategy_key == COPY_CONSENSUS:
        accepted, reason = consensus_ok, "consensus"
    elif strategy_key == COPY_ELITE:
        accepted, reason = elite_ok, "elite"
    elif strategy_key == COPY_ADAPTIVE:
        if regime == "GREEN":
            accepted = filtered_ok
            reason = "adaptive_green_filtered"
        else:
            accepted = elite_ok or consensus_ok
            reason = "adaptive_yellow_strict"
    elif strategy_key == COPY_V2_HYBRID:
        technical_ok, technical_score = _technical_hybrid_ok(history, regime, settings)
        accepted = filtered_ok and technical_ok
        reason = "v2_hybrid"
    else:
        raise ValueError(f"Unknown copy strategy {strategy_key!r}")

    score = 0.0
    if accepted:
        pf_component = min(3.0, quality.profit_factor or 0.0) * 10.0
        score = 40.0 + min(30.0, quality.win_rate * 30.0) + pf_component + min(20.0, consensus * 5.0)
        score += max(-10.0, min(10.0, -disadvantage * 100.0))
        score += min(20.0, technical_score * 0.2)
    return CopyDecision(
        accepted=accepted,
        score=score,
        leader_quality=quality,
        consensus_count=consensus,
        entry_disadvantage_pct=disadvantage,
        copy_latency_seconds=candidate.latency_seconds,
        reason=reason if accepted else f"{reason}_rejected",
        stop_pct=dynamic_stop,
        technical_score=technical_score,
    )
