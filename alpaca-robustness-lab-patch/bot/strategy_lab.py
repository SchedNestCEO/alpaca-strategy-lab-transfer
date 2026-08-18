from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .backtest import (
    BacktestResult,
    BacktestSettings,
    Backtester,
    adverse_exit_price,
    fetch_historical_bars,
    normalize_bars,
    parse_date,
    position_notional_cap,
)
from .config import Config
from .copy_signals import CopySignal, copy_signal_symbols, load_copy_signals_csv
from .indicators import RegimeAssessment, market_regime
from .risk import QTY_STEP
from .strategies.adaptive import regime_adaptive_signal
from .strategies.copytrading import (
    COPY_ADAPTIVE,
    COPY_CONSENSUS,
    COPY_ELITE,
    COPY_FILTERED,
    COPY_RAW,
    COPY_STRATEGY_KEYS,
    COPY_V2_HYBRID,
    KAY_STYLE_FILTERED,
    KAY_STYLE_RAW,
    CopyDecision,
    evaluate_copy_signal,
    pair_closed_leader_trades,
)
from .strategies.breakout import donchian_20_exit, donchian_55_signal
from .strategies.common import (
    StrategySignal,
    current_row,
    indicators,
    three_line_strike_pattern,
)
from .strategies.mean_reversion import (
    bearish_engulfing_signal,
    mean_reversion_exit,
    three_down_signal,
    three_line_strike_signal,
)
from .strategies.trend import (
    ma_20_100_signal,
    ma_20_exit,
    ma_50_200_signal,
    ma_50_exit,
)

UTC = timezone.utc
LAB_ETFS = ("SPY", "QQQ", "DIA", "IWM", "XLK", "XLF", "XLV", "XLI", "XLP", "XLY")
BASE_STRATEGY_KEYS = (
    "current_v2",
    "ma_20_100",
    "ma_50_200",
    "donchian_55",
    "three_down_mean_reversion",
    "bearish_engulfing_mean_reversion",
    "three_line_strike",
    "regime_adaptive",
)
# Backward-compatible registry constant. Copy strategies are appended when --copy-signals is supplied.
STRATEGY_KEYS = BASE_STRATEGY_KEYS
STRATEGY_LABELS = {
    "current_v2": "Current V2",
    "ma_20_100": "MA 20/100",
    "ma_50_200": "MA 50/200",
    "donchian_55": "Donchian 55",
    "three_down_mean_reversion": "3-Down Mean Rev",
    "bearish_engulfing_mean_reversion": "Engulfing Mean Rev",
    "three_line_strike": "Three-Line Strike",
    "regime_adaptive": "Regime Adaptive",
    COPY_RAW: "Copy Raw",
    COPY_FILTERED: "Copy Filtered",
    COPY_CONSENSUS: "Copy Consensus",
    COPY_ELITE: "Copy Elite",
    COPY_ADAPTIVE: "Copy Adaptive",
    COPY_V2_HYBRID: "Copy + V2 Hybrid",
    KAY_STYLE_RAW: "Kay-Style Raw",
    KAY_STYLE_FILTERED: "Kay-Style Filtered",
    "spy_buy_hold": "SPY Buy/Hold",
    "qqq_buy_hold": "QQQ Buy/Hold",
    "spy_qqq_50_50": "50/50 Benchmark",
}

TRADE_COLUMNS = [
    "symbol",
    "signal_date",
    "entry_date",
    "entry_price",
    "exit_date",
    "exit_price",
    "position_notional",
    "quantity",
    "stop_price",
    "target_price",
    "return_pct",
    "pnl_dollars",
    "holding_days",
    "holding_bars",
    "market_regime",
    "technical_score",
    "rsi",
    "atr_pct",
    "momentum20",
    "momentum60",
    "exit_reason",
    "ambiguous_stop_first",
    "strategy_route",
    "copy_latency_seconds",
    "entry_disadvantage_pct",
    "leader_return_pct",
    "follower_return_pct",
    "copyability_tax_pct",
    "leader_id",
    "leader_qualified_at_entry",
    "consensus_count",
]

EQUITY_COLUMNS = [
    "date",
    "cash",
    "open_position_value",
    "equity",
    "drawdown_pct",
    "positions_open",
    "market_regime",
]

COMPARISON_COLUMNS = [
    "strategy",
    "ending_capital",
    "total_return_pct",
    "annualized_return_pct",
    "maximum_drawdown_pct",
    "return_over_max_drawdown",
    "profit_factor",
    "win_rate",
    "total_trades",
    "worst_losing_streak",
    "percentage_of_time_invested",
    "average_open_positions",
    "portfolio_drawdown_halt_date",
    "ambiguous_daily_bar_exits",
    "copy_latency_seconds",
    "entry_disadvantage_pct",
    "leader_return_pct",
    "follower_return_pct",
    "copyability_tax_pct",
    "leader_id",
    "leader_qualified_at_entry",
    "consensus_count",
]


@dataclass(frozen=True)
class StrategyDefinition:
    key: str
    label: str
    signal_fn: Callable[[pd.DataFrame, str, pd.DataFrame | None], StrategySignal | None]
    exit_fn: Callable[[pd.DataFrame, "LabPosition"], bool] | None = None
    max_hold_bars: int | None = None
    count_three_line_occurrences: bool = False


@dataclass(frozen=True)
class LabPending:
    symbol: str
    signal_date: date
    entry_date: date
    reserved_cap: Decimal
    risk_capital: Decimal
    signal: StrategySignal
    regime: str
    rsi: float
    atr_pct: float
    momentum20: float
    momentum60: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LabPosition:
    symbol: str
    signal_date: date
    entry_date: date
    entry_price: Decimal
    quantity: Decimal
    position_notional: Decimal
    stop_price: Decimal
    target_price: Decimal | None
    market_regime: str
    technical_score: float
    rsi: float
    atr_pct: float
    momentum20: float
    momentum60: float
    route: str
    holding_bars: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LabExit:
    price: Decimal
    reason: str
    ambiguous_stop_first: bool = False


@dataclass
class LabResult:
    summary: dict[str, Any]
    trades: list[dict[str, Any]]
    equity_curve: list[dict[str, Any]]


@dataclass
class StrategyPeriodResults:
    full: LabResult | BacktestResult
    development: LabResult | BacktestResult
    holdout: LabResult | BacktestResult


@dataclass
class ComparisonResult:
    start: date
    end: date
    holdout_start: date
    initial_capital: Decimal
    slippage: Decimal
    data_feed: str
    strategies: dict[str, StrategyPeriodResults]
    benchmarks: dict[str, dict[str, dict[str, Any]]]
    output_dir: Path | None = None

    def write_outputs(self, root: Path) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output_dir = root / timestamp
        suffix = 1
        while output_dir.exists():
            output_dir = root / f"{timestamp}-{suffix}"
            suffix += 1
        output_dir.mkdir(parents=True)

        full_rows = comparison_rows(self.strategies, self.benchmarks, "full")
        holdout_rows = comparison_rows(self.strategies, self.benchmarks, "holdout")
        _write_csv(output_dir / "comparison.csv", COMPARISON_COLUMNS, full_rows)
        _write_csv(output_dir / "comparison_holdout.csv", COMPARISON_COLUMNS, holdout_rows)

        summary_payload = {
            "period": {"start": self.start.isoformat(), "end": self.end.isoformat()},
            "holdout_start": self.holdout_start.isoformat(),
            "starting_capital": float(self.initial_capital),
            "slippage": float(self.slippage),
            "historical_data_feed": self.data_feed,
            "rankings": rank_holdout_strategies(self.strategies),
            "strategies": {
                key: {
                    "full": value.full.summary,
                    "development": value.development.summary,
                    "holdout": value.holdout.summary,
                }
                for key, value in self.strategies.items()
            },
            "benchmarks": self.benchmarks,
        }
        with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary_payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

        for key, periods in self.strategies.items():
            strategy_dir = output_dir / key
            strategy_dir.mkdir()
            _write_csv(strategy_dir / "trades.csv", TRADE_COLUMNS, _normalize_trade_rows(periods.full.trades))
            _write_csv(strategy_dir / "equity_curve.csv", EQUITY_COLUMNS, periods.full.equity_curve)
            with (strategy_dir / "summary.json").open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "full": periods.full.summary,
                        "development": periods.development.summary,
                        "holdout": periods.holdout.summary,
                    },
                    handle,
                    indent=2,
                    sort_keys=True,
                )
                handle.write("\n")

        self.output_dir = output_dir
        return output_dir


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _normalize_trade_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        result = {column: row.get(column, "") for column in TRADE_COLUMNS}
        result["holding_bars"] = row.get("holding_bars", row.get("holding_days", ""))
        result["strategy_route"] = row.get("strategy_route", "current_v2")
        normalized.append(result)
    return normalized


def _decimal(value: Any) -> Decimal:
    return Decimal(str(value))


def _frame_through(frame: pd.DataFrame | None, through: date) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    return frame.loc[:through]


def _next_symbol_date(frame: pd.DataFrame, signal_date: date, end: date) -> date | None:
    later = frame.index[(frame.index > signal_date) & (frame.index <= end)]
    return later[0] if len(later) else None


def _adverse_entry(open_price: Decimal, slippage: Decimal) -> Decimal:
    return open_price * (Decimal("1") + slippage)


def _adverse_long_exit(price: Decimal, slippage: Decimal) -> Decimal:
    return price * (Decimal("1") - slippage)


def _qty_for_notional(notional: Decimal, price: Decimal) -> Decimal:
    if notional <= 0 or price <= 0:
        return Decimal("0")
    return (notional / price).quantize(QTY_STEP, rounding=ROUND_DOWN)


def resolve_stop_target_exit(
    *,
    open_price: Decimal,
    high: Decimal,
    low: Decimal,
    stop_price: Decimal,
    target_price: Decimal | None,
    slippage: Decimal,
) -> LabExit | None:
    if open_price <= stop_price:
        return LabExit(_adverse_long_exit(open_price, slippage), "stop_gap")
    if target_price is not None and open_price >= target_price:
        return LabExit(_adverse_long_exit(open_price, slippage), "target_gap")

    stop_touched = low <= stop_price
    target_touched = target_price is not None and high >= target_price
    if stop_touched and target_touched:
        return LabExit(
            _adverse_long_exit(stop_price, slippage),
            "stop",
            ambiguous_stop_first=True,
        )
    if stop_touched:
        return LabExit(_adverse_long_exit(stop_price, slippage), "stop")
    if target_touched and target_price is not None:
        return LabExit(_adverse_long_exit(target_price, slippage), "target")
    return None


def _signal_wrapper(fn: Callable[[pd.DataFrame, str], StrategySignal | None]) -> Callable[[pd.DataFrame, str, pd.DataFrame | None], StrategySignal | None]:
    def wrapped(history: pd.DataFrame, regime: str, spy_history: pd.DataFrame | None) -> StrategySignal | None:
        del spy_history
        return fn(history, regime)

    return wrapped


def _exit_wrapper(fn: Callable[[pd.DataFrame], bool]) -> Callable[[pd.DataFrame, LabPosition], bool]:
    def wrapped(history: pd.DataFrame, position: LabPosition) -> bool:
        del position
        return fn(history)

    return wrapped


def _adaptive_signal(history: pd.DataFrame, regime: str, spy_history: pd.DataFrame | None) -> StrategySignal | None:
    return regime_adaptive_signal(history, regime, spy_history)


def _adaptive_exit(history: pd.DataFrame, position: LabPosition) -> bool:
    route = position.route
    if route == "donchian_55":
        return donchian_20_exit(history)
    if route == "ma_20_100":
        return ma_20_exit(history)
    if route == "ma_50_200":
        return ma_50_exit(history)
    if route in {
        "three_down_mean_reversion",
        "bearish_engulfing_mean_reversion",
    }:
        return mean_reversion_exit(history)
    if route == "three_line_strike":
        return False
    return False


def strategy_definitions() -> dict[str, StrategyDefinition]:
    return {
        "ma_20_100": StrategyDefinition(
            "ma_20_100",
            STRATEGY_LABELS["ma_20_100"],
            _signal_wrapper(ma_20_100_signal),
            _exit_wrapper(ma_20_exit),
        ),
        "ma_50_200": StrategyDefinition(
            "ma_50_200",
            STRATEGY_LABELS["ma_50_200"],
            _signal_wrapper(ma_50_200_signal),
            _exit_wrapper(ma_50_exit),
        ),
        "donchian_55": StrategyDefinition(
            "donchian_55",
            STRATEGY_LABELS["donchian_55"],
            _signal_wrapper(donchian_55_signal),
            _exit_wrapper(donchian_20_exit),
        ),
        "three_down_mean_reversion": StrategyDefinition(
            "three_down_mean_reversion",
            STRATEGY_LABELS["three_down_mean_reversion"],
            _signal_wrapper(three_down_signal),
            _exit_wrapper(mean_reversion_exit),
            max_hold_bars=5,
        ),
        "bearish_engulfing_mean_reversion": StrategyDefinition(
            "bearish_engulfing_mean_reversion",
            STRATEGY_LABELS["bearish_engulfing_mean_reversion"],
            _signal_wrapper(bearish_engulfing_signal),
            _exit_wrapper(mean_reversion_exit),
            max_hold_bars=5,
        ),
        "three_line_strike": StrategyDefinition(
            "three_line_strike",
            STRATEGY_LABELS["three_line_strike"],
            _signal_wrapper(three_line_strike_signal),
            None,
            max_hold_bars=10,
            count_three_line_occurrences=True,
        ),
        "regime_adaptive": StrategyDefinition(
            "regime_adaptive",
            STRATEGY_LABELS["regime_adaptive"],
            _adaptive_signal,
            _adaptive_exit,
        ),
    }


def precompute_market_regimes(
    spy: pd.DataFrame,
    *,
    fast: int = 50,
    slow: int = 200,
) -> dict[date, RegimeAssessment]:
    """Vectorized equivalent of market_regime, using only data at-or-before each row."""
    if spy.empty:
        return {}
    close = spy["close"].astype(float)
    sma_fast = close.rolling(fast).mean()
    sma_slow = close.rolling(slow).mean()
    vol20 = close.pct_change().rolling(20).std() * (252 ** 0.5)
    slope20 = sma_slow / sma_slow.shift(20) - 1
    result: dict[date, RegimeAssessment] = {}
    for position, current_date in enumerate(spy.index):
        if position + 1 < slow + 25:
            continue
        c = float(close.iloc[position])
        f = float(sma_fast.iloc[position])
        sl = float(sma_slow.iloc[position])
        slope = float(slope20.iloc[position])
        vol = float(vol20.iloc[position])
        if any(pd.isna(value) for value in (c, f, sl, slope, vol)):
            continue
        score = 0.0
        if c > sl:
            score += 35
        if f > sl:
            score += 30
        if slope > 0:
            score += 20
        if vol < 0.25:
            score += 15
        elif vol < 0.35:
            score += 8
        if c < sl or vol >= 0.45:
            regime = "RED"
        elif score >= 85:
            regime = "GREEN"
        else:
            regime = "YELLOW"
        result[current_date] = RegimeAssessment(regime, score, c, f, sl, slope, vol)
    return result


class StrategySimulator:
    """Research-only daily-bar portfolio simulator. No broker/database dependencies."""

    def __init__(
        self,
        definition: StrategyDefinition,
        settings: BacktestSettings,
        bars: dict[str, pd.DataFrame],
        *,
        regime_fn: Callable[..., RegimeAssessment] = market_regime,
        prepared_bars: bool = False,
        precomputed_regimes: dict[date, RegimeAssessment] | None = None,
    ):
        self.definition = definition
        self.settings = settings
        self.bars = (
            bars
            if prepared_bars
            else {symbol: indicators(normalize_bars(frame)) for symbol, frame in bars.items()}
        )
        self.regime_fn = regime_fn
        self._precomputed_regimes = (
            precomputed_regimes
            if precomputed_regimes is not None
            else (
                precompute_market_regimes(
                    self.bars.get("SPY", pd.DataFrame()),
                    fast=settings.trend_fast_sma,
                    slow=settings.trend_slow_sma,
                )
                if regime_fn is market_regime
                else None
            )
        )
        self.cash = settings.initial_capital
        self.positions: dict[str, LabPosition] = {}
        self.pending_entries: dict[date, list[LabPending]] = defaultdict(list)
        self.pending_exits: dict[date, dict[str, str]] = defaultdict(dict)
        self.reserved_cash = Decimal("0")
        self.trades: list[dict[str, Any]] = []
        self.equity_curve: list[dict[str, Any]] = []
        self.last_exit: dict[str, date] = {}
        self.high_water = settings.initial_capital
        self.halt_date: date | None = None
        self.invested_days = 0
        self._last_equity = settings.initial_capital
        self._regime_cache: dict[date, RegimeAssessment | None] = {}
        self.three_line_occurrences = 0
        self._three_line_occurrences_by_date = (
            self._precompute_three_line_occurrences()
            if definition.count_three_line_occurrences
            else {}
        )

    def _precompute_three_line_occurrences(self) -> dict[date, int]:
        counts: dict[date, int] = defaultdict(int)
        for symbol in self.settings.watchlist:
            frame = self.bars.get(symbol)
            if frame is None or len(frame) < 4:
                continue
            for index in range(3, len(frame)):
                window = frame.iloc[index - 3 : index + 1]
                if three_line_strike_pattern(window):
                    counts[frame.index[index]] += 1
        return dict(counts)

    def run(self) -> LabResult:
        trading_dates = self._trading_dates()
        if not trading_dates:
            raise ValueError("No historical bars were returned for the requested period")

        for current_date in trading_dates:
            day_start_equity = self._last_equity
            self._execute_pending_exits(current_date)
            self._enter_pending(current_date)
            had_position = bool(self.positions)
            self._process_hard_exits(current_date)
            self._increment_holding_bars(current_date)
            self._count_pattern_occurrences(current_date)

            regime_assessment = self._regime_for(current_date)
            regime_name = regime_assessment.regime if regime_assessment else "UNKNOWN"
            equity = self._equity(current_date)
            if had_position or self.positions:
                self.invested_days += 1

            self.high_water = max(self.high_water, equity)
            drawdown = self._drawdown(equity)
            if self.halt_date is None and drawdown >= self.settings.max_high_water_drawdown_pct:
                self.halt_date = current_date

            self.equity_curve.append(
                {
                    "date": current_date.isoformat(),
                    "cash": float(self.cash),
                    "open_position_value": float(self._open_position_value(current_date)),
                    "equity": float(equity),
                    "drawdown_pct": float(drawdown),
                    "positions_open": len(self.positions),
                    "market_regime": regime_name,
                }
            )

            daily_drawdown = (
                max(Decimal("0"), (day_start_equity - equity) / day_start_equity)
                if day_start_equity > 0
                else Decimal("0")
            )
            self._schedule_signal_exits(current_date)
            if self.halt_date is None and daily_drawdown < self.settings.max_daily_drawdown_pct:
                self._schedule_entries(current_date, regime_assessment)
            self._last_equity = equity

        self._liquidate_at_end(trading_dates[-1])
        self._refresh_final_equity_row()
        summary = self._build_summary(trading_dates)
        return LabResult(summary=summary, trades=self.trades, equity_curve=self.equity_curve)

    def _trading_dates(self) -> list[date]:
        spy = self.bars.get("SPY")
        if spy is not None and not spy.empty:
            return [d for d in spy.index if self.settings.start <= d <= self.settings.end]
        dates: set[date] = set()
        for frame in self.bars.values():
            dates.update(d for d in frame.index if self.settings.start <= d <= self.settings.end)
        return sorted(dates)

    def _bar_for(self, symbol: str, current_date: date) -> pd.Series | None:
        frame = self.bars.get(symbol)
        if frame is None or current_date not in frame.index:
            return None
        return frame.loc[current_date]

    def _close_for(self, symbol: str, current_date: date) -> Decimal | None:
        history = _frame_through(self.bars.get(symbol), current_date)
        if history.empty:
            return None
        return _decimal(history.iloc[-1]["close"])

    def _open_position_value(self, current_date: date) -> Decimal:
        total = Decimal("0")
        for position in self.positions.values():
            close = self._close_for(position.symbol, current_date)
            if close is not None:
                total += position.quantity * close
        return total

    def _equity(self, current_date: date) -> Decimal:
        return self.cash + self._open_position_value(current_date)

    def _risk_capital(self, current_date: date) -> Decimal:
        equity = self._equity(current_date)
        cost_basis = sum(
            (position.quantity * position.entry_price for position in self.positions.values()),
            Decimal("0"),
        )
        realized_capital = self.cash + cost_basis
        return min(equity, realized_capital)

    def _drawdown(self, equity: Decimal) -> Decimal:
        if self.high_water <= 0:
            return Decimal("0")
        return max(Decimal("0"), (self.high_water - equity) / self.high_water)

    def _regime_for(self, current_date: date) -> RegimeAssessment | None:
        if current_date in self._regime_cache:
            return self._regime_cache[current_date]
        if self._precomputed_regimes is not None:
            result = self._precomputed_regimes.get(current_date)
            self._regime_cache[current_date] = result
            return result
        history = _frame_through(self.bars.get("SPY"), current_date)
        if len(history) < self.settings.trend_slow_sma + 25:
            self._regime_cache[current_date] = None
            return None
        try:
            result = self.regime_fn(
                history,
                self.settings.trend_fast_sma,
                self.settings.trend_slow_sma,
            )
        except (ValueError, IndexError, KeyError, TypeError):
            result = None
        self._regime_cache[current_date] = result
        return result

    def _cooldown_active(self, symbol: str, current_date: date) -> bool:
        previous = self.last_exit.get(symbol)
        if previous is None:
            return False
        return (current_date - previous).days * 24 < self.settings.cooldown_hours

    def _effective_existing_symbols(self) -> set[str]:
        scheduled_exits = {
            symbol
            for exits in self.pending_exits.values()
            for symbol in exits
        }
        active = set(self.positions) - scheduled_exits
        pending = {
            entry.symbol
            for entries in self.pending_entries.values()
            for entry in entries
        }
        return active | pending

    def _schedule_entries(
        self,
        current_date: date,
        regime_assessment: RegimeAssessment | None,
    ) -> None:
        if regime_assessment is None or regime_assessment.regime == "RED":
            return
        allowed_positions = (
            self.settings.max_positions
            if regime_assessment.regime == "GREEN"
            else self.settings.yellow_max_positions
        )
        existing_symbols = self._effective_existing_symbols()
        slots = max(0, allowed_positions - len(existing_symbols))
        if slots <= 0:
            return

        spy_history = _frame_through(self.bars.get("SPY"), current_date)
        candidates: list[tuple[float, str, StrategySignal, date, dict[str, float]]] = []
        for symbol in self.settings.watchlist:
            if symbol in existing_symbols or self._cooldown_active(symbol, current_date):
                continue
            frame = self.bars.get(symbol)
            if frame is None or frame.empty:
                continue
            history = _frame_through(frame, current_date)
            if history.empty:
                continue
            try:
                signal = self.definition.signal_fn(history, regime_assessment.regime, spy_history)
            except (ValueError, IndexError, KeyError, TypeError, ArithmeticError):
                continue
            if signal is None:
                continue
            entry_date = _next_symbol_date(frame, current_date, self.settings.end)
            if entry_date is None:
                continue
            features = self._features(history)
            candidates.append((signal.score, symbol, signal, entry_date, features))

        candidates.sort(key=lambda item: (-item[0], item[1]))
        risk_capital = self._risk_capital(current_date)
        scheduled_today = 0
        for _, symbol, signal, entry_date, features in candidates:
            if slots <= 0 or scheduled_today >= self.settings.max_new_trades_per_day:
                break
            available_cash = max(Decimal("0"), self.cash - self.reserved_cash)
            cap = position_notional_cap(
                risk_capital,
                available_cash,
                signal.stop_pct,
                allocation_pct=self.settings.position_alloc_pct,
                risk_per_trade_pct=self.settings.risk_per_trade_pct,
            )
            if cap < Decimal("1"):
                continue
            self.pending_entries[entry_date].append(
                LabPending(
                    symbol=symbol,
                    signal_date=current_date,
                    entry_date=entry_date,
                    reserved_cap=cap,
                    risk_capital=risk_capital,
                    signal=signal,
                    regime=regime_assessment.regime,
                    rsi=features["rsi"],
                    atr_pct=features["atr_pct"],
                    momentum20=features["momentum20"],
                    momentum60=features["momentum60"],
                )
            )
            self.reserved_cash += cap
            existing_symbols.add(symbol)
            slots -= 1
            scheduled_today += 1

    @staticmethod
    def _features(history: pd.DataFrame) -> dict[str, float]:
        prepared = current_row(history)
        if prepared is None:
            return {"rsi": 0.0, "atr_pct": 0.0, "momentum20": 0.0, "momentum60": 0.0}
        _, row = prepared
        return {
            "rsi": float(row["rsi14"]),
            "atr_pct": float(row["atr14"] / row["close"]),
            "momentum20": float(row["momentum20"]),
            "momentum60": float(row["momentum60"]),
        }

    def _enter_pending(self, current_date: date) -> None:
        intents = self.pending_entries.pop(current_date, [])
        for intent in intents:
            self.reserved_cash = max(Decimal("0"), self.reserved_cash - intent.reserved_cap)
            if intent.symbol in self.positions:
                continue
            bar = self._bar_for(intent.symbol, current_date)
            if bar is None:
                continue
            open_price = _decimal(bar["open"])
            if open_price <= 0:
                continue
            entry_price = _adverse_entry(open_price, self.settings.slippage)
            cap = min(intent.reserved_cap, self.cash)
            qty = _qty_for_notional(cap, entry_price)
            notional = qty * entry_price
            if qty <= 0 or notional < Decimal("1") or notional > self.cash:
                continue
            stop_price = entry_price * (Decimal("1") - intent.signal.stop_pct)
            target_price = (
                entry_price * (Decimal("1") + intent.signal.stop_pct * intent.signal.target_multiple)
                if intent.signal.target_multiple is not None
                else None
            )
            self.cash -= notional
            route = str(intent.signal.metadata.get("adaptive_route") or self.definition.key)
            self.positions[intent.symbol] = LabPosition(
                symbol=intent.symbol,
                signal_date=intent.signal_date,
                entry_date=current_date,
                entry_price=entry_price,
                quantity=qty,
                position_notional=notional,
                stop_price=stop_price,
                target_price=target_price,
                market_regime=intent.regime,
                technical_score=float(intent.signal.score),
                rsi=intent.rsi,
                atr_pct=intent.atr_pct,
                momentum20=intent.momentum20,
                momentum60=intent.momentum60,
                route=route,
                metadata=dict(intent.metadata),
            )

    def _execute_pending_exits(self, current_date: date) -> None:
        exits = self.pending_exits.pop(current_date, {})
        for symbol, reason in list(exits.items()):
            position = self.positions.get(symbol)
            if position is None:
                continue
            bar = self._bar_for(symbol, current_date)
            if bar is None:
                continue
            open_price = _decimal(bar["open"])
            if open_price <= position.stop_price:
                decision = LabExit(_adverse_long_exit(open_price, self.settings.slippage), "stop_gap")
            elif position.target_price is not None and open_price >= position.target_price:
                decision = LabExit(_adverse_long_exit(open_price, self.settings.slippage), "target_gap")
            else:
                decision = LabExit(_adverse_long_exit(open_price, self.settings.slippage), reason)
            self._close_position(position, current_date, decision)

    def _process_hard_exits(self, current_date: date) -> None:
        for position in list(self.positions.values()):
            bar = self._bar_for(position.symbol, current_date)
            if bar is None:
                continue
            decision = resolve_stop_target_exit(
                open_price=_decimal(bar["open"]),
                high=_decimal(bar["high"]),
                low=_decimal(bar["low"]),
                stop_price=position.stop_price,
                target_price=position.target_price,
                slippage=self.settings.slippage,
            )
            if decision is not None:
                self._close_position(position, current_date, decision)

    def _increment_holding_bars(self, current_date: date) -> None:
        for position in self.positions.values():
            if current_date >= position.entry_date and self._bar_for(position.symbol, current_date) is not None:
                position.holding_bars += 1

    def _schedule_signal_exits(self, current_date: date) -> None:
        for position in list(self.positions.values()):
            frame = self.bars.get(position.symbol)
            if frame is None:
                continue
            history = _frame_through(frame, current_date)
            if history.empty:
                continue
            should_exit = False
            reason = "signal_exit"
            try:
                if self.definition.exit_fn is not None:
                    should_exit = self.definition.exit_fn(history, position)
            except (ValueError, IndexError, KeyError, TypeError):
                should_exit = False
            max_hold = self._max_hold_for(position)
            if max_hold is not None and position.holding_bars >= max_hold:
                should_exit = True
                reason = "max_hold"
            if not should_exit:
                continue
            exit_date = _next_symbol_date(frame, current_date, self.settings.end)
            if exit_date is not None:
                self.pending_exits[exit_date][position.symbol] = reason

    def _max_hold_for(self, position: LabPosition) -> int | None:
        if self.definition.key != "regime_adaptive":
            return self.definition.max_hold_bars
        if position.route in {"three_down_mean_reversion", "bearish_engulfing_mean_reversion"}:
            return 5
        if position.route == "three_line_strike":
            return 10
        return None

    def _count_pattern_occurrences(self, current_date: date) -> None:
        if not self.definition.count_three_line_occurrences:
            return
        self.three_line_occurrences += self._three_line_occurrences_by_date.get(current_date, 0)

    def _close_position(self, position: LabPosition, exit_date: date, decision: LabExit) -> None:
        self.cash += position.quantity * decision.price
        pnl = position.quantity * (decision.price - position.entry_price)
        return_pct = float(decision.price / position.entry_price - Decimal("1"))
        self.trades.append(
            {
                "symbol": position.symbol,
                "signal_date": position.signal_date.isoformat(),
                "entry_date": position.entry_date.isoformat(),
                "entry_price": float(position.entry_price),
                "exit_date": exit_date.isoformat(),
                "exit_price": float(decision.price),
                "position_notional": float(position.position_notional),
                "quantity": float(position.quantity),
                "stop_price": float(position.stop_price),
                "target_price": float(position.target_price) if position.target_price is not None else "",
                "return_pct": return_pct,
                "pnl_dollars": float(pnl),
                "holding_days": max((exit_date - position.entry_date).days, 0),
                "holding_bars": position.holding_bars,
                "market_regime": position.market_regime,
                "technical_score": position.technical_score,
                "rsi": position.rsi,
                "atr_pct": position.atr_pct,
                "momentum20": position.momentum20,
                "momentum60": position.momentum60,
                "exit_reason": decision.reason,
                "ambiguous_stop_first": decision.ambiguous_stop_first,
                "strategy_route": position.route,
                "copy_latency_seconds": position.metadata.get("copy_latency_seconds", ""),
                "entry_disadvantage_pct": position.metadata.get("entry_disadvantage_pct", ""),
                "leader_return_pct": position.metadata.get("leader_return_pct", ""),
                "follower_return_pct": return_pct if position.metadata.get("is_copy_strategy") else "",
                "copyability_tax_pct": (
                    position.metadata.get("leader_return_pct") - return_pct
                    if position.metadata.get("is_copy_strategy") and position.metadata.get("leader_return_pct") is not None
                    else ""
                ),
                "leader_id": position.metadata.get("leader_id", ""),
                "leader_qualified_at_entry": position.metadata.get("leader_qualified_at_entry", ""),
                "consensus_count": position.metadata.get("consensus_count", ""),
            }
        )
        self.last_exit[position.symbol] = exit_date
        self.positions.pop(position.symbol, None)
        for exits in self.pending_exits.values():
            exits.pop(position.symbol, None)

    def _liquidate_at_end(self, current_date: date) -> None:
        for position in list(self.positions.values()):
            close = self._close_for(position.symbol, current_date)
            if close is None:
                continue
            self._close_position(
                position,
                current_date,
                LabExit(_adverse_long_exit(close, self.settings.slippage), "end_of_backtest"),
            )

    def _refresh_final_equity_row(self) -> None:
        if not self.equity_curve:
            return
        final = self.equity_curve[-1]
        final["cash"] = float(self.cash)
        final["open_position_value"] = 0.0
        final["equity"] = float(self.cash)
        final["positions_open"] = 0
        final["drawdown_pct"] = float(self._drawdown(self.cash))

    def _build_summary(self, trading_dates: list[date]) -> dict[str, Any]:
        ending = _decimal(self.equity_curve[-1]["equity"])
        starting = self.settings.initial_capital
        total_return = ending / starting - Decimal("1") if starting else Decimal("0")
        calendar_days = max((self.settings.end - self.settings.start).days, 1)
        annualized = (
            float((ending / starting) ** (Decimal("365.25") / Decimal(calendar_days)) - Decimal("1"))
            if ending > 0 and starting > 0
            else -1.0
        )
        returns = [float(trade["return_pct"]) for trade in self.trades]
        winners = [value for value in returns if value > 0]
        losers = [value for value in returns if value <= 0]
        gross_wins = sum(float(trade["pnl_dollars"]) for trade in self.trades if trade["pnl_dollars"] > 0)
        gross_losses = abs(sum(float(trade["pnl_dollars"]) for trade in self.trades if trade["pnl_dollars"] < 0))
        max_dd = max((float(row["drawdown_pct"]) for row in self.equity_curve), default=0.0)
        max_dd_row = next((row for row in self.equity_curve if float(row["drawdown_pct"]) == max_dd), None)
        losing_streak = 0
        worst_streak = 0
        for value in returns:
            if value <= 0:
                losing_streak += 1
                worst_streak = max(worst_streak, losing_streak)
            else:
                losing_streak = 0
        open_counts = [int(row["positions_open"]) for row in self.equity_curve]
        regime_counts = {
            regime: sum(1 for trade in self.trades if trade["market_regime"] == regime)
            for regime in ("GREEN", "YELLOW", "RED")
        }
        return {
            "strategy": self.definition.key,
            "label": self.definition.label,
            "backtest_period": {"start": self.settings.start.isoformat(), "end": self.settings.end.isoformat()},
            "starting_capital": float(starting),
            "ending_capital": float(ending),
            "total_return_pct": float(total_return),
            "annualized_return_pct": annualized,
            "total_trades": len(self.trades),
            "winning_trades": len(winners),
            "losing_trades": len(losers),
            "win_rate": len(winners) / len(returns) if returns else 0.0,
            "average_winner_pct": sum(winners) / len(winners) if winners else 0.0,
            "average_loser_pct": sum(losers) / len(losers) if losers else 0.0,
            "average_holding_days": (
                sum(int(trade["holding_days"]) for trade in self.trades) / len(self.trades)
                if self.trades else 0.0
            ),
            "profit_factor": gross_wins / gross_losses if gross_losses else None,
            "maximum_drawdown_pct": max_dd,
            "date_of_maximum_drawdown": max_dd_row["date"] if max_dd_row else None,
            "worst_losing_streak": worst_streak,
            "best_trade_pct": max(returns) if returns else 0.0,
            "worst_trade_pct": min(returns) if returns else 0.0,
            "percentage_of_time_invested": self.invested_days / len(trading_dates) if trading_dates else 0.0,
            "average_open_positions": sum(open_counts) / len(open_counts) if open_counts else 0.0,
            "green_regime_trades": regime_counts["GREEN"],
            "yellow_regime_trades": regime_counts["YELLOW"],
            "red_regime_trades": regime_counts["RED"],
            "ambiguous_daily_bar_exits": sum(1 for trade in self.trades if trade["ambiguous_stop_first"]),
            "slippage_assumption": float(self.settings.slippage),
            "historical_data_feed": self.settings.data_feed,
            "return_over_max_drawdown": float(total_return) / max_dd if max_dd else None,
            "portfolio_drawdown_halt_date": self.halt_date.isoformat() if self.halt_date else None,
            "three_line_strike_occurrences": self.three_line_occurrences,
            **self._copy_summary_metrics(),
        }

    def _copy_summary_metrics(self) -> dict[str, Any]:
        copy_trades = [trade for trade in self.trades if trade.get("leader_id") not in (None, "")]
        if not copy_trades:
            return {
                "copy_latency_seconds": None,
                "entry_disadvantage_pct": None,
                "leader_return_pct": None,
                "follower_return_pct": None,
                "copyability_tax_pct": None,
                "leader_id": None,
                "leader_qualified_at_entry": None,
                "consensus_count": None,
            }

        def avg_numeric(key: str) -> float | None:
            values = [float(row[key]) for row in copy_trades if row.get(key) not in (None, "")]
            return sum(values) / len(values) if values else None

        leaders = sorted({str(row["leader_id"]) for row in copy_trades if row.get("leader_id")})
        qualified = [bool(row.get("leader_qualified_at_entry")) for row in copy_trades]
        return {
            "copy_latency_seconds": avg_numeric("copy_latency_seconds"),
            "entry_disadvantage_pct": avg_numeric("entry_disadvantage_pct"),
            "leader_return_pct": avg_numeric("leader_return_pct"),
            "follower_return_pct": avg_numeric("follower_return_pct"),
            "copyability_tax_pct": avg_numeric("copyability_tax_pct"),
            "leader_id": ",".join(leaders),
            "leader_qualified_at_entry": (sum(qualified) / len(qualified)) if qualified else None,
            "consensus_count": avg_numeric("consensus_count"),
        }



def _copy_definition(key: str) -> StrategyDefinition:
    def never_signal(history: pd.DataFrame, regime: str, spy_history: pd.DataFrame | None) -> StrategySignal | None:
        del history, regime, spy_history
        return None

    return StrategyDefinition(key=key, label=STRATEGY_LABELS[key], signal_fn=never_signal)


class CopyStrategySimulator(StrategySimulator):
    """Research-only copy-signal simulator layered on the same portfolio risk engine.

    Daily bars cannot establish exact intraday ordering. A copy alert is therefore
    evaluated on its signal session (or the next available session for a non-session
    timestamp) and entered no earlier than the following session open. The fill is
    conservative for a long: it cannot be better than the supplied follower price.
    """

    def __init__(
        self,
        key: str,
        settings: BacktestSettings,
        bars: dict[str, pd.DataFrame],
        signals: list[CopySignal],
        *,
        regime_fn: Callable[..., RegimeAssessment] = market_regime,
    ):
        if key not in COPY_STRATEGY_KEYS:
            raise ValueError(f"Unknown copy strategy {key!r}")
        super().__init__(_copy_definition(key), settings, bars, regime_fn=regime_fn)
        self.copy_key = key
        self.copy_signals = list(signals)
        self.closed_leader_trades = pair_closed_leader_trades(self.copy_signals)
        self._signals_by_decision_date: dict[date, list[CopySignal]] = defaultdict(list)
        self._copy_exit_meta: dict[tuple[date, str], dict[str, Any]] = {}
        trading_dates = self._trading_dates()
        for signal in self.copy_signals:
            # Pre-period signals remain available to leader-quality calculations but
            # can never create a position in this independently capitalized period.
            if not (self.settings.start <= signal.signal_time.date() <= self.settings.end):
                continue
            decision_date = self._decision_date(signal.signal_time.date(), trading_dates)
            if decision_date is not None:
                self._signals_by_decision_date[decision_date].append(signal)
        for values in self._signals_by_decision_date.values():
            values.sort(key=lambda item: (item.signal_time, item.leader_id, item.symbol, item.side))

    @staticmethod
    def _decision_date(signal_date: date, trading_dates: list[date]) -> date | None:
        for candidate in trading_dates:
            if candidate >= signal_date:
                return candidate
        return None

    def _schedule_entries(
        self,
        current_date: date,
        regime_assessment: RegimeAssessment | None,
    ) -> None:
        if regime_assessment is None or regime_assessment.regime == "RED":
            return
        allowed_positions = (
            self.settings.max_positions
            if regime_assessment.regime == "GREEN"
            else self.settings.yellow_max_positions
        )
        existing_symbols = self._effective_existing_symbols()
        slots = max(0, allowed_positions - len(existing_symbols))
        if slots <= 0:
            return

        candidates: list[tuple[float, str, CopySignal, CopyDecision, date, dict[str, float]]] = []
        for signal in self._signals_by_decision_date.get(current_date, []):
            if signal.side != "BUY" or signal.symbol not in self.settings.watchlist:
                continue
            if signal.symbol in existing_symbols or self._cooldown_active(signal.symbol, current_date):
                continue
            frame = self.bars.get(signal.symbol)
            if frame is None or frame.empty:
                continue
            history = _frame_through(frame, current_date)
            if history.empty:
                continue
            try:
                decision = evaluate_copy_signal(
                    self.copy_key,
                    signal,
                    all_signals=self.copy_signals,
                    closed_trades=self.closed_leader_trades,
                    history=history,
                    regime=regime_assessment.regime,
                    settings=self.settings,
                )
            except (ValueError, IndexError, KeyError, TypeError, ArithmeticError):
                continue
            if not decision.accepted or decision.stop_pct is None:
                continue
            entry_date = _next_symbol_date(frame, current_date, self.settings.end)
            if entry_date is None:
                continue
            features = self._features(history)
            candidates.append((decision.score, signal.symbol, signal, decision, entry_date, features))

        candidates.sort(key=lambda item: (-item[0], item[1], item[2].leader_id))
        risk_capital = self._risk_capital(current_date)
        scheduled_today = 0
        for _, symbol, signal, decision, entry_date, features in candidates:
            if slots <= 0 or scheduled_today >= self.settings.max_new_trades_per_day:
                break
            available_cash = max(Decimal("0"), self.cash - self.reserved_cash)
            cap = position_notional_cap(
                risk_capital,
                available_cash,
                decision.stop_pct,
                allocation_pct=self.settings.position_alloc_pct,
                risk_per_trade_pct=self.settings.risk_per_trade_pct,
            )
            if cap < Decimal("1"):
                continue
            leader_trade = self._closed_trade_for_entry(signal)
            metadata = {
                "is_copy_strategy": True,
                "leader_id": signal.leader_id,
                "source_trade_id": signal.source_trade_id or "",
                "leader_entry_price": float(signal.leader_price),
                "follower_signal_price": float(signal.follower_price),
                "copy_latency_seconds": decision.copy_latency_seconds,
                "entry_disadvantage_pct": decision.entry_disadvantage_pct,
                "leader_qualified_at_entry": decision.leader_quality.qualified,
                "consensus_count": decision.consensus_count,
                "leader_return_pct": leader_trade.return_pct if leader_trade is not None else None,
                "copy_policy_reason": decision.reason,
            }
            self.pending_entries[entry_date].append(
                LabPending(
                    symbol=symbol,
                    signal_date=current_date,
                    entry_date=entry_date,
                    reserved_cap=cap,
                    risk_capital=risk_capital,
                    signal=StrategySignal(
                        stop_pct=decision.stop_pct,
                        target_multiple=None,
                        score=decision.score,
                        metadata={"adaptive_route": self.copy_key},
                    ),
                    regime=regime_assessment.regime,
                    rsi=features["rsi"],
                    atr_pct=features["atr_pct"],
                    momentum20=features["momentum20"],
                    momentum60=features["momentum60"],
                    metadata=metadata,
                )
            )
            self.reserved_cash += cap
            existing_symbols.add(symbol)
            slots -= 1
            scheduled_today += 1

    def _closed_trade_for_entry(self, entry: CopySignal):
        exact = [
            trade
            for trade in self.closed_leader_trades
            if trade.entry.identity == entry.identity
        ]
        return exact[0] if exact else None

    def _enter_pending(self, current_date: date) -> None:
        intents = self.pending_entries.pop(current_date, [])
        for intent in intents:
            self.reserved_cash = max(Decimal("0"), self.reserved_cash - intent.reserved_cap)
            if intent.symbol in self.positions:
                continue
            bar = self._bar_for(intent.symbol, current_date)
            if bar is None:
                continue
            open_price = _decimal(bar["open"])
            follower_signal_price = _decimal(intent.metadata.get("follower_signal_price", open_price))
            if open_price <= 0 or follower_signal_price <= 0:
                continue
            # Never award a long-copy fill better than what the follower could actually see.
            base_price = max(open_price, follower_signal_price)
            entry_price = _adverse_entry(base_price, self.settings.slippage)
            cap = min(intent.reserved_cap, self.cash)
            qty = _qty_for_notional(cap, entry_price)
            notional = qty * entry_price
            if qty <= 0 or notional < Decimal("1") or notional > self.cash:
                continue
            stop_price = entry_price * (Decimal("1") - intent.signal.stop_pct)
            self.cash -= notional
            metadata = dict(intent.metadata)
            leader_entry = _decimal(metadata.get("leader_entry_price", entry_price))
            metadata["entry_disadvantage_pct"] = (
                float(entry_price / leader_entry - Decimal("1")) if leader_entry > 0 else None
            )
            self.positions[intent.symbol] = LabPosition(
                symbol=intent.symbol,
                signal_date=intent.signal_date,
                entry_date=current_date,
                entry_price=entry_price,
                quantity=qty,
                position_notional=notional,
                stop_price=stop_price,
                target_price=None,
                market_regime=intent.regime,
                technical_score=float(intent.signal.score),
                rsi=intent.rsi,
                atr_pct=intent.atr_pct,
                momentum20=intent.momentum20,
                momentum60=intent.momentum60,
                route=self.copy_key,
                metadata=metadata,
            )

    def _schedule_signal_exits(self, current_date: date) -> None:
        sells = [signal for signal in self._signals_by_decision_date.get(current_date, []) if signal.side == "SELL"]
        if not sells:
            return
        for position in list(self.positions.values()):
            frame = self.bars.get(position.symbol)
            if frame is None:
                continue
            matched: CopySignal | None = None
            for signal in sells:
                if signal.symbol != position.symbol or signal.leader_id != position.metadata.get("leader_id"):
                    continue
                position_trade_id = str(position.metadata.get("source_trade_id") or "")
                signal_trade_id = signal.source_trade_id or ""
                if position_trade_id and signal_trade_id and position_trade_id != signal_trade_id:
                    continue
                matched = signal
                break
            if matched is None:
                continue
            exit_date = _next_symbol_date(frame, current_date, self.settings.end)
            if exit_date is None:
                continue
            leader_entry = _decimal(position.metadata.get("leader_entry_price", position.entry_price))
            leader_return = (
                float(matched.leader_price / leader_entry - Decimal("1"))
                if leader_entry > 0
                else None
            )
            position.metadata["leader_return_pct"] = leader_return
            self.pending_exits[exit_date][position.symbol] = "leader_exit"
            self._copy_exit_meta[(exit_date, position.symbol)] = {
                "follower_price": matched.follower_price,
                "leader_exit_price": matched.leader_price,
            }

    def _execute_pending_exits(self, current_date: date) -> None:
        exits = self.pending_exits.pop(current_date, {})
        for symbol, reason in list(exits.items()):
            position = self.positions.get(symbol)
            if position is None:
                continue
            bar = self._bar_for(symbol, current_date)
            if bar is None:
                continue
            open_price = _decimal(bar["open"])
            if open_price <= position.stop_price:
                decision = LabExit(_adverse_long_exit(open_price, self.settings.slippage), "stop_gap")
            else:
                meta = self._copy_exit_meta.pop((current_date, symbol), None)
                if meta is not None:
                    follower_price = _decimal(meta["follower_price"])
                    # Selling a long cannot receive a better fill than both the follower quote
                    # and the delayed daily-bar open in this conservative approximation.
                    base_price = min(open_price, follower_price)
                    decision = LabExit(_adverse_long_exit(base_price, self.settings.slippage), reason)
                else:
                    decision = LabExit(_adverse_long_exit(open_price, self.settings.slippage), reason)
            self._close_position(position, current_date, decision)


def _settings_for_period(base: BacktestSettings, *, start: date, end: date, watchlist: tuple[str, ...]) -> BacktestSettings:
    return replace(base, start=start, end=end, watchlist=watchlist)


def _current_v2_settings(base: BacktestSettings, cfg: Config, *, start: date, end: date) -> BacktestSettings:
    return replace(base, start=start, end=end, watchlist=tuple(dict.fromkeys((*cfg.watchlist, "SPY", "QQQ"))))


def _current_v2_result(settings: BacktestSettings, bars: dict[str, pd.DataFrame]) -> BacktestResult:
    result = Backtester(settings, bars).run()
    result.summary = {"strategy": "current_v2", "label": STRATEGY_LABELS["current_v2"], **result.summary}
    return result


def _simulate_periods(
    key: str,
    definition: StrategyDefinition | None,
    base_settings: BacktestSettings,
    cfg: Config,
    bars: dict[str, pd.DataFrame],
    lab_watchlist: tuple[str, ...],
    holdout_start: date,
    copy_signals: list[CopySignal] | None = None,
) -> StrategyPeriodResults:
    periods = {
        "full": (base_settings.start, base_settings.end),
        "development": (base_settings.start, holdout_start - timedelta(days=1)),
        "holdout": (holdout_start, base_settings.end),
    }
    results: dict[str, LabResult | BacktestResult] = {}
    for period_name, (period_start, period_end) in periods.items():
        if period_start >= period_end:
            raise ValueError(f"{period_name} period is empty; choose a holdout date inside the requested interval")
        if key == "current_v2":
            settings = _current_v2_settings(base_settings, cfg, start=period_start, end=period_end)
            results[period_name] = _current_v2_result(settings, bars)
        elif key in COPY_STRATEGY_KEYS:
            if copy_signals is None:
                raise ValueError(f"Copy strategy {key} requires --copy-signals")
            settings = _settings_for_period(base_settings, start=period_start, end=period_end, watchlist=lab_watchlist)
            # Qualification may use older closed leader trades from before a holdout candidate,
            # but never a close at-or-after that candidate. To preserve that history without
            # carrying capital or positions across the split, include all pre-period signals as
            # reference-only input while the simulator only acts on signals whose decision dates
            # fall inside settings.start/end.
            reference_signals = [signal for signal in copy_signals if signal.signal_time.date() <= period_end]
            results[period_name] = CopyStrategySimulator(key, settings, bars, reference_signals).run()
        else:
            if definition is None:
                raise ValueError(f"Missing strategy definition for {key}")
            settings = _settings_for_period(base_settings, start=period_start, end=period_end, watchlist=lab_watchlist)
            results[period_name] = StrategySimulator(definition, settings, bars).run()
    return StrategyPeriodResults(
        full=results["full"],
        development=results["development"],
        holdout=results["holdout"],
    )


def _benchmark_series(frame: pd.DataFrame, *, start: date, end: date, capital: Decimal) -> tuple[pd.Series, dict[str, Any]]:
    interval = frame.loc[(frame.index >= start) & (frame.index <= end)].copy()
    if interval.empty:
        return pd.Series(dtype=float), _empty_benchmark_summary(start, end, capital)
    first_open = float(interval.iloc[0]["open"])
    if first_open <= 0:
        return pd.Series(dtype=float), _empty_benchmark_summary(start, end, capital)
    equity = interval["close"].astype(float) / first_open * float(capital)
    equity.iloc[0] = float(capital) * float(interval.iloc[0]["close"]) / first_open
    return equity, _summary_from_equity(equity, start=start, end=end, capital=capital)


def _empty_benchmark_summary(start: date, end: date, capital: Decimal) -> dict[str, Any]:
    return {
        "backtest_period": {"start": start.isoformat(), "end": end.isoformat()},
        "starting_capital": float(capital),
        "ending_capital": float(capital),
        "total_return_pct": 0.0,
        "annualized_return_pct": 0.0,
        "maximum_drawdown_pct": 0.0,
        "return_over_max_drawdown": None,
        "profit_factor": None,
        "win_rate": None,
        "total_trades": None,
        "worst_losing_streak": None,
        "percentage_of_time_invested": 1.0,
        "average_open_positions": 1.0,
        "portfolio_drawdown_halt_date": None,
        "ambiguous_daily_bar_exits": 0,
    }


def _summary_from_equity(equity: pd.Series, *, start: date, end: date, capital: Decimal) -> dict[str, Any]:
    if equity.empty:
        return _empty_benchmark_summary(start, end, capital)
    ending = float(equity.iloc[-1])
    total_return = ending / float(capital) - 1 if capital else 0.0
    running_max = equity.cummax()
    dd = (running_max - equity) / running_max.replace(0, float("nan"))
    max_dd = float(dd.max()) if not dd.empty else 0.0
    days = max((end - start).days, 1)
    annualized = (ending / float(capital)) ** (365.25 / days) - 1 if ending > 0 and capital > 0 else -1.0
    return {
        "backtest_period": {"start": start.isoformat(), "end": end.isoformat()},
        "starting_capital": float(capital),
        "ending_capital": ending,
        "total_return_pct": total_return,
        "annualized_return_pct": annualized,
        "maximum_drawdown_pct": max_dd,
        "return_over_max_drawdown": total_return / max_dd if max_dd else None,
        "profit_factor": None,
        "win_rate": None,
        "total_trades": None,
        "worst_losing_streak": None,
        "percentage_of_time_invested": 1.0,
        "average_open_positions": 1.0,
        "portfolio_drawdown_halt_date": None,
        "ambiguous_daily_bar_exits": 0,
    }


def _benchmarks_for_period(
    bars: dict[str, pd.DataFrame],
    *,
    start: date,
    end: date,
    capital: Decimal,
) -> dict[str, dict[str, Any]]:
    spy_equity, spy_summary = _benchmark_series(bars["SPY"], start=start, end=end, capital=capital)
    qqq_equity, qqq_summary = _benchmark_series(bars["QQQ"], start=start, end=end, capital=capital)
    aligned = pd.concat([spy_equity.rename("spy"), qqq_equity.rename("qqq")], axis=1).dropna()
    if aligned.empty:
        combo_summary = _empty_benchmark_summary(start, end, capital)
    else:
        combo = aligned["spy"] * 0.5 + aligned["qqq"] * 0.5
        combo_summary = _summary_from_equity(combo, start=start, end=end, capital=capital)
    return {
        "spy_buy_hold": spy_summary,
        "qqq_buy_hold": qqq_summary,
        "spy_qqq_50_50": combo_summary,
    }


def benchmark_periods(
    bars: dict[str, pd.DataFrame],
    *,
    start: date,
    end: date,
    holdout_start: date,
    capital: Decimal,
) -> dict[str, dict[str, dict[str, Any]]]:
    ranges = {
        "full": (start, end),
        "development": (start, holdout_start - timedelta(days=1)),
        "holdout": (holdout_start, end),
    }
    combined: dict[str, dict[str, dict[str, Any]]] = {
        "spy_buy_hold": {},
        "qqq_buy_hold": {},
        "spy_qqq_50_50": {},
    }
    for period_name, (period_start, period_end) in ranges.items():
        summaries = _benchmarks_for_period(bars, start=period_start, end=period_end, capital=capital)
        for key, summary in summaries.items():
            combined[key][period_name] = summary
    return combined


def _comparison_row(key: str, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "strategy": STRATEGY_LABELS.get(key, key),
        "ending_capital": summary.get("ending_capital"),
        "total_return_pct": summary.get("total_return_pct"),
        "annualized_return_pct": summary.get("annualized_return_pct"),
        "maximum_drawdown_pct": summary.get("maximum_drawdown_pct"),
        "return_over_max_drawdown": summary.get("return_over_max_drawdown"),
        "profit_factor": summary.get("profit_factor"),
        "win_rate": summary.get("win_rate"),
        "total_trades": summary.get("total_trades"),
        "worst_losing_streak": summary.get("worst_losing_streak"),
        "percentage_of_time_invested": summary.get("percentage_of_time_invested"),
        "average_open_positions": summary.get("average_open_positions"),
        "portfolio_drawdown_halt_date": summary.get("portfolio_drawdown_halt_date"),
        "ambiguous_daily_bar_exits": summary.get("ambiguous_daily_bar_exits"),
        "copy_latency_seconds": summary.get("copy_latency_seconds"),
        "entry_disadvantage_pct": summary.get("entry_disadvantage_pct"),
        "leader_return_pct": summary.get("leader_return_pct"),
        "follower_return_pct": summary.get("follower_return_pct"),
        "copyability_tax_pct": summary.get("copyability_tax_pct"),
        "leader_id": summary.get("leader_id"),
        "leader_qualified_at_entry": summary.get("leader_qualified_at_entry"),
        "consensus_count": summary.get("consensus_count"),
    }


def comparison_rows(
    strategies: dict[str, StrategyPeriodResults],
    benchmarks: dict[str, dict[str, dict[str, Any]]],
    period: str,
) -> list[dict[str, Any]]:
    rows = [
        _comparison_row(key, getattr(result, period).summary)
        for key, result in strategies.items()
    ]
    rows.extend(_comparison_row(key, periods[period]) for key, periods in benchmarks.items())
    return rows


def rank_holdout_strategies(strategies: dict[str, StrategyPeriodResults]) -> list[dict[str, Any]]:
    """Survival-first holdout ranking with fixed, non-optimized gates.

    The source integration note deliberately says not to rank on ending balance alone.
    We therefore prefer no drawdown halt/ruin, a minimally useful sample, positive
    holdout return and PF>1, then risk-adjusted return and lower copyability tax.
    """
    minimum_trades = 10

    def sort_key(item: tuple[str, StrategyPeriodResults]) -> tuple[int, int, int, int, float, float, float]:
        summary = item[1].holdout.summary
        ending = float(summary.get("ending_capital") or 0.0)
        ret = float(summary.get("total_return_pct") or 0.0)
        ratio = summary.get("return_over_max_drawdown")
        pf = summary.get("profit_factor")
        trades = summary.get("total_trades")
        trade_count = int(trades or 0)
        halt = summary.get("portfolio_drawdown_halt_date")
        tax = summary.get("copyability_tax_pct")
        return (
            1 if ending > 0 and halt in (None, "") else 0,
            1 if trade_count >= minimum_trades else 0,
            1 if ret > 0 else 0,
            1 if pf is not None and float(pf) > 1.0 else 0,
            float(ratio) if ratio is not None else float("-inf"),
            ret,
            -float(tax) if tax is not None else 0.0,
        )

    ordered = sorted(strategies.items(), key=sort_key, reverse=True)
    return [
        {
            "rank": index,
            "strategy": key,
            "label": STRATEGY_LABELS[key],
            "holdout_return_pct": result.holdout.summary.get("total_return_pct"),
            "holdout_return_over_max_drawdown": result.holdout.summary.get("return_over_max_drawdown"),
            "holdout_profit_factor": result.holdout.summary.get("profit_factor"),
            "holdout_trades": result.holdout.summary.get("total_trades"),
            "holdout_halt_date": result.holdout.summary.get("portfolio_drawdown_halt_date"),
            "holdout_copyability_tax_pct": result.holdout.summary.get("copyability_tax_pct"),
            "adequate_trade_count": int(result.holdout.summary.get("total_trades") or 0) >= minimum_trades,
        }
        for index, (key, result) in enumerate(ordered, start=1)
    ]


def _fmt_pct(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.1%}"


def _fmt_num(value: Any, digits: int = 2) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def print_comparison(result: ComparisonResult) -> None:
    def table(period: str, title: str) -> None:
        print(title)
        print(f"{'STRATEGY':24} {'END':>8} {'RETURN':>9} {'MAX DD':>8} {'PF':>7} {'WIN%':>8} {'TRADES':>7} {'HALT':>12}")
        rows = comparison_rows(result.strategies, result.benchmarks, period)
        for row in rows:
            halt = row.get("portfolio_drawdown_halt_date") or "-"
            trades = row.get("total_trades")
            print(
                f"{str(row['strategy'])[:24]:24} "
                f"${float(row['ending_capital']):7.2f} "
                f"{_fmt_pct(row['total_return_pct']):>9} "
                f"{_fmt_pct(row['maximum_drawdown_pct']):>8} "
                f"{_fmt_num(row['profit_factor']):>7} "
                f"{_fmt_pct(row['win_rate']):>8} "
                f"{str(trades) if trades is not None else '-':>7} "
                f"{str(halt):>12}"
            )
        print()

    print(f"STRATEGY LAB: {result.start.isoformat()} to {result.end.isoformat()}")
    print(f"Starting capital: ${result.initial_capital:.2f} | Slippage: {result.slippage:.2%} | Feed: {result.data_feed}")
    print(f"Holdout starts: {result.holdout_start.isoformat()}\n")
    table("full", "FULL PERIOD")
    table("holdout", "HOLDOUT PERIOD")
    print("HOLDOUT RANKING (survival, sample size, positive return/PF, risk-adjusted return, copy tax)")
    for row in rank_holdout_strategies(result.strategies):
        print(
            f"  {row['rank']}. {row['label']}: "
            f"return {_fmt_pct(row['holdout_return_pct'])}, "
            f"return/max-DD {_fmt_num(row['holdout_return_over_max_drawdown'])}, "
            f"PF {_fmt_num(row['holdout_profit_factor'])}"
        )
    if result.output_dir:
        print(f"\nOutput directory: {result.output_dir}")


def run_comparison(
    cfg: Config,
    *,
    start_text: str,
    end_text: str,
    capital: str = "100",
    slippage: str = "0.001",
    holdout_start_text: str,
    copy_signals_path: str | None = None,
) -> int:
    start = parse_date(start_text)
    end = parse_date(end_text)
    holdout_start = parse_date(holdout_start_text)
    if start >= end:
        raise ValueError("--start must be before --end")
    if not (start < holdout_start < end):
        raise ValueError("--holdout-start must fall strictly inside the requested period")
    initial_capital = Decimal(str(capital))
    slippage_value = Decimal(str(slippage))
    if initial_capital <= 0:
        raise ValueError("--capital must be greater than zero")
    if slippage_value < 0 or slippage_value >= 1:
        raise ValueError("--slippage must be between 0 and 1")

    copy_signals = load_copy_signals_csv(copy_signals_path) if copy_signals_path else None
    copy_symbols = copy_signal_symbols(copy_signals or ())
    extended_watchlist = tuple(dict.fromkeys((*cfg.watchlist, *LAB_ETFS, *copy_symbols)))
    base_settings = BacktestSettings.from_config(
        cfg,
        start=start,
        end=end,
        capital=initial_capital,
        slippage=slippage_value,
    )
    bars = fetch_historical_bars(
        cfg.alpaca_api_key,
        cfg.alpaca_secret_key,
        extended_watchlist,
        start=start,
        end=end,
        feed_name=base_settings.data_feed,
    )
    if "SPY" not in bars or "QQQ" not in bars:
        raise ValueError("SPY and QQQ historical bars are required for comparison")

    definitions = strategy_definitions()
    strategies: dict[str, StrategyPeriodResults] = {}
    strategy_keys = (*BASE_STRATEGY_KEYS, *COPY_STRATEGY_KEYS) if copy_signals is not None else BASE_STRATEGY_KEYS
    for key in strategy_keys:
        definition = definitions.get(key)
        strategies[key] = _simulate_periods(
            key,
            definition,
            base_settings,
            cfg,
            bars,
            extended_watchlist,
            holdout_start,
            copy_signals=copy_signals,
        )

    benchmarks = benchmark_periods(
        bars,
        start=start,
        end=end,
        holdout_start=holdout_start,
        capital=initial_capital,
    )
    result = ComparisonResult(
        start=start,
        end=end,
        holdout_start=holdout_start,
        initial_capital=initial_capital,
        slippage=slippage_value,
        data_feed=base_settings.data_feed,
        strategies=strategies,
        benchmarks=benchmarks,
    )
    result.write_outputs(cfg.project_root / "backtests" / "comparisons")
    print_comparison(result)
    return 0
