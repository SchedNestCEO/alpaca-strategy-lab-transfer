from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

from .config import Config
from .indicators import RegimeAssessment, TechnicalAssessment, analyze_symbol, market_regime
from .risk import build_position_plan, dynamic_stop_pct

UTC = timezone.utc
WARMUP_CALENDAR_DAYS = 300
BENCHMARKS = ("SPY", "QQQ")
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
    "market_regime",
    "technical_score",
    "rsi",
    "atr_pct",
    "momentum20",
    "momentum60",
    "exit_reason",
    "ambiguous_stop_first",
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


@dataclass(frozen=True)
class BacktestSettings:
    start: date
    end: date
    initial_capital: Decimal
    slippage: Decimal = Decimal("0.001")
    position_alloc_pct: Decimal = Decimal("0.10")
    risk_per_trade_pct: Decimal = Decimal("0.005")
    max_positions: int = 3
    yellow_max_positions: int = 1
    max_new_trades_per_day: int = 3
    min_stop_pct: Decimal = Decimal("0.02")
    max_stop_pct: Decimal = Decimal("0.05")
    atr_stop_multiple: Decimal = Decimal("2")
    reward_risk_multiple: Decimal = Decimal("2")
    max_daily_drawdown_pct: Decimal = Decimal("0.01")
    max_high_water_drawdown_pct: Decimal = Decimal("0.05")
    cooldown_hours: int = 72
    trend_fast_sma: int = 50
    trend_slow_sma: int = 200
    pullback_sma: int = 20
    atr_period: int = 14
    rsi_period: int = 14
    max_atr_pct: float = 0.025
    min_technical_score: float = 75.0
    yellow_min_score: float = 85.0
    data_feed: str = "iex"
    watchlist: tuple[str, ...] = ()

    @classmethod
    def from_config(
        cls,
        cfg: Config,
        *,
        start: date,
        end: date,
        capital: Decimal,
        slippage: Decimal,
    ) -> "BacktestSettings":
        symbols = tuple(dict.fromkeys((*cfg.watchlist, *BENCHMARKS)))
        return cls(
            start=start,
            end=end,
            initial_capital=capital,
            slippage=slippage,
            position_alloc_pct=cfg.position_alloc_pct,
            risk_per_trade_pct=cfg.risk_per_trade_pct,
            max_positions=cfg.max_positions,
            yellow_max_positions=cfg.yellow_max_positions,
            max_new_trades_per_day=cfg.max_new_trades_per_day,
            min_stop_pct=cfg.min_stop_pct,
            max_stop_pct=cfg.max_stop_pct,
            atr_stop_multiple=cfg.atr_stop_multiple,
            reward_risk_multiple=cfg.reward_risk_multiple,
            max_daily_drawdown_pct=cfg.max_daily_drawdown_pct,
            max_high_water_drawdown_pct=cfg.max_high_water_drawdown_pct,
            cooldown_hours=cfg.cooldown_hours,
            trend_fast_sma=cfg.trend_fast_sma,
            trend_slow_sma=cfg.trend_slow_sma,
            pullback_sma=cfg.pullback_sma,
            atr_period=cfg.atr_period,
            rsi_period=cfg.rsi_period,
            max_atr_pct=cfg.max_atr_pct,
            min_technical_score=cfg.min_technical_score,
            yellow_min_score=cfg.yellow_min_score,
            data_feed=cfg.data_feed or "iex",
            watchlist=symbols,
        )


@dataclass(frozen=True)
class PendingEntry:
    symbol: str
    signal_date: date
    entry_date: date
    reserved_cap: Decimal
    risk_capital: Decimal
    stop_pct: Decimal
    target_pct: Decimal
    market_regime: str
    technical_score: float
    rsi: float
    atr_pct: float
    momentum20: float
    momentum60: float


@dataclass
class OpenPosition:
    symbol: str
    signal_date: date
    entry_date: date
    entry_price: Decimal
    quantity: Decimal
    position_notional: Decimal
    stop_price: Decimal
    target_price: Decimal
    market_regime: str
    technical_score: float
    rsi: float
    atr_pct: float
    momentum20: float
    momentum60: float


@dataclass(frozen=True)
class ExitDecision:
    price: Decimal
    reason: str
    ambiguous_stop_first: bool = False


@dataclass
class BacktestResult:
    summary: dict[str, Any]
    trades: list[dict[str, Any]]
    equity_curve: list[dict[str, Any]]
    output_dir: Path | None = None

    def write_outputs(self, root: Path = Path("backtests")) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output_dir = root / timestamp
        suffix = 1
        while output_dir.exists():
            output_dir = root / f"{timestamp}-{suffix}"
            suffix += 1
        output_dir.mkdir(parents=True)

        with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(self.summary, handle, indent=2, sort_keys=True)
            handle.write("\n")

        with (output_dir / "trades.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=TRADE_COLUMNS)
            writer.writeheader()
            writer.writerows(self.trades)

        with (output_dir / "equity_curve.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=EQUITY_COLUMNS)
            writer.writeheader()
            writer.writerows(self.equity_curve)

        self.output_dir = output_dir
        return output_dir


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid date {value!r}; use YYYY-MM-DD") from exc


def resolve_period(
    *,
    years: int | None,
    start_text: str | None,
    end_text: str | None,
) -> tuple[date, date]:
    if years is not None and start_text:
        raise ValueError("Use either --years or --start/--end, not both")
    if years is not None:
        if years <= 0:
            raise ValueError("--years must be greater than zero")
        end = parse_date(end_text) if end_text else datetime.now(UTC).date()
        start = end - timedelta(days=round(years * 365.25))
    else:
        if not start_text or not end_text:
            raise ValueError("Provide --years or both --start and --end")
        start, end = parse_date(start_text), parse_date(end_text)
    if start >= end:
        raise ValueError("Backtest start must be before end")
    return start, end


def normalize_bars(frame: pd.DataFrame) -> pd.DataFrame:
    """Return daily OHLCV data indexed by plain UTC calendar date."""
    if frame.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    work = frame.copy()
    if "timestamp" in work.columns:
        timestamp = pd.to_datetime(work.pop("timestamp"), utc=True)
    else:
        timestamp = pd.to_datetime(work.index, utc=True)
    work.index = timestamp.dt.date if isinstance(timestamp, pd.Series) else timestamp.date
    work.index.name = "date"
    needed = ["open", "high", "low", "close", "volume"]
    for column in needed:
        if column not in work.columns:
            work[column] = 0.0
        work[column] = pd.to_numeric(work[column], errors="coerce")
    selected = work[needed].sort_index()
    return selected.loc[~selected.index.duplicated(keep="last")]


def fetch_historical_bars(
    api_key: str,
    secret_key: str,
    symbols: Iterable[str],
    *,
    start: date,
    end: date,
    feed_name: str = "iex",
) -> dict[str, pd.DataFrame]:
    """Fetch only daily stock bars; this function never creates a broker client."""
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    try:
        feed = DataFeed(feed_name)
    except ValueError as exc:
        raise ValueError(f"Unsupported historical data feed {feed_name!r}") from exc

    client = StockHistoricalDataClient(api_key, secret_key)
    request_start = datetime.combine(
        start - timedelta(days=WARMUP_CALENDAR_DAYS), time.min, tzinfo=UTC
    )
    request_end = datetime.combine(end + timedelta(days=1), time.min, tzinfo=UTC)
    result: dict[str, pd.DataFrame] = {}

    for symbol in dict.fromkeys(symbols):
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=request_start,
            end=request_end,
            adjustment=Adjustment.ALL,
            feed=feed,
        )
        barset = client.get_stock_bars(request)
        result[symbol] = normalize_bars(barset.df.reset_index())
    return result


def next_trading_day(dates: Iterable[date], signal_date: date) -> date | None:
    for candidate in sorted(set(dates)):
        if candidate > signal_date:
            return candidate
    return None


def adverse_exit_price(price: Decimal, slippage: Decimal) -> Decimal:
    """Adverse fill for selling a long position."""
    return price * (Decimal("1") - slippage)


def position_notional_cap(
    risk_capital: Decimal,
    cash_available: Decimal,
    stop_pct: Decimal,
    *,
    allocation_pct: Decimal = Decimal("0.10"),
    risk_per_trade_pct: Decimal = Decimal("0.005"),
) -> Decimal:
    if risk_capital <= 0 or cash_available <= 0 or stop_pct <= 0:
        return Decimal("0")
    return min(
        risk_capital * allocation_pct,
        risk_capital * risk_per_trade_pct / stop_pct,
        cash_available,
    )


def regime_allows_entries(regime: str) -> bool:
    return regime in {"GREEN", "YELLOW"}


def drawdown_halt(equity: Decimal, high_water: Decimal, threshold: Decimal) -> bool:
    if high_water <= 0:
        return False
    return (high_water - equity) / high_water >= threshold


def resolve_daily_exit(
    *,
    open_price: Decimal,
    high: Decimal,
    low: Decimal,
    stop_price: Decimal,
    target_price: Decimal,
    slippage: Decimal,
) -> ExitDecision | None:
    """Resolve a long exit without inventing intraday ordering."""
    if open_price <= stop_price:
        return ExitDecision(adverse_exit_price(open_price, slippage), "stop_gap")
    if open_price >= target_price:
        return ExitDecision(adverse_exit_price(open_price, slippage), "target_gap")

    stop_touched = low <= stop_price
    target_touched = high >= target_price
    if stop_touched and target_touched:
        return ExitDecision(
            adverse_exit_price(stop_price, slippage),
            "stop",
            ambiguous_stop_first=True,
        )
    if stop_touched:
        return ExitDecision(adverse_exit_price(stop_price, slippage), "stop")
    if target_touched:
        return ExitDecision(adverse_exit_price(target_price, slippage), "target")
    return None


def _date_frame(frame: pd.DataFrame, through: date | None = None) -> pd.DataFrame:
    if through is None:
        return frame
    return frame.loc[frame.index <= through]


def _decimal(value: Any) -> Decimal:
    return Decimal(str(value))


class Backtester:
    """Standalone historical simulator with no broker or database access."""

    def __init__(
        self,
        settings: BacktestSettings,
        bars: dict[str, pd.DataFrame],
        *,
        analysis_fn: Callable[..., TechnicalAssessment] = analyze_symbol,
        regime_fn: Callable[..., RegimeAssessment] = market_regime,
    ):
        self.settings = settings
        self.bars = {symbol: normalize_bars(frame) for symbol, frame in bars.items()}
        self.analysis_fn = analysis_fn
        self.regime_fn = regime_fn
        self.cash = settings.initial_capital
        self.positions: dict[str, OpenPosition] = {}
        self.pending: dict[date, list[PendingEntry]] = defaultdict(list)
        self.reserved_cash = Decimal("0")
        self.trades: list[dict[str, Any]] = []
        self.equity_curve: list[dict[str, Any]] = []
        self.last_exit: dict[str, date] = {}
        self.high_water = settings.initial_capital
        self.halt_date: date | None = None
        self.invested_days = 0
        self._last_equity = settings.initial_capital
        self._regime_cache: dict[date, RegimeAssessment | None] = {}

    def run(self) -> BacktestResult:
        trading_dates = self._trading_dates()
        if not trading_dates:
            raise ValueError("No historical bars were returned for the requested period")
        for current_date in trading_dates:
            day_start_equity = self._last_equity
            self._enter_pending(current_date)
            had_position = bool(self.positions)
            self._process_exits(current_date)
            regime_assessment = self._regime_for(current_date)
            regime_name = regime_assessment.regime if regime_assessment else "UNKNOWN"
            equity = self._equity(current_date)
            if had_position or self.positions:
                self.invested_days += 1

            self.high_water = max(self.high_water, equity)
            drawdown = (
                max(Decimal("0"), (self.high_water - equity) / self.high_water)
                if self.high_water > 0
                else Decimal("0")
            )
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
            if self.halt_date is None and daily_drawdown < self.settings.max_daily_drawdown_pct:
                self._schedule_entries(current_date, regime_assessment)
            self._last_equity = equity

        self._liquidate_at_end(trading_dates[-1])
        self._refresh_final_equity_row()
        summary = self._build_summary(trading_dates)
        return BacktestResult(summary=summary, trades=self.trades, equity_curve=self.equity_curve)

    def _trading_dates(self) -> list[date]:
        dates: set[date] = set()
        for frame in self.bars.values():
            dates.update(
                d for d in frame.index
                if self.settings.start <= d <= self.settings.end
            )
        return sorted(dates)

    def _regime_for(self, current_date: date) -> RegimeAssessment | None:
        if current_date in self._regime_cache:
            return self._regime_cache[current_date]
        spy = self.bars.get("SPY")
        if spy is None:
            self._regime_cache[current_date] = None
            return None
        history = _date_frame(spy, current_date)
        if len(history) < self.settings.trend_slow_sma + 25:
            self._regime_cache[current_date] = None
            return None
        try:
            result = self.regime_fn(
                history,
                self.settings.trend_fast_sma,
                self.settings.trend_slow_sma,
            )
        except (ValueError, IndexError, KeyError):
            result = None
        self._regime_cache[current_date] = result
        return result

    def _close_for(self, symbol: str, current_date: date) -> Decimal | None:
        frame = _date_frame(self.bars.get(symbol, pd.DataFrame()), current_date)
        if frame.empty:
            return None
        return _decimal(frame.iloc[-1]["close"])

    def _bar_for(self, symbol: str, current_date: date) -> pd.Series | None:
        frame = self.bars.get(symbol)
        if frame is None or current_date not in frame.index:
            return None
        return frame.loc[current_date]

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

    def _cooldown_active(self, symbol: str, current_date: date) -> bool:
        previous = self.last_exit.get(symbol)
        if previous is None:
            return False
        return (current_date - previous).days * 24 < self.settings.cooldown_hours

    def _schedule_entries(
        self,
        current_date: date,
        regime_assessment: RegimeAssessment | None,
    ) -> None:
        if regime_assessment is None or not regime_allows_entries(regime_assessment.regime):
            return
        allowed_positions = (
            self.settings.max_positions
            if regime_assessment.regime == "GREEN"
            else self.settings.yellow_max_positions
        )
        existing_symbols = set(self.positions) | {
            entry.symbol
            for entries in self.pending.values()
            for entry in entries
        }
        slots = max(0, allowed_positions - len(existing_symbols))
        if slots <= 0:
            return

        min_score = (
            self.settings.min_technical_score
            if regime_assessment.regime == "GREEN"
            else self.settings.yellow_min_score
        )
        candidates: list[tuple[float, str, TechnicalAssessment, Decimal, date]] = []
        for symbol in self.settings.watchlist:
            if symbol in existing_symbols or self._cooldown_active(symbol, current_date):
                continue
            frame = self.bars.get(symbol)
            if frame is None:
                continue
            history = _date_frame(frame, current_date)
            if history.empty:
                continue
            try:
                technical = self.analysis_fn(
                    history,
                    rsi_period=self.settings.rsi_period,
                    atr_period=self.settings.atr_period,
                    pullback_sma=self.settings.pullback_sma,
                    fast_sma=self.settings.trend_fast_sma,
                    slow_sma=self.settings.trend_slow_sma,
                    max_atr_pct=self.settings.max_atr_pct,
                    min_score=min_score,
                )
            except (ValueError, IndexError, KeyError, TypeError):
                continue
            if not technical.eligible:
                continue
            stop_pct = dynamic_stop_pct(
                _decimal(technical.atr_pct),
                self.settings.atr_stop_multiple,
                self.settings.min_stop_pct,
                self.settings.max_stop_pct,
            )
            if stop_pct is None:
                continue
            entry_date = next_trading_day(frame.index, current_date)
            if entry_date is None or entry_date > self.settings.end:
                continue
            candidates.append(
                (
                    technical.score,
                    symbol,
                    technical,
                    stop_pct,
                    entry_date,
                )
            )

        candidates.sort(key=lambda item: (-item[0], item[1]))
        scheduled_today = 0
        risk_capital = self._risk_capital(current_date)
        for score, symbol, technical, stop_pct, entry_date in candidates:
            if slots <= 0 or scheduled_today >= self.settings.max_new_trades_per_day:
                break
            available_cash = max(Decimal("0"), self.cash - self.reserved_cash)
            cap = position_notional_cap(
                risk_capital,
                available_cash,
                stop_pct,
                allocation_pct=self.settings.position_alloc_pct,
                risk_per_trade_pct=self.settings.risk_per_trade_pct,
            )
            if cap < Decimal("1"):
                continue
            target_pct = stop_pct * self.settings.reward_risk_multiple
            self.pending[entry_date].append(
                PendingEntry(
                    symbol=symbol,
                    signal_date=current_date,
                    entry_date=entry_date,
                    reserved_cap=cap,
                    risk_capital=risk_capital,
                    stop_pct=stop_pct,
                    target_pct=target_pct,
                    market_regime=regime_assessment.regime,
                    technical_score=score,
                    rsi=technical.rsi,
                    atr_pct=technical.atr_pct,
                    momentum20=technical.momentum20,
                    momentum60=technical.momentum60,
                )
            )
            self.reserved_cash += cap
            existing_symbols.add(symbol)
            slots -= 1
            scheduled_today += 1

    def _enter_pending(self, current_date: date) -> None:
        entries = self.pending.pop(current_date, [])
        for intent in entries:
            self.reserved_cash = max(Decimal("0"), self.reserved_cash - intent.reserved_cap)
            bar = self._bar_for(intent.symbol, current_date)
            if bar is None or intent.symbol in self.positions:
                continue
            open_price = _decimal(bar["open"])
            if open_price <= 0:
                continue
            entry_price = open_price * (Decimal("1") + self.settings.slippage)
            available_cash = min(intent.reserved_cap, self.cash)
            try:
                plan = build_position_plan(
                    risk_capital=intent.risk_capital,
                    reference_price=entry_price,
                    atr_pct=_decimal(intent.atr_pct),
                    allocation_pct=self.settings.position_alloc_pct,
                    risk_per_trade_pct=self.settings.risk_per_trade_pct,
                    min_stop_pct=self.settings.min_stop_pct,
                    max_stop_pct=self.settings.max_stop_pct,
                    atr_stop_multiple=self.settings.atr_stop_multiple,
                    reward_risk_multiple=self.settings.reward_risk_multiple,
                    cash_available=available_cash,
                )
            except (ValueError, ArithmeticError):
                continue
            notional = plan.qty * entry_price
            if notional <= 0 or notional > self.cash:
                continue
            self.cash -= notional
            self.positions[intent.symbol] = OpenPosition(
                symbol=intent.symbol,
                signal_date=intent.signal_date,
                entry_date=current_date,
                entry_price=entry_price,
                quantity=plan.qty,
                position_notional=notional,
                stop_price=plan.stop_price,
                target_price=plan.take_profit_price,
                market_regime=intent.market_regime,
                technical_score=intent.technical_score,
                rsi=intent.rsi,
                atr_pct=intent.atr_pct,
                momentum20=intent.momentum20,
                momentum60=intent.momentum60,
            )

    def _process_exits(self, current_date: date) -> None:
        for symbol, position in list(self.positions.items()):
            bar = self._bar_for(symbol, current_date)
            if bar is None:
                continue
            decision = resolve_daily_exit(
                open_price=_decimal(bar["open"]),
                high=_decimal(bar["high"]),
                low=_decimal(bar["low"]),
                stop_price=position.stop_price,
                target_price=position.target_price,
                slippage=self.settings.slippage,
            )
            if decision is not None:
                self._close_position(position, current_date, decision)

    def _close_position(
        self,
        position: OpenPosition,
        exit_date: date,
        decision: ExitDecision,
    ) -> None:
        self.cash += position.quantity * decision.price
        return_pct = float(decision.price / position.entry_price - Decimal("1"))
        pnl = position.quantity * (decision.price - position.entry_price)
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
                "target_price": float(position.target_price),
                "return_pct": return_pct,
                "pnl_dollars": float(pnl),
                "holding_days": max((exit_date - position.entry_date).days, 0),
                "market_regime": position.market_regime,
                "technical_score": position.technical_score,
                "rsi": position.rsi,
                "atr_pct": position.atr_pct,
                "momentum20": position.momentum20,
                "momentum60": position.momentum60,
                "exit_reason": decision.reason,
                "ambiguous_stop_first": decision.ambiguous_stop_first,
            }
        )
        self.last_exit[position.symbol] = exit_date
        self.positions.pop(position.symbol, None)

    def _liquidate_at_end(self, current_date: date) -> None:
        for position in list(self.positions.values()):
            close = self._close_for(position.symbol, current_date)
            if close is None:
                continue
            decision = ExitDecision(
                adverse_exit_price(close, self.settings.slippage),
                "end_of_backtest",
            )
            self._close_position(position, current_date, decision)

    def _refresh_final_equity_row(self) -> None:
        if not self.equity_curve:
            return
        final = self.equity_curve[-1]
        final["cash"] = float(self.cash)
        final["open_position_value"] = 0.0
        final["equity"] = float(self.cash)
        final["positions_open"] = 0
        final["drawdown_pct"] = float(
            max(
                Decimal("0"),
                (self.high_water - _decimal(self.cash)) / self.high_water,
            )
            if self.high_water > 0
            else Decimal("0")
        )

    def _build_summary(self, trading_dates: list[date]) -> dict[str, Any]:
        ending = _decimal(self.equity_curve[-1]["equity"])
        starting = self.settings.initial_capital
        total_return = ending / starting - Decimal("1") if starting else Decimal("0")
        days = max((self.settings.end - self.settings.start).days, 1)
        annualized = (
            float((ending / starting) ** (Decimal("365.25") / Decimal(days)) - Decimal("1"))
            if ending > 0 and starting > 0
            else -1.0
        )
        returns = [float(trade["return_pct"]) for trade in self.trades]
        winners = [value for value in returns if value > 0]
        losers = [value for value in returns if value <= 0]
        gross_wins = sum(
            float(trade["pnl_dollars"]) for trade in self.trades if trade["pnl_dollars"] > 0
        )
        gross_losses = abs(
            sum(float(trade["pnl_dollars"]) for trade in self.trades if trade["pnl_dollars"] < 0)
        )
        max_dd = max((float(row["drawdown_pct"]) for row in self.equity_curve), default=0.0)
        max_dd_row = next(
            (row for row in self.equity_curve if float(row["drawdown_pct"]) == max_dd),
            None,
        )
        losing_streak = 0
        worst_streak = 0
        for value in returns:
            if value <= 0:
                losing_streak += 1
                worst_streak = max(worst_streak, losing_streak)
            else:
                losing_streak = 0
        invested_pct = self.invested_days / len(trading_dates) if trading_dates else 0.0
        open_counts = [int(row["positions_open"]) for row in self.equity_curve]
        benchmarks = {
            symbol: self._benchmark(symbol)
            for symbol in BENCHMARKS
        }
        regime_trades = {
            regime: sum(1 for trade in self.trades if trade["market_regime"] == regime)
            for regime in ("GREEN", "YELLOW", "RED")
        }
        return {
            "backtest_period": {
                "start": self.settings.start.isoformat(),
                "end": self.settings.end.isoformat(),
            },
            "starting_capital": float(starting),
            "ending_capital": float(ending),
            "total_return_pct": float(total_return),
            "annualized_return_pct": annualized,
            "spy_ending_value": benchmarks["SPY"]["ending_value"],
            "spy_total_return_pct": benchmarks["SPY"]["total_return_pct"],
            "qqq_ending_value": benchmarks["QQQ"]["ending_value"],
            "qqq_total_return_pct": benchmarks["QQQ"]["total_return_pct"],
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
            "percentage_of_time_invested": invested_pct,
            "average_open_positions": (
                sum(open_counts) / len(open_counts) if open_counts else 0.0
            ),
            "green_regime_trades": regime_trades["GREEN"],
            "yellow_regime_trades": regime_trades["YELLOW"],
            "red_regime_trades": regime_trades["RED"],
            "ambiguous_daily_bar_exits": sum(
                1 for trade in self.trades if trade["ambiguous_stop_first"]
            ),
            "slippage_assumption": float(self.settings.slippage),
            "historical_data_feed": self.settings.data_feed,
            "return_over_max_drawdown": (
                float(total_return) / max_dd if max_dd else None
            ),
            "portfolio_drawdown_halt_date": (
                self.halt_date.isoformat() if self.halt_date else None
            ),
        }

    def _benchmark(self, symbol: str) -> dict[str, float | None]:
        frame = self.bars.get(symbol)
        if frame is None:
            return {"ending_value": 0.0, "total_return_pct": 0.0}
        interval = frame.loc[
            (frame.index >= self.settings.start) & (frame.index <= self.settings.end)
        ]
        if interval.empty:
            return {"ending_value": 0.0, "total_return_pct": 0.0}
        first_open = _decimal(interval.iloc[0]["open"])
        last_close = _decimal(interval.iloc[-1]["close"])
        if first_open <= 0:
            return {"ending_value": 0.0, "total_return_pct": 0.0}
        ending = self.settings.initial_capital * last_close / first_open
        return {
            "ending_value": float(ending),
            "total_return_pct": float(last_close / first_open - Decimal("1")),
        }


def print_report(result: BacktestResult) -> None:
    summary = result.summary
    period = summary["backtest_period"]
    print("BACKTEST PERIOD")
    print(f"  {period['start']} to {period['end']}")
    print(f"Starting capital: ${summary['starting_capital']:.2f}")
    print(f"Ending capital: ${summary['ending_capital']:.2f}")
    print(f"Total return %: {summary['total_return_pct']:.2%}")
    print(f"Annualized return %: {summary['annualized_return_pct']:.2%}")
    print(f"SPY ending value: ${summary['spy_ending_value']:.2f}")
    print(f"SPY total return %: {summary['spy_total_return_pct']:.2%}")
    print(f"QQQ ending value: ${summary['qqq_ending_value']:.2f}")
    print(f"QQQ total return %: {summary['qqq_total_return_pct']:.2%}")
    print(f"Total trades: {summary['total_trades']}")
    print(f"Winning trades: {summary['winning_trades']}")
    print(f"Losing trades: {summary['losing_trades']}")
    print(f"Win rate: {summary['win_rate']:.2%}")
    print(f"Average winner %: {summary['average_winner_pct']:.2%}")
    print(f"Average loser %: {summary['average_loser_pct']:.2%}")
    print(f"Average holding days: {summary['average_holding_days']:.2f}")
    profit_factor = summary["profit_factor"]
    print(f"Profit factor: {profit_factor:.4f}" if profit_factor is not None else "Profit factor: n/a")
    print(f"Maximum drawdown %: {summary['maximum_drawdown_pct']:.2%}")
    print(f"Date of maximum drawdown: {summary['date_of_maximum_drawdown'] or 'n/a'}")
    print(f"Worst losing streak: {summary['worst_losing_streak']}")
    print(f"Best trade: {summary['best_trade_pct']:.2%}")
    print(f"Worst trade: {summary['worst_trade_pct']:.2%}")
    print(f"Percentage of time invested: {summary['percentage_of_time_invested']:.2%}")
    print(f"Average number of open positions: {summary['average_open_positions']:.2f}")
    print(f"GREEN regime trades: {summary['green_regime_trades']}")
    print(f"YELLOW regime trades: {summary['yellow_regime_trades']}")
    print(f"RED regime trades: {summary['red_regime_trades']}")
    print(f"Ambiguous daily-bar exits: {summary['ambiguous_daily_bar_exits']}")
    print(f"Slippage assumption: {summary['slippage_assumption']:.2%}")
    print(f"Historical data feed used: {summary['historical_data_feed']}")
    ratio = summary["return_over_max_drawdown"]
    print(f"Return / max drawdown: {ratio:.4f}" if ratio is not None else "Return / max drawdown: n/a")
    print(f"Portfolio drawdown halt date: {summary['portfolio_drawdown_halt_date'] or 'not triggered'}")
    if result.output_dir:
        print(f"Output directory: {result.output_dir}")


def run_backtest(
    cfg: Config,
    *,
    years: int | None = None,
    start_text: str | None = None,
    end_text: str | None = None,
    capital: str = "100",
    slippage: str = "0.001",
) -> int:
    start, end = resolve_period(years=years, start_text=start_text, end_text=end_text)
    initial_capital = Decimal(str(capital))
    slippage_value = Decimal(str(slippage))
    if initial_capital <= 0:
        raise ValueError("--capital must be greater than zero")
    if slippage_value < 0 or slippage_value >= 1:
        raise ValueError("--slippage must be between 0 and 1")
    settings = BacktestSettings.from_config(
        cfg,
        start=start,
        end=end,
        capital=initial_capital,
        slippage=slippage_value,
    )
    bars = fetch_historical_bars(
        cfg.alpaca_api_key,
        cfg.alpaca_secret_key,
        settings.watchlist,
        start=start,
        end=end,
        feed_name=settings.data_feed,
    )
    result = Backtester(settings, bars).run()
    result.write_outputs(cfg.project_root / "backtests")
    print_report(result)
    return 0