from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import pandas as pd

from .backtest import BacktestSettings, fetch_historical_bars, normalize_bars, parse_date
from .config import Config
from .strategies.common import indicators
from .strategy_lab import (
    LAB_ETFS,
    STRATEGY_LABELS,
    StrategyDefinition,
    StrategySimulator,
    precompute_market_regimes,
    strategy_definitions,
)

UTC = timezone.utc
DEFAULT_STRATEGIES = ("ma_20_100", "ma_50_200", "donchian_55")
DEFAULT_SLIPPAGES = (
    Decimal("0"),
    Decimal("0.001"),
    Decimal("0.0015"),
    Decimal("0.002"),
    Decimal("0.0025"),
    Decimal("0.005"),
)
DEFAULT_WINDOW_SLIPPAGES = (Decimal("0.001"), Decimal("0.0025"))
DEFAULT_ROLLING_MONTHS = (12, 24)
BASELINE_SLIPPAGE = Decimal("0.001")
STRESS_SLIPPAGE = Decimal("0.0025")
SEVERE_SLIPPAGE = Decimal("0.005")
MIN_HOLDOUT_TRADES = 30
MAX_BASELINE_WINDOW_HALT_RATE = 0.20
MAX_STRESS_WINDOW_HALT_RATE = 0.25
MIN_BASELINE_POSITIVE_WINDOW_RATE = 0.60
MIN_STRESS_POSITIVE_WINDOW_RATE = 0.50


@dataclass(frozen=True)
class WindowSpec:
    kind: str
    label: str
    start: date
    end: date


@dataclass
class RobustnessResult:
    start: date
    end: date
    holdout_start: date
    capital: Decimal
    slippages: tuple[Decimal, ...]
    window_slippages: tuple[Decimal, ...]
    strategies: tuple[str, ...]
    window_step_months: int
    rolling_months: tuple[int, ...]
    data_feed: str
    slippage_rows: list[dict[str, Any]]
    window_rows: list[dict[str, Any]]
    shadow_rows: list[dict[str, Any]]
    benchmark_rows: list[dict[str, Any]]
    summary_rows: list[dict[str, Any]]
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

        _write_csv(output_dir / "summary.csv", self.summary_rows)
        _write_csv(output_dir / "slippage_stress.csv", self.slippage_rows)
        _write_csv(output_dir / "rolling_and_calendar_windows.csv", self.window_rows)
        _write_csv(output_dir / "shadow_continuation.csv", self.shadow_rows)
        _write_csv(output_dir / "exposure_matched_benchmarks.csv", self.benchmark_rows)

        payload = {
            "methodology": {
                "research_only": True,
                "production_execution_modified": False,
                "strategies": list(self.strategies),
                "slippages": [float(value) for value in self.slippages],
                "window_slippages": [float(value) for value in self.window_slippages],
                "rolling_months": list(self.rolling_months),
                "window_step_months": self.window_step_months,
                "baseline_slippage": float(BASELINE_SLIPPAGE),
                "stress_slippage": float(STRESS_SLIPPAGE),
                "severe_slippage": float(SEVERE_SLIPPAGE),
                "minimum_holdout_trades": MIN_HOLDOUT_TRADES,
                "eligibility_gates": {
                    "baseline_full": "positive return and no 5% high-water halt",
                    "baseline_holdout": "positive return, PF>1, adequate trades, and no halt",
                    "stress_holdout": "positive return, PF>1, and no halt at 0.25% slippage",
                    "severe_holdout": "non-negative return, PF>=1, and no halt at 0.50% slippage",
                    "baseline_windows": (
                        f"median return>0, median PF>1, positive-window rate>={MIN_BASELINE_POSITIVE_WINDOW_RATE:.0%}, "
                        f"halt rate<={MAX_BASELINE_WINDOW_HALT_RATE:.0%}"
                    ),
                    "stress_windows": (
                        f"median return>=0, median PF>=1, positive-window rate>={MIN_STRESS_POSITIVE_WINDOW_RATE:.0%}, "
                        f"halt rate<={MAX_STRESS_WINDOW_HALT_RATE:.0%}"
                    ),
                },
                "shadow_continuation": (
                    "At the fixed 0.10% baseline only, the official 5% halt remains unchanged while a parallel "
                    "research shadow disables only the simulator's entry halt so post-breach hypothetical performance "
                    "can be diagnosed; it never submits orders."
                ),
                "exposure_matched_benchmarks": [
                    "30% SPY + 70% cash",
                    "30% QQQ + 70% cash",
                    "15% SPY + 15% QQQ + 70% cash",
                ],
            },
            "period": {
                "start": self.start.isoformat(),
                "end": self.end.isoformat(),
                "holdout_start": self.holdout_start.isoformat(),
            },
            "starting_capital": float(self.capital),
            "historical_data_feed": self.data_feed,
            "summary": self.summary_rows,
        }
        with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

        self.output_dir = output_dir
        return output_dir


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                columns.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def parse_decimal_list(text: str) -> tuple[Decimal, ...]:
    values: list[Decimal] = []
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        value = Decimal(raw)
        if value < 0 or value >= 1:
            raise ValueError("Slippage values must be between 0 and 1")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("At least one slippage value is required")
    return tuple(sorted(values))


def parse_int_list(text: str) -> tuple[int, ...]:
    values: list[int] = []
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        value = int(raw)
        if value <= 0:
            raise ValueError("Rolling window months must be positive")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("At least one rolling window length is required")
    return tuple(sorted(values))


def parse_strategy_list(text: str) -> tuple[str, ...]:
    definitions = strategy_definitions()
    values: list[str] = []
    for raw in text.split(","):
        key = raw.strip()
        if not key:
            continue
        if key not in definitions:
            raise ValueError(
                f"Unsupported robustness strategy {key!r}. "
                f"Choose from: {', '.join(sorted(definitions))}"
            )
        if key not in values:
            values.append(key)
    if not values:
        raise ValueError("At least one strategy is required")
    return tuple(values)


def _add_months(value: date, months: int) -> date:
    return (pd.Timestamp(value) + pd.DateOffset(months=months)).date()


def rolling_windows(start: date, end: date, *, months: int, step_months: int) -> list[WindowSpec]:
    if months <= 0 or step_months <= 0:
        raise ValueError("Rolling window and step sizes must be positive")
    windows: list[WindowSpec] = []
    cursor = start
    index = 1
    while True:
        window_end = _add_months(cursor, months) - timedelta(days=1)
        if window_end > end:
            break
        windows.append(
            WindowSpec(
                kind=f"rolling_{months}m",
                label=f"R{months:02d}-{index:02d}",
                start=cursor,
                end=window_end,
            )
        )
        cursor = _add_months(cursor, step_months)
        index += 1
    return windows


def calendar_year_windows(start: date, end: date) -> list[WindowSpec]:
    windows: list[WindowSpec] = []
    for year in range(start.year, end.year + 1):
        year_start = date(year, 1, 1)
        year_end = date(year, 12, 31)
        if year_start >= start and year_end <= end:
            windows.append(WindowSpec("calendar_year", str(year), year_start, year_end))
    return windows


def build_windows(
    start: date,
    end: date,
    *,
    rolling_months_values: Iterable[int],
    step_months: int,
) -> list[WindowSpec]:
    windows = calendar_year_windows(start, end)
    for months in rolling_months_values:
        windows.extend(rolling_windows(start, end, months=months, step_months=step_months))
    return windows


def _prepare_bars(bars: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {symbol: indicators(normalize_bars(frame)) for symbol, frame in bars.items()}


def _period_settings(
    base: BacktestSettings,
    *,
    start: date,
    end: date,
    slippage: Decimal,
    watchlist: tuple[str, ...],
    shadow: bool = False,
) -> BacktestSettings:
    # Shadow mode changes only the research simulator's halt threshold. All sizing,
    # stops, market-regime gates, daily-loss gates, and signal rules remain identical.
    high_water_limit = Decimal("999") if shadow else base.max_high_water_drawdown_pct
    return replace(
        base,
        start=start,
        end=end,
        slippage=slippage,
        watchlist=watchlist,
        max_high_water_drawdown_pct=high_water_limit,
    )


def _run_one(
    definition: StrategyDefinition,
    base: BacktestSettings,
    prepared_bars: dict[str, pd.DataFrame],
    watchlist: tuple[str, ...],
    *,
    start: date,
    end: date,
    slippage: Decimal,
    shadow: bool = False,
    precomputed_regimes: dict[date, Any] | None = None,
):
    settings = _period_settings(
        base,
        start=start,
        end=end,
        slippage=slippage,
        watchlist=watchlist,
        shadow=shadow,
    )
    return StrategySimulator(
        definition,
        settings,
        prepared_bars,
        prepared_bars=True,
        precomputed_regimes=precomputed_regimes,
    ).run()


def _metric_row(
    strategy: str,
    *,
    period: str,
    start: date,
    end: date,
    slippage: Decimal,
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "strategy": strategy,
        "label": STRATEGY_LABELS.get(strategy, strategy),
        "period": period,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "slippage": float(slippage),
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
    }


def _shadow_diagnostics(
    official_summary: dict[str, Any],
    shadow_result,
    *,
    threshold: Decimal,
) -> dict[str, Any]:
    curve = shadow_result.equity_curve
    breach = next((row for row in curve if float(row.get("drawdown_pct", 0.0)) >= float(threshold)), None)
    breach_date = date.fromisoformat(breach["date"]) if breach else None
    recovered = None
    days_to_recover = None
    post_breach_return = None
    if breach is not None:
        breach_index = curve.index(breach)
        prior_high = max(float(row["equity"]) for row in curve[: breach_index + 1])
        breach_equity = float(breach["equity"])
        recovery_row = next(
            (row for row in curve[breach_index + 1 :] if float(row["equity"]) >= prior_high),
            None,
        )
        recovered = recovery_row is not None
        if recovery_row is not None and breach_date is not None:
            days_to_recover = (date.fromisoformat(recovery_row["date"]) - breach_date).days
        ending = float(shadow_result.summary.get("ending_capital") or 0.0)
        post_breach_return = ending / breach_equity - 1 if breach_equity > 0 else None
    return {
        "official_halt_date": official_summary.get("portfolio_drawdown_halt_date"),
        "official_return_pct": official_summary.get("total_return_pct"),
        "official_trades": official_summary.get("total_trades"),
        "shadow_first_5pct_breach_date": breach["date"] if breach else None,
        "shadow_return_pct": shadow_result.summary.get("total_return_pct"),
        "shadow_max_drawdown_pct": shadow_result.summary.get("maximum_drawdown_pct"),
        "shadow_profit_factor": shadow_result.summary.get("profit_factor"),
        "shadow_trades": shadow_result.summary.get("total_trades"),
        "shadow_recovered_prior_high": recovered,
        "shadow_days_to_recover": days_to_recover,
        "shadow_post_breach_return_pct": post_breach_return,
    }


def _aligned_relative(frame: pd.DataFrame, *, start: date, end: date, slippage: Decimal) -> pd.Series:
    interval = frame.loc[(frame.index >= start) & (frame.index <= end)].copy()
    if interval.empty:
        return pd.Series(dtype=float)
    first_open = float(interval.iloc[0]["open"])
    if first_open <= 0:
        return pd.Series(dtype=float)
    entry = first_open * (1.0 + float(slippage))
    relative = interval["close"].astype(float) / entry
    if len(relative):
        relative.iloc[-1] = float(interval.iloc[-1]["close"]) * (1.0 - float(slippage)) / entry
    return relative


def _equity_summary(equity: pd.Series, *, capital: Decimal, start: date, end: date) -> dict[str, Any]:
    if equity.empty:
        return {
            "ending_capital": float(capital),
            "total_return_pct": 0.0,
            "annualized_return_pct": 0.0,
            "maximum_drawdown_pct": 0.0,
            "return_over_max_drawdown": None,
        }
    ending = float(equity.iloc[-1])
    starting = float(capital)
    ret = ending / starting - 1 if starting else 0.0
    running_max = equity.cummax()
    dd = (running_max - equity) / running_max.replace(0, float("nan"))
    max_dd = float(dd.max()) if not dd.empty else 0.0
    days = max((end - start).days, 1)
    annualized = (ending / starting) ** (365.25 / days) - 1 if ending > 0 and starting > 0 else -1.0
    return {
        "ending_capital": ending,
        "total_return_pct": ret,
        "annualized_return_pct": annualized,
        "maximum_drawdown_pct": max_dd,
        "return_over_max_drawdown": ret / max_dd if max_dd else None,
    }


def exposure_matched_benchmarks(
    bars: dict[str, pd.DataFrame],
    *,
    start: date,
    end: date,
    capital: Decimal,
    slippage: Decimal,
) -> list[dict[str, Any]]:
    spy = _aligned_relative(bars["SPY"], start=start, end=end, slippage=slippage)
    qqq = _aligned_relative(bars["QQQ"], start=start, end=end, slippage=slippage)
    aligned = pd.concat([spy.rename("spy"), qqq.rename("qqq")], axis=1).dropna()
    definitions = (
        ("spy_30_cash_70", "30% SPY + 70% cash", 0.30, 0.0),
        ("qqq_30_cash_70", "30% QQQ + 70% cash", 0.0, 0.30),
        ("spy15_qqq15_cash70", "15% SPY + 15% QQQ + 70% cash", 0.15, 0.15),
    )
    rows: list[dict[str, Any]] = []
    for key, label, spy_weight, qqq_weight in definitions:
        if aligned.empty:
            equity = pd.Series(dtype=float)
        else:
            cash_weight = 1.0 - spy_weight - qqq_weight
            equity = float(capital) * (
                cash_weight + spy_weight * aligned["spy"] + qqq_weight * aligned["qqq"]
            )
        summary = _equity_summary(equity, capital=capital, start=start, end=end)
        rows.append(
            {
                "benchmark": key,
                "label": label,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "slippage": float(slippage),
                **summary,
            }
        )
    return rows


def _median(values: list[float]) -> float | None:
    return median(values) if values else None


def _window_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "window_count": 0,
            "positive_rate": None,
            "halt_rate": None,
            "median_return": None,
            "worst_return": None,
            "median_profit_factor": None,
            "worst_drawdown": None,
        }
    returns = [float(row["total_return_pct"] or 0.0) for row in rows]
    pfs = [float(row["profit_factor"]) for row in rows if row.get("profit_factor") is not None]
    dds = [float(row["maximum_drawdown_pct"] or 0.0) for row in rows]
    return {
        "window_count": len(rows),
        "positive_rate": sum(1 for value in returns if value > 0) / len(rows),
        "halt_rate": sum(1 for row in rows if row.get("portfolio_drawdown_halt_date")) / len(rows),
        "median_return": _median(returns),
        "worst_return": min(returns),
        "median_profit_factor": _median(pfs),
        "worst_drawdown": max(dds) if dds else None,
    }


def _find_slippage_row(
    rows: list[dict[str, Any]], strategy: str, period: str, slippage: Decimal
) -> dict[str, Any] | None:
    target = float(slippage)
    return next(
        (
            row
            for row in rows
            if row["strategy"] == strategy
            and row["period"] == period
            and math.isclose(float(row["slippage"]), target, abs_tol=1e-12)
        ),
        None,
    )


def _window_subset(
    rows: list[dict[str, Any]],
    strategy: str,
    slippage: Decimal,
    *,
    rolling_only: bool = False,
) -> list[dict[str, Any]]:
    target = float(slippage)
    return [
        row
        for row in rows
        if row["strategy"] == strategy
        and math.isclose(float(row["slippage"]), target, abs_tol=1e-12)
        and (not rolling_only or str(row.get("period", "")).startswith("rolling_"))
    ]


def _safe_pf(row: dict[str, Any] | None) -> float | None:
    if not row or row.get("profit_factor") is None:
        return None
    return float(row["profit_factor"])


def _gate_summary(
    strategy: str,
    slippage_rows: list[dict[str, Any]],
    window_rows: list[dict[str, Any]],
    shadow_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_full = _find_slippage_row(slippage_rows, strategy, "full", BASELINE_SLIPPAGE)
    baseline_holdout = _find_slippage_row(slippage_rows, strategy, "holdout", BASELINE_SLIPPAGE)
    stress_holdout = _find_slippage_row(slippage_rows, strategy, "holdout", STRESS_SLIPPAGE)
    severe_holdout = _find_slippage_row(slippage_rows, strategy, "holdout", SEVERE_SLIPPAGE)
    base_windows = _window_stats(
        _window_subset(window_rows, strategy, BASELINE_SLIPPAGE, rolling_only=True)
    )
    stress_windows = _window_stats(
        _window_subset(window_rows, strategy, STRESS_SLIPPAGE, rolling_only=True)
    )
    base_calendar = _window_stats([
        row for row in _window_subset(window_rows, strategy, BASELINE_SLIPPAGE)
        if row.get("period") == "calendar_year"
    ])

    base_full_gate = bool(
        baseline_full
        and float(baseline_full.get("total_return_pct") or 0.0) > 0
        and not baseline_full.get("portfolio_drawdown_halt_date")
    )
    base_holdout_pf = _safe_pf(baseline_holdout)
    base_holdout_gate = bool(
        baseline_holdout
        and int(baseline_holdout.get("total_trades") or 0) >= MIN_HOLDOUT_TRADES
        and float(baseline_holdout.get("total_return_pct") or 0.0) > 0
        and base_holdout_pf is not None
        and base_holdout_pf > 1.0
        and not baseline_holdout.get("portfolio_drawdown_halt_date")
    )
    stress_pf = _safe_pf(stress_holdout)
    stress_holdout_gate = bool(
        stress_holdout
        and float(stress_holdout.get("total_return_pct") or 0.0) > 0
        and stress_pf is not None
        and stress_pf > 1.0
        and not stress_holdout.get("portfolio_drawdown_halt_date")
    )
    severe_pf = _safe_pf(severe_holdout)
    severe_holdout_gate = bool(
        severe_holdout
        and float(severe_holdout.get("total_return_pct") or 0.0) >= 0
        and severe_pf is not None
        and severe_pf >= 1.0
        and not severe_holdout.get("portfolio_drawdown_halt_date")
    )
    base_window_gate = bool(
        base_windows["window_count"]
        and (base_windows["median_return"] or 0.0) > 0
        and base_windows["median_profit_factor"] is not None
        and float(base_windows["median_profit_factor"]) > 1.0
        and (base_windows["positive_rate"] or 0.0) >= MIN_BASELINE_POSITIVE_WINDOW_RATE
        and (base_windows["halt_rate"] or 0.0) <= MAX_BASELINE_WINDOW_HALT_RATE
    )
    stress_window_gate = bool(
        stress_windows["window_count"]
        and (stress_windows["median_return"] or 0.0) >= 0
        and stress_windows["median_profit_factor"] is not None
        and float(stress_windows["median_profit_factor"]) >= 1.0
        and (stress_windows["positive_rate"] or 0.0) >= MIN_STRESS_POSITIVE_WINDOW_RATE
        and (stress_windows["halt_rate"] or 0.0) <= MAX_STRESS_WINDOW_HALT_RATE
    )
    gates = {
        "gate_baseline_full_survival": base_full_gate,
        "gate_baseline_holdout_edge": base_holdout_gate,
        "gate_stress_holdout_edge": stress_holdout_gate,
        "gate_severe_holdout_nonnegative": severe_holdout_gate,
        "gate_baseline_window_stability": base_window_gate,
        "gate_stress_window_stability": stress_window_gate,
    }
    passed = sum(1 for value in gates.values() if value)
    base_positive = float(base_windows["positive_rate"] or 0.0)
    stress_positive = float(stress_windows["positive_rate"] or 0.0)
    stability_score = 100.0 * (0.70 * passed / len(gates) + 0.15 * base_positive + 0.15 * stress_positive)

    shadow_full = next(
        (
            row
            for row in shadow_rows
            if row["strategy"] == strategy
            and row["period"] == "full"
            and math.isclose(float(row["slippage"]), float(BASELINE_SLIPPAGE), abs_tol=1e-12)
        ),
        None,
    )
    return {
        "strategy": strategy,
        "label": STRATEGY_LABELS.get(strategy, strategy),
        "eligible": all(gates.values()),
        "gates_passed": passed,
        "gates_total": len(gates),
        "stability_score": stability_score,
        **gates,
        "baseline_full_return_pct": baseline_full.get("total_return_pct") if baseline_full else None,
        "baseline_full_pf": baseline_full.get("profit_factor") if baseline_full else None,
        "baseline_full_halt_date": baseline_full.get("portfolio_drawdown_halt_date") if baseline_full else None,
        "baseline_holdout_return_pct": baseline_holdout.get("total_return_pct") if baseline_holdout else None,
        "baseline_holdout_pf": baseline_holdout.get("profit_factor") if baseline_holdout else None,
        "baseline_holdout_trades": baseline_holdout.get("total_trades") if baseline_holdout else None,
        "stress_holdout_return_pct": stress_holdout.get("total_return_pct") if stress_holdout else None,
        "stress_holdout_pf": stress_holdout.get("profit_factor") if stress_holdout else None,
        "severe_holdout_return_pct": severe_holdout.get("total_return_pct") if severe_holdout else None,
        "severe_holdout_pf": severe_holdout.get("profit_factor") if severe_holdout else None,
        "baseline_window_count": base_windows["window_count"],
        "baseline_window_positive_rate": base_windows["positive_rate"],
        "baseline_window_halt_rate": base_windows["halt_rate"],
        "baseline_window_median_return_pct": base_windows["median_return"],
        "baseline_window_worst_return_pct": base_windows["worst_return"],
        "baseline_window_median_pf": base_windows["median_profit_factor"],
        "baseline_window_worst_drawdown_pct": base_windows["worst_drawdown"],
        "stress_window_count": stress_windows["window_count"],
        "stress_window_positive_rate": stress_windows["positive_rate"],
        "stress_window_halt_rate": stress_windows["halt_rate"],
        "stress_window_median_return_pct": stress_windows["median_return"],
        "stress_window_worst_return_pct": stress_windows["worst_return"],
        "stress_window_median_pf": stress_windows["median_profit_factor"],
        "baseline_calendar_year_count": base_calendar["window_count"],
        "baseline_calendar_year_positive_rate": base_calendar["positive_rate"],
        "baseline_calendar_year_median_return_pct": base_calendar["median_return"],
        "baseline_calendar_year_worst_return_pct": base_calendar["worst_return"],
        "shadow_baseline_full_return_pct": shadow_full.get("shadow_return_pct") if shadow_full else None,
        "shadow_baseline_first_5pct_breach_date": shadow_full.get("shadow_first_5pct_breach_date") if shadow_full else None,
        "shadow_baseline_recovered_prior_high": shadow_full.get("shadow_recovered_prior_high") if shadow_full else None,
        "shadow_baseline_post_breach_return_pct": shadow_full.get("shadow_post_breach_return_pct") if shadow_full else None,
    }


def _ranking_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        1 if row["eligible"] else 0,
        int(row["gates_passed"]),
        float(row["stability_score"]),
        float(row.get("stress_holdout_return_pct") or -999),
        float(row.get("baseline_holdout_return_pct") or -999),
    )


def _closest_required(slippages: tuple[Decimal, ...], required: Decimal) -> None:
    if required not in slippages:
        raise ValueError(
            f"Robustness eligibility requires slippage {required}; include it in --slippages"
        )


def print_robustness(result: RobustnessResult) -> None:
    print("ROBUSTNESS LAB")
    print(f"Period: {result.start.isoformat()} to {result.end.isoformat()} | Holdout: {result.holdout_start.isoformat()}")
    print(f"Capital: ${result.capital:.2f} | Feed: {result.data_feed}")
    print("Strategies: " + ", ".join(STRATEGY_LABELS.get(key, key) for key in result.strategies))
    print("Slippage grid: " + ", ".join(f"{value:.2%}" for value in result.slippages))
    print(
        "Window grid: "
        + ", ".join(f"{months}m" for months in result.rolling_months)
        + f" rolling every {result.window_step_months} months + complete calendar years"
    )
    print()
    print(f"{'STRATEGY':20} {'ELIGIBLE':>9} {'GATES':>7} {'SCORE':>7} {'BASE HO':>9} {'0.25% HO':>9} {'0.50% HO':>9} {'WIN+':>7} {'HALTS':>7}")
    for row in sorted(result.summary_rows, key=_ranking_key, reverse=True):
        print(
            f"{str(row['label'])[:20]:20} "
            f"{('YES' if row['eligible'] else 'NO'):>9} "
            f"{str(row['gates_passed']) + '/' + str(row['gates_total']):>7} "
            f"{float(row['stability_score']):6.1f} "
            f"{_fmt_pct(row.get('baseline_holdout_return_pct')):>9} "
            f"{_fmt_pct(row.get('stress_holdout_return_pct')):>9} "
            f"{_fmt_pct(row.get('severe_holdout_return_pct')):>9} "
            f"{_fmt_pct(row.get('baseline_window_positive_rate')):>7} "
            f"{_fmt_pct(row.get('baseline_window_halt_rate')):>7}"
        )
    print("\nEligibility is a fixed research gate; it never arms or changes paper execution.")
    if result.output_dir:
        print(f"Output directory: {result.output_dir}")


def _fmt_pct(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.1%}"


def run_robustness(
    cfg: Config,
    *,
    start_text: str,
    end_text: str,
    holdout_start_text: str,
    capital: str = "100",
    strategies_text: str = ",".join(DEFAULT_STRATEGIES),
    slippages_text: str = ",".join(str(value) for value in DEFAULT_SLIPPAGES),
    window_slippages_text: str = ",".join(str(value) for value in DEFAULT_WINDOW_SLIPPAGES),
    rolling_months_text: str = ",".join(str(value) for value in DEFAULT_ROLLING_MONTHS),
    window_step_months: int = 12,
) -> int:
    start = parse_date(start_text)
    end = parse_date(end_text)
    holdout_start = parse_date(holdout_start_text)
    if start >= end:
        raise ValueError("--start must be before --end")
    if not (start < holdout_start < end):
        raise ValueError("--holdout-start must fall strictly inside the requested period")
    initial_capital = Decimal(str(capital))
    if initial_capital <= 0:
        raise ValueError("--capital must be greater than zero")
    if window_step_months <= 0:
        raise ValueError("--window-step-months must be greater than zero")

    strategies = parse_strategy_list(strategies_text)
    slippages = parse_decimal_list(slippages_text)
    window_slippages = parse_decimal_list(window_slippages_text)
    rolling_months_values = parse_int_list(rolling_months_text)
    for required in (BASELINE_SLIPPAGE, STRESS_SLIPPAGE, SEVERE_SLIPPAGE):
        _closest_required(slippages, required)
    for required in (BASELINE_SLIPPAGE, STRESS_SLIPPAGE):
        if required not in window_slippages:
            raise ValueError(
                f"Robustness window eligibility requires slippage {required}; include it in --window-slippages"
            )

    watchlist = tuple(dict.fromkeys((*cfg.watchlist, *LAB_ETFS)))
    base = BacktestSettings.from_config(
        cfg,
        start=start,
        end=end,
        capital=initial_capital,
        slippage=BASELINE_SLIPPAGE,
    )
    print("Fetching historical bars once for the full Robustness Lab...", flush=True)
    raw_bars = fetch_historical_bars(
        cfg.alpaca_api_key,
        cfg.alpaca_secret_key,
        watchlist,
        start=start,
        end=end,
        feed_name=base.data_feed,
    )
    if "SPY" not in raw_bars or "QQQ" not in raw_bars:
        raise ValueError("SPY and QQQ historical bars are required for robustness testing")
    prepared_bars = _prepare_bars(raw_bars)
    regime_map = precompute_market_regimes(
        prepared_bars["SPY"],
        fast=base.trend_fast_sma,
        slow=base.trend_slow_sma,
    )
    definitions = strategy_definitions()
    windows = build_windows(
        start,
        end,
        rolling_months_values=rolling_months_values,
        step_months=window_step_months,
    )

    main_periods = (WindowSpec("full", "full", start, end), WindowSpec("holdout", "holdout", holdout_start, end))
    slippage_rows: list[dict[str, Any]] = []
    shadow_rows: list[dict[str, Any]] = []
    benchmark_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []

    main_jobs = len(strategies) * len(slippages) * len(main_periods)
    # Shadow continuation is diagnostic, so run it only at the fixed 0.10% baseline
    # rather than multiplying the already-large stress grid.
    shadow_jobs = len(strategies) * len(main_periods)
    window_jobs = len(strategies) * len(window_slippages) * len(windows)
    print(
        f"Planned simulations: {main_jobs} official main + {shadow_jobs} shadow main + {window_jobs} rolling/calendar",
        flush=True,
    )

    completed = 0
    for strategy in strategies:
        definition = definitions[strategy]
        for slip in slippages:
            for period in main_periods:
                official = _run_one(
                    definition,
                    base,
                    prepared_bars,
                    watchlist,
                    start=period.start,
                    end=period.end,
                    slippage=slip,
                    precomputed_regimes=regime_map,
                )
                slippage_rows.append(
                    _metric_row(
                        strategy,
                        period=period.kind,
                        start=period.start,
                        end=period.end,
                        slippage=slip,
                        summary=official.summary,
                    )
                )
                completed += 1
                if slip == BASELINE_SLIPPAGE:
                    shadow = _run_one(
                        definition,
                        base,
                        prepared_bars,
                        watchlist,
                        start=period.start,
                        end=period.end,
                        slippage=slip,
                        shadow=True,
                        precomputed_regimes=regime_map,
                    )
                    shadow_rows.append(
                        {
                            "strategy": strategy,
                            "label": STRATEGY_LABELS.get(strategy, strategy),
                            "period": period.kind,
                            "start": period.start.isoformat(),
                            "end": period.end.isoformat(),
                            "slippage": float(slip),
                            **_shadow_diagnostics(
                                official.summary,
                                shadow,
                                threshold=base.max_high_water_drawdown_pct,
                            ),
                        }
                    )
                    completed += 1
                print(
                    f"[{completed}/{main_jobs + shadow_jobs + window_jobs}] "
                    f"{STRATEGY_LABELS.get(strategy, strategy)} {period.kind} {slip:.2%}",
                    flush=True,
                )

    for slip in slippages:
        for bench_period in main_periods:
            for row in exposure_matched_benchmarks(
                prepared_bars,
                start=bench_period.start,
                end=bench_period.end,
                capital=initial_capital,
                slippage=slip,
            ):
                benchmark_rows.append({"window_type": bench_period.kind, "window_label": bench_period.label, **row})

    for slip in window_slippages:
        for window in windows:
            for strategy in strategies:
                definition = definitions[strategy]
                result = _run_one(
                    definition,
                    base,
                    prepared_bars,
                    watchlist,
                    start=window.start,
                    end=window.end,
                    slippage=slip,
                    precomputed_regimes=regime_map,
                )
                row = _metric_row(
                    strategy,
                    period=window.kind,
                    start=window.start,
                    end=window.end,
                    slippage=slip,
                    summary=result.summary,
                )
                row["window_label"] = window.label
                window_rows.append(row)
                completed += 1
                if completed % 25 == 0 or completed == main_jobs + shadow_jobs + window_jobs:
                    print(
                        f"[{completed}/{main_jobs + shadow_jobs + window_jobs}] robustness simulations complete",
                        flush=True,
                    )
            for row in exposure_matched_benchmarks(
                prepared_bars,
                start=window.start,
                end=window.end,
                capital=initial_capital,
                slippage=slip,
            ):
                benchmark_rows.append({"window_type": window.kind, "window_label": window.label, **row})

    summary_rows = [
        _gate_summary(strategy, slippage_rows, window_rows, shadow_rows)
        for strategy in strategies
    ]
    result = RobustnessResult(
        start=start,
        end=end,
        holdout_start=holdout_start,
        capital=initial_capital,
        slippages=slippages,
        window_slippages=window_slippages,
        strategies=strategies,
        window_step_months=window_step_months,
        rolling_months=rolling_months_values,
        data_feed=base.data_feed,
        slippage_rows=slippage_rows,
        window_rows=window_rows,
        shadow_rows=shadow_rows,
        benchmark_rows=benchmark_rows,
        summary_rows=summary_rows,
    )
    result.write_outputs(cfg.project_root / "backtests" / "robustness")
    print_robustness(result)
    return 0
