from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable

UTC = timezone.utc

CONGRESS_REQUIRED_COLUMNS = (
    "member_id",
    "symbol",
    "side",
    "transaction_time",
    "disclosure_time",
)
CONGRESS_OPTIONAL_COLUMNS = (
    "amount_low",
    "amount_high",
    "source_id",
    "verified",
)
CONGRESS_ALLOWED_COLUMNS = frozenset((*CONGRESS_REQUIRED_COLUMNS, *CONGRESS_OPTIONAL_COLUMNS))

NEWS_REQUIRED_COLUMNS = (
    "symbol",
    "signal_date",
    "assessment_time",
    "verdict",
)
NEWS_OPTIONAL_COLUMNS = ("reason", "headline_count", "source_id")
NEWS_ALLOWED_COLUMNS = frozenset((*NEWS_REQUIRED_COLUMNS, *NEWS_OPTIONAL_COLUMNS))


@dataclass(frozen=True)
class CongressDisclosure:
    member_id: str
    symbol: str
    side: str
    transaction_time: datetime
    disclosure_time: datetime
    amount_low: Decimal | None = None
    amount_high: Decimal | None = None
    source_id: str | None = None
    verified: bool | None = None
    row_number: int | None = None


@dataclass(frozen=True)
class HistoricalNewsAssessment:
    symbol: str
    signal_date: date
    assessment_time: datetime
    verdict: str
    reason: str = ""
    headline_count: int | None = None
    source_id: str | None = None
    row_number: int | None = None


def _parse_timestamp(raw: str, *, field: str, row_number: int) -> datetime:
    value = raw.strip()
    if not value:
        raise ValueError(f"row {row_number}: {field} is required")
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"row {row_number}: {field} must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"row {row_number}: {field} must be timezone-aware")
    return parsed.astimezone(UTC)


def _parse_date(raw: str, *, field: str, row_number: int) -> date:
    value = raw.strip()
    if not value:
        raise ValueError(f"row {row_number}: {field} is required")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"row {row_number}: {field} must be YYYY-MM-DD") from exc


def _parse_optional_decimal(raw: str | None, *, field: str, row_number: int) -> Decimal | None:
    if raw is None or not raw.strip():
        return None
    try:
        value = Decimal(raw.strip())
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"row {row_number}: {field} must be a decimal number") from exc
    if not value.is_finite() or value < 0:
        raise ValueError(f"row {row_number}: {field} must be non-negative")
    return value


def _parse_verified(raw: str | None, *, row_number: int) -> bool | None:
    if raw is None or not raw.strip():
        return None
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"row {row_number}: verified must be true/false when provided")


def _normalize_symbol(raw: str, *, row_number: int) -> str:
    symbol = raw.strip().upper()
    if not symbol or not symbol.replace(".", "").replace("-", "").isalnum():
        raise ValueError(f"row {row_number}: symbol is invalid")
    return symbol


def _normalize_congress_side(raw: str, *, row_number: int) -> str:
    value = raw.strip().upper()
    aliases = {
        "PURCHASE": "BUY",
        "PURCHASED": "BUY",
        "BUY": "BUY",
        "SALE": "SELL",
        "SOLD": "SELL",
        "SELL": "SELL",
    }
    normalized = aliases.get(value)
    if normalized is None:
        raise ValueError(f"row {row_number}: side must be BUY/PURCHASE or SELL/SALE")
    return normalized


def load_congress_signals_csv(path: str | Path) -> list[CongressDisclosure]:
    """Load historical public-disclosure signals with strict no-lookahead fields.

    The transaction timestamp records when the lawmaker traded. The disclosure timestamp
    records when the information became public. Research logic uses disclosure_time only;
    transaction_time can never make a signal available early.
    """
    source = Path(path)
    if not source.exists() or not source.is_file():
        raise ValueError(f"congress-signal CSV does not exist: {source}")

    with source.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        missing = [column for column in CONGRESS_REQUIRED_COLUMNS if column not in fieldnames]
        if missing:
            raise ValueError(f"congress-signal CSV missing required columns: {', '.join(missing)}")
        unknown = [column for column in fieldnames if column not in CONGRESS_ALLOWED_COLUMNS]
        if unknown:
            raise ValueError(f"congress-signal CSV has unknown columns: {', '.join(unknown)}")

        result: list[CongressDisclosure] = []
        seen: set[tuple[str, str, str, str, str]] = set()
        for row_number, row in enumerate(reader, start=2):
            member_id = (row.get("member_id") or "").strip()
            if not member_id:
                raise ValueError(f"row {row_number}: member_id is required")
            symbol = _normalize_symbol(row.get("symbol") or "", row_number=row_number)
            side = _normalize_congress_side(row.get("side") or "", row_number=row_number)
            transaction_time = _parse_timestamp(
                row.get("transaction_time") or "", field="transaction_time", row_number=row_number
            )
            disclosure_time = _parse_timestamp(
                row.get("disclosure_time") or "", field="disclosure_time", row_number=row_number
            )
            if disclosure_time < transaction_time:
                raise ValueError(f"row {row_number}: disclosure_time cannot precede transaction_time")
            amount_low = _parse_optional_decimal(row.get("amount_low"), field="amount_low", row_number=row_number)
            amount_high = _parse_optional_decimal(row.get("amount_high"), field="amount_high", row_number=row_number)
            if amount_low is not None and amount_high is not None and amount_high < amount_low:
                raise ValueError(f"row {row_number}: amount_high cannot be less than amount_low")
            source_id = (row.get("source_id") or "").strip() or None
            verified = _parse_verified(row.get("verified"), row_number=row_number)
            key = (member_id, symbol, side, source_id or "", disclosure_time.isoformat())
            if key in seen:
                raise ValueError(f"row {row_number}: duplicate congress disclosure")
            seen.add(key)
            result.append(
                CongressDisclosure(
                    member_id=member_id,
                    symbol=symbol,
                    side=side,
                    transaction_time=transaction_time,
                    disclosure_time=disclosure_time,
                    amount_low=amount_low,
                    amount_high=amount_high,
                    source_id=source_id,
                    verified=verified,
                    row_number=row_number,
                )
            )

    result.sort(key=lambda item: (item.disclosure_time, item.member_id, item.symbol, item.side))
    return result


def congress_signal_symbols(signals: Iterable[CongressDisclosure]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(signal.symbol for signal in signals))


def congress_score_at(
    signals: Iterable[CongressDisclosure],
    *,
    symbol: str,
    signal_date: date,
    lookback_days: int = 90,
) -> float:
    """Return a small 0-100 confidence score from already-public PURCHASE disclosures.

    To remain conservative with daily bars, disclosures dated on the signal day are not
    used because their intraday public availability relative to the strategy decision is
    ambiguous. Each distinct member with a verified-or-unspecified BUY disclosure in the
    prior lookback window contributes 25 points, capped at 100. The score can only affect
    ranking; it never creates a trade or bypasses a hard gate.
    """
    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive")
    start_date = signal_date - timedelta(days=lookback_days)
    members = {
        item.member_id
        for item in signals
        if item.symbol == symbol.upper()
        and item.side == "BUY"
        and item.verified is not False
        and start_date <= item.disclosure_time.date() < signal_date
    }
    return float(min(100, 25 * len(members)))


def load_news_assessments_csv(path: str | Path) -> list[HistoricalNewsAssessment]:
    """Load precomputed historical news-veto decisions.

    Each row is the verdict that was available for a specific symbol and strategy signal
    date. Missing assessments fail closed in the research simulator. This avoids calling
    a live model during a historical backtest and makes the run deterministic/reproducible.
    """
    source = Path(path)
    if not source.exists() or not source.is_file():
        raise ValueError(f"news-assessment CSV does not exist: {source}")

    with source.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        missing = [column for column in NEWS_REQUIRED_COLUMNS if column not in fieldnames]
        if missing:
            raise ValueError(f"news-assessment CSV missing required columns: {', '.join(missing)}")
        unknown = [column for column in fieldnames if column not in NEWS_ALLOWED_COLUMNS]
        if unknown:
            raise ValueError(f"news-assessment CSV has unknown columns: {', '.join(unknown)}")

        result: list[HistoricalNewsAssessment] = []
        seen: set[tuple[str, date]] = set()
        for row_number, row in enumerate(reader, start=2):
            symbol = _normalize_symbol(row.get("symbol") or "", row_number=row_number)
            signal_date = _parse_date(row.get("signal_date") or "", field="signal_date", row_number=row_number)
            assessment_time = _parse_timestamp(
                row.get("assessment_time") or "", field="assessment_time", row_number=row_number
            )
            if assessment_time.date() != signal_date:
                raise ValueError(
                    f"row {row_number}: assessment_time date must match signal_date for deterministic daily-bar replay"
                )
            verdict = (row.get("verdict") or "").strip().upper()
            if verdict not in {"SAFE", "RISKY"}:
                raise ValueError(f"row {row_number}: verdict must be SAFE or RISKY")
            reason = (row.get("reason") or "").strip()
            headline_count_raw = (row.get("headline_count") or "").strip()
            headline_count = None
            if headline_count_raw:
                try:
                    headline_count = int(headline_count_raw)
                except ValueError as exc:
                    raise ValueError(f"row {row_number}: headline_count must be an integer") from exc
                if headline_count < 0:
                    raise ValueError(f"row {row_number}: headline_count must be non-negative")
            source_id = (row.get("source_id") or "").strip() or None
            key = (symbol, signal_date)
            if key in seen:
                raise ValueError(f"row {row_number}: duplicate news assessment for {symbol} on {signal_date}")
            seen.add(key)
            result.append(
                HistoricalNewsAssessment(
                    symbol=symbol,
                    signal_date=signal_date,
                    assessment_time=assessment_time,
                    verdict=verdict,
                    reason=reason,
                    headline_count=headline_count,
                    source_id=source_id,
                    row_number=row_number,
                )
            )

    result.sort(key=lambda item: (item.signal_date, item.symbol))
    return result


def news_assessment_map(
    assessments: Iterable[HistoricalNewsAssessment],
) -> dict[tuple[str, date], HistoricalNewsAssessment]:
    return {(item.symbol, item.signal_date): item for item in assessments}
