from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable

UTC = timezone.utc
REQUIRED_COLUMNS = (
    "leader_id",
    "symbol",
    "side",
    "leader_time",
    "leader_price",
    "signal_time",
    "follower_price",
)
OPTIONAL_COLUMNS = ("source_trade_id", "verified")
ALLOWED_COLUMNS = frozenset((*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS))


@dataclass(frozen=True)
class CopySignal:
    leader_id: str
    symbol: str
    side: str
    leader_time: datetime
    leader_price: Decimal
    signal_time: datetime
    follower_price: Decimal
    source_trade_id: str | None = None
    verified: bool | None = None
    row_number: int | None = None

    @property
    def latency_seconds(self) -> float:
        return (self.signal_time - self.leader_time).total_seconds()

    @property
    def entry_disadvantage_pct(self) -> float:
        if self.leader_price <= 0:
            return 0.0
        if self.side == "BUY":
            return float(self.follower_price / self.leader_price - Decimal("1"))
        return float(Decimal("1") - self.follower_price / self.leader_price)

    @property
    def identity(self) -> tuple[str, str, str, str]:
        return (
            self.leader_id,
            self.symbol,
            self.source_trade_id or "",
            self.signal_time.isoformat(),
        )


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


def _parse_price(raw: str, *, field: str, row_number: int) -> Decimal:
    value = raw.strip()
    try:
        parsed = Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"row {row_number}: {field} must be a decimal number") from exc
    if not parsed.is_finite() or parsed <= 0:
        raise ValueError(f"row {row_number}: {field} must be greater than zero")
    return parsed


def _parse_verified(raw: str | None, *, row_number: int) -> bool | None:
    if raw is None or not raw.strip():
        return None
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"row {row_number}: verified must be true/false when provided")


def _normalize_side(raw: str, *, row_number: int) -> str:
    value = raw.strip().upper()
    aliases = {"LONG": "BUY", "OPEN": "BUY", "EXIT": "SELL", "CLOSE": "SELL"}
    value = aliases.get(value, value)
    if value not in {"BUY", "SELL"}:
        raise ValueError(f"row {row_number}: side must be BUY or SELL")
    return value


def load_copy_signals_csv(path: str | Path) -> list[CopySignal]:
    """Load the copy-signal schema with fail-closed validation.

    The loader intentionally rejects unknown columns so a provider schema change cannot
    silently alter a research run. Timestamps are normalized to UTC after requiring an
    explicit timezone offset.
    """
    source = Path(path)
    if not source.exists() or not source.is_file():
        raise ValueError(f"copy-signal CSV does not exist: {source}")

    with source.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
        if missing:
            raise ValueError(f"copy-signal CSV missing required columns: {', '.join(missing)}")
        unknown = [column for column in fieldnames if column not in ALLOWED_COLUMNS]
        if unknown:
            raise ValueError(f"copy-signal CSV has unknown columns: {', '.join(unknown)}")

        result: list[CopySignal] = []
        seen: set[tuple[str, str, str, str, str]] = set()
        for row_number, row in enumerate(reader, start=2):
            leader_id = (row.get("leader_id") or "").strip()
            symbol = (row.get("symbol") or "").strip().upper()
            if not leader_id:
                raise ValueError(f"row {row_number}: leader_id is required")
            if not symbol or not symbol.replace(".", "").replace("-", "").isalnum():
                raise ValueError(f"row {row_number}: symbol is invalid")
            side = _normalize_side(row.get("side") or "", row_number=row_number)
            leader_time = _parse_timestamp(row.get("leader_time") or "", field="leader_time", row_number=row_number)
            signal_time = _parse_timestamp(row.get("signal_time") or "", field="signal_time", row_number=row_number)
            if signal_time < leader_time:
                raise ValueError(f"row {row_number}: signal_time cannot precede leader_time")
            leader_price = _parse_price(row.get("leader_price") or "", field="leader_price", row_number=row_number)
            follower_price = _parse_price(row.get("follower_price") or "", field="follower_price", row_number=row_number)
            source_trade_id = (row.get("source_trade_id") or "").strip() or None
            verified = _parse_verified(row.get("verified"), row_number=row_number)
            duplicate_key = (
                leader_id,
                symbol,
                side,
                source_trade_id or "",
                signal_time.isoformat(),
            )
            if duplicate_key in seen:
                raise ValueError(f"row {row_number}: duplicate copy signal")
            seen.add(duplicate_key)
            result.append(
                CopySignal(
                    leader_id=leader_id,
                    symbol=symbol,
                    side=side,
                    leader_time=leader_time,
                    leader_price=leader_price,
                    signal_time=signal_time,
                    follower_price=follower_price,
                    source_trade_id=source_trade_id,
                    verified=verified,
                    row_number=row_number,
                )
            )

    result.sort(key=lambda signal: (signal.signal_time, signal.leader_id, signal.symbol, signal.side))
    return result


def copy_signal_symbols(signals: Iterable[CopySignal]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(signal.symbol for signal in signals))
