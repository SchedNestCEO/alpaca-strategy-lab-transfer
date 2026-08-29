from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib import error, request


@dataclass(frozen=True)
class AveDarisService:
    id: str
    name: str
    endpoint: str
    protocol: str
    price_usd: float
    trust_score: float
    success_rate: float
    median_latency_ms: float
    score: float | None = None


class AveDarisClient:
    """Read-only AveDaris routing client for the Alpaca PAPER research system.

    This client is intentionally incapable of placing brokerage orders, changing risk
    limits, arming execution, or bypassing the existing market/news/risk gates.
    It only discovers/ranks external research or data services through AveDaris.
    """

    def __init__(self, base_url: str, *, timeout_seconds: float = 5.0) -> None:
        base = base_url.strip().rstrip("/")
        if not base:
            raise ValueError("AveDaris base URL is required")
        if not (base.startswith("https://") or base.startswith("http://localhost")):
            raise ValueError("AveDaris URL must use HTTPS outside localhost")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.base_url = base
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_env(cls) -> AveDarisClient | None:
        if os.getenv("AVEDARIS_ENABLED", "false").strip().lower() != "true":
            return None
        url = os.getenv("AVEDARIS_URL", "").strip()
        if not url:
            raise ValueError("AVEDARIS_URL is required when AVEDARIS_ENABLED=true")
        timeout = float(os.getenv("AVEDARIS_TIMEOUT_SECONDS", "5"))
        return cls(url, timeout_seconds=timeout)

    def health(self) -> dict[str, Any]:
        return self._request_json("GET", "/health")

    def discover(
        self,
        capability: str,
        *,
        max_price_usd: float = 0.05,
        min_trust_score: float = 70.0,
        protocols: tuple[str, ...] = ("mpp", "x402", "free"),
        limit: int = 5,
    ) -> list[AveDarisService]:
        payload = self._routing_payload(
            capability,
            max_price_usd=max_price_usd,
            min_trust_score=min_trust_score,
            protocols=protocols,
            limit=limit,
        )
        data = self._request_json("POST", "/v1/discover", payload)
        return [self._parse_service(item) for item in data.get("services", [])]

    def route(
        self,
        capability: str,
        *,
        max_price_usd: float = 0.05,
        min_trust_score: float = 70.0,
        protocols: tuple[str, ...] = ("mpp", "x402", "free"),
    ) -> AveDarisService | None:
        payload = self._routing_payload(
            capability,
            max_price_usd=max_price_usd,
            min_trust_score=min_trust_score,
            protocols=protocols,
            limit=1,
        )
        try:
            data = self._request_json("POST", "/v1/route", payload)
        except RuntimeError as exc:
            if "HTTP 404" in str(exc):
                return None
            raise
        raw = data.get("service")
        return self._parse_service(raw) if isinstance(raw, dict) else None

    @staticmethod
    def _routing_payload(
        capability: str,
        *,
        max_price_usd: float,
        min_trust_score: float,
        protocols: tuple[str, ...],
        limit: int,
    ) -> dict[str, Any]:
        capability = capability.strip()
        if not capability:
            raise ValueError("capability is required")
        if max_price_usd < 0:
            raise ValueError("max_price_usd must be >= 0")
        if not 0 <= min_trust_score <= 100:
            raise ValueError("min_trust_score must be between 0 and 100")
        if limit < 1 or limit > 20:
            raise ValueError("limit must be between 1 and 20")
        allowed = {"mpp", "x402", "free"}
        invalid = set(protocols) - allowed
        if invalid:
            raise ValueError(f"unsupported AveDaris protocol(s): {', '.join(sorted(invalid))}")
        return {
            "capability": capability,
            "maxPriceUsd": float(max_price_usd),
            "minTrustScore": float(min_trust_score),
            "protocols": list(protocols),
            "limit": limit,
        }

    def _request_json(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"accept": "application/json"}
        if body is not None:
            headers["content-type"] = "application/json"
        req = request.Request(self.base_url + path, data=body, headers=headers, method=method)
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
                parsed = json.loads(raw) if raw else {}
                if not isinstance(parsed, dict):
                    raise RuntimeError("AveDaris returned a non-object JSON response")
                return parsed
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"AveDaris HTTP {exc.code}: {detail[:500]}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"AveDaris unavailable: {exc.reason}") from exc

    @staticmethod
    def _parse_service(raw: dict[str, Any]) -> AveDarisService:
        return AveDarisService(
            id=str(raw.get("id", "")),
            name=str(raw.get("name", "")),
            endpoint=str(raw.get("endpoint", "")),
            protocol=str(raw.get("protocol", "")),
            price_usd=float(raw.get("priceUsd", 0.0)),
            trust_score=float(raw.get("trustScore", 0.0)),
            success_rate=float(raw.get("successRate", 0.0)),
            median_latency_ms=float(raw.get("medianLatencyMs", 0.0)),
            score=float(raw["score"]) if raw.get("score") is not None else None,
        )
