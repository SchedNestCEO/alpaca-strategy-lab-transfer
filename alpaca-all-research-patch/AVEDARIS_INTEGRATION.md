# AveDaris integration — Alpaca as first network client

The existing Adaptive Alpaca Paper Trader is AveDaris's first internal consumer. This does **not** create another trading bot.

## Responsibility split

### Alpaca system owns

- market scanning and research
- paper-only broker execution
- deterministic position/risk limits
- news veto and market-regime gates
- backtests, robustness tests and learning experiments
- the existing PAPER arm/disarm controls

### AveDaris owns

- external service discovery
- provider ranking
- price/trust/protocol constraints
- provider reputation
- economic routing
- later: testnet paid execution and verifiable service receipts

## Safety boundary

`bot/avedaris_client.py` is a discovery/routing client only. It has no Alpaca credentials and no methods for broker orders, position sizing, execution arming, bank transfers, or withdrawals.

AveDaris results are **inputs**, never authority. A provider selected by AveDaris cannot bypass any existing Alpaca hard gate.

The trading system remains paper-only.

## Environment

AveDaris is opt-in and disabled by default:

```text
AVEDARIS_ENABLED=false
AVEDARIS_URL=https://<avedaris-worker>.workers.dev
AVEDARIS_TIMEOUT_SECONDS=5
```

No AveDaris payment key belongs in the Alpaca app. In v0.1, payment credentials remain isolated inside AveDaris's Cloudflare BuyerAgent.

## Example

```python
from bot.avedaris_client import AveDarisClient

client = AveDarisClient.from_env()
if client:
    provider = client.route(
        "market-news-verification",
        max_price_usd=0.02,
        min_trust_score=80,
    )
```

Future research adapters can use this selection to choose a data or analysis provider. The returned information must still pass the Alpaca system's normal validation and risk gates.

## First closed-loop target

1. Alpaca identifies a need for a research capability.
2. Alpaca asks AveDaris to route it.
3. AveDaris selects an eligible provider by capability, trust, reliability and cost.
4. AveDaris BuyerAgent performs a testnet-only paid request.
5. Alpaca receives the result as research input.
6. Alpaca independently accepts/rejects the input under its existing rules.
7. AveDaris records the service outcome and updates provider reputation.

Production payment delegation is intentionally not enabled yet.
