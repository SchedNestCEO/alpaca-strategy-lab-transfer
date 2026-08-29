# Polymarket Shadow Copy

Standalone research-only service inside the Alpaca strategy repository. It does **not** place Polymarket orders and does not modify Alpaca execution.

Tracked wallet: `0xf705fa045201391d9632b7f3cde06a5e24453ca7`

The worker polls the public Polymarket Data API for new wallet trades, filters for Crypto markets, records the wallet entry, then captures the executable side of the CLOB order book at +30 seconds, +2 minutes, +5 minutes, and +15 minutes. Positive slippage means the follower would receive a worse entry than the tracked wallet.

Endpoints:
- `/health` service health
- `/summary` running copyability statistics and recent tracked trades

Environment variables:
- `POLYMARKET_WALLET` target wallet
- `POLL_SECONDS` default 5
- `BOOTSTRAP_LOOKBACK_SECONDS` default 30
- `DB_PATH` SQLite path (default `/tmp/polymarket-shadow-copy.db`)
- `PORT` supplied by Railway

For durable long-term observations, attach persistent storage or migrate the SQLite tables to Postgres. The initial Railway deployment intentionally starts research-only with no signing keys or trade-execution credentials.
