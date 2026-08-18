# Adaptive Alpaca Paper Trader — Phone-First / Replit Build

A **paper-trading-only** Alpaca research app designed to be controlled from an iPhone while execution runs in the cloud.

This is not a guaranteed-profit system. It is a controlled experiment that combines deterministic risk rules, trend/pullback signals, an AI news **veto**, compounding position sizing, a persistent experience database, and a shadow learning model.

## Safety boundary

- The broker is permanently created with `TradingClient(..., paper=True)`.
- There is **no live-trading switch** in this project.
- New paper orders require two independent conditions:
  1. `DRY_RUN=false` in environment/secrets.
  2. `execution_enabled=true`, which you explicitly arm from the password-protected dashboard by typing `PAPER`.
- Pause automatically disarms execution.
- Flatten automatically pauses and disarms execution.
- The learning model cannot change risk limits or bypass the news/market/position gates.
- No bank-transfer or withdrawal permissions are included.

## Strategy in this build

The strategy is intentionally different from the original RSI<=35 dip buyer.

### Market regime

SPY is classified as:

- **GREEN**: strong trend / normal volatility — up to 3 positions.
- **YELLOW**: mixed conditions — up to 1 position and a higher technical-score threshold.
- **RED**: price below long trend or extreme volatility — no new long positions.

### Candidate rules

Candidates are scored using:

- price above 200-day SMA
- 50-day SMA above 200-day SMA
- rising 200-day SMA
- positive 20-day momentum
- positive 60-day momentum
- controlled pullback near the 20/50-day average
- RSI in a controlled range rather than deeply oversold
- ATR/volatility constraint
- reasonable volume participation
- optional congressional score as a **small bonus only**

The OpenAI news layer is veto-only. A `RISKY` or uncertain response blocks the trade.

### Compounding

New position sizing is based on **compounding risk capital**:

```text
risk capital = min(current equity, current equity - unrealized P/L)
```

This means:

- realized gains can increase future dollar position sizes;
- unrealized gains do **not** increase new position sizes;
- unrealized losses reduce sizing immediately.

Position notional is the smaller of:

```text
10% of risk capital
or
0.5% account risk / stop distance
or
available cash
```

So if the account grows, positions grow proportionally without intentionally increasing percentage risk.

### Stops / targets

The stop is volatility-aware:

```text
stop distance = max(2%, 2 x ATR%)
```

If normal volatility would require a stop beyond 5%, the position is rejected.

Default profit target is 2x the stop distance. This build intentionally keeps broker-side bracket protection rather than attempting an unsafe fractional trailing-stop conversion.

A stop trigger is not a guaranteed execution price; gaps/slippage are possible.

## Learning system

Every qualifying safe candidate is saved once per symbol/day — including opportunities the bot does not ultimately trade because of slots, cooldowns, or execution locks.

After the configured forward horizon (default 10 trading bars), the app labels the observation using historical daily bars:

- target reached before stop -> positive label
- stop reached before target -> negative label
- neither -> sign of horizon closing return
- target and stop both touched inside one daily bar -> excluded as ambiguous

A logistic-regression **challenger** is trained on a time-ordered train/test split after enough observations exist. The dashboard shows its AUC/Brier metrics.

Promotion is manual. Even after promotion, the model will not affect candidate ranking unless `USE_PRODUCTION_MODEL=true` is explicitly set in the environment.

This prevents the bot from rewriting its own safety rules after a handful of trades.

## Persistent database

Locally, if `DATABASE_URL` is blank, the app uses SQLite at `data/bot.db`.

For Replit deployments, add Replit's SQL Database and use its `DATABASE_URL`. Do **not** rely on deployment-local files for history/state.

The database stores:

- pause / execution-arm state
- strategy-start flag / account high-water mark
- equity snapshots
- candidate features and decisions
- labeled outcomes
- model versions / validation metrics
- submitted paper-order records
- event history

## Mobile dashboard

The dashboard shows:

- paper account value
- return from the configured baseline
- compounding capital
- cash / unrealized P&L
- high-water drawdown
- market regime
- open positions
- candidate decisions
- recent paper orders
- challenger model metrics
- event history

Phone controls:

- Pause + disarm
- Resume scans
- Run scan now
- Arm/disarm paper orders
- Flatten the PAPER account (requires typing `PAPER`)
- Promote a challenger model (requires typing `PAPER`)

## Quick local setup

```bash
cp .env.example .env
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 run.py init-db
python3 run.py status
python3 run.py scan
python3 run.py web
```

Then open `http://localhost:5000`.

## Replit commands

Dashboard web process:

```bash
python3 run.py web
```

Hourly scheduled cycle:

```bash
python3 run.py cycle
```

A cycle:

1. checks the database pause state;
2. checks Alpaca's market clock;
3. scans and optionally paper-trades;
4. labels outcomes that have matured;
5. checks once per week whether a new challenger can be trained.

Useful manual commands:

```bash
python3 run.py status
python3 run.py scan
python3 run.py label
python3 run.py train
python3 run.py promote MODEL_ID --confirm PAPER
python3 run.py flatten --confirm PAPER
```

## Initial $100 experiment

Before the first scan, create/use a **clean Alpaca paper account with about $100 starting equity** and no unrelated positions/orders, then use that paper account's API keys. Alpaca currently creates paper accounts with $100,000 by default unless you configure a different starting balance. This app refuses to initialize its strategy when first-run paper equity is outside the configured baseline tolerance (`BASELINE_TOLERANCE_PCT`, default 2%).

Keep `DRY_RUN=true` initially. In dry-run mode the app scans, evaluates news, stores decisions, and learns, but cannot submit paper orders.

## Congress signal

`bot/congress.py` intentionally does **not** scrape congressional disclosure sites. It is a provider hook for a future licensed/approved data source.

Until one is integrated, congressional scores are zero and have no effect. Even when populated, the score contributes only a small ranking bonus; it cannot create a trade or bypass any hard gate.

## Replit persistence warning

Do not evaluate the deployment using SQLite. Replit deployments should use the managed SQL database so the dashboard and scheduled job share persistent state.

## Project layout

```text
bot/
  broker.py        Alpaca paper-only wrapper
  config.py        environment + safety validation
  congress.py      licensed-data hook (zero by default)
  dashboard.py     authenticated mobile web UI
  db.py            SQLAlchemy persistence
  indicators.py    trend/regime/ATR/RSI calculations
  learner.py       outcome labeling + challenger model
  main.py          scan/risk/execution workflow
  market_data.py   Alpaca historical/latest market data
  news_filter.py   OpenAI veto-only news classifier
  risk.py          compounding sizing + bracket plan

templates/         mobile dashboard HTML
static/            phone-first CSS
scripts/           Replit run helpers
tests/             offline unit tests
run.py             CLI/web entrypoint
```

## What this project does NOT do

- live trading
- margin / leverage
- options
- short selling
- automatic ACH/bank withdrawals
- direct congressional-site scraping
- autonomous risk-rule rewriting
- guaranteed 5% maximum realized loss
- guaranteed profit

Use paper trading and walk-forward testing before considering any separate live implementation.

## Optional copytrading Strategy Lab input

The Strategy Lab can also evaluate a strict external copy-signal CSV without adding any broker-order path. This is **research-only** and does not change the hard-coded Alpaca paper execution boundary.

Example:

```bash
python3 run.py compare \
  --start 2021-08-18 \
  --end 2026-08-17 \
  --capital 100 \
  --slippage 0.001 \
  --holdout-start 2024-01-01 \
  --copy-signals data/copy_signals.csv
```

Schema example: `data/copy_signals.example.csv`. Required columns are `leader_id,symbol,side,leader_time,leader_price,signal_time,follower_price`; optional columns are `source_trade_id,verified`. Timestamps must be timezone-aware ISO-8601 values.

When `--copy-signals` is supplied, the comparison adds: Copy Raw, Copy Filtered, Copy Consensus, Copy Elite, Copy Adaptive, Copy + V2 Hybrid, Kay-Style Raw, and Kay-Style Filtered. Every variant keeps the existing RED-regime block, volatility stop rejection, 0.5% planned-risk cap, 10% notional cap, daily loss gate, and 5% high-water halt. Follower fills are never allowed to use the leader's price. In daily-bar mode, entries/exits are delayed conservatively to a later session open and bounded adversely by the supplied follower price.

The integration note names rolling leader-quality and anti-FOMO gates but does not prescribe numeric thresholds. The implementation therefore uses fixed, documented research defaults in `bot/strategies/copytrading.py`; they are not grid-searched or tuned on the holdout. `KAY_STYLE_RAW` and `KAY_STYLE_FILTERED` are intentionally implemented as aliases of the raw/filtered copy policies until Kay-specific rule differences are supplied, rather than inventing unsupported rules.


## Regime hypothesis variants

The research registry also includes three explicitly post-hoc GREEN-only challengers: `ma_20_100_green`, `ma_50_200_green`, and `donchian_55_green`. They change only the entry regime from `GREEN or YELLOW` to `GREEN`; all risk, stop, exit, cooldown, slippage, and drawdown rules remain unchanged. See `REGIME_HYPOTHESIS.md`. These variants are exploratory because they were defined after inspecting earlier robustness/trade-forensics results.

## All-research screen

The research build can screen all 28 implemented price/regime/copy/news/congress variants in one command when real timestamped historical input files are available. See `ALL_RESEARCH_INTEGRATION.md`. The command is `python3 run.py research-suite ...`; it requires copy, congressional-disclosure, and historical news-assessment CSVs and screens them at 0.10%, 0.25%, and 0.50% adverse slippage by default. It never arms or changes PAPER execution.
