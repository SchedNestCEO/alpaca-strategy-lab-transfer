# Copytrading Strategy Lab Patch

This patch is research-only. It does not submit broker orders and does not modify the paper-only execution safety boundary.

## Additions

- `bot/strategies/copytrading.py`
  - `COPY_RAW`
  - `COPY_FILTERED`
  - `COPY_CONSENSUS`
  - `COPY_ELITE`
  - `COPY_ADAPTIVE`
  - `COPY_V2_HYBRID`
  - `KAY_STYLE_RAW`
  - `KAY_STYLE_FILTERED`
  - leader rolling quality gates
  - anti-FOMO entry disadvantage gates
  - copy latency and copyability-tax metrics
  - risk sizing compatible with the existing 0.5% planned-risk / 10% notional / 5% stop-rejection rules
- `bot/copy_signals.py`: strict CSV signal loader
- `tests/test_copytrading.py`: offline unit tests
- `data/copy_signals.example.csv`: import schema example

## Strategy Lab integration points

1. Add `--copy-signals PATH` to the existing `run.py compare` CLI.
2. Load the CSV through `load_copy_signals_csv`.
3. Add the eight strategy names to the simulator registry.
4. For every copy strategy, execute using the follower-available price, never the leader's price.
5. Compute leader qualification only from trades closed before the candidate signal timestamp.
6. In daily-bar mode, if intraday follower pricing is unavailable, execute no earlier than the next session open.
7. Preserve the existing RED-regime no-new-long rule, volatility stop rejection, 0.5% planned account-risk cap, 10% notional cap, and 5% high-water halt.
8. Add these comparison columns:
   - `copy_latency_seconds`
   - `entry_disadvantage_pct`
   - `leader_return_pct`
   - `follower_return_pct`
   - `copyability_tax_pct`
   - `leader_id`
   - `leader_qualified_at_entry`
   - `consensus_count`

## Winner selection

Do not pick the strategy with the highest ending balance alone. Rank candidates on an untouched holdout using a survival-first score, for example:

1. Reject if the 5% halt triggers substantially earlier than peers or if account ruin occurs.
2. Require adequate trade count.
3. Prefer positive holdout expectancy and profit factor > 1.
4. Then compare holdout return, max drawdown, and copyability tax.
5. Keep training-period results descriptive only; do not retune on the holdout.

## Signal CSV schema

Required columns:

`leader_id,symbol,side,leader_time,leader_price,signal_time,follower_price`

Optional:

`source_trade_id,verified`

All timestamps should be timezone-aware ISO-8601 strings.
