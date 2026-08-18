# All-Research Integration

This update combines every implemented research family into one screening command while leaving the broker execution path unchanged.

## Strategy families screened together

- Existing price strategies: Current V2, MA 20/100, MA 50/200, Donchian 55, 3-Down mean reversion, bearish-engulfing mean reversion, Three-Line Strike, Regime Adaptive.
- GREEN-only post-hoc variants: MA 20/100, MA 50/200, Donchian 55.
- Copytrading: COPY_RAW, COPY_FILTERED, COPY_CONSENSUS, COPY_ELITE, COPY_ADAPTIVE, COPY_V2_HYBRID, KAY_STYLE_RAW, KAY_STYLE_FILTERED.
- Historical news-veto overlays on MA 20/100, MA 50/200, and Donchian 55.
- Historical congressional-confidence overlays on the same three strategies.
- Combined historical news + congressional overlays on the same three strategies.

That is 28 research strategies when all three historical input files are supplied.

## Safety and causality rules

- Research only; no broker order submission is added.
- Congress never creates a trade. It contributes at most the same +5 ranking points used by the production scanner.
- Congressional information becomes usable only after `disclosure_time`, never at `transaction_time`.
- To avoid daily-bar intraday ambiguity, congressional disclosures dated on the candidate signal day are not used until a later trading day.
- News remains veto-only. A missing historical news assessment fails closed for news-overlay strategies.
- Historical news decisions are precomputed and timestamped so backtests are deterministic and do not call a live model during replay.
- RED still blocks new longs. Existing stop, sizing, daily-loss, cooldown, position-count, and 5% high-water rules remain unchanged.
- The repeatedly inspected 2024-2026 validation period is descriptive only for new variants; it is not described as untouched out-of-sample evidence.

## Required historical inputs

### Copy signals

Use the existing schema in `data/copy_signals.example.csv`.

### Congressional disclosures

`data/congress_signals.example.csv` shows the strict schema:

`member_id,symbol,side,transaction_time,disclosure_time`

Optional:

`amount_low,amount_high,source_id,verified`

All timestamps must be timezone-aware ISO-8601 strings. `disclosure_time` must be on or after `transaction_time`.

### Historical AI-news assessments

`data/news_assessments.example.csv` shows the strict schema:

`symbol,signal_date,assessment_time,verdict`

Optional:

`reason,headline_count,source_id`

`verdict` must be `SAFE` or `RISKY`. `assessment_time` must be timezone-aware and its calendar date must equal `signal_date`. These rows should be generated from only the headlines available at that historical decision time using the same conservative veto policy as production.

## One-command screen

```bash
python3 run.py research-suite \
  --start 2021-08-18 \
  --end 2026-08-17 \
  --capital 100 \
  --holdout-start 2024-01-01 \
  --copy-signals data/copy_signals.csv \
  --congress-signals data/congress_signals.csv \
  --news-assessments data/news_assessments.csv
```

The default all-strategy screen runs at 0.10%, 0.25%, and 0.50% adverse slippage. It writes a combined `suite_holdout_screen.csv` under `backtests/research_suite/<timestamp>/` plus the full per-slippage comparison outputs.

This screen is a funnel, not a promotion rule. Any survivor still goes through the full Robustness Lab and then forward PAPER testing with `DRY_RUN=true` until explicitly changed by the user.
