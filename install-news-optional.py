from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if new in text:
        print(f"already patched: {path}")
        return
    if old not in text:
        raise SystemExit(f"Expected pattern not found in {path}; aborting without changing it")
    backup = p.with_suffix(p.suffix + ".newsopt.bak")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"patched: {path}")


replace_once(
    "run.py",
    'suite.add_argument("--news-assessments", required=True, help="Strict historical news-veto assessment CSV")',
    'suite.add_argument("--news-assessments", help="Optional strict historical news-veto assessment CSV; omit when historical news is unavailable")',
)
replace_once(
    "run.py",
    'validate_data.add_argument("--news-assessments", required=True)',
    'validate_data.add_argument("--news-assessments")',
)
replace_once(
    "run.py",
    '"news": len(load_news_assessments_csv(args.news_assessments)),',
    '"news": len(load_news_assessments_csv(args.news_assessments)) if args.news_assessments else 0,',
)
replace_once(
    "bot/strategy_lab.py",
    '    congress_signals_path: str,\n    news_assessments_path: str,\n    dividend_events_path: str,',
    '    congress_signals_path: str | None,\n    news_assessments_path: str | None,\n    dividend_events_path: str,',
)
replace_once(
    "bot/strategy_lab.py",
    '    load_congress_signals_csv(congress_signals_path)\n    load_news_assessments_csv(news_assessments_path)\n    load_dividend_events_csv(dividend_events_path)',
    '    if congress_signals_path:\n        load_congress_signals_csv(congress_signals_path)\n    if news_assessments_path:\n        load_news_assessments_csv(news_assessments_path)\n    load_dividend_events_csv(dividend_events_path)',
)
replace_once(
    "bot/strategy_lab.py",
    '            "congress_signals": str(congress_signals_path),\n            "news_assessments": str(news_assessments_path),',
    '            "congress_signals": str(congress_signals_path) if congress_signals_path else None,\n            "news_assessments": str(news_assessments_path) if news_assessments_path else None,',
)

print("news optional patch applied")
