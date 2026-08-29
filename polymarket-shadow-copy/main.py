import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests

WALLET = os.getenv("POLYMARKET_WALLET", "0xf705fa045201391d9632b7f3cde06a5e24453ca7").lower()
POLL_SECONDS = float(os.getenv("POLL_SECONDS", "5"))
DATA_API = os.getenv("POLYMARKET_DATA_API", "https://data-api.polymarket.com")
CLOB_API = os.getenv("POLYMARKET_CLOB_API", "https://clob.polymarket.com")
DB_PATH = os.getenv("DB_PATH", "/tmp/polymarket-shadow-copy.db")
PORT = int(os.getenv("PORT", "8080"))
DELAYS = (30, 120, 300, 900)
BOOTSTRAP_LOOKBACK_SECONDS = int(os.getenv("BOOTSTRAP_LOOKBACK_SECONDS", "30"))

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("polymarket-shadow-copy")
http = requests.Session()
http.headers.update({"User-Agent": "polymarket-shadow-copy-research/1.0"})


def now_ts():
    return int(time.time())


def db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    conn = db()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS trades (
      trade_key TEXT PRIMARY KEY,
      tx_hash TEXT,
      wallet TEXT NOT NULL,
      timestamp INTEGER NOT NULL,
      condition_id TEXT,
      asset TEXT NOT NULL,
      side TEXT,
      outcome TEXT,
      title TEXT,
      slug TEXT,
      entry_price REAL NOT NULL,
      size REAL,
      detected_at INTEGER NOT NULL,
      raw_json TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS snapshots (
      trade_key TEXT NOT NULL,
      delay_seconds INTEGER NOT NULL,
      target_timestamp INTEGER NOT NULL,
      captured_at INTEGER,
      midpoint REAL,
      best_bid REAL,
      best_ask REAL,
      copy_price REAL,
      slippage REAL,
      status TEXT NOT NULL DEFAULT 'pending',
      error TEXT,
      PRIMARY KEY (trade_key, delay_seconds),
      FOREIGN KEY (trade_key) REFERENCES trades(trade_key)
    );
    """)
    conn.commit()
    conn.close()


def api_get(url, params=None, timeout=10):
    r = http.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def trade_key(t):
    stable = "|".join(str(t.get(k, "")) for k in ("transactionHash", "timestamp", "asset", "side", "price", "size"))
    return hashlib.sha256(stable.encode()).hexdigest()


def fetch_recent_trades():
    # Polymarket Data API: public user trade history.
    data = api_get(f"{DATA_API}/trades", {"user": WALLET, "limit": 100})
    return data if isinstance(data, list) else data.get("data", [])


def is_crypto_trade(t):
    # Avoid false positives: accept explicit category metadata first, then well-known crypto market text.
    category = str(t.get("category") or t.get("eventCategory") or "").lower()
    if category == "crypto" or "crypto" in category:
        return True
    text = " ".join(str(t.get(k) or "") for k in ("title", "question", "slug", "eventSlug", "description")).lower()
    tokens = ("bitcoin", " btc", "btc-", "ethereum", " eth", "eth-", "solana", " sol", "sol-", "crypto", "xrp", "dogecoin", "doge")
    return any(x in f" {text}" for x in tokens)


def normalize_trade(t):
    asset = str(t.get("asset") or t.get("tokenId") or t.get("token_id") or "")
    if not asset:
        return None
    try:
        ts = int(float(t.get("timestamp") or t.get("createdAt") or 0))
        price = float(t.get("price"))
    except (TypeError, ValueError):
        return None
    if ts <= 0 or not (0 <= price <= 1):
        return None
    try:
        size = float(t.get("size")) if t.get("size") is not None else None
    except (TypeError, ValueError):
        size = None
    return {
        "trade_key": trade_key(t), "tx_hash": t.get("transactionHash"), "timestamp": ts,
        "condition_id": t.get("conditionId") or t.get("condition_id"), "asset": asset,
        "side": str(t.get("side") or "BUY").upper(), "outcome": t.get("outcome"),
        "title": t.get("title") or t.get("question"), "slug": t.get("slug") or t.get("eventSlug"),
        "entry_price": price, "size": size, "raw_json": json.dumps(t, separators=(",", ":"), default=str),
    }


def insert_trade(t):
    conn = db()
    cur = conn.execute("""INSERT OR IGNORE INTO trades
      (trade_key,tx_hash,wallet,timestamp,condition_id,asset,side,outcome,title,slug,entry_price,size,detected_at,raw_json)
      VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
      (t["trade_key"], t["tx_hash"], WALLET, t["timestamp"], t["condition_id"], t["asset"], t["side"],
       t["outcome"], t["title"], t["slug"], t["entry_price"], t["size"], now_ts(), t["raw_json"]))
    inserted = cur.rowcount == 1
    if inserted:
        for delay in DELAYS:
            conn.execute("INSERT OR IGNORE INTO snapshots(trade_key,delay_seconds,target_timestamp) VALUES(?,?,?)",
                         (t["trade_key"], delay, t["timestamp"] + delay))
    conn.commit(); conn.close()
    return inserted


def order_book(asset):
    book = api_get(f"{CLOB_API}/book", {"token_id": asset})
    bids = book.get("bids") or []
    asks = book.get("asks") or []
    best_bid = max((float(x["price"]) for x in bids), default=None)
    best_ask = min((float(x["price"]) for x in asks), default=None)
    midpoint = (best_bid + best_ask) / 2 if best_bid is not None and best_ask is not None else (best_bid if best_bid is not None else best_ask)
    return midpoint, best_bid, best_ask


def capture_due_snapshots():
    conn = db()
    due = conn.execute("""SELECT s.*, t.asset, t.side, t.entry_price FROM snapshots s JOIN trades t USING(trade_key)
                          WHERE s.status='pending' AND s.target_timestamp <= ? ORDER BY s.target_timestamp LIMIT 100""", (now_ts(),)).fetchall()
    conn.close()
    for row in due:
        try:
            mid, bid, ask = order_book(row["asset"])
            side = (row["side"] or "BUY").upper()
            copy_price = ask if side == "BUY" else bid
            if copy_price is None:
                copy_price = mid
            # Positive slippage means a follower receives a worse price than the tracked wallet.
            slip = None if copy_price is None else ((copy_price - row["entry_price"]) if side == "BUY" else (row["entry_price"] - copy_price))
            conn = db()
            conn.execute("""UPDATE snapshots SET captured_at=?,midpoint=?,best_bid=?,best_ask=?,copy_price=?,slippage=?,status='captured',error=NULL
                            WHERE trade_key=? AND delay_seconds=?""",
                         (now_ts(), mid, bid, ask, copy_price, slip, row["trade_key"], row["delay_seconds"]))
            conn.commit(); conn.close()
            log.info("snapshot %ss %s copy=%s slippage=%s", row["delay_seconds"], row["trade_key"][:10], copy_price, slip)
        except Exception as e:
            log.warning("snapshot error %s: %s", row["trade_key"][:10], e)
            # Keep pending so transient API errors retry; expose last error for diagnostics.
            conn = db(); conn.execute("UPDATE snapshots SET error=? WHERE trade_key=? AND delay_seconds=?", (str(e)[:500], row["trade_key"], row["delay_seconds"])); conn.commit(); conn.close()


def poller():
    initialized = False
    while True:
        try:
            trades = fetch_recent_trades()
            cutoff = now_ts() - BOOTSTRAP_LOOKBACK_SECONDS if not initialized else 0
            for raw in sorted(trades, key=lambda x: int(float(x.get("timestamp") or 0))):
                t = normalize_trade(raw)
                if not t or t["timestamp"] < cutoff or not is_crypto_trade(raw):
                    continue
                if insert_trade(t):
                    log.info("NEW CRYPTO TRADE %s %s %s @ %.4f size=%s", t["title"], t["side"], t["outcome"], t["entry_price"], t["size"])
            initialized = True
            capture_due_snapshots()
        except Exception as e:
            log.exception("poll loop error: %s", e)
        time.sleep(POLL_SECONDS)


def summary():
    conn = db()
    trades = conn.execute("SELECT COUNT(*) n FROM trades").fetchone()["n"]
    rows = conn.execute("""SELECT delay_seconds, COUNT(*) n, AVG(slippage) avg_slippage,
                           SUM(CASE WHEN slippage <= 0 THEN 1 ELSE 0 END) favorable
                           FROM snapshots WHERE status='captured' GROUP BY delay_seconds ORDER BY delay_seconds""").fetchall()
    recent = conn.execute("SELECT trade_key,timestamp,title,outcome,side,entry_price,size FROM trades ORDER BY timestamp DESC LIMIT 20").fetchall()
    conn.close()
    return {"status":"ok", "wallet":WALLET, "tracked_crypto_trades":trades,
            "copyability":[dict(x) for x in rows], "recent_trades":[dict(x) for x in recent],
            "timestamp":datetime.now(timezone.utc).isoformat()}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ("/", "/health", "/summary"):
            self.send_response(404); self.end_headers(); return
        payload = {"status":"ok"} if self.path == "/health" else summary()
        body = json.dumps(payload, default=str).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def log_message(self, fmt, *args):
        return


if __name__ == "__main__":
    init_db()
    threading.Thread(target=poller, daemon=True).start()
    log.info("tracking %s every %ss; snapshots=%s", WALLET, POLL_SECONDS, DELAYS)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
