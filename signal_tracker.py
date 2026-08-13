"""
OptionsView Forward Tracker v1

Creates/updates:
- data/optionsview_signals.csv : one row per UNIQUE virtual trade
- data/optionsview_marks.csv   : one mark per open signal per scan

Does NOT change the entry algorithm.
It reads the current scan snapshot produced by alerts_engine.py.

Forward-test strategies tracked simultaneously:
A) HOLD_TO_EXPIRY
B) TP50
C) TP50_STOP2X

Important:
- Current option mark uses a conservative close debit:
    short ask - long bid
- Since GitHub scans only twice per day, TP/STOP detection is discrete,
  not intraday-perfect.
- Expiration payoff is calculated from underlying close at/after expiration.
"""

from __future__ import annotations

import csv
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

SNAPSHOT = Path("optionsview_alerts_scan.csv")
SIGNALS = Path("data/optionsview_signals.csv")
MARKS = Path("data/optionsview_marks.csv")

CONTRACT_MULTIPLIER = 100

SIGNAL_FIELDS = [
    "signal_id","opportunity_key","first_seen_utc","last_seen_pass_utc",
    "ticker","side","spread","short_strike","long_strike","expiration",
    "entry_dte","entry_price","entry_credit","commission_rt","net_max_loss",
    "entry_win_probability","entry_prob_otm","entry_delta","entry_net_roc",
    "entry_expected_move","entry_em_distance","entry_iv_rank","entry_short_iv",
    "entry_oi","entry_bid_ask_pct","entry_liquidity","entry_rsi",
    "entry_technical_score","entry_options_quality_score",
    "hold_status","hold_exit_utc","hold_exit_reason","hold_pnl_net",
    "tp50_status","tp50_exit_utc","tp50_exit_reason","tp50_exit_debit","tp50_pnl_net",
    "tp50_stop2x_status","tp50_stop2x_exit_utc","tp50_stop2x_exit_reason",
    "tp50_stop2x_exit_debit","tp50_stop2x_pnl_net",
    "latest_mark_utc","latest_underlying_price","latest_close_debit",
    "latest_unrealized_pnl_net","latest_dte",
]

MARK_FIELDS = [
    "mark_time_utc","signal_id","opportunity_key","ticker","side","expiration",
    "short_strike","long_strike","underlying_price","close_debit_conservative",
    "short_bid","short_ask","long_bid","long_ask","unrealized_pnl_net",
    "dte","mark_status","mark_error",
]


def safe_float(x, default=None):
    try:
        if x in (None, ""):
            return default
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def parse_bool(x) -> bool:
    return str(x).strip().lower() in {"true", "1", "yes"}


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]], fields: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def append_csv(path: Path, rows: List[Dict[str, object]], fields: List[str]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def normalize_strike(x) -> str:
    v = safe_float(x)
    if v is None:
        return ""
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:g}"


def opportunity_key(row: Dict[str, str]) -> str:
    return "|".join([
        str(row.get("ticker", "")).upper(),
        str(row.get("expiration", "")),
        str(row.get("signal", "")).upper(),
        normalize_strike(row.get("short_strike")),
        normalize_strike(row.get("long_strike")),
    ])


def make_signal_id(row: Dict[str, str], seen_utc: str) -> str:
    dt = pd.to_datetime(seen_utc, utc=True, errors="coerce")
    stamp = dt.strftime("%Y%m%dT%H%M") if not pd.isna(dt) else datetime.now(timezone.utc).strftime("%Y%m%dT%H%M")
    side_letter = "P" if str(row.get("signal", "")).upper() == "PUT" else "C"
    return (
        f"{str(row.get('ticker','')).upper()}-{stamp}-"
        f"{normalize_strike(row.get('short_strike'))}{side_letter}"
        f"{normalize_strike(row.get('long_strike'))}{side_letter}"
    )


def create_signal(row: Dict[str, str], seen_utc: str) -> Dict[str, object]:
    return {
        "signal_id": make_signal_id(row, seen_utc),
        "opportunity_key": opportunity_key(row),
        "first_seen_utc": seen_utc,
        "last_seen_pass_utc": seen_utc,
        "ticker": str(row.get("ticker", "")).upper(),
        "side": str(row.get("signal", "")).upper(),
        "spread": row.get("spread", ""),
        "short_strike": row.get("short_strike", ""),
        "long_strike": row.get("long_strike", ""),
        "expiration": row.get("expiration", ""),
        "entry_dte": row.get("dte", ""),
        "entry_price": row.get("price", ""),
        "entry_credit": row.get("credit", ""),
        "commission_rt": row.get("commission_rt", ""),
        "net_max_loss": row.get("net_max_loss", ""),
        "entry_win_probability": row.get("win_probability", ""),
        "entry_prob_otm": row.get("prob_otm", ""),
        "entry_delta": row.get("delta", ""),
        "entry_net_roc": row.get("net_roc", ""),
        "entry_expected_move": row.get("expected_move", ""),
        "entry_em_distance": row.get("em_distance", ""),
        "entry_iv_rank": row.get("iv_rank", ""),
        "entry_short_iv": row.get("short_iv", ""),
        "entry_oi": row.get("oi", ""),
        "entry_bid_ask_pct": row.get("bid_ask_spread_pct", ""),
        "entry_liquidity": row.get("liquidity", ""),
        "entry_rsi": row.get("rsi", ""),
        "entry_technical_score": row.get("technical_score", ""),
        "entry_options_quality_score": row.get("options_quality_score", ""),
        "hold_status": "OPEN","hold_exit_utc": "","hold_exit_reason": "","hold_pnl_net": "",
        "tp50_status": "OPEN","tp50_exit_utc": "","tp50_exit_reason": "",
        "tp50_exit_debit": "","tp50_pnl_net": "",
        "tp50_stop2x_status": "OPEN","tp50_stop2x_exit_utc": "",
        "tp50_stop2x_exit_reason": "","tp50_stop2x_exit_debit": "",
        "tp50_stop2x_pnl_net": "",
        "latest_mark_utc": "","latest_underlying_price": "","latest_close_debit": "",
        "latest_unrealized_pnl_net": "","latest_dte": "",
    }


def current_underlying_price(ticker: str) -> Optional[float]:
    try:
        hist = yf.Ticker(ticker).history(period="5d", interval="1d", auto_adjust=False)
        if hist is None or hist.empty:
            return None
        return safe_float(hist["Close"].dropna().iloc[-1])
    except Exception:
        return None


def get_contract_row(df: pd.DataFrame, strike: float) -> Optional[pd.Series]:
    if df is None or df.empty or "strike" not in df.columns:
        return None
    strikes = pd.to_numeric(df["strike"], errors="coerce")
    mask = (strikes - strike).abs() < 1e-8
    if not mask.any():
        return None
    return df.loc[mask].iloc[0]


def current_spread_mark(signal: Dict[str, str]) -> Tuple[Optional[Dict[str, float]], str]:
    ticker = signal["ticker"]
    side = signal["side"]
    expiration = signal["expiration"]
    short_strike = safe_float(signal["short_strike"])
    long_strike = safe_float(signal["long_strike"])

    if short_strike is None or long_strike is None or not expiration:
        return None, "missing strikes/expiration"

    try:
        tk = yf.Ticker(ticker)
        if expiration not in list(tk.options):
            return None, "expiration not available in current option chain"

        chain = tk.option_chain(expiration)
        df = chain.puts if side == "PUT" else chain.calls

        short = get_contract_row(df, short_strike)
        long = get_contract_row(df, long_strike)
        if short is None or long is None:
            return None, "exact strike not found"

        sbid = safe_float(short.get("bid"))
        sask = safe_float(short.get("ask"))
        lbid = safe_float(long.get("bid"))
        lask = safe_float(long.get("ask"))

        if None in (sbid, sask, lbid, lask):
            return None, "missing bid/ask"

        close_debit = max(0.0, sask - lbid)

        return {
            "short_bid": sbid,"short_ask": sask,
            "long_bid": lbid,"long_ask": lask,
            "close_debit": round(close_debit, 4),
        }, ""
    except Exception as e:
        return None, str(e)


def expiration_underlying_close(ticker: str, expiration: str) -> Optional[float]:
    try:
        exp = pd.Timestamp(expiration)
        end = exp + pd.Timedelta(days=5)
        hist = yf.Ticker(ticker).history(
            start=(exp - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=False,
        )
        if hist is None or hist.empty:
            return None

        idx = pd.to_datetime(hist.index).tz_localize(None).normalize()
        exact = hist.loc[idx == exp.normalize()]
        if not exact.empty:
            return safe_float(exact["Close"].iloc[-1])

        prior = hist.loc[idx <= exp.normalize()]
        if not prior.empty:
            return safe_float(prior["Close"].iloc[-1])

        return None
    except Exception:
        return None


def intrinsic_debit_at_expiration(signal: Dict[str, str], underlying: float) -> float:
    side = signal["side"]
    short = safe_float(signal["short_strike"], 0.0)
    long_ = safe_float(signal["long_strike"], 0.0)

    if side == "CALL":
        short_intrinsic = max(0.0, underlying - short)
        long_intrinsic = max(0.0, underlying - long_)
    else:
        short_intrinsic = max(0.0, short - underlying)
        long_intrinsic = max(0.0, long_ - underlying)

    return max(0.0, short_intrinsic - long_intrinsic)


def pnl_net(entry_credit: float, exit_debit: float, commission_rt: float) -> float:
    return round((entry_credit - exit_debit) * CONTRACT_MULTIPLIER - commission_rt, 2)


def close_strategy(signal: Dict[str, object], prefix: str, when: str, reason: str, debit: float):
    entry_credit = safe_float(signal.get("entry_credit"), 0.0)
    commission = safe_float(signal.get("commission_rt"), 0.0)
    signal[f"{prefix}_status"] = "CLOSED"
    signal[f"{prefix}_exit_utc"] = when
    signal[f"{prefix}_exit_reason"] = reason
    if prefix != "hold":
        signal[f"{prefix}_exit_debit"] = round(debit, 4)
    signal[f"{prefix}_pnl_net"] = pnl_net(entry_credit, debit, commission)


def process_expiration(signal: Dict[str, object], now_utc: str) -> bool:
    expiration = str(signal.get("expiration", ""))
    if not expiration:
        return False

    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    exp = pd.Timestamp(expiration).normalize()

    if today < exp:
        return False

    if (
        signal.get("hold_status") != "OPEN"
        and signal.get("tp50_status") != "OPEN"
        and signal.get("tp50_stop2x_status") != "OPEN"
    ):
        return True

    final_price = expiration_underlying_close(str(signal["ticker"]), expiration)
    if final_price is None:
        return False

    debit = intrinsic_debit_at_expiration(signal, final_price)

    if signal.get("hold_status") == "OPEN":
        close_strategy(signal, "hold", now_utc, "EXPIRATION", debit)
    if signal.get("tp50_status") == "OPEN":
        close_strategy(signal, "tp50", now_utc, "EXPIRATION", debit)
    if signal.get("tp50_stop2x_status") == "OPEN":
        close_strategy(signal, "tp50_stop2x", now_utc, "EXPIRATION", debit)

    return True


def mark_signal(signal: Dict[str, object], now_utc: str) -> Dict[str, object]:
    ticker = str(signal["ticker"])
    expiration = str(signal["expiration"])
    exp = pd.Timestamp(expiration).normalize()
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    dte = int((exp - today).days)

    underlying = current_underlying_price(ticker)

    if dte <= 0:
        ok = process_expiration(signal, now_utc)
        final_underlying = expiration_underlying_close(ticker, expiration)
        return {
            "mark_time_utc": now_utc,"signal_id": signal["signal_id"],
            "opportunity_key": signal["opportunity_key"],"ticker": ticker,
            "side": signal["side"],"expiration": expiration,
            "short_strike": signal["short_strike"],"long_strike": signal["long_strike"],
            "underlying_price": final_underlying if final_underlying is not None else (underlying or ""),
            "close_debit_conservative": "","short_bid": "","short_ask": "",
            "long_bid": "","long_ask": "","unrealized_pnl_net": "","dte": dte,
            "mark_status": "EXPIRATION_FINALIZED" if ok else "EXPIRATION_PENDING_DATA",
            "mark_error": "" if ok else "Could not obtain expiration underlying close",
        }

    quote, err = current_spread_mark(signal)
    if quote is None:
        return {
            "mark_time_utc": now_utc,"signal_id": signal["signal_id"],
            "opportunity_key": signal["opportunity_key"],"ticker": ticker,
            "side": signal["side"],"expiration": expiration,
            "short_strike": signal["short_strike"],"long_strike": signal["long_strike"],
            "underlying_price": underlying or "","close_debit_conservative": "",
            "short_bid": "","short_ask": "","long_bid": "","long_ask": "",
            "unrealized_pnl_net": "","dte": dte,"mark_status": "NO_MARK",
            "mark_error": err,
        }

    close_debit = quote["close_debit"]
    entry_credit = safe_float(signal.get("entry_credit"), 0.0)
    commission = safe_float(signal.get("commission_rt"), 0.0)
    unreal = pnl_net(entry_credit, close_debit, commission)

    signal["latest_mark_utc"] = now_utc
    signal["latest_underlying_price"] = underlying if underlying is not None else ""
    signal["latest_close_debit"] = close_debit
    signal["latest_unrealized_pnl_net"] = unreal
    signal["latest_dte"] = dte

    if signal.get("tp50_status") == "OPEN" and close_debit <= entry_credit * 0.50:
        close_strategy(signal, "tp50", now_utc, "TP50_OBSERVED", close_debit)

    if signal.get("tp50_stop2x_status") == "OPEN":
        if close_debit <= entry_credit * 0.50:
            close_strategy(signal, "tp50_stop2x", now_utc, "TP50_OBSERVED", close_debit)
        elif close_debit >= entry_credit * 2.0:
            close_strategy(signal, "tp50_stop2x", now_utc, "STOP2X_OBSERVED", close_debit)

    return {
        "mark_time_utc": now_utc,"signal_id": signal["signal_id"],
        "opportunity_key": signal["opportunity_key"],"ticker": ticker,
        "side": signal["side"],"expiration": expiration,
        "short_strike": signal["short_strike"],"long_strike": signal["long_strike"],
        "underlying_price": underlying if underlying is not None else "",
        "close_debit_conservative": close_debit,
        "short_bid": quote["short_bid"],"short_ask": quote["short_ask"],
        "long_bid": quote["long_bid"],"long_ask": quote["long_ask"],
        "unrealized_pnl_net": unreal,"dte": dte,
        "mark_status": "MARKED","mark_error": "",
    }


def main():
    if not SNAPSHOT.exists():
        raise SystemExit("optionsview_alerts_scan.csv not found")

    snapshot = read_csv(SNAPSHOT)
    if not snapshot:
        raise SystemExit("Snapshot is empty")

    now_utc = datetime.now(timezone.utc).isoformat()
    signals = read_csv(SIGNALS)

    open_by_key = {}
    for s in signals:
        if (
            s.get("hold_status") == "OPEN"
            or s.get("tp50_status") == "OPEN"
            or s.get("tp50_stop2x_status") == "OPEN"
        ):
            open_by_key[s.get("opportunity_key", "")] = s

    new_count = 0
    for row in snapshot:
        is_pass = (
            str(row.get("history_status", "")).upper() == "PASS"
            or parse_bool(row.get("executable"))
        )
        if not is_pass:
            continue

        key = opportunity_key(row)
        if not key or key in open_by_key:
            if key in open_by_key:
                open_by_key[key]["last_seen_pass_utc"] = now_utc
            continue

        sig = create_signal(row, now_utc)
        signals.append(sig)
        open_by_key[key] = sig
        new_count += 1

    marks = []
    marked_count = 0
    for s in signals:
        if not (
            s.get("hold_status") == "OPEN"
            or s.get("tp50_status") == "OPEN"
            or s.get("tp50_stop2x_status") == "OPEN"
        ):
            continue
        marks.append(mark_signal(s, now_utc))
        marked_count += 1

    write_csv(SIGNALS, signals, SIGNAL_FIELDS)
    append_csv(MARKS, marks, MARK_FIELDS)

    open_signals = sum(
        1 for s in signals
        if (
            s.get("hold_status") == "OPEN"
            or s.get("tp50_status") == "OPEN"
            or s.get("tp50_stop2x_status") == "OPEN"
        )
    )

    print(
        f"TRACKER CHECK: new_signals={new_count} "
        f"marked={marked_count} total_signals={len(signals)} open={open_signals}"
    )


if __name__ == "__main__":
    main()
