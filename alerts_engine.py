import os
import time
import csv
import requests
from datetime import datetime, timezone

from optionsview_core_engine import analyze_ticker, DEFAULT_TICKERS, CORE_VERSION

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
TICKERS = [t.strip().upper() for t in os.getenv("TICKERS", DEFAULT_TICKERS).split(",") if t.strip()]

# Now this threshold is estimated WIN probability, not setup score.
MIN_WIN_PROB = float(os.getenv("MIN_WIN_PROB", "68"))
MAX_ALERTS = int(os.getenv("MAX_ALERTS", "6"))
MAX_RISK_PER_TRADE = float(os.getenv("MAX_RISK_PER_TRADE", "500"))

# Prevent a Telegram burst of highly correlated trades.
MAX_PER_BUCKET = int(os.getenv("MAX_PER_BUCKET", "2"))

BUCKETS = {
    "MEGA_TECH": {"MSFT","AAPL","GOOGL","META","AMZN","NVDA","AVGO","ADBE","CRM","QQQ"},
    "CONSUMER": {"MCD","PEP","PG","KO","WMT","COST","HD","LOW","TGT","SBUX","MDLZ"},
    "FINANCIALS": {"JPM","MA","V","BLK","SCHW","XLF"},
    "INDEX": {"SPY","IWM"},
    "HEALTH": {"JNJ","XLV"},
    "COMM": {"CMCSA"},
    "STAPLES_ETF": {"XLP"},
}

def bucket_for(ticker):
    for name, members in BUCKETS.items():
        if ticker in members:
            return name
    return "OTHER"


def signal_id(r):
    """Stable human-readable ID for this scan signal."""
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    s1 = str(r.short_strike).replace(".0", "").replace(".", "p")
    s2 = str(r.long_strike).replace(".0", "").replace(".", "p")
    side = "P" if r.signal == "PUT" else "C"
    return f"{r.ticker}-{date}-{s1}{side}{s2}{side}"


def send_telegram(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram secrets missing. Message not sent:")
        print(text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    r = requests.post(url, data={"chat_id":TELEGRAM_CHAT_ID,"text":text}, timeout=25)
    if r.status_code >= 300:
        print("Telegram error:",r.status_code,r.text)


def fmt(x,suffix=""):
    if x is None:
        return "N/A"
    try:
        return f"{float(x):g}{suffix}"
    except Exception:
        return str(x)


def diversify(results):
    chosen=[]
    counts={}
    # sorted before this function
    for r in results:
        b=bucket_for(r.ticker)
        key=(b,r.signal)
        if counts.get(key,0) >= MAX_PER_BUCKET:
            continue
        chosen.append(r)
        counts[key]=counts.get(key,0)+1
        if len(chosen)>=MAX_ALERTS:
            break
    return chosen


def alert_message(results):
    now=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines=[
        f"🔥 OptionsView ALGO v1.3 · {now}",
        f"Core: {CORE_VERSION}",
        f"Universo: {len(TICKERS)} | Umbral prob. éxito: {MIN_WIN_PROB:g}%",
        ""
    ]
    for r in results:
        sid = signal_id(r)
        adj = r.probability_adjustment if r.probability_adjustment is not None else 0
        adj_txt = f"{adj:+.1f} pp"
        lines += [
            f"🚨 {r.ticker} | Prob. éxito estimada: {fmt(r.win_probability,'%')} | PASS ✅",
            f"ID: {sid}",
            f"{r.spread} · {r.dte} DTE · Máx {r.contracts} contrato(s)",
            f"Precio: ${fmt(r.price)} | RSI: {fmt(r.rsi)}",
            f"Prob OTM base: {fmt(r.prob_otm,'%')} | Ajuste ALGO: {adj_txt}",
            f"Short delta: {fmt(abs(r.delta))}",
            f"Crédito detectado: ${fmt(r.credit)} | NO ENTRAR < ${fmt(r.min_entry_credit)}",
            f"ROC neto: {fmt(r.net_roc,'%')} | Riesgo neto: ${fmt(r.net_max_loss)}",
            f"EM: ±${fmt(r.expected_move)} | Short: {fmt(r.em_distance)}× EM",
            f"HVR/IVR proxy: {fmt(r.iv_rank,'%')} | IV short: {fmt(r.short_iv,'%')}",
            f"OI: {fmt(r.oi)} | Bid/Ask: {fmt(r.bid_ask_spread_pct,'%')} | Liquidez: {r.liquidity}",
            f"Earnings: {r.earnings_status} ({r.earnings_date})",
            f"Calidad opciones: {fmt(r.options_quality_score,'/100')} | Técnico: {fmt(r.technical_score,'/100')}",
            f"Delta source: {r.delta_source}",
            ""
        ]
    lines += [
        "⚠️ Probabilidad ESTIMADA, todavía no calibrada. Ejecuta solo si IBKR confirma strikes/vencimiento, delta razonable y crédito >= mínimo indicado."
    ]
    return "\n".join(lines).strip()


def no_alert_message(top_reviewed):
    now=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines=[
        f"✅ OptionsView ALGO v1.3: sin setups ejecutables >= {MIN_WIN_PROB:g}%",
        f"Core: {CORE_VERSION}",
        f"Universo revisado: {len(TICKERS)}",
        "",
        "Top revisados:"
    ]
    for r in top_reviewed[:5]:
        p = fmt(r.win_probability,"%") if r.win_probability is not None else "N/A"
        lines.append(f"{r.ticker}: P(win) {p} | {r.signal} | {r.reject_reason or 'No ejecutable'}")
    return "\n".join(lines)


def write_csv(scan_rows):
    """Write exactly one row per ticker attempted in this scan."""
    if not scan_rows:
        return

    # Union of all keys so ERROR rows and normal rows fit in one CSV.
    fieldnames = []
    seen = set()
    for row in scan_rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with open("optionsview_alerts_scan.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in scan_rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})

def main():
    all_results = []
    scan_rows = []
    scan_time = datetime.now(timezone.utc).isoformat()

    for ticker in TICKERS:
        ticker = ticker.strip().upper()
        try:
            r = analyze_ticker(
                ticker,
                min_score=MIN_WIN_PROB,
                max_risk=MAX_RISK_PER_TRADE
            )
            all_results.append(r)

            row = r.to_dict()
            row["scan_time_utc"] = scan_time
            row["signal_id"] = (
                signal_id(r)
                if r.short_strike is not None and r.long_strike is not None
                else ""
            )

            # Explicit historical classification.
            if r.executable:
                history_status = "PASS"
            elif r.data_status == "ERROR":
                history_status = "ERROR"
            elif r.signal == "NO TRADE":
                history_status = "NO_SETUP"
            elif "chain" in (r.reject_reason or "").lower() or "cadena" in (r.reject_reason or "").lower():
                history_status = "NO_VALID_CHAIN"
            else:
                history_status = "REJECT"

            row["history_status"] = history_status
            row["scan_error"] = ""
            scan_rows.append(row)

            print(
                f"{ticker}: status={history_status} "
                f"Pwin={r.win_probability} tech={r.technical_score} "
                f"optQ={r.options_quality_score} exec={r.executable} "
                f"reason={r.reject_reason}"
            )

        except Exception as e:
            # CRITICAL: an exception must still create one historical row.
            error_row = {
                "ticker": ticker,
                "core_version": CORE_VERSION,
                "data_status": "ERROR",
                "price": "",
                "rsi": "",
                "signal": "ERROR",
                "technical_score": "",
                "options_quality_score": "",
                "win_probability": "",
                "probability_adjustment": "",
                "min_entry_credit": "",
                "contracts": 0,
                "spread": "",
                "short_strike": "",
                "long_strike": "",
                "expiration": "",
                "dte": "",
                "earnings_date": "",
                "earnings_days": "",
                "earnings_status": "UNKNOWN",
                "credit": "",
                "commission_rt": "",
                "max_loss": "",
                "net_max_loss": "",
                "roc": "",
                "net_roc": "",
                "prob_otm": "",
                "delta": "",
                "delta_source": "",
                "iv_rank": "",
                "short_iv": "",
                "expected_move": "",
                "em_distance": "",
                "liquidity": "",
                "oi": "",
                "bid_ask_spread_pct": "",
                "reject_reason": "Exception during analyze_ticker",
                "executable": False,
                "final_score": "",
                "options_score": "",
                "scan_time_utc": scan_time,
                "signal_id": "",
                "history_status": "ERROR",
                "scan_error": str(e),
            }
            scan_rows.append(error_row)
            print(f"{ticker}: ERROR {e}")

        time.sleep(0.6)

    # Historical invariant: 1 row per requested ticker.
    print(f"HISTORY CHECK: requested={len(TICKERS)} rows={len(scan_rows)}")
    if len(scan_rows) != len(TICKERS):
        print("WARNING: historical row count does not match ticker count")

    write_csv(scan_rows)

    executable = [r for r in all_results if r.executable]
    executable.sort(
        key=lambda x: (
            x.win_probability or 0,
            x.options_quality_score or 0,
            x.net_roc or 0
        ),
        reverse=True
    )
    selected = diversify(executable)

    top_reviewed = sorted(
        all_results,
        key=lambda x: (
            x.win_probability or 0,
            x.technical_score or 0
        ),
        reverse=True
    )

    if selected:
        send_telegram(alert_message(selected))
    else:
        send_telegram(no_alert_message(top_reviewed))


if __name__ == "__main__":
    main()
