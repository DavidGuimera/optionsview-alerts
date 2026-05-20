import os
import time
import csv
import requests
from datetime import datetime

from optionsview_core_engine import analyze_ticker, DEFAULT_TICKERS, CORE_VERSION

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
TICKERS = [t.strip().upper() for t in os.getenv("TICKERS", DEFAULT_TICKERS).split(",") if t.strip()]
MIN_SCORE = int(os.getenv("MIN_SCORE", "60"))
MAX_ALERTS = int(os.getenv("MAX_ALERTS", "10"))


def send_telegram(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram secrets missing. Message not sent:")
        print(text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    r = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=25)
    if r.status_code >= 300:
        print("Telegram error:", r.status_code, r.text)


def fmt(x, suffix=""):
    if x is None:
        return "N/A"
    try:
        return f"{float(x):g}{suffix}"
    except Exception:
        return str(x)


def alert_message(results):
    today = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"🔥 OptionsView PRO Alerts · {today}",
        f"Core: {CORE_VERSION}",
        f"Universo revisado: {len(TICKERS)} tickers",
        f"Setups ejecutables >= {MIN_SCORE}%: {len(results)}",
        ""
    ]
    for r in results:
        lines += [
            f"🚨 {r.ticker} {r.final_score}% | {r.spread}",
            f"Técnico: {r.technical_score}% | Opciones: {r.options_score}% | Contratos máx: {r.contracts}",
            f"Tipo: {r.signal} | Riesgo: {'Medio' if r.final_score < 75 else 'Bajo/Medio'}",
            f"Crédito: ${fmt(r.credit)} | ROC: {fmt(r.roc, '%')} | Prob OTM: {fmt(r.prob_otm, '%')} | Delta: {fmt(r.delta)}",
            f"IVR/HVR: {fmt(r.iv_rank, '%')} | Liquidez: {r.liquidity} | OI: {fmt(r.oi)} | Bid/Ask: {fmt(r.bid_ask_spread_pct, '%')}",
            f"Exp: {r.expiration} ({r.dte} DTE) | Earnings: {r.earnings_status} ({r.earnings_date})",
            f"Precio: ${fmt(r.price)} | RSI: {fmt(r.rsi)}",
            ""
        ]
    return "\n".join(lines).strip()


def no_alert_message(top_reviewed):
    today = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"✅ OptionsView: sin setups ejecutables >= {MIN_SCORE}%",
        f"Core: {CORE_VERSION}",
        f"Universo revisado: {len(TICKERS)} tickers",
        "",
        "Top revisados:"
    ]
    for r in top_reviewed[:5]:
        score = r.final_score if r.final_score else r.technical_score
        reason = r.reject_reason or "No ejecutable"
        lines.append(f"{r.ticker}: {score}% | {r.signal} | {reason}")
    return "\n".join(lines)


def write_csv(all_results):
    with open("optionsview_alerts_scan.csv", "w", newline="") as f:
        fieldnames = list(all_results[0].to_dict().keys()) if all_results else ["empty"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow(r.to_dict())


def main():
    all_results = []
    for ticker in TICKERS:
        try:
            r = analyze_ticker(ticker, min_score=MIN_SCORE)
            all_results.append(r)
            print(f"{ticker}: final={r.final_score} tech={r.technical_score} opt={r.options_score} exec={r.executable} reason={r.reject_reason}")
        except Exception as e:
            print(f"{ticker}: ERROR {e}")
        time.sleep(0.6)

    if all_results:
        write_csv(all_results)

    executable = [r for r in all_results if r.executable]
    executable.sort(key=lambda x: x.final_score, reverse=True)
    top_reviewed = sorted(all_results, key=lambda x: (x.final_score or x.technical_score), reverse=True)

    if executable:
        send_telegram(alert_message(executable[:MAX_ALERTS]))
    else:
        send_telegram(no_alert_message(top_reviewed))


if __name__ == "__main__":
    main()
