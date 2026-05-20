"""
OptionsView Alerts - Unified Engine
Usa el mismo motor que la APP: optionsview_core_engine.py
"""

from __future__ import annotations

import os
import time
import requests
import pandas as pd

from optionsview_core_engine import (
    FULL_TICKERS,
    analyze_ticker,
    format_telegram_alert,
    parse_tickers,
)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
TICKERS_TEXT = os.getenv("TICKERS", FULL_TICKERS)
MIN_SCORE = int(os.getenv("MIN_SCORE", "60"))
MAX_ALERTS = int(os.getenv("MAX_ALERTS", "10"))
PERIOD = os.getenv("PERIOD", "1y")
DELAY_SECONDS = float(os.getenv("DELAY_SECONDS", "0.7"))


def send_telegram(message: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram secrets not configured. Message preview:")
        print(message)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    r = requests.post(url, data=payload, timeout=25)
    r.raise_for_status()


def main() -> None:
    tickers = parse_tickers(TICKERS_TEXT)
    rows = []

    print(f"Scanning {len(tickers)} tickers with unified engine...")
    for ticker in tickers:
        row = analyze_ticker(ticker, period=PERIOD, deep_options=True)
        rows.append(row)
        print(
            f"{ticker}: Final={row.get('Final Score %')} | "
            f"Tech={row.get('Technical Score %')} | "
            f"Options={row.get('Options Score %')} | "
            f"Signal={row.get('Señal')} | Data={row.get('Data Status')} | OptionsStatus={row.get('Options Status')}"
        )
        time.sleep(DELAY_SECONDS)

    df = pd.DataFrame(rows)
    if df.empty:
        send_telegram("⚠️ OptionsView: no se pudieron analizar tickers.")
        return

    # Evita alertas si datos u opciones son dudosos. Si quieres alertas solo técnicas, cambia esta condición.
    alerts = df[
        (df["Final Score %"].fillna(0) >= MIN_SCORE)
        & (df["Data Status"].astype(str) == "OK")
        & (df["Señal"].astype(str) != "NO TRADE")
        & (df["Earnings Status"].astype(str) != "DANGER")
    ].copy()

    alerts = alerts.sort_values(["Final Score %", "Technical Score %"], ascending=False).head(MAX_ALERTS)

    if alerts.empty:
        top = df.sort_values("Final Score %", ascending=False).head(5)
        summary = "\n".join(
            f"{r['Ticker']}: {r.get('Final Score %', 0)}% | {r.get('Señal', '')}"
            for _, r in top.iterrows()
        )
        send_telegram(f"✅ OptionsView: sin setups >= {MIN_SCORE}%\n\nTop revisados:\n{summary}")
        return

    for _, row in alerts.iterrows():
        send_telegram(format_telegram_alert(row.to_dict()))


if __name__ == "__main__":
    main()
