import os
import time
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from ta.momentum import RSIIndicator

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

TICKERS = os.getenv(
    "TICKERS",
    "MCD,PEP,PG,COST,SBUX,MSFT,AAPL,GOOGL,JPM,V,SPY,QQQ"
).split(",")

MIN_SCORE = int(os.getenv("MIN_SCORE", "60"))
MAX_ALERTS = int(os.getenv("MAX_ALERTS", "10"))

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }

    requests.post(url, data=payload, timeout=20)

def download_data(ticker, period="6mo"):
    ticker = ticker.strip().upper()

    try:
        tk = yf.Ticker(ticker)

        data = tk.history(
            period=period,
            interval="1d",
            auto_adjust=True
        )

        if data is None or data.empty:
            return pd.DataFrame()

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [
                c[0] if isinstance(c, tuple) else c
                for c in data.columns
            ]

        data = data.loc[:, ~data.columns.duplicated()]

        needed = ["Open", "High", "Low", "Close"]

        if not all(col in data.columns for col in needed):
            return pd.DataFrame()

        data = data.dropna(subset=needed)

        return data

    except Exception:
        return pd.DataFrame()

def calculate_score(data):
    try:
        close = data["Close"]

        rsi = RSIIndicator(close, window=14).rsi().iloc[-1]

        sma20 = close.rolling(20).mean().iloc[-1]
        sma50 = close.rolling(50).mean().iloc[-1]

        price = close.iloc[-1]

        score = 50

        if rsi < 35:
            score += 12

        if price > sma20:
            score += 10

        if sma20 > sma50:
            score += 10

        momentum = ((price / close.iloc[-20]) - 1) * 100

        if momentum > 0:
            score += 8

        volatility = close.pct_change().std() * 100

        if volatility < 2.5:
            score += 8

        return min(round(score), 100)

    except Exception:
        return 0

def build_alert(ticker, data, score):
    try:
        price = round(data["Close"].iloc[-1], 2)

        if score >= 75:
            contracts = "2 contratos"
        elif score >= 65:
            contracts = "1 contrato"
        else:
            contracts = "NO TRADE"

        return (
            f"🔥 <b>{ticker}</b>\n"
            f"Precio: ${price}\n"
            f"Score: {score}%\n"
            f"Sizing: {contracts}"
        )

    except Exception:
        return None

results = []

for ticker in TICKERS:

    data = download_data(ticker)

    if data.empty:
        continue

    score = calculate_score(data)

    if score >= MIN_SCORE:
        results.append((ticker, score, data))

    time.sleep(0.4)

results = sorted(results, key=lambda x: x[1], reverse=True)

for ticker, score, data in results[:MAX_ALERTS]:

    msg = build_alert(ticker, data, score)

    if msg:
        send_telegram(msg)
