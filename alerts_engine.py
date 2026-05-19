"""
OptionsView PRO - Alerts Engine para GitHub Actions
Escaneo cada hora + anti-duplicados 24h.
"""

import os
import time
import json
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

MIN_SETUP_SCORE = 65
MAX_TICKERS = 230
DEFAULT_PUT_RSI = 32
DEFAULT_CALL_RSI = 68
SUPPORT_RESISTANCE_WINDOW = 60

ALERT_MEMORY_FILE = "alert_memory.json"
ALERT_COOLDOWN_HOURS = 24


def send_telegram(message):
    if not BOT_TOKEN or not CHAT_ID:
        print("Faltan variables Telegram")
        return False

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message}

    try:
        r = requests.post(url, data=payload, timeout=10)
        print(r.text)
        return r.status_code == 200
    except Exception as e:
        print("Error Telegram:", e)
        return False


def load_alert_memory():
    if not os.path.exists(ALERT_MEMORY_FILE):
        return {}
    try:
        with open(ALERT_MEMORY_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def save_alert_memory(memory):
    try:
        with open(ALERT_MEMORY_FILE, "w") as f:
            json.dump(memory, f)
    except Exception as e:
        print("Error guardando memoria:", e)


def recently_alerted(ticker, memory):
    if ticker not in memory:
        return False

    try:
        last_time = datetime.fromisoformat(memory[ticker])
        return datetime.utcnow() - last_time < timedelta(hours=ALERT_COOLDOWN_HOURS)
    except Exception:
        return False


def get_universe():
    tickers = [
        "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","TSLA",
        "MCD","PEP","PG","KO","COST","WMT","HD","LOW","JNJ",
        "ABBV","MRK","UNH","XOM","CVX","PM","MDLZ","TROW",
        "JPM","BAC","GS","MS","V","MA",
        "AMD","AVGO","QCOM","CSCO","ADBE","CRM","INTU","PANW",
        "AMAT","LRCX","KLAC","MU","ADI",
        "SBUX","NKE","CMCSA","DIS","NFLX",
        "CAT","HON","GE","RTX","DE",
        "LLY","ISRG","TMO","ABT","GILD","VRTX"
    ]
    return tickers[:MAX_TICKERS]


def safe_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_stochastic(data, k_period=14):
    low_min = data["Low"].rolling(k_period).min()
    high_max = data["High"].rolling(k_period).max()
    return 100 * ((data["Close"] - low_min) / (high_max - low_min))


def calculate_iv_rank_proxy(data):
    try:
        returns = data["Close"].pct_change().dropna()
        hv20 = returns.rolling(20).std() * np.sqrt(252) * 100
        hv20 = hv20.dropna()

        if hv20.empty:
            return np.nan

        current = safe_float(hv20.iloc[-1])
        low = safe_float(hv20.min())
        high = safe_float(hv20.max())

        if high == low:
            return np.nan

        return round(((current - low) / (high - low)) * 100, 1)
    except Exception:
        return np.nan


def trend_from_data(data):
    try:
        if data.empty or len(data) < 60:
            return "Neutral"

        close = data["Close"].dropna()
        ema21 = close.ewm(span=21, adjust=False).mean().iloc[-1]
        sma50 = close.rolling(50).mean().iloc[-1]
        price = close.iloc[-1]

        if price > ema21 > sma50:
            return "Alcista"
        if price < ema21 < sma50:
            return "Bajista"
        return "Neutral"
    except Exception:
        return "Neutral"


def download_data(ticker, period="1y", interval="1d"):
    try:
        data = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False
        )

        if data is None or data.empty:
            return pd.DataFrame()

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        return data.dropna()
    except Exception as e:
        print(f"Error descargando {ticker}: {e}")
        return pd.DataFrame()


def score_setup(signal, price, rsi_val, stoch_val, dist_support, dist_resistance,
                sma200, trend_1d, trend_4h, trend_1h, iv_rank):

    if signal == "NO TRADE":
        return 0

    score = 45

    if signal == "PUT / Bull Put Spread":
        if not pd.isna(sma200):
            score += 18 if price > sma200 else -22

        if trend_1d == "Alcista":
            score += 12
        if trend_4h == "Alcista":
            score += 10
        if trend_1h == "Alcista":
            score += 4

        if dist_support <= 1.5:
            score += 16
        elif dist_support <= 3:
            score += 10
        elif dist_support > 8:
            score -= 10

        if rsi_val <= 30:
            score += 6
        elif rsi_val >= 55:
            score -= 6

        if stoch_val <= 20:
            score += 3
        elif stoch_val >= 80:
            score -= 3

    elif signal == "CALL Spread":
        if not pd.isna(sma200):
            score += 18 if price < sma200 else -12

        if trend_1d == "Bajista":
            score += 12
        if trend_4h == "Bajista":
            score += 10
        if trend_1h == "Bajista":
            score += 4

        if dist_resistance <= 1.5:
            score += 16
        elif dist_resistance <= 4:
            score += 10
        elif dist_resistance > 8:
            score -= 10

        if rsi_val >= 70:
            score += 6
        elif rsi_val <= 45:
            score -= 6

        if stoch_val >= 80:
            score += 3
        elif stoch_val <= 20:
            score -= 3

    if not pd.isna(iv_rank):
        if iv_rank >= 60:
            score += 15
        elif iv_rank >= 40:
            score += 8
        elif iv_rank < 20:
            score -= 15

    return int(max(0, min(100, round(score))))


def analyze_ticker(ticker):
    data = download_data(ticker)

    if data.empty or len(data) < 200:
        return None

    data["RSI"] = calculate_rsi(data["Close"])
    data["STOCH"] = calculate_stochastic(data)
    data["SMA200"] = data["Close"].rolling(200).mean()

    latest = data.iloc[-1]

    price = safe_float(latest["Close"])
    rsi_val = safe_float(latest["RSI"])
    stoch_val = safe_float(latest["STOCH"])
    sma200 = safe_float(latest["SMA200"])

    support = safe_float(data.tail(SUPPORT_RESISTANCE_WINDOW)["Low"].min())
    resistance = safe_float(data.tail(SUPPORT_RESISTANCE_WINDOW)["High"].max())

    dist_support = ((price - support) / support) * 100 if support else np.nan
    dist_resistance = ((resistance - price) / price) * 100 if resistance else np.nan

    data_4h = download_data(ticker, "6mo", "4h")
    data_1h = download_data(ticker, "2mo", "1h")

    trend_1d = trend_from_data(data)
    trend_4h = trend_from_data(data_4h)
    trend_1h = trend_from_data(data_1h)

    iv_rank = calculate_iv_rank_proxy(data)

    signal = "NO TRADE"

    if not pd.isna(rsi_val) and not pd.isna(dist_support):
        if rsi_val <= DEFAULT_PUT_RSI and dist_support <= 3:
            signal = "PUT / Bull Put Spread"

    if signal == "NO TRADE" and not pd.isna(rsi_val) and not pd.isna(dist_resistance):
        if rsi_val >= DEFAULT_CALL_RSI and dist_resistance <= 4:
            signal = "CALL Spread"

    score = score_setup(
        signal,
        price,
        rsi_val,
        stoch_val,
        dist_support,
        dist_resistance,
        sma200,
        trend_1d,
        trend_4h,
        trend_1h,
        iv_rank
    )

    if score < MIN_SETUP_SCORE:
        return None

    return {
        "ticker": ticker,
        "signal": signal,
        "score": score,
        "price": round(price, 2),
        "rsi": round(rsi_val, 1),
        "iv_rank": iv_rank,
        "trend_1d": trend_1d,
        "trend_4h": trend_4h,
        "trend_1h": trend_1h
    }


def alert_message(setup):
    tipo = "🔴 PREMIUM" if setup["score"] >= 70 else "🟡 WATCHLIST"

    return (
        f"{tipo} OptionsView PRO\n\n"
        f"{setup['ticker']}\n"
        f"{setup['signal']}\n\n"
        f"⭐ Setup Score: {setup['score']}%\n"
        f"💲 Precio: {setup['price']}\n"
        f"📉 RSI: {setup['rsi']}\n"
        f"🔥 IV Rank: {setup['iv_rank']}%\n\n"
        f"📈 1D: {setup['trend_1d']}\n"
        f"📈 4H: {setup['trend_4h']}\n"
        f"📈 1H: {setup['trend_1h']}\n\n"
        f"⚠️ Revisar cadena real antes de entrar"
    )


def scan_once():
    tickers = get_universe()
    memory = load_alert_memory()

    print(f"Escaneando {len(tickers)} tickers...")

    setups = []

    for i, ticker in enumerate(tickers, 1):
        try:
            setup = analyze_ticker(ticker)

            if setup:
                if recently_alerted(setup["ticker"], memory):
                    print(f"Duplicado evitado: {setup['ticker']}")
                    continue

                setups.append(setup)
                send_telegram(alert_message(setup))

                memory[setup["ticker"]] = datetime.utcnow().isoformat()
                save_alert_memory(memory)

                print(f"ALERTA: {setup['ticker']} {setup['score']}%")

            if i % 10 == 0:
                print(f"Procesados {i}/{len(tickers)}")

            time.sleep(0.2)

        except Exception as e:
            print(f"Error {ticker}: {e}")

    print(f"Escaneo terminado. Setups encontrados: {len(setups)}")
    return setups


if __name__ == "__main__":
    setups = scan_once()

    send_telegram(
        f"✅ Escaneo finalizado. "
        f"Setups nuevos encontrados: {len(setups)}"
    )
