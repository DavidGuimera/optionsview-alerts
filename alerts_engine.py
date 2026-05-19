"""
OptionsView PRO - Alerts Engine para GitHub Actions
Escanea tickers TOP y envía alertas Telegram automáticamente.

Preparado para:
✅ GitHub Actions
✅ Telegram
✅ Sin Mac encendido
✅ Gratis

Secrets necesarios en GitHub:
TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID
"""

import os
import time
import requests
import numpy as np
import pandas as pd
import yfinance as yf

# =====================================================
# CONFIGURACIÓN
# =====================================================

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

MIN_SETUP_SCORE = 65
MAX_TICKERS = 230

DEFAULT_PUT_RSI = 32
DEFAULT_CALL_RSI = 68

SUPPORT_RESISTANCE_WINDOW = 60

# =====================================================
# TELEGRAM
# =====================================================

def send_telegram(message):

    if not BOT_TOKEN or not CHAT_ID:
        print("Faltan variables Telegram")
        return False

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    payload = {
        "chat_id": CHAT_ID,
        "text": message
    }

    try:
        r = requests.post(url, data=payload, timeout=10)
        print(r.text)
        return r.status_code == 200

    except Exception as e:
        print("Error Telegram:", e)
        return False

# =====================================================
# UNIVERSO DE TICKERS
# =====================================================

def get_universe():

    tickers = [

        # Mega caps
        "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","TSLA",

        # Dividendos / calidad
        "MCD","PEP","PG","KO","COST","WMT","HD","LOW","JNJ",
        "ABBV","MRK","UNH","XOM","CVX","PM","MDLZ","TROW",

        # Financieras
        "JPM","BAC","GS","MS","V","MA",

        # Tecnología
        "AMD","AVGO","QCOM","CSCO","ADBE","CRM","INTU","PANW",
        "AMAT","LRCX","KLAC","MU","ADI",

        # Consumo
        "SBUX","NKE","CMCSA","DIS","NFLX",

        # Industriales
        "CAT","HON","GE","RTX","DE",

        # Healthcare
        "LLY","ISRG","TMO","ABT","GILD","VRTX"

    ]

    return tickers[:MAX_TICKERS]

# =====================================================
# FUNCIONES AUXILIARES
# =====================================================

def safe_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan

# =====================================================
# RSI
# =====================================================

def calculate_rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1/period,
        min_periods=period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1/period,
        min_periods=period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    return rsi

# =====================================================
# ESTOCÁSTICO
# =====================================================

def calculate_stochastic(data, k_period=14):

    low_min = data["Low"].rolling(k_period).min()

    high_max = data["High"].rolling(k_period).max()

    stoch = 100 * (
        (data["Close"] - low_min) /
        (high_max - low_min)
    )

    return stoch

# =====================================================
# IV RANK ESTIMADO
# =====================================================

def calculate_iv_rank_proxy(data):

    try:

        returns = data["Close"].pct_change().dropna()

        hv20 = (
            returns
            .rolling(20)
            .std()
            * np.sqrt(252)
            * 100
        )

        hv20 = hv20.dropna()

        if hv20.empty:
            return np.nan

        current = safe_float(hv20.iloc[-1])

        low = safe_float(hv20.min())
        high = safe_float(hv20.max())

        if high == low:
            return np.nan

        iv_rank = ((current - low) / (high - low)) * 100

        return round(iv_rank, 1)

    except Exception:
        return np.nan

# =====================================================
# TENDENCIA
# =====================================================

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

        elif price < ema21 < sma50:
            return "Bajista"

        else:
            return "Neutral"

    except Exception:
        return "Neutral"

# =====================================================
# DESCARGA DATA
# =====================================================

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

# =====================================================
# SCORE
# =====================================================

def score_setup(
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
):

    if signal == "NO TRADE":
        return 0

    score = 45

    # =========================================
    # PUT SPREADS
    # =========================================

    if signal == "PUT / Bull Put Spread":

        if not pd.isna(sma200):

            if price > sma200:
                score += 18
            else:
                score -= 22

        # Multi timeframe

        if trend_1d == "Alcista":
            score += 12

        if trend_4h == "Alcista":
            score += 10

        if trend_1h == "Alcista":
            score += 4

        # Soporte

        if dist_support <= 1.5:
            score += 16

        elif dist_support <= 3:
            score += 10

        elif dist_support > 8:
            score -= 10

        # RSI menos peso

        if rsi_val <= 30:
            score += 6

        elif rsi_val >= 55:
            score -= 6

        # Estocástico MUCHO menos peso

        if stoch_val <= 20:
            score += 3

        elif stoch_val >= 80:
            score -= 3

    # =========================================
    # CALL SPREADS
    # =========================================

    elif signal == "CALL Spread":

        if not pd.isna(sma200):

            if price < sma200:
                score += 18
            else:
                score -= 12

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

    # =========================================
    # IV RANK
    # =========================================

    if not pd.isna(iv_rank):

        if iv_rank >= 60:
            score += 15

        elif iv_rank >= 40:
            score += 8

        elif iv_rank < 20:
            score -= 15

    return int(max(0, min(100, round(score))))

# =====================================================
# ANALYZE TICKER
# =====================================================

def analyze_ticker(ticker):

    data = download_data(ticker)

    if data.empty or len(data) < 200:
        return None

    data["RSI"] = calculate_rsi(data["Close"])
    data["STOCH"] = calculate_stochastic(data)

    data["SMA20"] = data["Close"].rolling(20).mean()
    data["SMA50"] = data["Close"].rolling(50).mean()
    data["SMA200"] = data["Close"].rolling(200).mean()

    latest = data.iloc[-1]

    price = safe_float(latest["Close"])

    rsi_val = safe_float(latest["RSI"])

    stoch_val = safe_float(latest["STOCH"])

    sma200 = safe_float(latest["SMA200"])

    support = safe_float(
        data.tail(SUPPORT_RESISTANCE_WINDOW)["Low"].min()
    )

    resistance = safe_float(
        data.tail(SUPPORT_RESISTANCE_WINDOW)["High"].max()
    )

    dist_support = (
        ((price - support) / support) * 100
        if support else np.nan
    )

    dist_resistance = (
        ((resistance - price) / price) * 100
        if resistance else np.nan
    )

    # =========================================
    # MULTI TIMEFRAME
    # =========================================

    data_4h = download_data(ticker, "6mo", "4h")

    data_1h = download_data(ticker, "2mo", "1h")

    trend_1d = trend_from_data(data)

    trend_4h = trend_from_data(data_4h)

    trend_1h = trend_from_data(data_1h)

    iv_rank = calculate_iv_rank_proxy(data)

    signal = "NO TRADE"

    # PUTS

    if (
        not pd.isna(rsi_val)
        and not pd.isna(dist_support)
    ):

        if (
            rsi_val <= DEFAULT_PUT_RSI
            and dist_support <= 3
        ):

            signal = "PUT / Bull Put Spread"

    # CALLS

    if (
        signal == "NO TRADE"
        and not pd.isna(rsi_val)
        and not pd.isna(dist_resistance)
    ):

        if (
            rsi_val >= DEFAULT_CALL_RSI
            and dist_resistance <= 4
        ):

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

# =====================================================
# MENSAJE ALERTA
# =====================================================

def alert_message(setup):

    return (
        f"🚨 OptionsView PRO ALERT\n\n"
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

# =====================================================
# ESCANEO
# =====================================================

def scan_once():

    tickers = get_universe()

    print(f"Escaneando {len(tickers)} tickers...")

    setups = []

    for i, ticker in enumerate(tickers, 1):

        try:

            setup = analyze_ticker(ticker)

            if setup:

                setups.append(setup)

                send_telegram(
                    alert_message(setup)
                )

                print(
                    f"ALERTA: {setup['ticker']} "
                    f"{setup['score']}%"
                )

            if i % 10 == 0:

                print(
                    f"Procesados {i}/{len(tickers)}"
                )

            time.sleep(0.2)

        except Exception as e:

            print(f"Error {ticker}: {e}")

    print(
        f"Escaneo terminado. "
        f"Setups encontrados: {len(setups)}"
    )

    return setups

# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    send_telegram(
        "✅ OptionsView PRO: escaneo GitHub iniciado"
    )

    setups = scan_once()

    send_telegram(
        f"✅ Escaneo finalizado. "
        f"Setups encontrados: {len(setups)}"
    )
