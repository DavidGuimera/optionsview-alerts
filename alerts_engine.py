import os
import time
import requests
import numpy as np
import pandas as pd
import yfinance as yf

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

MIN_SETUP_SCORE = 65
SCAN_INTERVAL_MINUTES = 5
MAX_TICKERS = 230

DEFAULT_PUT_RSI = 32
DEFAULT_CALL_RSI = 68

def send_telegram(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message}
    try:
        r = requests.post(url, data=payload, timeout=10)
        print(r.text)
        return r.status_code == 200
    except Exception as e:
        print("Error Telegram:", e)
        return False

def safe_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan

def get_sp500():
    try:
        df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        return df["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
    except Exception:
        return ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","COST","PEP","PG","MCD","V","MA","KO","HD","LOW","WMT","JNJ","JPM"]

def get_nasdaq100():
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
        for table in tables:
            for col in table.columns:
                if "ticker" in str(col).lower() or "symbol" in str(col).lower():
                    tickers = table[col].astype(str).str.replace(".", "-", regex=False).tolist()
                    tickers = [t for t in tickers if t and t.lower() != "nan"]
                    if len(tickers) > 20:
                        return tickers
    except Exception:
        pass
    return ["AAPL","MSFT","NVDA","AMZN","META","TSLA","COST","NFLX","PEP","ADBE","AMD","AVGO","QCOM","SBUX"]

def get_universe():
    tickers = get_sp500() + get_nasdaq100()
    clean = []
    seen = set()
    for t in tickers:
        t = t.upper().strip()
        if t and t not in seen:
            clean.append(t)
            seen.add(t)
    return clean[:MAX_TICKERS]

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def stochastic(data, k_period=14):
    low_min = data["Low"].rolling(k_period).min()
    high_max = data["High"].rolling(k_period).max()
    return 100 * ((data["Close"] - low_min) / (high_max - low_min))

def iv_rank_proxy(data):
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
        close = data["Close"].dropna()
        if len(close) < 60:
            return "Neutral"
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

def download(ticker, period="1y", interval="1d"):
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True, threads=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data.dropna(subset=["Open","High","Low","Close"])
    except Exception:
        return pd.DataFrame()

def score_setup(signal, price, rsi_val, stoch_val, dist_support, dist_resistance, sma200, trend_1d, trend_4h, trend_1h, iv_rank):
    if signal == "NO TRADE":
        return 0

    score = 45

    if signal == "PUT / Bull Put Spread":
        score += 18 if price > sma200 else -22
        score += 12 if trend_1d in ["Alcista", "Alcista fuerte"] else -15 if trend_1d in ["Bajista", "Bajista fuerte"] else 0
        score += 10 if trend_4h == "Alcista" else -10 if trend_4h == "Bajista" else 0
        score += 4 if trend_1h == "Alcista" else -4 if trend_1h == "Bajista" else 0
        score += 16 if dist_support <= 1.5 else 10 if dist_support <= 3 else -10 if dist_support > 8 else 0
        score += 6 if rsi_val <= 30 else -6 if rsi_val >= 55 else 0
        score += 3 if stoch_val <= 20 else -3 if stoch_val >= 80 else 0

    if signal == "CALL Spread":
        score += 18 if price < sma200 else -12
        score += 12 if trend_1d in ["Bajista", "Bajista fuerte"] else -12 if trend_1d in ["Alcista", "Alcista fuerte"] else 0
        score += 10 if trend_4h == "Bajista" else -10 if trend_4h == "Alcista" else 0
        score += 4 if trend_1h == "Bajista" else -4 if trend_1h == "Alcista" else 0
        score += 16 if dist_resistance <= 1.5 else 10 if dist_resistance <= 4 else -10 if dist_resistance > 8 else 0
        score += 6 if rsi_val >= 70 else -6 if rsi_val <= 45 else 0
        score += 3 if stoch_val >= 80 else -3 if stoch_val <= 20 else 0

    if not pd.isna(iv_rank):
        score += 15 if iv_rank >= 60 else 8 if iv_rank >= 40 else -15 if iv_rank < 20 else 0

    return int(max(0, min(100, round(score))))

def analyze_ticker(ticker):
    data = download(ticker, "1y", "1d")
    if data.empty or len(data) < 200:
        return None

    data["RSI"] = rsi(data["Close"])
    data["STOCH"] = stochastic(data)
    data["SMA20"] = data["Close"].rolling(20).mean()
    data["SMA50"] = data["Close"].rolling(50).mean()
    data["SMA200"] = data["Close"].rolling(200).mean()

    latest = data.iloc[-1]
    price = safe_float(latest["Close"])
    rsi_val = safe_float(latest["RSI"])
    stoch_val = safe_float(latest["STOCH"])
    sma20 = safe_float(latest["SMA20"])
    sma50 = safe_float(latest["SMA50"])
    sma200 = safe_float(latest["SMA200"])

    support = safe_float(data.tail(60)["Low"].min())
    resistance = safe_float(data.tail(60)["High"].max())

    dist_support = ((price - support) / support) * 100 if support else np.nan
    dist_resistance = ((resistance - price) / price) * 100 if price else np.nan

    if price > sma20 > sma50 > sma200:
        trend_1d = "Alcista fuerte"
    elif price > sma50 and price > sma200:
        trend_1d = "Alcista"
    elif price < sma20 < sma50 < sma200:
        trend_1d = "Bajista fuerte"
    elif price < sma50 and price < sma200:
        trend_1d = "Bajista"
    else:
        trend_1d = "Neutral"

    data_4h = download(ticker, "6mo", "4h")
    data_1h = download(ticker, "2mo", "1h")
    trend_4h = trend_from_data(data_4h)
    trend_1h = trend_from_data(data_1h)

    iv_rank = iv_rank_proxy(data)

    signal = "NO TRADE"
    if rsi_val <= DEFAULT_PUT_RSI and dist_support <= 3:
        signal = "PUT / Bull Put Spread"
    elif rsi_val >= DEFAULT_CALL_RSI and dist_resistance <= 4:
        signal = "CALL Spread"

    score = score_setup(signal, price, rsi_val, stoch_val, dist_support, dist_resistance, sma200, trend_1d, trend_4h, trend_1h, iv_rank)

    if score < MIN_SETUP_SCORE:
        return None

    return {
        "ticker": ticker,
        "price": round(price, 2),
        "signal": signal,
        "score": score,
        "rsi": round(rsi_val, 1),
        "stoch": round(stoch_val, 1),
        "iv_rank": iv_rank,
        "trend_1d": trend_1d,
        "trend_4h": trend_4h,
        "trend_1h": trend_1h,
        "support": round(support, 2),
        "resistance": round(resistance, 2),
        "dist_support": round(dist_support, 2),
        "dist_resistance": round(dist_resistance, 2)
    }

sent = set()

def alert_message(s):
    tipo = "🔴 PREMIUM" if s["score"] >= 70 else "🟡 WATCHLIST"
    return (
        f"{tipo} OptionsView PRO\n"
        f"{s['ticker']} · {s['signal']}\n"
        f"Setup score: {s['score']}%\n"
        f"Precio: {s['price']}\n"
        f"RSI: {s['rsi']} | Stoch: {s['stoch']}\n"
        f"IV Rank est.: {s['iv_rank']}%\n"
        f"1D: {s['trend_1d']} | 4H: {s['trend_4h']} | 1H: {s['trend_1h']}\n"
        f"Soporte: {s['support']} | Resistencia: {s['resistance']}\n"
        f"Dist soporte: {s['dist_support']}% | Dist resistencia: {s['dist_resistance']}%\n"
        f"Revisar earnings y cadena real antes de entrar."
    )

def scan_once():
    tickers = get_universe()
    print(f"Escaneando {len(tickers)} tickers...")
    setups = []

    for i, ticker in enumerate(tickers, 1):
        try:
            setup = analyze_ticker(ticker)
            if setup:
                setups.append(setup)
                key = f"{setup['ticker']}-{setup['signal']}-{setup['score']}"
                if key not in sent:
                    sent.add(key)
                    send_telegram(alert_message(setup))
                    print(f"ALERTA: {setup['ticker']} {setup['score']}%")
            if i % 25 == 0:
                print(f"Procesados {i}/{len(tickers)}")
            time.sleep(0.2)
        except Exception as e:
            print(f"Error {ticker}: {e}")

    print(f"Escaneo terminado. Setups encontrados: {len(setups)}")
    return setups

send_telegram("✅ OptionsView PRO: scanner SP500 + NASDAQ100 iniciado")

while True:
    print("Nuevo escaneo...")
    scan_once()
    print(f"Esperando {SCAN_INTERVAL_MINUTES} minutos...")
    time.sleep(SCAN_INTERVAL_MINUTES * 60)
