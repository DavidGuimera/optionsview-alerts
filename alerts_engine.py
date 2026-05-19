import os
import time
import json
import requests
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf


# =====================================================
# CONFIG - MISMA LÓGICA QUE LA APP LOCAL
# =====================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

MIN_SETUP_SCORE = int(os.getenv("MIN_SETUP_SCORE", "60"))
MAX_TICKERS = int(os.getenv("MAX_TICKERS", "230"))

DEFAULT_PERIOD = "1y"
DEFAULT_PUT_RSI = 35
DEFAULT_CALL_RSI = 65
DEFAULT_DISTANCE_SUPPORT = 4.0
DEFAULT_DISTANCE_RESISTANCE = 5.0
SUPPORT_RESISTANCE_WINDOW = 60
DEFAULT_SPREAD_WIDTH = 5

ALERT_MEMORY_FILE = "alert_memory.json"
ALERT_COOLDOWN_HOURS = 24


# =====================================================
# TELEGRAM
# =====================================================

def send_telegram(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Faltan TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}

    try:
        r = requests.post(url, data=payload, timeout=10)
        print(r.text)
        return r.status_code == 200
    except Exception as e:
        print("Error Telegram:", e)
        return False


# =====================================================
# ANTI-DUPLICADOS 24H
# =====================================================

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


# =====================================================
# SP500 + NASDAQ100
# =====================================================

def get_sp500_tickers():
    fallback = [
        "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","BRK-B","LLY","AVGO",
        "JPM","V","MA","XOM","UNH","COST","WMT","HD","PG","JNJ","ABBV","NFLX",
        "BAC","KO","MRK","CVX","CRM","AMD","PEP","TMO","ADBE","LIN","WFC","MCD",
        "CSCO","ABT","QCOM","PM","IBM","TXN","GE","DHR","INTU","AMGN","CAT","VZ",
        "NOW","ISRG","DIS","NEE","PFE","RTX","CMCSA","SPGI","UBER","UNP","LOW",
        "GS","PGR","BKNG","T","AXP","HON","BLK","ETN","TJX","COP","BA","SYK",
        "SCHW","VRTX","LMT","C","MDT","ADP","ELV","PANW","DE","FI","SBUX"
    ]

    try:
        df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        tickers = df["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
        return tickers if tickers else fallback
    except Exception as e:
        print("Fallback SP500:", e)
        return fallback


def get_nasdaq100_tickers():
    fallback = [
        "AAPL","MSFT","NVDA","AMZN","META","AVGO","GOOGL","GOOG","TSLA","COST",
        "NFLX","ASML","TMUS","CSCO","PEP","ADBE","AMD","LIN","INTU","TXN",
        "QCOM","AMGN","HON","ISRG","AMAT","BKNG","CMCSA","PANW","ADP","VRTX",
        "GILD","SBUX","MU","ADI","LRCX","MELI","KLAC","MDLZ","CRWD","REGN",
        "INTC","CTAS","PYPL","CEG","MAR","ORLY","SNPS","CDNS","CSX","ABNB"
    ]

    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
        for table in tables:
            for col in table.columns:
                if "ticker" in str(col).lower() or "symbol" in str(col).lower():
                    tickers = table[col].astype(str).str.replace(".", "-", regex=False).tolist()
                    tickers = [t for t in tickers if t and t.lower() != "nan"]
                    if len(tickers) > 20:
                        return tickers
        return fallback
    except Exception as e:
        print("Fallback NASDAQ100:", e)
        return fallback


def get_universe():
    tickers = get_sp500_tickers() + get_nasdaq100_tickers()
    clean = []
    seen = set()

    for ticker in tickers:
        ticker = str(ticker).upper().strip()
        if ticker and ticker not in seen:
            clean.append(ticker)
            seen.add(ticker)

    return clean[:MAX_TICKERS]


# =====================================================
# INDICADORES
# =====================================================

def safe_float(value):
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_stochastic(data, k_period=14, d_period=3):
    low_min = data["Low"].rolling(window=k_period).min()
    high_max = data["High"].rolling(window=k_period).max()
    k = 100 * ((data["Close"] - low_min) / (high_max - low_min))
    d = k.rolling(window=d_period).mean()
    return k, d


def download_data(ticker, period="1y", interval="1d"):
    ticker = str(ticker).strip().upper()

    for attempt in range(3):
        try:
            data = yf.download(
                ticker,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=True,
                threads=False,
            )

            if data is not None and not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)

                data = data.dropna(subset=["Open", "High", "Low", "Close"])

                if not data.empty:
                    return data
        except Exception:
            pass

        time.sleep(0.35 + attempt * 0.35)

    return pd.DataFrame()


def prepare_data(ticker, period="1y"):
    data = download_data(ticker, period, "1d")

    if data.empty:
        return data

    data["RSI"] = calculate_rsi(data["Close"])
    data["STO_K"], data["STO_D"] = calculate_stochastic(data)

    data["EMA21"] = data["Close"].ewm(span=21, adjust=False).mean()
    data["SMA20"] = data["Close"].rolling(20).mean()
    data["SMA50"] = data["Close"].rolling(50).mean()
    data["SMA200"] = data["Close"].rolling(200).mean()

    return data


def get_support_resistance(data, window=60):
    recent = data.tail(window)
    return safe_float(recent["Low"].min()), safe_float(recent["High"].max())


def calculate_iv_rank_proxy(data):
    try:
        if data is None or data.empty or len(data) < 40:
            return np.nan, np.nan

        returns = data["Close"].pct_change().dropna()
        hv20 = returns.rolling(20).std() * np.sqrt(252) * 100
        hv20 = hv20.dropna()

        if hv20.empty:
            return np.nan, np.nan

        current_hv = safe_float(hv20.iloc[-1])
        hv_min = safe_float(hv20.min())
        hv_max = safe_float(hv20.max())

        if pd.isna(current_hv) or pd.isna(hv_min) or pd.isna(hv_max) or hv_max == hv_min:
            return round(current_hv, 1), np.nan

        iv_rank = ((current_hv - hv_min) / (hv_max - hv_min)) * 100
        return round(current_hv, 1), round(max(0, min(100, iv_rank)), 1)

    except Exception:
        return np.nan, np.nan


def classify_trend(price, sma20, sma50, sma200):
    if pd.isna(sma200):
        if price > sma20 > sma50:
            return "Alcista"
        if price < sma20 < sma50:
            return "Bajista"
        return "Lateral / Mixta"

    if price > sma20 > sma50 > sma200:
        return "Alcista fuerte"

    if price > sma50 and price > sma200:
        return "Alcista"

    if price < sma20 < sma50 < sma200:
        return "Bajista fuerte"

    if price < sma50 and price < sma200:
        return "Bajista"

    return "Lateral / Mixta"


def detect_trend(data):
    try:
        if data is None or data.empty or len(data) < 60:
            return "Neutral"

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

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


def round_to_option_strike(price):
    if pd.isna(price):
        return np.nan

    if price >= 500:
        step = 10
    elif price >= 200:
        step = 5
    elif price >= 100:
        step = 2.5
    elif price >= 50:
        step = 1
    else:
        step = 0.5

    return round(price / step) * step


def format_strike(value):
    if pd.isna(value):
        return ""

    return str(int(value)) if float(value).is_integer() else str(value)


# =====================================================
# SCORE - MISMO CÁLCULO QUE LA APP MODIFICADA
# =====================================================

def calculate_setup_quality(signal, rsi, sto_k, dist_support, dist_resistance, trend, price, sma20, sma50, sma200, iv_rank=np.nan, trend_4h="Neutral", trend_1h="Neutral"):
    if signal in ["NO TRADE", "ERROR"]:
        return 0

    score = 45

    bullish_trend = trend in ["Alcista", "Alcista fuerte"]
    bearish_trend = trend in ["Bajista", "Bajista fuerte"]

    if signal == "PUT / Bull Put Spread":
        if not pd.isna(sma200):
            score += 18 if price > sma200 else -18

        if bullish_trend:
            score += 12
        elif bearish_trend:
            score -= 12

        if trend_4h == "Alcista":
            score += 9
        elif trend_4h == "Bajista":
            score -= 8

        if trend_1h == "Alcista":
            score += 4
        elif trend_1h == "Bajista":
            score -= 4

        if dist_support <= 1.5:
            score += 16
        elif dist_support <= 3:
            score += 12
        elif dist_support <= 4:
            score += 8
        elif dist_support > 8:
            score -= 10

        if rsi <= 30:
            score += 8
        elif rsi <= 35:
            score += 5
        elif rsi >= 55:
            score -= 6

        if sto_k <= 20:
            score += 3
        elif sto_k <= 35:
            score += 1
        elif sto_k >= 80:
            score -= 3

    elif signal == "CALL Spread":
        if not pd.isna(sma200):
            score += 18 if price < sma200 else -10

        if bearish_trend:
            score += 12
        elif bullish_trend:
            score -= 10

        if trend_4h == "Bajista":
            score += 9
        elif trend_4h == "Alcista":
            score -= 8

        if trend_1h == "Bajista":
            score += 4
        elif trend_1h == "Alcista":
            score -= 4

        if dist_resistance <= 1.5:
            score += 16
        elif dist_resistance <= 4:
            score += 12
        elif dist_resistance <= 5:
            score += 8
        elif dist_resistance > 9:
            score -= 10

        if rsi >= 70:
            score += 8
        elif rsi >= 65:
            score += 5
        elif rsi <= 45:
            score -= 6

        if sto_k >= 80:
            score += 3
        elif sto_k >= 65:
            score += 1
        elif sto_k <= 20:
            score -= 3

    if not pd.isna(iv_rank):
        if iv_rank >= 80:
            score += 10
        elif iv_rank >= 60:
            score += 15
        elif iv_rank >= 40:
            score += 8
        elif iv_rank >= 25:
            score += 3
        elif iv_rank < 20:
            score -= 12

    return int(max(0, min(100, round(score))))


def quality_label(score):
    if score >= 85:
        return "Excepcional"
    if score >= 75:
        return "Muy bueno"
    if score >= 60:
        return "Bueno"
    if score >= 50:
        return "Aceptable"
    if score > 0:
        return "Flojo"
    return "Sin setup"


def position_recommendation(quality, risk):
    try:
        quality = float(quality)
    except Exception:
        quality = 0

    contracts = 0
    label = "Sin entrada"

    if 50 <= quality < 65:
        contracts = 1
        label = "Entrada moderada"
    elif 65 <= quality < 75:
        contracts = 1
        label = "Buen setup"
    elif 75 <= quality < 85:
        contracts = 2
        label = "Entrada fuerte"
    elif quality >= 85:
        contracts = 2
        label = "Setup excepcional"

    if risk == "Alto":
        contracts = max(0, contracts - 1)
        if contracts == 0:
            label = "Evitar / revisar"
        else:
            label = f"Riesgo alto: reducir a {contracts} contrato"

    return contracts, label


def advanced_risk_filter(signal, price, sma20, sma50, sma200, rsi, sto_k):
    reasons = []
    risk_level = "Bajo"

    if signal == "PUT / Bull Put Spread":
        if not pd.isna(sma200) and price < sma200:
            reasons.append("Precio bajo SMA200")
        if not pd.isna(sma20) and not pd.isna(sma50) and sma20 < sma50:
            reasons.append("SMA20 bajo SMA50")
        if not pd.isna(sma20) and not pd.isna(sma50) and not pd.isna(sma200) and sma20 < sma50 < sma200:
            reasons.append("Tendencia bajista fuerte")
        if rsi < 25:
            reasons.append("RSI extremadamente bajo")

    if signal == "CALL Spread":
        if not pd.isna(sma20) and not pd.isna(sma50) and not pd.isna(sma200) and price > sma20 > sma50 > sma200:
            reasons.append("Tendencia alcista fuerte")
        if rsi > 78:
            reasons.append("RSI extremadamente alto")

    if len(reasons) >= 3:
        risk_level = "Alto"
    elif len(reasons) >= 1:
        risk_level = "Medio"

    return risk_level, reasons


# =====================================================
# ANÁLISIS
# =====================================================

def analyze_ticker(ticker):
    data = prepare_data(ticker, DEFAULT_PERIOD)

    if data.empty or len(data) < 80:
        return None

    latest = data.iloc[-1]

    price = safe_float(latest["Close"])
    rsi = safe_float(latest["RSI"])
    sto_k = safe_float(latest["STO_K"])

    sma20 = safe_float(latest["SMA20"])
    sma50 = safe_float(latest["SMA50"])
    sma200 = safe_float(latest["SMA200"])

    support, resistance = get_support_resistance(data, SUPPORT_RESISTANCE_WINDOW)

    dist_support = ((price - support) / support) * 100 if support else np.nan
    dist_resistance = ((resistance - price) / price) * 100 if price else np.nan

    trend = classify_trend(price, sma20, sma50, sma200)

    put_setup = rsi <= DEFAULT_PUT_RSI and dist_support <= DEFAULT_DISTANCE_SUPPORT
    call_setup = rsi >= DEFAULT_CALL_RSI and dist_resistance <= DEFAULT_DISTANCE_RESISTANCE

    signal = "NO TRADE"

    if put_setup:
        signal = "PUT / Bull Put Spread"

    if call_setup:
        signal = "CALL Spread"

    if put_setup and call_setup:
        signal = "NO TRADE"

    estrategia = ""

    if signal == "PUT / Bull Put Spread":
        base_put = min(support, price * 0.93)
        put_short = round_to_option_strike(base_put)
        put_long = put_short - DEFAULT_SPREAD_WIDTH
        estrategia = f"{format_strike(put_short)}/{format_strike(put_long)} Bull Put Spread"

    elif signal == "CALL Spread":
        base_call = max(resistance, price * 1.07)
        call_short = round_to_option_strike(base_call)
        call_long = call_short + DEFAULT_SPREAD_WIDTH
        estrategia = f"{format_strike(call_short)}/{format_strike(call_long)} Bear Call Spread"

    risk, reasons = advanced_risk_filter(signal, price, sma20, sma50, sma200, rsi, sto_k)

    _, iv_rank_est = calculate_iv_rank_proxy(data)

    try:
        data_4h = download_data(ticker, "6mo", "4h")
        data_1h = download_data(ticker, "2mo", "1h")
        trend_4h = detect_trend(data_4h)
        trend_1h = detect_trend(data_1h)
    except Exception:
        trend_4h = "Neutral"
        trend_1h = "Neutral"

    quality = calculate_setup_quality(
        signal,
        rsi,
        sto_k,
        dist_support,
        dist_resistance,
        trend,
        price,
        sma20,
        sma50,
        sma200,
        iv_rank_est,
        trend_4h,
        trend_1h,
    )

    contracts, entry_label = position_recommendation(quality, risk)

    if quality < MIN_SETUP_SCORE:
        return None

    return {
        "Ticker": ticker,
        "Precio": round(price, 2),
        "Señal": signal,
        "Calidad setup %": quality,
        "Calidad": quality_label(quality),
        "Riesgo": risk,
        "Contratos sugeridos": contracts,
        "Entrada sugerida": entry_label,
        "Estrategia sugerida": estrategia,
        "RSI": round(rsi, 2),
        "Estocástico K": round(sto_k, 2),
        "Soporte 60D": round(support, 2),
        "Resistencia 60D": round(resistance, 2),
        "Distancia soporte %": round(dist_support, 2),
        "Distancia resistencia %": round(dist_resistance, 2),
        "IV Rank estimado %": iv_rank_est,
        "Trend 4H": trend_4h,
        "Trend 1H": trend_1h,
        "Advertencia": "; ".join(reasons),
    }


# =====================================================
# MENSAJE TELEGRAM
# =====================================================

def alert_message(row):
    score = safe_float(row.get("Calidad setup %", 0))

    if score >= 85:
        tipo = "🔥 SETUP ÉLITE"
    elif score >= 75:
        tipo = "🔴 PREMIUM"
    elif score >= 65:
        tipo = "🟡 BUEN SETUP"
    else:
        tipo = "🟡 WATCHLIST"

    return (
        f"{tipo} OptionsView PRO\n\n"
        f"{row.get('Ticker')} · {row.get('Señal')}\n"
        f"Setup score APP: {row.get('Calidad setup %')}% · {row.get('Calidad')}\n"
        f"Precio: {row.get('Precio')}\n"
        f"Estrategia: {row.get('Estrategia sugerida') or 'Sin spread'}\n"
        f"Contratos: {row.get('Contratos sugeridos')} · {row.get('Entrada sugerida')}\n\n"
        f"RSI: {row.get('RSI')} · Stoch K: {row.get('Estocástico K')}\n"
        f"IV Rank est.: {row.get('IV Rank estimado %')}%\n"
        f"Soporte: {row.get('Soporte 60D')} · Resistencia: {row.get('Resistencia 60D')}\n"
        f"Dist soporte: {row.get('Distancia soporte %')}% · Dist resistencia: {row.get('Distancia resistencia %')}%\n\n"
        f"Riesgo: {row.get('Riesgo')}\n"
        f"4H: {row.get('Trend 4H')} · 1H: {row.get('Trend 1H')}\n"
        f"Advertencia: {row.get('Advertencia') or 'Sin advertencias'}\n\n"
        f"Revisar earnings y cadena real antes de entrar."
    )


# =====================================================
# SCAN
# =====================================================

def scan_once():
    tickers = get_universe()
    memory = load_alert_memory()
    setups = []

    print(f"Escaneando {len(tickers)} tickers SP500 + NASDAQ100 con lógica igual a la app...")

    for i, ticker in enumerate(tickers, 1):
        try:
            row = analyze_ticker(ticker)

            if row:
                ticker_key = row["Ticker"]

                if recently_alerted(ticker_key, memory):
                    print(f"Duplicado evitado 24h: {ticker_key}")
                else:
                    setups.append(row)
                    send_telegram(alert_message(row))
                    memory[ticker_key] = datetime.utcnow().isoformat()
                    save_alert_memory(memory)
                    print(f"ALERTA: {ticker_key} {row.get('Calidad setup %')}%")

            if i % 10 == 0:
                print(f"Procesados {i}/{len(tickers)}")

            time.sleep(0.2)

        except Exception as e:
            print(f"Error {ticker}: {e}")

    print(f"Escaneo terminado. Setups nuevos: {len(setups)}")
    return setups


if __name__ == "__main__":
    setups = scan_once()
    send_telegram(f"✅ Escaneo finalizado. Setups nuevos encontrados: {len(setups)}")
