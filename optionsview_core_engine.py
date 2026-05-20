"""
OptionsView CORE ENGINE
Motor único de scoring para APP y GitHub Alerts.

Objetivo:
- La APP y GitHub importan este mismo archivo.
- El % de setup sale de analyze_ticker().
- Se separa Technical Score, Options Score y Final Score.

Instalación:
pip install yfinance pandas numpy requests dash plotly
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from math import erf, log, sqrt
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# =====================================================
# CONFIGURACIÓN GLOBAL COMPARTIDA
# =====================================================

DEFAULT_TICKERS_FAST = "MCD,PEP,PG,COST,SBUX,MSFT,AAPL,GOOGL,JPM,V,SPY,QQQ"

FULL_TICKERS = (
    "MCD,PEP,PG,KO,JNJ,WMT,COST,HD,LOW,TGT,"
    "SBUX,MDLZ,CMCSA,"
    "MSFT,AAPL,GOOGL,META,AMZN,NVDA,AVGO,ADBE,CRM,"
    "JPM,MA,V,BLK,SCHW,"
    "SPY,QQQ,IWM,XLP,XLV,XLF"
)

RISK_FREE_RATE = 0.045
CONTRACT_MULTIPLIER = 100
DEFAULT_DTE = 40
TARGET_DTE_MIN = 25
TARGET_DTE_MAX = 50
SUPPORT_RESISTANCE_WINDOW = 60
PUT_RSI_LIMIT = 35
CALL_RSI_LIMIT = 65
MAX_DIST_SUPPORT = 4.0
MAX_DIST_RESISTANCE = 5.0

# Modo conservador para tu operativa actual
MIN_FINAL_SCORE_ALERT = 60


# =====================================================
# UTILIDADES
# =====================================================

def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip() == "":
            return default
        out = float(value)
        if pd.isna(out):
            return default
        return out
    except Exception:
        return default


def clamp(value: float, low: float = 0, high: float = 100) -> float:
    if pd.isna(value):
        return low
    return max(low, min(high, value))


def normal_cdf(x: float) -> float:
    return 0.5 * (1 + erf(x / sqrt(2)))


def clean_ticker(ticker: str) -> str:
    return str(ticker).strip().upper().replace(".", "-")


def parse_tickers(text: str) -> List[str]:
    if not text:
        text = DEFAULT_TICKERS_FAST
    text = text.strip()
    if text.upper() in ["FAVORITAS", "FAVORITES", "FULL", "UNIVERSO"]:
        text = FULL_TICKERS
    if text.upper() in ["DEFAULT", "FAST"]:
        text = DEFAULT_TICKERS_FAST
    items = [clean_ticker(x) for x in text.replace(";", ",").split(",") if x.strip()]
    seen = set()
    out = []
    for t in items:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


# =====================================================
# DATOS SAFE
# =====================================================

def download_price_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Descarga diaria segura ticker por ticker. Evita mezclar META/COST, MultiIndex y columnas duplicadas."""
    ticker = clean_ticker(ticker)
    try:
        tk = yf.Ticker(ticker)
        data = tk.history(period=period, interval="1d", auto_adjust=True)
        if data is None or data.empty:
            return pd.DataFrame()

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]

        data = data.loc[:, ~data.columns.duplicated()].copy()
        needed = ["Open", "High", "Low", "Close"]
        if not all(c in data.columns for c in needed):
            return pd.DataFrame()
        data = data.dropna(subset=needed)
        if data.empty:
            return pd.DataFrame()

        # Validación básica anti datos corruptos
        last_close = safe_float(data["Close"].iloc[-1])
        if pd.isna(last_close) or last_close <= 0:
            return pd.DataFrame()
        return data
    except Exception:
        return pd.DataFrame()


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def prepare_price_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    data = download_price_data(ticker, period)
    if data.empty:
        return data
    close = data["Close"]
    data["RSI"] = calculate_rsi(close)
    data["SMA20"] = close.rolling(20).mean()
    data["SMA50"] = close.rolling(50).mean()
    data["SMA200"] = close.rolling(200).mean()
    data["EMA21"] = close.ewm(span=21, adjust=False).mean()
    data["RET"] = close.pct_change()
    data["HV20"] = data["RET"].rolling(20).std() * np.sqrt(252) * 100
    return data


def get_support_resistance(data: pd.DataFrame, window: int = SUPPORT_RESISTANCE_WINDOW) -> Tuple[float, float]:
    if data.empty:
        return np.nan, np.nan
    recent = data.tail(window)
    return safe_float(recent["Low"].min()), safe_float(recent["High"].max())


def calculate_iv_rank_proxy(data: pd.DataFrame) -> Tuple[float, float]:
    """Proxy estable. Si no hay histórico IV real, usa HV20 como régimen de volatilidad."""
    try:
        hv = data["HV20"].dropna()
        if len(hv) < 10:
            return np.nan, np.nan
        current = safe_float(hv.iloc[-1])
        hv_min = safe_float(hv.min())
        hv_max = safe_float(hv.max())
        if pd.isna(current) or pd.isna(hv_min) or pd.isna(hv_max) or hv_max == hv_min:
            return round(current, 1), np.nan
        ivr = ((current - hv_min) / (hv_max - hv_min)) * 100
        return round(current, 1), round(clamp(ivr), 1)
    except Exception:
        return np.nan, np.nan


def classify_trend(price: float, sma20: float, sma50: float, sma200: float) -> str:
    if pd.isna(price):
        return "No disponible"
    if not pd.isna(sma200):
        if price > sma20 > sma50 > sma200:
            return "Alcista fuerte"
        if price > sma50 and price > sma200:
            return "Alcista"
        if price < sma20 < sma50 < sma200:
            return "Bajista fuerte"
        if price < sma50 and price < sma200:
            return "Bajista"
    else:
        if price > sma20 > sma50:
            return "Alcista"
        if price < sma20 < sma50:
            return "Bajista"
    return "Lateral / Mixta"


def detect_setup(price: float, rsi: float, support: float, resistance: float) -> Tuple[str, float, float]:
    dist_support = ((price - support) / support) * 100 if support and not pd.isna(support) else np.nan
    dist_resistance = ((resistance - price) / price) * 100 if price and resistance and not pd.isna(resistance) else np.nan
    put_setup = not pd.isna(rsi) and not pd.isna(dist_support) and rsi <= PUT_RSI_LIMIT and dist_support <= MAX_DIST_SUPPORT
    call_setup = not pd.isna(rsi) and not pd.isna(dist_resistance) and rsi >= CALL_RSI_LIMIT and dist_resistance <= MAX_DIST_RESISTANCE
    if put_setup and not call_setup:
        return "PUT / Bull Put Spread", round(dist_support, 2), round(dist_resistance, 2)
    if call_setup and not put_setup:
        return "CALL / Bear Call Spread", round(dist_support, 2), round(dist_resistance, 2)
    return "NO TRADE", round(dist_support, 2) if not pd.isna(dist_support) else np.nan, round(dist_resistance, 2) if not pd.isna(dist_resistance) else np.nan


# =====================================================
# SCORE TÉCNICO
# =====================================================

def calculate_technical_score(signal: str, price: float, rsi: float, sma20: float, sma50: float, sma200: float,
                              support: float, resistance: float, dist_support: float, dist_resistance: float,
                              trend: str) -> Tuple[int, List[str]]:
    reasons = []
    if signal == "NO TRADE":
        return 0, ["No hay setup técnico claro"]

    score = 50

    if signal.startswith("PUT"):
        if not pd.isna(sma200):
            if price > sma200:
                score += 12; reasons.append("Precio sobre SMA200")
            else:
                score -= 14; reasons.append("Precio bajo SMA200")
        if trend in ["Alcista", "Alcista fuerte"]:
            score += 10; reasons.append("Tendencia compatible con put credit spread")
        elif trend in ["Bajista", "Bajista fuerte"]:
            score -= 12; reasons.append("Tendencia bajista contra el setup")
        if not pd.isna(dist_support):
            if dist_support <= 1.5:
                score += 16; reasons.append("Muy cerca de soporte")
            elif dist_support <= 3:
                score += 12; reasons.append("Cerca de soporte")
            elif dist_support <= 4:
                score += 7; reasons.append("Soporte razonablemente cerca")
        if not pd.isna(rsi):
            if rsi <= 30:
                score += 10; reasons.append("RSI sobreventa fuerte")
            elif rsi <= 35:
                score += 6; reasons.append("RSI sobreventa moderada")
            elif rsi > 50:
                score -= 4

    elif signal.startswith("CALL"):
        if not pd.isna(sma200):
            if price < sma200:
                score += 10; reasons.append("Precio bajo SMA200")
            else:
                score -= 5; reasons.append("Precio sobre SMA200: cuidado con call spread")
        if trend in ["Bajista", "Bajista fuerte"]:
            score += 10; reasons.append("Tendencia compatible con bear call spread")
        elif trend in ["Alcista", "Alcista fuerte"]:
            score -= 6; reasons.append("Tendencia alcista contra call spread")
        if not pd.isna(dist_resistance):
            if dist_resistance <= 1.5:
                score += 16; reasons.append("Muy cerca de resistencia")
            elif dist_resistance <= 4:
                score += 12; reasons.append("Cerca de resistencia")
            elif dist_resistance <= 5:
                score += 7; reasons.append("Resistencia razonablemente cerca")
        if not pd.isna(rsi):
            if rsi >= 70:
                score += 10; reasons.append("RSI sobrecompra fuerte")
            elif rsi >= 65:
                score += 6; reasons.append("RSI sobrecompra moderada")
            elif rsi < 50:
                score -= 4

    return int(round(clamp(score))), reasons


# =====================================================
# OPCIONES / SMART SPREAD
# =====================================================

def estimate_delta_prob(price: float, strike: float, iv_pct: float, dte: int, option_type: str) -> Tuple[float, float]:
    try:
        if price <= 0 or strike <= 0 or iv_pct <= 0 or dte <= 0:
            return np.nan, np.nan
        t = dte / 365
        sigma = iv_pct / 100
        d1 = (log(price / strike) + (RISK_FREE_RATE + 0.5 * sigma ** 2) * t) / (sigma * sqrt(t))
        d2 = d1 - sigma * sqrt(t)
        if option_type == "put":
            delta = normal_cdf(d1) - 1
            prob_otm = normal_cdf(d2) * 100
        else:
            delta = normal_cdf(d1)
            prob_otm = normal_cdf(-d2) * 100
        return round(delta, 2), round(prob_otm, 1)
    except Exception:
        return np.nan, np.nan


def get_earnings_info(ticker: str) -> Tuple[str, float, str]:
    """Devuelve texto, días a earnings y estado."""
    try:
        tk = yf.Ticker(ticker)
        cal = tk.calendar
        candidates = []
        if isinstance(cal, dict):
            for key in ["Earnings Date", "Earnings High", "Earnings Low"]:
                val = cal.get(key)
                if isinstance(val, (list, tuple)):
                    candidates.extend(val)
                elif val is not None:
                    candidates.append(val)
        elif isinstance(cal, pd.DataFrame) and not cal.empty:
            vals = cal.values.flatten().tolist()
            candidates.extend(vals)
        parsed = []
        for c in candidates:
            try:
                ts = pd.Timestamp(c).tz_localize(None)
                if not pd.isna(ts):
                    parsed.append(ts)
            except Exception:
                continue
        if not parsed:
            return "No disponible", np.nan, "UNKNOWN"
        today = pd.Timestamp.today().normalize()
        future = sorted([x for x in parsed if x >= today])
        if not future:
            return "No disponible", np.nan, "UNKNOWN"
        ed = future[0]
        days = int((ed.normalize() - today).days)
        if days <= 7:
            status = "DANGER"
        elif days <= 14:
            status = "CAUTION"
        else:
            status = "OK"
        return ed.strftime("%Y-%m-%d"), days, status
    except Exception:
        return "No disponible", np.nan, "UNKNOWN"


def choose_expiration(tk: yf.Ticker) -> Optional[str]:
    try:
        expirations = list(tk.options)
        if not expirations:
            return None
        today = pd.Timestamp.today().normalize()
        best = None
        best_score = -1e9
        for exp in expirations:
            dte = int((pd.Timestamp(exp) - today).days)
            if dte < 14:
                continue
            # preferimos 25-50 DTE, pero permitimos fallback
            window_bonus = 100 if TARGET_DTE_MIN <= dte <= TARGET_DTE_MAX else 0
            closeness = -abs(dte - DEFAULT_DTE)
            score = window_bonus + closeness
            if score > best_score:
                best_score = score
                best = exp
        return best or expirations[0]
    except Exception:
        return None


def clean_chain(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    for col in ["strike", "bid", "ask", "impliedVolatility", "openInterest", "volume"]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["mid"] = (out["bid"] + out["ask"]) / 2
    out["spread_pct"] = np.where(out["mid"] > 0, (out["ask"] - out["bid"]) / out["mid"] * 100, np.nan)
    out["iv_pct"] = out["impliedVolatility"] * 100
    out = out.dropna(subset=["strike"])
    return out.sort_values("strike")


def build_best_spread(ticker: str, signal: str, price: float) -> Dict[str, Any]:
    base = {
        "Options Status": "UNAVAILABLE",
        "Smart Spread": "",
        "Spread Type": "",
        "Expiration": "",
        "DTE": np.nan,
        "Short Strike": np.nan,
        "Long Strike": np.nan,
        "Credit": np.nan,
        "Max Profit": np.nan,
        "Max Loss": np.nan,
        "ROC %": np.nan,
        "Delta": np.nan,
        "Prob OTM %": np.nan,
        "IV %": np.nan,
        "Expected Move": np.nan,
        "Liquidity": "No disponible",
        "OI": np.nan,
        "Bid/Ask %": np.nan,
        "Options Notes": "Sin cadena de opciones",
    }
    if signal == "NO TRADE" or pd.isna(price) or price <= 0:
        base["Options Notes"] = "No hay setup técnico; no se construye spread"
        return base
    try:
        tk = yf.Ticker(ticker)
        exp = choose_expiration(tk)
        if not exp:
            return base
        today = pd.Timestamp.today().normalize()
        dte = int((pd.Timestamp(exp) - today).days)
        chain = tk.option_chain(exp)
        side = "put" if signal.startswith("PUT") else "call"
        options = clean_chain(chain.puts if side == "put" else chain.calls)
        if options.empty:
            return base

        widths = [2.5, 5, 10]
        candidates = []
        if side == "put":
            shorts = options[(options["strike"] < price) & (options["bid"] > 0) & (options["ask"] > 0)].copy()
            shorts = shorts.sort_values("strike", ascending=False).head(25)
        else:
            shorts = options[(options["strike"] > price) & (options["bid"] > 0) & (options["ask"] > 0)].copy()
            shorts = shorts.sort_values("strike").head(25)

        for _, srow in shorts.iterrows():
            short = safe_float(srow["strike"])
            bid = safe_float(srow["bid"])
            ask = safe_float(srow["ask"])
            iv = safe_float(srow["iv_pct"])
            oi = safe_float(srow["openInterest"])
            ba = safe_float(srow["spread_pct"])
            if pd.isna(iv) or iv <= 0 or bid <= 0:
                continue
            delta, prob = estimate_delta_prob(price, short, iv, dte, side)
            if pd.isna(prob):
                continue
            # Strike objetivo: delta aprox 0.15-0.30, prob OTM 70-88
            if prob < 65 or prob > 92:
                continue
            for width in widths:
                long_target = short - width if side == "put" else short + width
                options["dist_long"] = (options["strike"] - long_target).abs()
                lrow = options.sort_values("dist_long").iloc[0]
                long = safe_float(lrow["strike"])
                real_width = abs(short - long)
                if real_width <= 0:
                    continue
                long_ask = safe_float(lrow["ask"])
                if pd.isna(long_ask) or long_ask <= 0:
                    continue
                credit = round(max(0, bid - long_ask), 2)
                if credit <= 0:
                    continue
                max_profit = credit * CONTRACT_MULTIPLIER
                max_loss = (real_width - credit) * CONTRACT_MULTIPLIER
                if max_loss <= 0:
                    continue
                roc = round(max_profit / max_loss * 100, 1)
                if roc < 5 or roc > 60:
                    continue
                em = round(price * (iv / 100) * sqrt(max(dte, 1) / 365), 2)
                # score interno de spread
                spread_score = 50
                # Prob OTM
                if 75 <= prob <= 88:
                    spread_score += 18
                elif 70 <= prob < 75:
                    spread_score += 10
                elif prob >= 88:
                    spread_score += 6
                else:
                    spread_score -= 8
                # ROC
                if 10 <= roc <= 25:
                    spread_score += 18
                elif 7 <= roc < 10:
                    spread_score += 8
                elif roc > 30:
                    spread_score -= 6
                # liquidez
                if not pd.isna(oi) and oi >= 300:
                    spread_score += 12
                elif not pd.isna(oi) and oi >= 100:
                    spread_score += 8
                elif pd.isna(oi) or oi < 50:
                    spread_score -= 10
                if not pd.isna(ba) and ba <= 15:
                    spread_score += 8
                elif not pd.isna(ba) and ba <= 30:
                    spread_score += 2
                elif not pd.isna(ba) and ba > 30:
                    spread_score -= 10
                # distancia vs expected move
                if side == "put":
                    outside_em = short < price - em
                else:
                    outside_em = short > price + em
                if outside_em:
                    spread_score += 8
                # Delta
                abs_delta = abs(delta)
                if 0.12 <= abs_delta <= 0.28:
                    spread_score += 8
                elif abs_delta > 0.35:
                    spread_score -= 10

                liquidity = "Alta" if (not pd.isna(oi) and oi >= 100 and (pd.isna(ba) or ba <= 20)) else "Media" if (pd.isna(ba) or ba <= 35) else "Baja"
                label = "PCS" if side == "put" else "CCS"
                smart = f"{ticker} {short:g}/{long:g} {label}"
                candidates.append({
                    **base,
                    "Options Status": "OK",
                    "Smart Spread": smart,
                    "Spread Type": label,
                    "Expiration": exp,
                    "DTE": dte,
                    "Short Strike": short,
                    "Long Strike": long,
                    "Credit": credit,
                    "Max Profit": round(max_profit, 2),
                    "Max Loss": round(max_loss, 2),
                    "ROC %": roc,
                    "Delta": delta,
                    "Prob OTM %": prob,
                    "IV %": round(iv, 1),
                    "Expected Move": em,
                    "Liquidity": liquidity,
                    "OI": oi,
                    "Bid/Ask %": round(ba, 1) if not pd.isna(ba) else np.nan,
                    "Options Notes": "Spread calculado desde cadena yfinance",
                    "_spread_score": clamp(spread_score),
                })
        if not candidates:
            base["Options Notes"] = "No se encontró spread líquido con criterios mínimos"
            return base
        # Priorizamos calidad, luego ROC moderado y probabilidad
        candidates = sorted(candidates, key=lambda x: (x["_spread_score"], x["Prob OTM %"], x["ROC %"]), reverse=True)
        best = candidates[0]
        best.pop("_spread_score", None)
        return best
    except Exception as e:
        base["Options Notes"] = f"Error opciones: {e}"
        return base


def calculate_options_score(options: Dict[str, Any], iv_rank: float, earnings_status: str) -> Tuple[Optional[int], List[str]]:
    reasons = []
    if options.get("Options Status") != "OK":
        return None, [str(options.get("Options Notes", "Opciones no disponibles"))]

    score = 50
    prob = safe_float(options.get("Prob OTM %"))
    roc = safe_float(options.get("ROC %"))
    oi = safe_float(options.get("OI"))
    ba = safe_float(options.get("Bid/Ask %"))
    delta = abs(safe_float(options.get("Delta")))
    liq = str(options.get("Liquidity", ""))

    if not pd.isna(iv_rank):
        if iv_rank >= 60:
            score += 14; reasons.append("IVR alto/favorable")
        elif iv_rank >= 40:
            score += 8; reasons.append("IVR aceptable")
        elif iv_rank < 20:
            score -= 12; reasons.append("IVR bajo")

    if not pd.isna(prob):
        if prob >= 80:
            score += 14; reasons.append("Prob OTM alta")
        elif prob >= 72:
            score += 9; reasons.append("Prob OTM aceptable")
        elif prob < 68:
            score -= 10; reasons.append("Prob OTM baja")

    if not pd.isna(roc):
        if 10 <= roc <= 25:
            score += 14; reasons.append("ROC sano")
        elif 7 <= roc < 10:
            score += 6; reasons.append("ROC moderado")
        elif roc < 6:
            score -= 8; reasons.append("ROC bajo")
        elif roc > 35:
            score -= 7; reasons.append("ROC demasiado agresivo")

    if liq == "Alta":
        score += 10; reasons.append("Liquidez alta")
    elif liq == "Media":
        score += 4; reasons.append("Liquidez media")
    elif liq == "Baja":
        score -= 12; reasons.append("Liquidez baja")

    if not pd.isna(oi) and oi < 50:
        score -= 8; reasons.append("OI bajo")
    if not pd.isna(ba) and ba > 30:
        score -= 8; reasons.append("Bid/ask amplio")
    if not pd.isna(delta):
        if 0.12 <= delta <= 0.28:
            score += 6; reasons.append("Delta adecuada")
        elif delta > 0.35:
            score -= 10; reasons.append("Delta agresiva")

    if earnings_status == "DANGER":
        score -= 25; reasons.append("Earnings demasiado cerca")
    elif earnings_status == "CAUTION":
        score -= 10; reasons.append("Earnings en zona de precaución")

    return int(round(clamp(score))), reasons


def combine_scores(technical_score: int, options_score: Optional[int], earnings_status: str) -> int:
    if technical_score <= 0:
        return 0
    if options_score is None:
        # Si la cadena falla, no hundimos a 0. Damos score técnico penalizado.
        final = technical_score - 8
    else:
        final = technical_score * 0.55 + options_score * 0.45
    if earnings_status == "DANGER":
        final -= 25
    elif earnings_status == "CAUTION":
        final -= 8
    return int(round(clamp(final)))


def position_sizing(final_score: int, earnings_status: str, liquidity: str) -> Tuple[int, str]:
    if final_score < 60 or earnings_status == "DANGER" or liquidity == "Baja":
        return 0, "NO TRADE"
    if 60 <= final_score < 75:
        return 1, "1 contrato máximo"
    if 75 <= final_score < 85:
        return 2, "2 contratos máximo"
    return 2, "2 contratos máximo · revisar manualmente si quieres más"


def quality_label(score: int) -> str:
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


# =====================================================
# ANÁLISIS PRINCIPAL: ÚNICA FUENTE DE VERDAD
# =====================================================

def analyze_ticker(ticker: str, period: str = "1y", deep_options: bool = True) -> Dict[str, Any]:
    ticker = clean_ticker(ticker)
    row: Dict[str, Any] = {
        "Ticker": ticker,
        "Data Status": "ERROR",
        "Precio": np.nan,
        "Señal": "ERROR",
        "Technical Score %": 0,
        "Options Score %": np.nan,
        "Final Score %": 0,
        "Calidad setup %": 0,
        "Calidad": "Error",
    }
    data = prepare_price_data(ticker, period)
    if data.empty or len(data) < 60:
        row.update({"Señal": "ERROR", "Data Status": "Sin datos suficientes", "Alertas setup": "Error datos"})
        return row

    latest = data.iloc[-1]
    prev = data.iloc[-2] if len(data) > 1 else latest
    price = safe_float(latest["Close"])
    prev_price = safe_float(prev["Close"])
    change = price - prev_price if not pd.isna(price) and not pd.isna(prev_price) else np.nan
    change_pct = (change / prev_price) * 100 if prev_price and not pd.isna(change) else np.nan
    rsi = safe_float(latest.get("RSI"))
    sma20 = safe_float(latest.get("SMA20"))
    sma50 = safe_float(latest.get("SMA50"))
    sma200 = safe_float(latest.get("SMA200"))
    support, resistance = get_support_resistance(data)
    signal, dist_support, dist_resistance = detect_setup(price, rsi, support, resistance)
    trend = classify_trend(price, sma20, sma50, sma200)
    hv20, iv_rank = calculate_iv_rank_proxy(data)
    earnings_date, earnings_days, earnings_status = get_earnings_info(ticker)

    tech_score, tech_reasons = calculate_technical_score(
        signal, price, rsi, sma20, sma50, sma200, support, resistance, dist_support, dist_resistance, trend
    )

    options = build_best_spread(ticker, signal, price) if deep_options else {
        "Options Status": "SKIPPED", "Options Notes": "Deep options desactivado", "Liquidity": "No disponible"
    }
    opt_score, opt_reasons = calculate_options_score(options, iv_rank, earnings_status)
    final = combine_scores(tech_score, opt_score, earnings_status)
    contracts, sizing_text = position_sizing(final, earnings_status, str(options.get("Liquidity", "No disponible")))

    alerts = []
    if earnings_status == "DANGER":
        alerts.append(f"⚠ Earnings cerca: {earnings_date}")
    elif earnings_status == "CAUTION":
        alerts.append(f"⚠ Earnings en {earnings_days} días")
    if opt_score is None and signal != "NO TRADE":
        alerts.append("⚠ Opciones no disponibles: score basado en técnico penalizado")
    if safe_float(options.get("Bid/Ask %")) > 30:
        alerts.append("⚠ Bid/ask amplio")
    if safe_float(options.get("OI")) < 50 and not pd.isna(safe_float(options.get("OI"))):
        alerts.append("⚠ OI bajo")
    if not alerts:
        alerts.append("✅ Sin alertas graves")

    row.update({
        "Data Status": "OK",
        "Precio": round(price, 2),
        "Cambio": round(change, 2) if not pd.isna(change) else np.nan,
        "%": round(change_pct, 2) if not pd.isna(change_pct) else np.nan,
        "Señal": signal,
        "Technical Score %": tech_score,
        "Options Score %": opt_score if opt_score is not None else np.nan,
        "Final Score %": final,
        "Calidad setup %": final,
        "Calidad": quality_label(final),
        "Contratos sugeridos": contracts,
        "Entrada sugerida": sizing_text,
        "Riesgo": "Alto" if final < 50 or earnings_status == "DANGER" else "Medio" if final < 70 or earnings_status == "CAUTION" else "Bajo",
        "RSI": round(rsi, 2) if not pd.isna(rsi) else np.nan,
        "Trend": trend,
        "SMA20": round(sma20, 2) if not pd.isna(sma20) else np.nan,
        "SMA50": round(sma50, 2) if not pd.isna(sma50) else np.nan,
        "SMA200": round(sma200, 2) if not pd.isna(sma200) else np.nan,
        "Soporte 60D": round(support, 2) if not pd.isna(support) else np.nan,
        "Resistencia 60D": round(resistance, 2) if not pd.isna(resistance) else np.nan,
        "Distancia soporte %": dist_support,
        "Distancia resistencia %": dist_resistance,
        "HV 20D %": hv20,
        "IV Rank estimado %": iv_rank,
        "Earnings próximos": earnings_date,
        "Earnings días": earnings_days,
        "Earnings Status": earnings_status,
        "Estrategia sugerida": options.get("Smart Spread", ""),
        "Smart Spread": options.get("Smart Spread", ""),
        "Spread Type": options.get("Spread Type", ""),
        "Expiración analizada": options.get("Expiration", ""),
        "DTE": options.get("DTE", np.nan),
        "Crédito estimado": options.get("Credit", np.nan),
        "Max Profit": options.get("Max Profit", np.nan),
        "Max Loss": options.get("Max Loss", np.nan),
        "ROI/ROC estimado %": options.get("ROC %", np.nan),
        "Delta estimada": options.get("Delta", np.nan),
        "Prob OTM estimada %": options.get("Prob OTM %", np.nan),
        "IV estimada %": options.get("IV %", np.nan),
        "Expected Move": options.get("Expected Move", np.nan),
        "Liquidez": options.get("Liquidity", "No disponible"),
        "Open interest": options.get("OI", np.nan),
        "Bid/Ask spread %": options.get("Bid/Ask %", np.nan),
        "Options Status": options.get("Options Status", "UNAVAILABLE"),
        "Options Notes": options.get("Options Notes", ""),
        "Technical Reasons": " | ".join(tech_reasons[:5]),
        "Options Reasons": " | ".join(opt_reasons[:5]),
        "Alertas setup": " | ".join(alerts),
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    return row


def analyze_universe(tickers: List[str] | str, period: str = "1y", deep_options: bool = True, max_items: Optional[int] = None) -> pd.DataFrame:
    if isinstance(tickers, str):
        ticker_list = parse_tickers(tickers)
    else:
        ticker_list = [clean_ticker(x) for x in tickers]
    if max_items:
        ticker_list = ticker_list[:max_items]
    rows = []
    for t in ticker_list:
        rows.append(analyze_ticker(t, period=period, deep_options=deep_options))
    return pd.DataFrame(rows)


def format_telegram_alert(row: Dict[str, Any]) -> str:
    ticker = row.get("Ticker", "")
    score = row.get("Final Score %", row.get("Calidad setup %", 0))
    tech = row.get("Technical Score %", 0)
    opt = row.get("Options Score %", np.nan)
    spread = row.get("Smart Spread") or row.get("Estrategia sugerida") or "Sin spread"
    roc = row.get("ROI/ROC estimado %", np.nan)
    prob = row.get("Prob OTM estimada %", np.nan)
    delta = row.get("Delta estimada", np.nan)
    ivr = row.get("IV Rank estimado %", np.nan)
    earn = row.get("Earnings próximos", "No disponible")
    earn_status = row.get("Earnings Status", "UNKNOWN")
    contracts = row.get("Entrada sugerida", "")
    liq = row.get("Liquidez", "")
    data_status = row.get("Data Status", "")
    opt_status = row.get("Options Status", "")

    def fmt(v, suffix=""):
        return "N/A" if pd.isna(safe_float(v)) else f"{safe_float(v):g}{suffix}"

    return (
        f"🔥 <b>{ticker}</b> | <b>{score}%</b>\n"
        f"{spread}\n"
        f"Tech: {tech}% | Options: {fmt(opt, '%')} | {contracts}\n"
        f"ROC: {fmt(roc, '%')} | Prob OTM: {fmt(prob, '%')} | Delta: {fmt(delta)}\n"
        f"IVR: {fmt(ivr, '%')} | Liquidez: {liq}\n"
        f"Earnings: {earn} ({earn_status})\n"
        f"Data: {data_status} | Options: {opt_status}"
    )


if __name__ == "__main__":
    df = analyze_universe(DEFAULT_TICKERS_FAST, deep_options=True)
    cols = ["Ticker", "Precio", "Señal", "Technical Score %", "Options Score %", "Final Score %", "Smart Spread", "ROI/ROC estimado %", "Prob OTM estimada %"]
    print(df[[c for c in cols if c in df.columns]].to_string(index=False))
