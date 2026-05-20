"""
OptionsView PRO - GitHub Alerts PRO
Alertas automáticas para premium selling: IVR + Earnings Safety + Smart Spread Builder.

GitHub Secrets necesarios para Telegram:
- TELEGRAM_BOT_TOKEN
- TELEGRAM_CHAT_ID

Variables opcionales del workflow:
- TICKERS: "MCD,PEP,PG,COST,NVDA,AAPL,MSFT,V,KO,SBUX"
- MIN_SCORE: "60"
- MAX_ALERTS: "10"
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import erf, log, sqrt
from typing import Any

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# =========================
# CONFIGURACIÓN
# =========================

DEFAULT_TICKERS = "MCD,PEP,PG,COST,NVDA,AAPL,MSFT,V,KO,SBUX"
DEFAULT_PERIOD = "1y"
SUPPORT_RESISTANCE_WINDOW = 60
DEFAULT_PUT_RSI = 35
DEFAULT_CALL_RSI = 65
DEFAULT_DISTANCE_SUPPORT = 4.0
DEFAULT_DISTANCE_RESISTANCE = 5.0
DEFAULT_DTE_DAYS = 40
RISK_FREE_RATE = 0.045
CONTRACT_MULTIPLIER = 100

EARNINGS_HARD_BLOCK_DAYS = 7
EARNINGS_SOFT_BLOCK_DAYS = 14
MIN_IVR_FOR_PREMIUM_SELLING = 25
IDEAL_IVR_MIN = 40
MAX_RISK_PER_TRADE_USD = 900
MAX_CONTRACTS_HARD_CAP = 3

SMART_WIDTHS = [2.5, 5, 10]
SMART_MIN_PROB_OTM = 68
SMART_MIN_ROC = 8
SMART_MAX_BA_SPREAD = 30
SMART_MIN_OI = 50
SMART_MIN_CREDIT = 0.12
SMART_MAX_DELTA_ABS = 0.35
SMART_MIN_DELTA_ABS = 0.08

MIN_SCORE = int(os.getenv("MIN_SCORE", "60"))
MAX_ALERTS = int(os.getenv("MAX_ALERTS", "10"))


# =========================
# UTILIDADES
# =========================

def safe_float(value: Any) -> float:
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def fmt(value: Any, decimals: int = 1, suffix: str = "") -> str:
    value = safe_float(value)
    if pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}{suffix}"


def normal_cdf(x: float) -> float:
    return 0.5 * (1 + erf(x / sqrt(2)))


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_stochastic(data: pd.DataFrame, k_period: int = 14, d_period: int = 3):
    low_min = data["Low"].rolling(window=k_period).min()
    high_max = data["High"].rolling(window=k_period).max()
    k = 100 * ((data["Close"] - low_min) / (high_max - low_min))
    d = k.rolling(window=d_period).mean()
    return k, d


def download_data(ticker: str, period: str = DEFAULT_PERIOD, interval: str = "1d") -> pd.DataFrame:
    ticker = ticker.strip().upper()
    for attempt in range(3):
        try:
            data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True, threads=False)
            if data is not None and not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                return data.dropna(subset=["Open", "High", "Low", "Close"])
        except Exception:
            pass
        time.sleep(0.4 + attempt * 0.4)
    return pd.DataFrame()


def prepare_data(ticker: str, period: str = DEFAULT_PERIOD) -> pd.DataFrame:
    data = download_data(ticker, period)
    if data.empty:
        return data
    data["RSI"] = calculate_rsi(data["Close"])
    data["STO_K"], data["STO_D"] = calculate_stochastic(data)
    data["SMA20"] = data["Close"].rolling(20).mean()
    data["SMA50"] = data["Close"].rolling(50).mean()
    data["SMA200"] = data["Close"].rolling(200).mean()
    return data


def get_support_resistance(data: pd.DataFrame, window: int = SUPPORT_RESISTANCE_WINDOW):
    recent = data.tail(window)
    return safe_float(recent["Low"].min()), safe_float(recent["High"].max())


def classify_trend(price, sma20, sma50, sma200) -> str:
    if not pd.isna(sma200) and price > sma20 > sma50 > sma200:
        return "Alcista fuerte"
    if not pd.isna(sma200) and price < sma20 < sma50 < sma200:
        return "Bajista fuerte"
    if price > sma50 and (pd.isna(sma200) or price > sma200):
        return "Alcista"
    if price < sma50 and (pd.isna(sma200) or price < sma200):
        return "Bajista"
    return "Lateral / Mixta"


def calculate_hv_rank_proxy(data: pd.DataFrame):
    try:
        returns = data["Close"].pct_change().dropna()
        hv20 = returns.rolling(20).std().dropna() * sqrt(252) * 100
        if hv20.empty:
            return np.nan, np.nan
        current_hv = safe_float(hv20.iloc[-1])
        hv_min = safe_float(hv20.min())
        hv_max = safe_float(hv20.max())
        if pd.isna(current_hv) or hv_max == hv_min:
            return round(current_hv, 1), np.nan
        rank = (current_hv - hv_min) / (hv_max - hv_min) * 100
        return round(current_hv, 1), round(max(0, min(100, rank)), 1)
    except Exception:
        return np.nan, np.nan


def estimate_delta_probability(price, strike, iv, dte_days, option_type="put"):
    try:
        if pd.isna(price) or pd.isna(strike) or pd.isna(iv) or iv <= 0 or dte_days <= 0:
            return np.nan, np.nan
        t = dte_days / 365
        sigma = iv / 100
        d1 = (log(price / strike) + (RISK_FREE_RATE + 0.5 * sigma**2) * t) / (sigma * sqrt(t))
        d2 = d1 - sigma * sqrt(t)
        if option_type == "put":
            delta = normal_cdf(d1) - 1
            prob_otm = normal_cdf(d2)
        else:
            delta = normal_cdf(d1)
            prob_otm = normal_cdf(-d2)
        return round(delta, 2), round(prob_otm * 100, 1)
    except Exception:
        return np.nan, np.nan


def get_earnings_info(ticker: str):
    """Devuelve fecha de earnings aproximada desde yfinance y nivel de seguridad."""
    try:
        tk = yf.Ticker(ticker)
        today = pd.Timestamp.today().normalize()
        dates = []

        try:
            cal = tk.calendar
            if isinstance(cal, dict):
                ed = cal.get("Earnings Date")
                if isinstance(ed, (list, tuple)):
                    dates += [pd.Timestamp(x).tz_localize(None).normalize() for x in ed if x is not None]
                elif ed is not None:
                    dates.append(pd.Timestamp(ed).tz_localize(None).normalize())
            elif isinstance(cal, pd.DataFrame) and not cal.empty:
                for value in cal.values.flatten():
                    try:
                        dt = pd.Timestamp(value).tz_localize(None).normalize()
                        if today <= dt <= today + pd.Timedelta(days=120):
                            dates.append(dt)
                    except Exception:
                        pass
        except Exception:
            pass

        future = sorted([d for d in dates if d >= today])
        if not future:
            return "No disponible", np.nan, "Desconocido", 0

        edate = future[0]
        days = int((edate - today).days)
        if days <= EARNINGS_HARD_BLOCK_DAYS:
            return edate.strftime("%Y-%m-%d"), days, "Bloqueado", -35
        if days <= EARNINGS_SOFT_BLOCK_DAYS:
            return edate.strftime("%Y-%m-%d"), days, "Precaución", -18
        return edate.strftime("%Y-%m-%d"), days, "OK", 0
    except Exception:
        return "No disponible", np.nan, "Desconocido", 0


def option_chain_for_best_expiration(ticker: str, target_min=25, target_max=50):
    tk = yf.Ticker(ticker)
    expirations = list(tk.options)
    if not expirations:
        return None, None, None
    today = pd.Timestamp.today().normalize()
    candidates = []
    for exp in expirations:
        dte = (pd.Timestamp(exp) - today).days
        if target_min <= dte <= target_max:
            try:
                chain = tk.option_chain(exp)
                oi = pd.to_numeric(chain.calls.get("openInterest", 0), errors="coerce").fillna(0).sum()
                oi += pd.to_numeric(chain.puts.get("openInterest", 0), errors="coerce").fillna(0).sum()
                candidates.append((exp, dte, oi))
            except Exception:
                candidates.append((exp, dte, 0))
    if not candidates:
        for exp in expirations:
            dte = (pd.Timestamp(exp) - today).days
            if dte > 15:
                candidates.append((exp, dte, 0))
    if not candidates:
        return None, None, None
    exp, dte, _ = sorted(candidates, key=lambda x: (x[2], -abs(x[1] - DEFAULT_DTE_DAYS)), reverse=True)[0]
    return exp, dte, tk.option_chain(exp)


def clean_chain(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Strike"] = pd.to_numeric(out.get("strike"), errors="coerce")
    out["Bid"] = pd.to_numeric(out.get("bid"), errors="coerce")
    out["Ask"] = pd.to_numeric(out.get("ask"), errors="coerce")
    out["Mid"] = ((out["Bid"] + out["Ask"]) / 2).round(2)
    out["Spread %"] = np.where(out["Mid"] > 0, ((out["Ask"] - out["Bid"]) / out["Mid"] * 100).round(1), np.nan)
    out["IV %"] = (pd.to_numeric(out.get("impliedVolatility"), errors="coerce") * 100).round(1)
    out["OI"] = pd.to_numeric(out.get("openInterest"), errors="coerce")
    out["Vol"] = pd.to_numeric(out.get("volume"), errors="coerce")
    return out[["Strike", "Bid", "Ask", "Mid", "Spread %", "IV %", "OI", "Vol"]].dropna(subset=["Strike"])


def liquidity_label(oi, spread_pct):
    oi = safe_float(oi)
    spread_pct = safe_float(spread_pct)
    if not pd.isna(oi) and not pd.isna(spread_pct) and oi >= 100 and spread_pct <= 15:
        return "Alta"
    if not pd.isna(oi) and not pd.isna(spread_pct) and oi >= 50 and spread_pct <= 30:
        return "Media"
    return "Baja"


def smart_trade_builder(ticker: str, price: float, signal: str, support=np.nan, resistance=np.nan):
    """Busca el mejor spread vertical real por seguridad, ROC, probabilidad y liquidez."""
    empty = {
        "Smart Spread": "Sin spread", "Smart Short": np.nan, "Smart Long": np.nan,
        "Smart Credit": np.nan, "Smart ROC %": np.nan, "Smart Prob OTM %": np.nan,
        "Smart Delta": np.nan, "Smart Max Loss": np.nan, "Smart Liquidez": "No disponible",
        "Smart Exp": "", "Smart DTE": np.nan, "Smart Score": 0, "Smart Razón": "Sin cadena"
    }
    if signal not in ["PUT / Bull Put Spread", "CALL Spread"]:
        empty["Smart Razón"] = "Sin señal técnica"
        return empty

    try:
        exp, dte, chain = option_chain_for_best_expiration(ticker)
        if chain is None:
            return empty
        side = "PUT" if signal == "PUT / Bull Put Spread" else "CALL"
        df = clean_chain(chain.puts if side == "PUT" else chain.calls)
        if df.empty:
            return empty

        candidates = []
        for width in SMART_WIDTHS:
            for _, short in df.iterrows():
                short_strike = safe_float(short["Strike"])
                if side == "PUT":
                    if short_strike >= price:
                        continue
                    long_target = short_strike - width
                    if not pd.isna(support) and short_strike > support * 1.015:
                        continue
                    option_type = "put"
                else:
                    if short_strike <= price:
                        continue
                    long_target = short_strike + width
                    if not pd.isna(resistance) and short_strike < resistance * 0.985:
                        continue
                    option_type = "call"

                long = df.iloc[(df["Strike"] - long_target).abs().argsort()[:1]].iloc[0]
                actual_width = abs(short_strike - safe_float(long["Strike"]))
                if actual_width <= 0:
                    continue

                credit = safe_float(short["Bid"]) - safe_float(long["Ask"])
                if pd.isna(credit) or credit < SMART_MIN_CREDIT:
                    continue

                iv = safe_float(short["IV %"])
                delta, prob_otm = estimate_delta_probability(price, short_strike, iv, dte, option_type)
                delta_abs = abs(safe_float(delta))
                if pd.isna(delta_abs) or delta_abs < SMART_MIN_DELTA_ABS or delta_abs > SMART_MAX_DELTA_ABS:
                    continue

                max_profit = credit * CONTRACT_MULTIPLIER
                max_loss = (actual_width - credit) * CONTRACT_MULTIPLIER
                if max_loss <= 0:
                    continue
                roc = max_profit / max_loss * 100
                if roc < SMART_MIN_ROC or prob_otm < SMART_MIN_PROB_OTM:
                    continue

                oi = safe_float(short["OI"])
                ba = safe_float(short["Spread %"])
                if (not pd.isna(oi) and oi < SMART_MIN_OI) or (not pd.isna(ba) and ba > SMART_MAX_BA_SPREAD):
                    continue

                expected_move = price * (iv / 100) * sqrt(dte / 365) if not pd.isna(iv) else np.nan
                outside_em_bonus = 0
                if not pd.isna(expected_move):
                    if side == "PUT" and short_strike < price - expected_move:
                        outside_em_bonus = 8
                    elif side == "CALL" and short_strike > price + expected_move:
                        outside_em_bonus = 8

                liq = liquidity_label(oi, ba)
                liq_score = 15 if liq == "Alta" else 8 if liq == "Media" else -10
                prob_score = min(25, max(0, (prob_otm - 65) * 1.2))
                roc_score = min(22, max(0, roc * 0.8))
                delta_score = 12 if 0.12 <= delta_abs <= 0.25 else 6
                width_score = 5 if actual_width <= 5 else 2
                smart_score = liq_score + prob_score + roc_score + delta_score + width_score + outside_em_bonus

                candidates.append({
                    "Smart Spread": f"{ticker} {int(short_strike) if short_strike.is_integer() else short_strike}/{int(safe_float(long['Strike'])) if safe_float(long['Strike']).is_integer() else safe_float(long['Strike'])} {'PCS' if side == 'PUT' else 'CCS'}",
                    "Smart Short": short_strike,
                    "Smart Long": safe_float(long["Strike"]),
                    "Smart Credit": round(credit, 2),
                    "Smart ROC %": round(roc, 1),
                    "Smart Prob OTM %": prob_otm,
                    "Smart Delta": delta,
                    "Smart Max Loss": round(max_loss, 2),
                    "Smart Liquidez": liq,
                    "Smart Exp": exp,
                    "Smart DTE": dte,
                    "Smart Score": round(smart_score, 1),
                    "Smart Razón": "OK"
                })

        if not candidates:
            empty["Smart Razón"] = "No cumple liquidez/ROC/probabilidad"
            return empty
        return sorted(candidates, key=lambda x: (x["Smart Score"], x["Smart Prob OTM %"], x["Smart ROC %"]), reverse=True)[0]
    except Exception as e:
        empty["Smart Razón"] = f"Error: {e}"
        return empty


def calculate_score(signal, rsi, sto_k, dist_support, dist_resistance, trend, price, sma20, sma50, sma200, iv_rank, earnings_penalty):
    if signal == "NO TRADE":
        return 0
    score = 45
    bullish = trend in ["Alcista", "Alcista fuerte"]
    bearish = trend in ["Bajista", "Bajista fuerte"]

    if signal == "PUT / Bull Put Spread":
        if not pd.isna(sma200): score += 18 if price > sma200 else -18
        score += 12 if bullish else -12 if bearish else 0
        if dist_support <= 1.5: score += 16
        elif dist_support <= 3: score += 12
        elif dist_support <= 4: score += 8
        elif dist_support > 8: score -= 10
        if rsi <= 30: score += 8
        elif rsi <= 35: score += 5
        elif rsi >= 55: score -= 6
        if sto_k <= 20: score += 3
        elif sto_k <= 35: score += 1
        elif sto_k >= 80: score -= 3

    if signal == "CALL Spread":
        if not pd.isna(sma200): score += 18 if price < sma200 else -10
        score += 12 if bearish else -10 if bullish else 0
        if dist_resistance <= 1.5: score += 16
        elif dist_resistance <= 4: score += 12
        elif dist_resistance <= 5: score += 8
        elif dist_resistance > 9: score -= 10
        if rsi >= 70: score += 8
        elif rsi >= 65: score += 5
        elif rsi <= 45: score -= 6
        if sto_k >= 80: score += 3
        elif sto_k >= 65: score += 1
        elif sto_k <= 20: score -= 3

    if not pd.isna(iv_rank):
        if iv_rank >= 80: score += 6
        elif iv_rank >= 60: score += 15
        elif iv_rank >= 40: score += 10
        elif iv_rank >= 25: score += 3
        else: score -= 14

    score += earnings_penalty
    return int(max(0, min(100, round(score))))


def risk_filter(signal, price, sma20, sma50, sma200, rsi):
    reasons = []
    if signal == "PUT / Bull Put Spread":
        if not pd.isna(sma200) and price < sma200: reasons.append("Precio bajo SMA200")
        if not pd.isna(sma20) and not pd.isna(sma50) and sma20 < sma50: reasons.append("SMA20 bajo SMA50")
        if rsi < 25: reasons.append("RSI extremadamente bajo")
    elif signal == "CALL Spread":
        if not pd.isna(sma20) and not pd.isna(sma50) and not pd.isna(sma200) and price > sma20 > sma50 > sma200:
            reasons.append("Tendencia alcista fuerte")
        if rsi > 78: reasons.append("RSI extremadamente alto")
    level = "Alto" if len(reasons) >= 3 else "Medio" if reasons else "Bajo"
    return level, reasons


def position_recommendation(score, risk, iv_rank=np.nan, earnings_risk="Desconocido", prob_otm=np.nan, max_loss=np.nan):
    contracts = 0
    if score >= 50 and score < 75:
        contracts = 1
    elif score >= 75 and score < 85:
        contracts = 2
    elif score >= 85:
        contracts = 3

    if not pd.isna(iv_rank) and iv_rank < MIN_IVR_FOR_PREMIUM_SELLING:
        contracts = max(0, contracts - 1)
    if not pd.isna(prob_otm) and prob_otm < 70:
        contracts = max(0, contracts - 1)
    if risk == "Alto":
        contracts = max(0, contracts - 1)
    if earnings_risk == "Precaución":
        contracts = max(0, contracts - 1)
    if earnings_risk == "Bloqueado":
        contracts = 0
    if not pd.isna(max_loss) and max_loss > 0:
        contracts = min(contracts, int(MAX_RISK_PER_TRADE_USD // max_loss))
    contracts = max(0, min(MAX_CONTRACTS_HARD_CAP, contracts))
    return contracts


def analyze_ticker(ticker: str):
    data = prepare_data(ticker)
    if data.empty or len(data) < 80:
        return None
    latest = data.iloc[-1]
    prev = data.iloc[-2]
    price = safe_float(latest["Close"])
    prev_price = safe_float(prev["Close"])
    change_pct = (price - prev_price) / prev_price * 100 if prev_price else np.nan
    rsi = safe_float(latest["RSI"])
    sto_k = safe_float(latest["STO_K"])
    sma20 = safe_float(latest["SMA20"])
    sma50 = safe_float(latest["SMA50"])
    sma200 = safe_float(latest["SMA200"])
    support, resistance = get_support_resistance(data)
    dist_support = ((price - support) / support) * 100 if support else np.nan
    dist_resistance = ((resistance - price) / price) * 100 if price else np.nan
    trend = classify_trend(price, sma20, sma50, sma200)
    hv20, hv_rank = calculate_hv_rank_proxy(data)

    put_setup = rsi <= DEFAULT_PUT_RSI and dist_support <= DEFAULT_DISTANCE_SUPPORT
    call_setup = rsi >= DEFAULT_CALL_RSI and dist_resistance <= DEFAULT_DISTANCE_RESISTANCE
    signal = "NO TRADE"
    if put_setup and not call_setup:
        signal = "PUT / Bull Put Spread"
    elif call_setup and not put_setup:
        signal = "CALL Spread"

    earnings_date, days_to_earnings, earnings_risk, earnings_penalty = get_earnings_info(ticker)
    score = calculate_score(signal, rsi, sto_k, dist_support, dist_resistance, trend, price, sma20, sma50, sma200, hv_rank, earnings_penalty)
    risk, reasons = risk_filter(signal, price, sma20, sma50, sma200, rsi)
    smart = smart_trade_builder(ticker, price, signal, support, resistance)

    # Ajuste final: si hay Smart Builder válido, usar su probabilidad/riesgo para contratos.
    contracts = position_recommendation(
        score,
        risk,
        hv_rank,
        earnings_risk,
        smart.get("Smart Prob OTM %", np.nan),
        smart.get("Smart Max Loss", np.nan),
    )

    # Alert quality: solo mandar si hay setup ejecutable.
    executable = (
        signal != "NO TRADE"
        and score >= MIN_SCORE
        and contracts > 0
        and smart.get("Smart Spread") != "Sin spread"
        and smart.get("Smart Liquidez") in ["Alta", "Media"]
        and earnings_risk != "Bloqueado"
    )

    return {
        "Ticker": ticker,
        "Precio": round(price, 2),
        "%": round(change_pct, 2),
        "Señal": signal,
        "Score": score,
        "Riesgo": risk,
        "RSI": round(rsi, 1),
        "Trend": trend,
        "HV 20D %": hv20,
        "IVR/HVR %": hv_rank,
        "Earnings": earnings_date,
        "Días earnings": days_to_earnings,
        "Earnings Safety": earnings_risk,
        "Contratos": contracts,
        "Razones riesgo": "; ".join(reasons),
        "Ejecutable": executable,
        **smart,
    }


def alert_line(row: dict) -> str:
    side = "PUT" if row["Señal"] == "PUT / Bull Put Spread" else "CALL"
    em = row.get("Earnings Safety", "N/A")
    earn = row.get("Earnings", "N/A")
    return (
        f"🚨 *{row['Ticker']}* {row['Score']}% | {row.get('Smart Spread')}\n"
        f"Tipo: {side} | Contratos máx: *{row['Contratos']}* | Riesgo: {row['Riesgo']}\n"
        f"ROC: {fmt(row.get('Smart ROC %'), 1, '%')} | Prob OTM: {fmt(row.get('Smart Prob OTM %'), 1, '%')} | Delta: {fmt(row.get('Smart Delta'), 2)}\n"
        f"IVR/HVR: {fmt(row.get('IVR/HVR %'), 1, '%')} | Liquidez: {row.get('Smart Liquidez')} | Crédito: ${fmt(row.get('Smart Credit'), 2)}\n"
        f"Exp: {row.get('Smart Exp')} ({fmt(row.get('Smart DTE'), 0)} DTE) | Earnings: {em} ({earn})\n"
        f"Precio: ${fmt(row.get('Precio'), 2)} | RSI: {fmt(row.get('RSI'), 1)}"
    )


def send_telegram(message: str):
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        print("Telegram no configurado. Mensaje generado:\n")
        print(message)
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    r = requests.post(url, json=payload, timeout=20)
    r.raise_for_status()


def main():
    tickers_text = os.getenv("TICKERS", DEFAULT_TICKERS)
    tickers = [t.strip().upper() for t in tickers_text.replace(";", ",").split(",") if t.strip()]
    print(f"Escaneando {len(tickers)} tickers: {', '.join(tickers)}")

    rows = []
    for ticker in tickers:
        try:
            row = analyze_ticker(ticker)
            if row:
                rows.append(row)
                print(f"{ticker}: score={row['Score']} executable={row['Ejecutable']} spread={row.get('Smart Spread')}")
        except Exception as e:
            print(f"{ticker}: error {e}")
        time.sleep(0.5)

    if not rows:
        send_telegram("OptionsView PRO: sin datos suficientes hoy.")
        return

    df = pd.DataFrame(rows)
    executable = df[df["Ejecutable"] == True].copy()
    if executable.empty:
        summary = (
            f"✅ OptionsView PRO escaneado · {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            f"No hay setups ejecutables con score ≥ {MIN_SCORE}.\n"
            f"Mejor candidato: {df.sort_values('Score', ascending=False).iloc[0]['Ticker']} "
            f"{int(df.sort_values('Score', ascending=False).iloc[0]['Score'])}%"
        )
        send_telegram(summary)
        return

    executable = executable.sort_values(["Score", "Smart Score", "Smart Prob OTM %"], ascending=False).head(MAX_ALERTS)
    header = f"🔥 *OptionsView PRO Alerts* · {datetime.now().strftime('%Y-%m-%d %H:%M')}\nSetups ejecutables: {len(executable)}\n\n"
    message = header + "\n\n".join(alert_line(row) for _, row in executable.iterrows())
    send_telegram(message)

    # Guarda CSV como artifact en GitHub Actions
    df.to_csv("optionsview_alerts_scan.csv", index=False)
    print("Guardado optionsview_alerts_scan.csv")


if __name__ == "__main__":
    main()
