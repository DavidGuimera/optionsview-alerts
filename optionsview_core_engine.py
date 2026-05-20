"""
OptionsView Core Engine v2026.05.20-unified-2
Unico motor de calculo para APP y GitHub Alerts.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from math import erf, log, sqrt
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

CORE_VERSION = "2026.05.20-unified-2"

DEFAULT_TICKERS_FAST = "MCD,PEP,PG,COST,SBUX,MSFT,AAPL,GOOGL,JPM,V,SPY,QQQ"
FULL_TICKERS = "MCD,PEP,PG,KO,JNJ,WMT,COST,HD,LOW,TGT,SBUX,MDLZ,CMCSA,MSFT,AAPL,GOOGL,META,AMZN,NVDA,AVGO,ADBE,CRM,JPM,MA,V,BLK,SCHW,SPY,QQQ,IWM,XLP,XLV,XLF"
CONTRACT_MULTIPLIER = 100
RISK_FREE_RATE = 0.045
TARGET_DTE = int(os.getenv("TARGET_DTE", "35"))


def safe_float(x: Any, default: float = np.nan) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, (pd.Series, pd.DataFrame)):
            return default
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return default
        return v
    except Exception:
        return default


def normalize_ticker(ticker: str) -> str:
    return str(ticker).strip().upper().replace(" ", "")


def parse_tickers(text: str) -> List[str]:
    text = text or DEFAULT_TICKERS_FAST
    if text.strip().upper() in ["FAVORITAS", "FAVORITES", "FULL"]:
        text = FULL_TICKERS
    out = []
    seen = set()
    for t in text.replace(";", ",").split(","):
        t = normalize_ticker(t)
        if t and t not in seen:
            out.append(t)
            seen.add(t)
    return out


def download_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Descarga segura ticker por ticker. Evita mezclar columnas entre tickers."""
    ticker = normalize_ticker(ticker)
    for attempt in range(3):
        try:
            tk = yf.Ticker(ticker)
            data = tk.history(period=period, interval="1d", auto_adjust=True)
            if data is None or data.empty:
                raise ValueError("empty history")
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]
            data = data.loc[:, ~data.columns.duplicated()].copy()
            needed = ["Open", "High", "Low", "Close"]
            if not all(c in data.columns for c in needed):
                raise ValueError("missing OHLC")
            data = data.dropna(subset=needed)
            if len(data) < 80:
                raise ValueError("not enough rows")
            for c in ["Open", "High", "Low", "Close", "Volume"]:
                if c in data.columns:
                    data[c] = pd.to_numeric(data[c], errors="coerce")
            data = data.dropna(subset=needed)
            return data
        except Exception:
            time.sleep(0.25 + attempt * 0.25)
    return pd.DataFrame()


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    close = pd.to_numeric(close, errors="coerce")
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def prepare_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    data = download_data(ticker, period)
    if data.empty:
        return data
    close = data["Close"]
    data["RSI"] = calculate_rsi(close)
    data["SMA20"] = close.rolling(20).mean()
    data["SMA50"] = close.rolling(50).mean()
    data["SMA200"] = close.rolling(200).mean()
    data["HV20"] = close.pct_change().rolling(20).std() * np.sqrt(252) * 100
    return data


def support_resistance(data: pd.DataFrame, window: int = 60) -> Tuple[float, float]:
    recent = data.tail(window)
    return safe_float(recent["Low"].min()), safe_float(recent["High"].max())


def hv_rank(data: pd.DataFrame) -> Tuple[float, float]:
    hv = pd.to_numeric(data.get("HV20", pd.Series(dtype=float)), errors="coerce").dropna()
    if hv.empty:
        return np.nan, np.nan
    current = safe_float(hv.iloc[-1])
    mn = safe_float(hv.min())
    mx = safe_float(hv.max())
    if np.isnan(current) or np.isnan(mn) or np.isnan(mx) or mx == mn:
        return round(current, 1), np.nan
    rank = max(0, min(100, (current - mn) / (mx - mn) * 100))
    return round(current, 1), round(rank, 1)


def next_earnings(ticker: str) -> Tuple[str, int, str]:
    try:
        tk = yf.Ticker(ticker)
        cal = getattr(tk, "calendar", None)
        dt = None
        if isinstance(cal, dict):
            val = cal.get("Earnings Date") or cal.get("EarningsDate")
            if isinstance(val, list) and val:
                dt = val[0]
            elif val is not None:
                dt = val
        if dt is None:
            edf = tk.get_earnings_dates(limit=4)
            if edf is not None and not edf.empty:
                for idx in edf.index:
                    ts = pd.Timestamp(idx).tz_localize(None) if getattr(pd.Timestamp(idx), 'tzinfo', None) else pd.Timestamp(idx)
                    if ts.date() >= datetime.utcnow().date():
                        dt = ts
                        break
        if dt is None:
            return "No disponible", 999, "UNKNOWN"
        ts = pd.Timestamp(dt).tz_localize(None) if getattr(pd.Timestamp(dt), 'tzinfo', None) else pd.Timestamp(dt)
        days = (ts.normalize() - pd.Timestamp.utcnow().tz_localize(None).normalize()).days
        status = "DANGER" if days <= 10 else "CAUTION" if days <= 21 else "OK"
        return str(ts.date()), int(days), status
    except Exception:
        return "No disponible", 999, "UNKNOWN"


def normal_cdf(x: float) -> float:
    return 0.5 * (1 + erf(x / sqrt(2)))


def delta_prob(price: float, strike: float, iv_pct: float, dte: int, side: str) -> Tuple[float, float]:
    try:
        if price <= 0 or strike <= 0 or iv_pct <= 0 or dte <= 0:
            return np.nan, np.nan
        t = dte / 365
        sigma = iv_pct / 100
        d1 = (log(price / strike) + (RISK_FREE_RATE + 0.5 * sigma**2) * t) / (sigma * sqrt(t))
        d2 = d1 - sigma * sqrt(t)
        if side == "PUT":
            return round(normal_cdf(d1) - 1, 2), round(normal_cdf(d2) * 100, 1)
        return round(normal_cdf(d1), 2), round(normal_cdf(-d2) * 100, 1)
    except Exception:
        return np.nan, np.nan


def nearest_expiration(ticker: str, target_dte: int = TARGET_DTE) -> Tuple[Optional[str], int]:
    try:
        exps = list(yf.Ticker(ticker).options)
        if not exps:
            return None, 0
        today = pd.Timestamp.utcnow().tz_localize(None).normalize()
        candidates = []
        for exp in exps:
            dte = (pd.Timestamp(exp) - today).days
            if dte >= 20:
                candidates.append((exp, dte, abs(dte - target_dte)))
        if not candidates:
            exp = exps[0]
            dte = (pd.Timestamp(exp) - today).days
            return exp, int(dte)
        exp, dte, _ = sorted(candidates, key=lambda x: x[2])[0]
        return exp, int(dte)
    except Exception:
        return None, 0


def step_for_price(price: float) -> float:
    if price >= 500:
        return 5.0
    if price >= 200:
        return 5.0
    if price >= 100:
        return 2.5
    if price >= 50:
        return 1.0
    return 0.5


def round_strike(x: float, price: float) -> float:
    step = step_for_price(price)
    return round(x / step) * step


def format_strike(x: Any) -> str:
    v = safe_float(x)
    if np.isnan(v):
        return "-"
    return str(int(v)) if float(v).is_integer() else str(v)


def choose_signal(rsi: float, price: float, support: float, resistance: float) -> Tuple[str, str]:
    dist_support = ((price - support) / support) * 100 if support else np.nan
    dist_resistance = ((resistance - price) / price) * 100 if price else np.nan
    put_setup = rsi <= 35 and (not np.isnan(dist_support) and dist_support <= 5)
    call_setup = rsi >= 65 and (not np.isnan(dist_resistance) and dist_resistance <= 7)
    if call_setup and not put_setup:
        return "CALL", "CALL / Bear Call Spread"
    if put_setup and not call_setup:
        return "PUT", "PUT / Bull Put Spread"
    return "NONE", "NO TRADE"


def technical_score(side: str, rsi: float, price: float, sma20: float, sma50: float, sma200: float, support: float, resistance: float) -> int:
    if side == "NONE":
        return 0
    score = 50
    if side == "CALL":
        if rsi >= 75: score += 16
        elif rsi >= 70: score += 12
        elif rsi >= 65: score += 8
        if not np.isnan(resistance):
            dist = ((resistance - price) / price) * 100
            if dist <= 2: score += 10
            elif dist <= 5: score += 7
            elif dist <= 7: score += 4
        if not np.isnan(sma20) and price < sma20: score += 5
        if not np.isnan(sma50) and price < sma50: score += 5
        if not np.isnan(sma200) and price > sma200: score -= 6
    elif side == "PUT":
        if rsi <= 25: score += 16
        elif rsi <= 30: score += 12
        elif rsi <= 35: score += 8
        if not np.isnan(support):
            dist = ((price - support) / support) * 100
            if dist <= 2: score += 10
            elif dist <= 4: score += 7
            elif dist <= 5: score += 4
        if not np.isnan(sma20) and price > sma20: score += 5
        if not np.isnan(sma50) and price > sma50: score += 5
        if not np.isnan(sma200) and price < sma200: score -= 6
    return int(max(0, min(100, round(score))))


def clean_chain(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    for c in ["strike", "bid", "ask", "impliedVolatility", "volume", "openInterest"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["strike"])
    return out


def option_metrics(ticker: str, side: str, price: float, support: float, resistance: float) -> Dict[str, Any]:
    base = {
        "options_available": False, "expiration": "-", "dte": 0, "spread": "-", "short_strike": np.nan,
        "long_strike": np.nan, "credit": np.nan, "roc": np.nan, "prob_otm": np.nan, "delta": np.nan,
        "iv": np.nan, "oi": np.nan, "volume": np.nan, "bidask": np.nan, "liquidity": "No disponible",
        "options_score": np.nan,
    }
    if side == "NONE":
        base["options_score"] = 0
        return base
    try:
        exp, dte = nearest_expiration(ticker)
        if not exp:
            return base
        tk = yf.Ticker(ticker)
        chain = tk.option_chain(exp)
        df = clean_chain(chain.calls if side == "CALL" else chain.puts)
        if df.empty:
            return base
        width = 5.0 if price >= 100 else 2.5
        if side == "CALL":
            target_short = round_strike(max(resistance if not np.isnan(resistance) else price * 1.05, price * 1.04), price)
            target_long = target_short + width
        else:
            target_short = round_strike(min(support if not np.isnan(support) else price * 0.95, price * 0.96), price)
            target_long = target_short - width
        short = df.iloc[(df["strike"] - target_short).abs().argsort()[:1]].iloc[0]
        long = df.iloc[(df["strike"] - target_long).abs().argsort()[:1]].iloc[0]
        short_strike = safe_float(short["strike"])
        long_strike = safe_float(long["strike"])
        spread_width = abs(short_strike - long_strike)
        credit = max(0, safe_float(short.get("bid")) - safe_float(long.get("ask")))
        max_loss = max(0.01, spread_width - credit)
        roc = (credit / max_loss) * 100 if credit > 0 else 0
        iv = safe_float(short.get("impliedVolatility")) * 100
        oi = safe_float(short.get("openInterest"))
        vol = safe_float(short.get("volume"))
        bid = safe_float(short.get("bid")); ask = safe_float(short.get("ask"))
        mid = (bid + ask) / 2 if ask > 0 and bid >= 0 else np.nan
        bidask = ((ask - bid) / mid) * 100 if not np.isnan(mid) and mid > 0 else np.nan
        delta, prob_otm = delta_prob(price, short_strike, iv, dte, side)
        liquidity = "Alta" if oi >= 100 and (np.isnan(bidask) or bidask <= 20) else "Media" if oi >= 50 else "Baja"

        oscore = 50
        if prob_otm >= 85: oscore += 16
        elif prob_otm >= 75: oscore += 10
        elif prob_otm >= 65: oscore += 4
        elif not np.isnan(prob_otm): oscore -= 10
        if 8 <= roc <= 25: oscore += 10
        elif roc > 25: oscore += 4
        elif roc > 0: oscore += 2
        if liquidity == "Alta": oscore += 10
        elif liquidity == "Media": oscore += 4
        else: oscore -= 8
        if 20 <= iv <= 80: oscore += 5
        if side == "CALL" and not np.isnan(delta):
            if 0.12 <= delta <= 0.28: oscore += 8
            elif delta > 0.35: oscore -= 12
        if side == "PUT" and not np.isnan(delta):
            ad = abs(delta)
            if 0.12 <= ad <= 0.28: oscore += 8
            elif ad > 0.35: oscore -= 12
        oscore = int(max(0, min(100, round(oscore))))
        base.update({
            "options_available": True,
            "expiration": exp,
            "dte": dte,
            "spread": f"{format_strike(short_strike)}/{format_strike(long_strike)} {'CCS' if side == 'CALL' else 'PCS'}",
            "short_strike": short_strike,
            "long_strike": long_strike,
            "credit": round(credit, 2),
            "roc": round(roc, 1),
            "prob_otm": round(prob_otm, 1) if not np.isnan(prob_otm) else np.nan,
            "delta": round(delta, 2) if not np.isnan(delta) else np.nan,
            "iv": round(iv, 1) if not np.isnan(iv) else np.nan,
            "oi": int(oi) if not np.isnan(oi) else np.nan,
            "volume": int(vol) if not np.isnan(vol) else np.nan,
            "bidask": round(bidask, 1) if not np.isnan(bidask) else np.nan,
            "liquidity": liquidity,
            "options_score": oscore,
        })
        return base
    except Exception as e:
        base["options_error"] = str(e)
        return base


def position_size(final_score: int, earnings_status: str, liquidity: str) -> Tuple[int, str]:
    if earnings_status == "DANGER" or final_score < 60:
        return 0, "NO TRADE"
    if liquidity == "Baja":
        return 0, "NO TRADE por liquidez"
    if 60 <= final_score < 75:
        return 1, "1 contrato max"
    if 75 <= final_score < 85:
        return 2, "2 contratos max"
    return 2, "2 contratos max + revisar manualmente"


def analyze_ticker(ticker: str, period: str = "1y", deep_options: bool = True) -> Dict[str, Any]:
    ticker = normalize_ticker(ticker)
    data = prepare_data(ticker, period)
    if data.empty:
        return {"ticker": ticker, "data_status": "ERROR", "final_score": 0, "signal": "ERROR", "price": np.nan, "core_version": CORE_VERSION}
    latest = data.iloc[-1]
    prev = data.iloc[-2]
    price = safe_float(latest["Close"])
    change = price - safe_float(prev["Close"])
    change_pct = change / safe_float(prev["Close"]) * 100 if safe_float(prev["Close"]) else np.nan
    rsi = safe_float(latest.get("RSI"))
    sma20 = safe_float(latest.get("SMA20")); sma50 = safe_float(latest.get("SMA50")); sma200 = safe_float(latest.get("SMA200"))
    support, resistance = support_resistance(data)
    hv, ivr = hv_rank(data)
    side, signal = choose_signal(rsi, price, support, resistance)
    tscore = technical_score(side, rsi, price, sma20, sma50, sma200, support, resistance)
    earnings_date, earnings_days, earnings_status = next_earnings(ticker)
    om = option_metrics(ticker, side, price, support, resistance) if deep_options else {"options_score": np.nan, "options_available": False, "liquidity": "No disponible", "spread": "-"}
    oscore = safe_float(om.get("options_score"))
    if np.isnan(oscore):
        # Si no hay opciones, no tiramos el score a cero: usamos técnico con descuento.
        final = int(round(tscore * 0.85)) if side != "NONE" else 0
    else:
        final = int(round(tscore * 0.58 + oscore * 0.42)) if side != "NONE" else 0
    # IV/HV rank como ajuste suave, no destructivo.
    if side != "NONE" and not np.isnan(ivr):
        if ivr >= 60: final += 5
        elif ivr < 20: final -= 5
    if earnings_status == "DANGER": final -= 25
    elif earnings_status == "CAUTION": final -= 10
    final = int(max(0, min(100, final)))
    contracts, sizing = position_size(final, earnings_status, om.get("liquidity", "No disponible"))
    return {
        "ticker": ticker,
        "core_version": CORE_VERSION,
        "data_status": "OK",
        "price": round(price, 2),
        "change": round(change, 2),
        "change_pct": round(change_pct, 2) if not np.isnan(change_pct) else np.nan,
        "rsi": round(rsi, 1) if not np.isnan(rsi) else np.nan,
        "sma20": round(sma20, 2) if not np.isnan(sma20) else np.nan,
        "sma50": round(sma50, 2) if not np.isnan(sma50) else np.nan,
        "sma200": round(sma200, 2) if not np.isnan(sma200) else np.nan,
        "support": round(support, 2) if not np.isnan(support) else np.nan,
        "resistance": round(resistance, 2) if not np.isnan(resistance) else np.nan,
        "hv20": hv,
        "ivr": ivr,
        "side": side,
        "signal": signal,
        "technical_score": tscore,
        "options_score": round(oscore, 1) if not np.isnan(oscore) else np.nan,
        "final_score": final,
        "earnings_date": earnings_date,
        "earnings_days": earnings_days,
        "earnings_status": earnings_status,
        "contracts": contracts,
        "sizing": sizing,
        **om,
    }


def scan_tickers(tickers: List[str], period: str = "1y", deep_options: bool = True) -> List[Dict[str, Any]]:
    rows = []
    for t in tickers:
        rows.append(analyze_ticker(t, period=period, deep_options=deep_options))
        time.sleep(0.2)
    return rows
