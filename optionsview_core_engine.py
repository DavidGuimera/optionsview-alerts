"""
OptionsView Core Engine v4 - EXECUTABLE ALERTS ONLY
- Shared engine for GitHub alerts.
- Sends only setups with valid option-chain data.
- Rejects fake/fallback values: credit 0, ROC 0, Delta 0, Prob OTM 100, missing OI, bad bid/ask.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from math import erf, log, sqrt
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import yfinance as yf

CORE_VERSION = "2026.05.20-executable-v4"
DEFAULT_TICKERS = "MCD,PEP,PG,KO,JNJ,WMT,COST,HD,LOW,TGT,SBUX,MDLZ,CMCSA,MSFT,AAPL,GOOGL,META,AMZN,NVDA,AVGO,ADBE,CRM,JPM,MA,V,BLK,SCHW,SPY,QQQ,IWM,XLP,XLV,XLF"
CONTRACT_MULTIPLIER = 100
RISK_FREE_RATE = 0.045
TARGET_DTE = 35
MIN_DTE = 25
MAX_DTE = 50
SPREAD_WIDTHS = [2.5, 5, 10]


def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return default
        return v
    except Exception:
        return default


def normal_cdf(x: float) -> float:
    return 0.5 * (1 + erf(x / sqrt(2)))


def download_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Ticker-by-ticker only. Avoids yfinance multi-ticker contamination."""
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period=period, interval="1d", auto_adjust=True)
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]
        needed = ["Open", "High", "Low", "Close"]
        if not all(c in df.columns for c in needed):
            return pd.DataFrame()
        df = df.dropna(subset=needed)
        return df
    except Exception:
        return pd.DataFrame()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_iv_rank_proxy(hist: pd.DataFrame) -> float:
    try:
        returns = hist["Close"].pct_change().dropna()
        hv20 = returns.rolling(20).std() * np.sqrt(252) * 100
        hv20 = hv20.dropna()
        if len(hv20) < 30:
            return np.nan
        cur = safe_float(hv20.iloc[-1])
        mn = safe_float(hv20.min())
        mx = safe_float(hv20.max())
        if np.isnan(cur) or np.isnan(mn) or np.isnan(mx) or mx <= mn:
            return np.nan
        return round(max(0, min(100, (cur - mn) / (mx - mn) * 100)), 1)
    except Exception:
        return np.nan


def get_next_earnings(ticker: str) -> tuple[str, Optional[int]]:
    try:
        tk = yf.Ticker(ticker)
        cal = getattr(tk, "calendar", None)
        raw = None
        if isinstance(cal, dict):
            raw = cal.get("Earnings Date") or cal.get("EarningsDate")
        elif isinstance(cal, pd.DataFrame) and not cal.empty:
            for idx in cal.index:
                if "earn" in str(idx).lower():
                    raw = cal.loc[idx].iloc[0]
                    break
        if isinstance(raw, (list, tuple)) and raw:
            raw = raw[0]
        if raw is None or str(raw).lower() in ["nan", "nat", "none"]:
            return "No disponible", None
        dt = pd.to_datetime(raw, errors="coerce")
        if pd.isna(dt):
            return str(raw), None
        today = pd.Timestamp.today().normalize()
        days = int((dt.tz_localize(None).normalize() - today).days)
        return dt.strftime("%Y-%m-%d"), days
    except Exception:
        return "No disponible", None


def choose_expiration(tk: yf.Ticker) -> tuple[Optional[str], Optional[int]]:
    try:
        expirations = list(tk.options)
        if not expirations:
            return None, None
        today = pd.Timestamp.today().normalize()
        candidates = []
        for exp in expirations:
            dte = int((pd.Timestamp(exp) - today).days)
            if MIN_DTE <= dte <= MAX_DTE:
                candidates.append((exp, dte, abs(dte - TARGET_DTE)))
        if not candidates:
            return None, None
        candidates.sort(key=lambda x: x[2])
        return candidates[0][0], candidates[0][1]
    except Exception:
        return None, None


def estimate_delta_prob(price: float, strike: float, iv_pct: float, dte: int, side: str) -> tuple[float, float]:
    try:
        if price <= 0 or strike <= 0 or iv_pct <= 0 or dte <= 0:
            return np.nan, np.nan
        t = dte / 365
        sigma = iv_pct / 100
        d1 = (log(price / strike) + (RISK_FREE_RATE + 0.5 * sigma ** 2) * t) / (sigma * sqrt(t))
        d2 = d1 - sigma * sqrt(t)
        if side == "PUT":
            delta = normal_cdf(d1) - 1
            prob_otm = normal_cdf(d2) * 100
        else:
            delta = normal_cdf(d1)
            prob_otm = normal_cdf(-d2) * 100
        return round(delta, 2), round(prob_otm, 1)
    except Exception:
        return np.nan, np.nan


def clean_chain(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    for col in ["strike", "bid", "ask", "impliedVolatility", "openInterest", "volume"]:
        out[col] = pd.to_numeric(out.get(col, np.nan), errors="coerce")
    out = out.dropna(subset=["strike", "bid", "ask", "impliedVolatility"])
    return out


@dataclass
class SetupResult:
    ticker: str
    core_version: str
    data_status: str
    price: float
    rsi: float
    signal: str
    technical_score: int
    options_score: Optional[int]
    final_score: int
    contracts: int
    spread: str
    short_strike: Optional[float]
    long_strike: Optional[float]
    expiration: str
    dte: Optional[int]
    earnings_date: str
    earnings_days: Optional[int]
    earnings_status: str
    credit: Optional[float]
    max_loss: Optional[float]
    roc: Optional[float]
    prob_otm: Optional[float]
    delta: Optional[float]
    iv_rank: Optional[float]
    liquidity: str
    oi: Optional[float]
    bid_ask_spread_pct: Optional[float]
    reject_reason: str
    executable: bool

    def to_dict(self):
        return asdict(self)


def technical_signal_and_score(hist: pd.DataFrame) -> dict:
    close = hist["Close"]
    price = safe_float(close.iloc[-1])
    r = safe_float(rsi(close).iloc[-1])
    sma20 = safe_float(close.rolling(20).mean().iloc[-1])
    sma50 = safe_float(close.rolling(50).mean().iloc[-1])
    sma200 = safe_float(close.rolling(200).mean().iloc[-1])
    support = safe_float(hist["Low"].tail(60).min())
    resistance = safe_float(hist["High"].tail(60).max())
    dist_support = ((price - support) / support * 100) if support > 0 else np.nan
    dist_res = ((resistance - price) / price * 100) if price > 0 else np.nan

    signal = "NO TRADE"
    score = 0

    # Bull Put Spread candidate: oversold / pullback near support, not broken trend.
    put_candidate = (r <= 42 and dist_support <= 6.0)
    # Bear Call Spread candidate: overbought / near resistance.
    call_candidate = (r >= 62 and dist_res <= 8.0)

    if put_candidate:
        signal = "PUT"
        score = 50
        if r <= 35: score += 12
        elif r <= 42: score += 7
        if not np.isnan(sma200): score += 10 if price > sma200 else -10
        if not np.isnan(sma20) and not np.isnan(sma50): score += 8 if sma20 >= sma50 else -5
        if dist_support <= 3: score += 10
        elif dist_support <= 6: score += 5
    elif call_candidate:
        signal = "CALL"
        score = 50
        if r >= 70: score += 12
        elif r >= 62: score += 7
        if not np.isnan(sma200): score += 8 if price < sma200 else 0
        if not np.isnan(sma20) and not np.isnan(sma50): score += 8 if sma20 <= sma50 else 0
        if dist_res <= 4: score += 10
        elif dist_res <= 8: score += 5
    else:
        score = 0

    return {
        "price": round(price, 2), "rsi": round(r, 1), "signal": signal,
        "technical_score": int(max(0, min(100, round(score))))
    }


def build_best_spread(ticker: str, price: float, side: str, iv_rank: float) -> dict:
    tk = yf.Ticker(ticker)
    exp, dte = choose_expiration(tk)
    if not exp:
        return {"ok": False, "reason": "Sin expiración válida 25-50 DTE"}
    try:
        chain = tk.option_chain(exp)
    except Exception:
        return {"ok": False, "reason": "No se pudo descargar option chain"}

    df = clean_chain(chain.puts if side == "PUT" else chain.calls)
    if df.empty:
        return {"ok": False, "reason": "Cadena de opciones vacía"}

    candidates = []
    for width in SPREAD_WIDTHS:
        for _, short in df.iterrows():
            ss = safe_float(short["strike"])
            if side == "PUT" and ss >= price:
                continue
            if side == "CALL" and ss <= price:
                continue
            long_target = ss - width if side == "PUT" else ss + width
            long_df = df.iloc[(df["strike"] - long_target).abs().argsort()[:1]]
            if long_df.empty:
                continue
            long = long_df.iloc[0]
            ls = safe_float(long["strike"])
            actual_width = abs(ss - ls)
            if actual_width <= 0:
                continue
            short_bid = safe_float(short["bid"])
            short_ask = safe_float(short["ask"])
            long_ask = safe_float(long["ask"])
            oi = safe_float(short.get("openInterest", np.nan))
            vol = safe_float(short.get("volume", np.nan))
            iv = safe_float(short.get("impliedVolatility", np.nan)) * 100
            if short_bid <= 0 or short_ask <= 0 or long_ask < 0 or iv <= 0:
                continue
            credit = round(short_bid - long_ask, 2)
            if credit <= 0:
                continue
            mid = (short_bid + short_ask) / 2
            ba_pct = round((short_ask - short_bid) / mid * 100, 1) if mid > 0 else np.nan
            max_loss = round((actual_width - credit) * CONTRACT_MULTIPLIER, 2)
            roc = round((credit * CONTRACT_MULTIPLIER / max_loss) * 100, 1) if max_loss > 0 else np.nan
            delta, prob_otm = estimate_delta_prob(price, ss, iv, dte, side)

            # Strict executable filters. No fake/fallback values.
            if np.isnan(delta) or np.isnan(prob_otm) or np.isnan(roc) or np.isnan(ba_pct):
                continue
            if prob_otm >= 99.5 or abs(delta) < 0.03:  # usually fake or useless.
                continue
            if prob_otm < 65 or prob_otm > 92:
                continue
            if abs(delta) > 0.35:
                continue
            if roc < 6 or roc > 40:
                continue
            if oi < 100:
                continue
            if ba_pct > 30:
                continue

            liquidity = "Alta" if oi >= 300 and ba_pct <= 15 else "Media"
            score = 50
            if 70 <= prob_otm <= 85: score += 15
            elif prob_otm > 85: score += 8
            if 8 <= roc <= 20: score += 15
            elif roc > 20: score += 8
            if 0.10 <= abs(delta) <= 0.28: score += 10
            if liquidity == "Alta": score += 8
            if not np.isnan(iv_rank):
                if iv_rank >= 60: score += 8
                elif iv_rank < 20: score -= 10

            candidates.append({
                "short": ss, "long": ls, "width": actual_width, "credit": credit,
                "max_loss": max_loss, "roc": roc, "delta": delta, "prob_otm": prob_otm,
                "oi": oi, "volume": vol, "ba_pct": ba_pct, "iv": round(iv, 1),
                "expiration": exp, "dte": dte, "liquidity": liquidity,
                "options_score": int(max(0, min(100, round(score))))
            })

    if not candidates:
        return {"ok": False, "reason": "Sin spread ejecutable: filtros de crédito/ROC/prob/OI/bid-ask"}

    candidates.sort(key=lambda c: (c["options_score"], c["roc"], c["oi"]), reverse=True)
    best = candidates[0]
    best["ok"] = True
    return best


def contracts_for_score(score: int) -> int:
    if score >= 75:
        return 2
    if score >= 60:
        return 1
    return 0


def analyze_ticker(ticker: str, min_score: int = 60) -> SetupResult:
    ticker = ticker.strip().upper()
    hist = download_history(ticker)
    if hist.empty or len(hist) < 80:
        return SetupResult(ticker, CORE_VERSION, "ERROR", np.nan, np.nan, "NO TRADE", 0, None, 0, 0, "", None, None, "", None, "No disponible", None, "UNKNOWN", None, None, None, None, None, None, "No disponible", None, None, "Sin datos históricos suficientes", False)

    tech = technical_signal_and_score(hist)
    price = tech["price"]
    rsiv = tech["rsi"]
    signal = tech["signal"]
    technical_score = tech["technical_score"]
    iv_rank = calculate_iv_rank_proxy(hist)
    earnings_date, earnings_days = get_next_earnings(ticker)
    earnings_status = "OK"
    if earnings_days is not None and 0 <= earnings_days <= 14:
        earnings_status = "BLOQUEADO"

    if signal == "NO TRADE" or technical_score <= 0:
        return SetupResult(ticker, CORE_VERSION, "OK", price, rsiv, "NO TRADE", technical_score, None, 0, 0, "", None, None, "", None, earnings_date, earnings_days, earnings_status, None, None, None, None, None, iv_rank, "No disponible", None, None, "Sin setup técnico", False)
    if earnings_status == "BLOQUEADO":
        return SetupResult(ticker, CORE_VERSION, "OK", price, rsiv, signal, technical_score, None, 0, 0, "", None, None, "", None, earnings_date, earnings_days, earnings_status, None, None, None, None, None, iv_rank, "No disponible", None, None, "Earnings demasiado cerca", False)

    opt = build_best_spread(ticker, price, signal, iv_rank)
    if not opt.get("ok"):
        # Do not create executable alert if options are invalid. Keep technical score visible only.
        return SetupResult(ticker, CORE_VERSION, "OK_NO_EXECUTABLE_OPTIONS", price, rsiv, signal, technical_score, None, technical_score, 0, "", None, None, "", None, earnings_date, earnings_days, earnings_status, None, None, None, None, None, iv_rank, "No ejecutable", None, None, opt.get("reason", "Opciones no válidas"), False)

    options_score = int(opt["options_score"])
    final = int(round(technical_score * 0.45 + options_score * 0.55))
    contracts = contracts_for_score(final)
    spread = f"{opt['short']:g}/{opt['long']:g} {'PCS' if signal == 'PUT' else 'CCS'}"
    executable = final >= min_score and contracts > 0
    reject = "" if executable else f"Score final menor que {min_score}"

    return SetupResult(
        ticker=ticker, core_version=CORE_VERSION, data_status="OK_EXECUTABLE_OPTIONS",
        price=price, rsi=rsiv, signal=signal, technical_score=technical_score,
        options_score=options_score, final_score=final, contracts=contracts,
        spread=spread, short_strike=opt["short"], long_strike=opt["long"], expiration=opt["expiration"], dte=opt["dte"],
        earnings_date=earnings_date, earnings_days=earnings_days, earnings_status=earnings_status,
        credit=opt["credit"], max_loss=opt["max_loss"], roc=opt["roc"], prob_otm=opt["prob_otm"], delta=opt["delta"],
        iv_rank=iv_rank, liquidity=opt["liquidity"], oi=opt["oi"], bid_ask_spread_pct=opt["ba_pct"],
        reject_reason=reject, executable=executable
    )
