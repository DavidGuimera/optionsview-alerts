"""
OptionsView Algo v1 — GitHub / yfinance forward-test engine

IMPORTANT
- win_probability is an ESTIMATED probability, not a guaranteed/calibrated probability.
- It starts from option-implied Prob OTM and applies small, capped adjustments.
- Technical/options scores remain diagnostic only.
- Hard filters decide executability BEFORE an alert can be sent.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from math import erf, log, sqrt
from typing import Optional
import numpy as np
import pandas as pd
import yfinance as yf

CORE_VERSION = "2026.08.12-algo-v1.3.1-100"

DEFAULT_TICKERS = "MCD,PEP,PG,KO,JNJ,WMT,COST,HD,LOW,TGT,SBUX,MDLZ,CMCSA,MSFT,AAPL,GOOGL,META,AMZN,NVDA,AVGO,ADBE,CRM,ORCL,CSCO,AMD,INTC,MU,QCOM,TXN,AMAT,JPM,MA,V,BLK,SCHW,BAC,WFC,C,GS,MS,XOM,CVX,COP,SLB,OXY,UNH,ABBV,MRK,PFE,LLY,GILD,AMGN,CVS,CAT,DE,GE,HON,UPS,BA,RTX,LMT,DIS,NFLX,TSLA,NKE,BKNG,ABNB,GM,F,NEE,DUK,SO,LIN,FCX,NEM,PLD,AMT,SPY,QQQ,IWM,DIA,XLP,XLV,XLF,XLE,XLI,XLK,XLY,XLU,SMH,SOXX,TLT,GLD,SLV,USO,EEM,FXI,ARKK,IBM,PYPL"

CONTRACT_MULTIPLIER = 100
RISK_FREE_RATE = 0.045

TARGET_DTE = 35
MIN_DTE = 25
MAX_DTE = 50
SPREAD_WIDTHS = [2.5, 5, 10]

# Hard execution filters
MIN_PROB_OTM = 65.0
MAX_PROB_OTM = 92.0
MIN_SHORT_DELTA = 0.08
MAX_SHORT_DELTA = 0.30
MIN_OI = 100
MAX_BID_ASK_PCT = 25.0
MIN_NET_ROC = 6.0
MAX_NET_ROC = 40.0
MIN_CREDIT = 0.20
MIN_EM_DISTANCE = 0.55       # short strike must be >= 0.55 expected moves away
EARNINGS_BLOCK_DAYS = 14
COMMISSION_PER_CONTRACT_LEG = 0.65  # estimate; 2 legs x open+close
DEFAULT_MAX_RISK_PER_TRADE = 500.0
MIN_EXECUTION_CREDIT_RATIO = 0.94  # do not enter if live credit falls below 94% of scanned credit


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


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def normal_cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def download_history(ticker, period="1y"):
    try:
        df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]
        needed = ["Open", "High", "Low", "Close"]
        if not all(c in df.columns for c in needed):
            return pd.DataFrame()
        return df.dropna(subset=needed)
    except Exception:
        return pd.DataFrame()


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_iv_rank_proxy(hist):
    """HV-rank proxy. Keep name visible as HVR/IVR proxy; it is NOT true option IV Rank."""
    try:
        ret = hist["Close"].pct_change().dropna()
        hv20 = (ret.rolling(20).std() * np.sqrt(252) * 100).dropna()
        if len(hv20) < 30:
            return np.nan
        cur, mn, mx = map(safe_float, [hv20.iloc[-1], hv20.min(), hv20.max()])
        if any(np.isnan(x) for x in [cur, mn, mx]) or mx <= mn:
            return np.nan
        return round(clamp((cur-mn)/(mx-mn)*100, 0, 100), 1)
    except Exception:
        return np.nan


def get_next_earnings(ticker):
    try:
        cal = getattr(yf.Ticker(ticker), "calendar", None)
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


def choose_expiration(tk):
    try:
        today = pd.Timestamp.today().normalize()
        candidates = []
        for exp in list(tk.options):
            dte = int((pd.Timestamp(exp) - today).days)
            if MIN_DTE <= dte <= MAX_DTE:
                candidates.append((exp, dte, abs(dte-TARGET_DTE)))
        if not candidates:
            return None, None
        candidates.sort(key=lambda x: x[2])
        return candidates[0][0], candidates[0][1]
    except Exception:
        return None, None


def estimate_delta_prob(price, strike, iv_pct, dte, side):
    """BS-style estimate because yfinance does not provide reliable live Greeks."""
    try:
        if min(price, strike, iv_pct, dte) <= 0:
            return np.nan, np.nan
        t = dte/365
        sigma = iv_pct/100
        d1 = (log(price/strike) + (RISK_FREE_RATE + 0.5*sigma**2)*t)/(sigma*sqrt(t))
        d2 = d1 - sigma*sqrt(t)
        if side == "PUT":
            delta = normal_cdf(d1)-1
            prob_otm = normal_cdf(d2)*100
        else:
            delta = normal_cdf(d1)
            prob_otm = normal_cdf(-d2)*100
        return round(delta, 3), round(prob_otm, 1)
    except Exception:
        return np.nan, np.nan


def clean_chain(df):
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    for col in ["strike","bid","ask","impliedVolatility","openInterest","volume"]:
        out[col] = pd.to_numeric(out.get(col, np.nan), errors="coerce")
    return out.dropna(subset=["strike","bid","ask","impliedVolatility"])


def technical_context(hist, side):
    close = hist["Close"]
    price = safe_float(close.iloc[-1])
    rv = safe_float(rsi(close).iloc[-1])
    sma20 = safe_float(close.rolling(20).mean().iloc[-1])
    sma50 = safe_float(close.rolling(50).mean().iloc[-1])
    sma200 = safe_float(close.rolling(200).mean().iloc[-1])
    mom20 = safe_float((price/close.iloc[-21]-1)*100) if len(close) >= 21 else np.nan
    support = safe_float(hist["Low"].tail(60).min())
    resistance = safe_float(hist["High"].tail(60).max())
    ds = ((price-support)/support*100) if support > 0 else np.nan
    dr = ((resistance-price)/price*100) if price > 0 else np.nan

    # diagnostic score only
    score = 50
    if side == "PUT":
        if rv <= 35: score += 12
        elif rv <= 42: score += 7
        if not np.isnan(sma200): score += 8 if price > sma200 else -10
        if not np.isnan(sma20) and not np.isnan(sma50): score += 6 if sma20 >= sma50 else -5
        if not np.isnan(ds): score += 8 if ds <= 3 else (4 if ds <= 6 else 0)
    else:
        if rv >= 70: score += 12
        elif rv >= 62: score += 7
        if not np.isnan(sma20) and not np.isnan(sma50): score += 6 if sma20 <= sma50 else 0
        if not np.isnan(dr): score += 8 if dr <= 4 else (4 if dr <= 8 else 0)

    return dict(price=round(price,2), rsi=round(rv,1), sma20=sma20, sma50=sma50,
                sma200=sma200, momentum20=mom20, dist_support=ds,
                dist_resistance=dr, technical_score=int(clamp(round(score),0,100)))


def detect_signal(hist):
    close = hist["Close"]
    price = safe_float(close.iloc[-1])
    rv = safe_float(rsi(close).iloc[-1])
    support = safe_float(hist["Low"].tail(60).min())
    resistance = safe_float(hist["High"].tail(60).max())
    ds = ((price-support)/support*100) if support > 0 else np.nan
    dr = ((resistance-price)/price*100) if price > 0 else np.nan
    if rv <= 42 and ds <= 6:
        return "PUT"
    if rv >= 62 and dr <= 8:
        return "CALL"
    return "NO TRADE"


def estimate_win_probability(prob_otm, side, tech, iv_rank, em_distance):
    """
    Estimated win probability.
    Base = option-implied Prob OTM.
    Adjustments are deliberately small/capped until Nov calibration.
    """
    p = float(prob_otm)
    adj = 0.0
    rv = tech["rsi"]
    sma20, sma50, sma200 = tech["sma20"], tech["sma50"], tech["sma200"]
    price = tech["price"]

    if side == "PUT":
        if 30 <= rv <= 42: adj += 1.5
        if not np.isnan(sma200) and price > sma200: adj += 1.0
        if not np.isnan(sma20) and not np.isnan(sma50) and sma20 >= sma50: adj += 0.5
        if tech["dist_support"] is not None and not np.isnan(tech["dist_support"]) and tech["dist_support"] <= 3: adj += 1.0
    else:
        if 62 <= rv <= 78: adj += 1.0
        if not np.isnan(sma20) and not np.isnan(sma50) and sma20 <= sma50: adj += 1.0
        if tech["dist_resistance"] is not None and not np.isnan(tech["dist_resistance"]) and tech["dist_resistance"] <= 4: adj += 1.0

    if not np.isnan(iv_rank):
        if iv_rank >= 60: adj += 0.5
        elif iv_rank < 15: adj -= 1.0

    if em_distance >= 1.0: adj += 1.5
    elif em_distance < 0.70: adj -= 1.5

    # Never allow technical heuristics to move probability more than +/-5 points.
    adj = clamp(adj, -5.0, 5.0)
    return round(clamp(p + adj, 55.0, 90.0), 1)


@dataclass
class SetupResult:
    ticker: str
    core_version: str
    data_status: str
    price: float
    rsi: float
    signal: str
    technical_score: int
    options_quality_score: Optional[int]
    win_probability: Optional[float]
    probability_adjustment: Optional[float]
    min_entry_credit: Optional[float]
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
    commission_rt: Optional[float]
    max_loss: Optional[float]
    net_max_loss: Optional[float]
    roc: Optional[float]
    net_roc: Optional[float]
    prob_otm: Optional[float]
    delta: Optional[float]
    delta_source: str
    iv_rank: Optional[float]
    short_iv: Optional[float]
    expected_move: Optional[float]
    em_distance: Optional[float]
    liquidity: str
    oi: Optional[float]
    bid_ask_spread_pct: Optional[float]
    reject_reason: str
    executable: bool

    @property
    def final_score(self):
        # backward compatibility with old alerts/app code
        return int(round(self.win_probability or 0))

    @property
    def options_score(self):
        return self.options_quality_score

    def to_dict(self):
        d = asdict(self)
        d["final_score"] = self.final_score
        d["options_score"] = self.options_score
        return d


def build_best_spread(ticker, price, side, iv_rank, tech, max_risk=DEFAULT_MAX_RISK_PER_TRADE):
    tk = yf.Ticker(ticker)
    exp, dte = choose_expiration(tk)
    if not exp:
        return {"ok":False, "reason":"Sin expiración válida 25-50 DTE"}
    try:
        chain = tk.option_chain(exp)
    except Exception:
        return {"ok":False, "reason":"No se pudo descargar option chain"}

    df = clean_chain(chain.puts if side=="PUT" else chain.calls)
    if df.empty:
        return {"ok":False, "reason":"Cadena de opciones vacía"}

    candidates = []
    for width in SPREAD_WIDTHS:
        for _, short in df.iterrows():
            ss = safe_float(short["strike"])
            if side=="PUT" and ss >= price: continue
            if side=="CALL" and ss <= price: continue

            target = ss-width if side=="PUT" else ss+width
            long_df = df.iloc[(df["strike"]-target).abs().argsort()[:1]]
            if long_df.empty: continue
            long = long_df.iloc[0]
            ls = safe_float(long["strike"])
            actual_width = abs(ss-ls)
            if actual_width <= 0: continue

            sbid, sask = safe_float(short["bid"]), safe_float(short["ask"])
            lask = safe_float(long["ask"])
            oi = safe_float(short.get("openInterest", np.nan))
            iv = safe_float(short.get("impliedVolatility", np.nan))*100

            if any(np.isnan(x) for x in [sbid,sask,lask,oi,iv]): continue
            if sbid <= 0 or sask <= 0 or lask < 0 or iv <= 0: continue

            credit = round(sbid-lask, 2)  # conservative executable estimate
            if credit < MIN_CREDIT: continue

            mid = (sbid+sask)/2
            ba = round((sask-sbid)/mid*100,1) if mid>0 else np.nan
            delta, prob = estimate_delta_prob(price, ss, iv, dte, side)

            gross_max_loss = round((actual_width-credit)*100,2)
            commission_rt = round(COMMISSION_PER_CONTRACT_LEG*4,2)
            net_max_loss = round(gross_max_loss+commission_rt,2)
            gross_profit = credit*100
            net_profit = gross_profit-commission_rt
            roc = round(gross_profit/gross_max_loss*100,1) if gross_max_loss>0 else np.nan
            net_roc = round(net_profit/net_max_loss*100,1) if net_max_loss>0 else np.nan

            expected_move = round(price*(iv/100)*sqrt(dte/365),2)
            strike_distance = (price-ss) if side=="PUT" else (ss-price)
            em_distance = round(strike_distance/expected_move,2) if expected_move>0 else np.nan

            # HARD DATA + EXECUTION FILTERS
            if any(np.isnan(x) for x in [delta,prob,ba,roc,net_roc,expected_move,em_distance]): continue
            if prob >= 99.5 or abs(delta) < 0.03: continue
            if not (MIN_PROB_OTM <= prob <= MAX_PROB_OTM): continue
            if not (MIN_SHORT_DELTA <= abs(delta) <= MAX_SHORT_DELTA): continue
            if oi < MIN_OI: continue
            if ba > MAX_BID_ASK_PCT: continue
            if not (MIN_NET_ROC <= net_roc <= MAX_NET_ROC): continue
            if em_distance < MIN_EM_DISTANCE: continue
            if net_max_loss > max_risk: continue

            liquidity = "Alta" if oi>=300 and ba<=15 else "Media"
            quality = 50
            if 70<=prob<=85: quality += 15
            if 8<=net_roc<=20: quality += 15
            if 0.10<=abs(delta)<=0.25: quality += 10
            if liquidity=="Alta": quality += 8
            if em_distance>=1.0: quality += 7
            if not np.isnan(iv_rank) and iv_rank>=50: quality += 5
            quality = int(clamp(round(quality),0,100))

            winp = estimate_win_probability(prob, side, tech, iv_rank, em_distance)

            candidates.append(dict(
                short=ss,long=ls,credit=credit,commission_rt=commission_rt,
                max_loss=gross_max_loss,net_max_loss=net_max_loss,roc=roc,net_roc=net_roc,
                delta=delta,prob_otm=prob,oi=oi,ba_pct=ba,iv=round(iv,1),
                expected_move=expected_move,em_distance=em_distance,
                expiration=exp,dte=dte,liquidity=liquidity,
                options_quality_score=quality,win_probability=winp
            ))

    if not candidates:
        return {"ok":False, "reason":"Sin spread ejecutable: filtros de datos/delta/crédito/ROC/OI/bid-ask/EM/riesgo"}

    candidates.sort(key=lambda c:(c["win_probability"],c["options_quality_score"],c["net_roc"],c["oi"]), reverse=True)
    best = candidates[0]
    best["ok"] = True
    return best


def contracts_for_trade(win_probability, net_max_loss, max_risk=DEFAULT_MAX_RISK_PER_TRADE):
    if not win_probability or not net_max_loss or net_max_loss <= 0:
        return 0
    by_risk = int(max_risk // net_max_loss)
    if by_risk <= 0: return 0
    # Conservative during forward-test
    if win_probability >= 80 and by_risk >= 2:
        return 2
    return 1


def empty_result(ticker, reason, price=np.nan, rv=np.nan, signal="NO TRADE",
                 tech_score=0, iv_rank=np.nan, earnings_date="No disponible",
                 earnings_days=None, earnings_status="UNKNOWN", status="ERROR"):
    # Use keyword arguments deliberately.
    # This prevents positional-argument mismatches when SetupResult gains new fields.
    return SetupResult(
        ticker=ticker,
        core_version=CORE_VERSION,
        data_status=status,
        price=price,
        rsi=rv,
        signal=signal,
        technical_score=tech_score,
        options_quality_score=None,
        win_probability=None,
        probability_adjustment=None,
        min_entry_credit=None,
        contracts=0,
        spread="",
        short_strike=None,
        long_strike=None,
        expiration="",
        dte=None,
        earnings_date=earnings_date,
        earnings_days=earnings_days,
        earnings_status=earnings_status,
        credit=None,
        commission_rt=None,
        max_loss=None,
        net_max_loss=None,
        roc=None,
        net_roc=None,
        prob_otm=None,
        delta=None,
        delta_source="N/A",
        iv_rank=iv_rank,
        short_iv=None,
        expected_move=None,
        em_distance=None,
        liquidity="No disponible",
        oi=None,
        bid_ask_spread_pct=None,
        reject_reason=reason,
        executable=False,
    )


def analyze_ticker(ticker, min_score=60, max_risk=DEFAULT_MAX_RISK_PER_TRADE):
    """
    min_score is retained for compatibility, but now means MINIMUM ESTIMATED WIN PROBABILITY.
    """
    ticker = ticker.strip().upper()
    hist = download_history(ticker)
    if hist.empty or len(hist)<80:
        return empty_result(ticker,"Sin datos históricos suficientes")

    signal = detect_signal(hist)
    if signal=="NO TRADE":
        price = round(safe_float(hist["Close"].iloc[-1]),2)
        rv = round(safe_float(rsi(hist["Close"]).iloc[-1]),1)
        return empty_result(ticker,"Sin setup técnico",price,rv,status="OK")

    tech = technical_context(hist,signal)
    price, rv, tech_score = tech["price"],tech["rsi"],tech["technical_score"]
    iv_rank = calculate_iv_rank_proxy(hist)
    edate, edays = get_next_earnings(ticker)
    estatus = "BLOQUEADO" if edays is not None and 0<=edays<=EARNINGS_BLOCK_DAYS else "OK"

    if estatus=="BLOQUEADO":
        return empty_result(ticker,"Earnings demasiado cerca",price,rv,signal,tech_score,iv_rank,edate,edays,estatus,status="OK")

    opt = build_best_spread(ticker,price,signal,iv_rank,tech,max_risk=max_risk)
    if not opt.get("ok"):
        return empty_result(ticker,opt.get("reason","Opciones no válidas"),price,rv,signal,tech_score,iv_rank,edate,edays,estatus,status="OK_NO_EXECUTABLE_OPTIONS")

    winp = float(opt["win_probability"])
    prob_adj = round(winp - float(opt["prob_otm"]), 1)
    min_entry_credit = round(float(opt["credit"]) * MIN_EXECUTION_CREDIT_RATIO, 2)
    contracts = contracts_for_trade(winp,opt["net_max_loss"],max_risk=max_risk)
    spread = f"{opt['short']:g}/{opt['long']:g} {'PCS' if signal=='PUT' else 'CCS'}"
    executable = winp >= float(min_score) and contracts>0
    reject = "" if executable else f"Probabilidad estimada menor que {min_score}%"

    return SetupResult(
        ticker=ticker,core_version=CORE_VERSION,data_status="OK_EXECUTABLE_OPTIONS",
        price=price,rsi=rv,signal=signal,technical_score=tech_score,
        options_quality_score=opt["options_quality_score"],win_probability=winp,
        probability_adjustment=prob_adj,min_entry_credit=min_entry_credit,
        contracts=contracts,spread=spread,short_strike=opt["short"],long_strike=opt["long"],
        expiration=opt["expiration"],dte=opt["dte"],earnings_date=edate,earnings_days=edays,
        earnings_status=estatus,credit=opt["credit"],commission_rt=opt["commission_rt"],
        max_loss=opt["max_loss"],net_max_loss=opt["net_max_loss"],roc=opt["roc"],net_roc=opt["net_roc"],
        prob_otm=opt["prob_otm"],delta=opt["delta"],
        delta_source="BS estimate from yfinance IV (not broker Greek)",
        iv_rank=iv_rank,short_iv=opt["iv"],expected_move=opt["expected_move"],
        em_distance=opt["em_distance"],liquidity=opt["liquidity"],oi=opt["oi"],
        bid_ask_spread_pct=opt["ba_pct"],reject_reason=reject,executable=executable
    )
