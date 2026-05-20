import os
import requests
from optionsview_core_engine import CORE_VERSION, FULL_TICKERS, parse_tickers, scan_tickers

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TICKERS = parse_tickers(os.getenv("TICKERS", FULL_TICKERS))
MIN_SCORE = int(os.getenv("MIN_SCORE", "60"))
MAX_ALERTS = int(os.getenv("MAX_ALERTS", "10"))


def send(msg: str):
    if not TOKEN or not CHAT_ID:
        print(msg)
        return
    requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage", data={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"}, timeout=20)

rows = scan_tickers(TICKERS, period="1y", deep_options=True)
rows = sorted(rows, key=lambda r: r.get("final_score", 0), reverse=True)
setups = [r for r in rows if r.get("final_score", 0) >= MIN_SCORE and r.get("contracts", 0) > 0]

if setups:
    lines = [f"🔥 <b>OptionsView Unified</b> · Core {CORE_VERSION}", f"Setups >= {MIN_SCORE}%: {len(setups)}"]
    for r in setups[:MAX_ALERTS]:
        lines.append(
            f"\n🚨 <b>{r['ticker']}</b> {r['final_score']}% | {r.get('spread','-')}\n"
            f"Técnico: {r.get('technical_score')}% | Opciones: {r.get('options_score')}% | Contratos: {r.get('contracts')}\n"
            f"ROC: {r.get('roc')}% | Prob OTM: {r.get('prob_otm')}% | Delta: {r.get('delta')}\n"
            f"IVR/HVR: {r.get('ivr')}% | Liquidez: {r.get('liquidity')} | Crédito: ${r.get('credit')}\n"
            f"Exp: {r.get('expiration')} ({r.get('dte')} DTE) | Earnings: {r.get('earnings_status')} ({r.get('earnings_date')})\n"
            f"Precio: ${r.get('price')} | RSI: {r.get('rsi')}"
        )
    send("\n".join(lines))
else:
    top = rows[:5]
    lines = [f"✅ OptionsView Unified · sin setups >= {MIN_SCORE}%", f"Core: {CORE_VERSION}", "\nTop revisados:"]
    for r in top:
        lines.append(f"{r.get('ticker')}: {r.get('final_score')}% | T:{r.get('technical_score')} O:{r.get('options_score')} | {r.get('signal')}")
    send("\n".join(lines))
