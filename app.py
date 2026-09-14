import os
import time
import json
import threading
from datetime import datetime, timezone

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import ta

try:
    from growwapi import GrowwAPI, GrowwFeed
except Exception:
    GrowwAPI = None
    GrowwFeed = None

# ============================================================
# SOVENDRA MARKET AI V3 — GROWW LIVE / READ-ONLY EDITION
# IMPORTANT SAFETY DESIGN:
# - NO place_order / modify_order / cancel_order calls exist.
# - NO smart-order/GTT/OCO calls exist.
# - The app only reads market data + public instrument metadata.
# - Groww credentials must be stored in Streamlit Secrets.
# - If live data is stale/missing, trade decision is disabled.
# ============================================================

st.set_page_config(
    page_title="Sovendra Market AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

REFRESH_SECONDS = 2
STALE_SECONDS = 8

QUICK_SYMBOLS = {
    "NIFTY 50": ("NSE", "CASH", "NIFTY"),
    "BANKNIFTY": ("NSE", "CASH", "BANKNIFTY"),
    "RELIANCE": ("NSE", "CASH", "RELIANCE"),
    "TCS": ("NSE", "CASH", "TCS"),
    "HDFCBANK": ("NSE", "CASH", "HDFCBANK"),
    "SBIN": ("NSE", "CASH", "SBIN"),
    "TATA POWER": ("NSE", "CASH", "TATAPOWER"),
    "ITC": ("NSE", "CASH", "ITC"),
}

INDEXES = {
    "NIFTY 50": ("NSE", "CASH", "NIFTY"),
    "BANKNIFTY": ("NSE", "CASH", "BANKNIFTY"),
}

st.markdown(
    """
    <style>
    .block-container {padding-top: 1rem; padding-bottom: 2rem;}
    .live {font-weight:700;}
    .safe {padding:.75rem 1rem;border-radius:12px;border:1px solid rgba(0,0,0,.12);}
    @media(max-width:768px){.block-container{padding-left:.7rem;padding-right:.7rem;}}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Secrets / Groww client
# -------------------------
def get_secret(name: str):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name)


def groww_client():
    if GrowwAPI is None:
        return None
    token = get_secret("GROWW_ACCESS_TOKEN")
    if not token:
        return None
    try:
        return GrowwAPI(token)
    except Exception:
        return None


groww = groww_client()

@st.cache_resource(show_spinner=False)
def feed_client():
    if groww is None or GrowwFeed is None:
        return None
    try:
        return GrowwFeed(groww)
    except Exception:
        return None

feed = feed_client()

# -------------------------
# Time / validation
# -------------------------
def now_ist():
    return pd.Timestamp.now(tz="Asia/Kolkata")


def market_open():
    n = now_ist()
    if n.weekday() >= 5:
        return False
    m = n.hour * 60 + n.minute
    return 555 <= m < 930


def age_seconds(ts_ms):
    if ts_ms is None or not np.isfinite(float(ts_ms)):
        return np.inf
    return max(0.0, (time.time() * 1000 - float(ts_ms)) / 1000.0)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

# -------------------------
# Read-only Groww calls
# -------------------------
def read_ltp(exchange, segment, symbol):
    # Preferred path: Groww live feed. It returns an exchange timestamp.
    if feed is not None:
        try:
            token = find_exchange_token(exchange, segment, symbol) or symbol
            instruments = [{"exchange": exchange, "segment": segment, "exchange_token": token}]
            feed.subscribe_ltp(instruments)
            data = feed.get_ltp()
            node = (data.get("ltp", {}).get(exchange, {}).get(segment, {}).get(str(token), {})
                    if isinstance(data, dict) else {})
            if isinstance(node, dict) and np.isfinite(safe_float(node.get("ltp"))):
                return {"ltp": safe_float(node.get("ltp")), "ts_ms": safe_float(node.get("tsInMillis")), "source": "Groww WebSocket feed"}
        except Exception:
            pass

    # Fallback: Groww real-time LTP REST endpoint. Still read-only.
    if groww is not None:
        try:
            key = f"{exchange}_{symbol}"
            data = groww.get_ltp(segment=segment, exchange_trading_symbols=(key,))
            payload = data.get("ltp", data) if isinstance(data, dict) else data
            if isinstance(payload, dict):
                value = payload.get(key)
                if value is None:
                    value = payload.get(symbol)
                value = safe_float(value)
                if np.isfinite(value):
                    return {"ltp": value, "ts_ms": time.time() * 1000, "source": "Groww live LTP API"}
        except Exception:
            pass
    return None


def read_quote(exchange, segment, symbol):
    if groww is None:
        return None
    try:
        data = groww.get_quote(exchange=exchange, segment=segment, trading_symbol=symbol)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def download_instruments():
    """Public Groww instrument master; no account credentials are used."""
    url = "https://growwapi-assets.groww.in/instruments/instrument.csv"
    try:
        df = pd.read_csv(url)
        return df
    except Exception:
        return pd.DataFrame()


def find_exchange_token(exchange, segment, trading_symbol):
    df = download_instruments()
    if df.empty:
        return None
    required = {"exchange", "segment", "trading_symbol", "exchange_token"}
    if not required.issubset(df.columns):
        return None
    x = df[
        (df["exchange"].astype(str) == exchange)
        & (df["segment"].astype(str) == segment)
        & (df["trading_symbol"].astype(str) == trading_symbol)
    ]
    if x.empty:
        return None
    return str(x.iloc[0]["exchange_token"])

# -------------------------
# Historical candles from Groww
# -------------------------
def historical_candles(exchange, segment, symbol, interval_minutes=5, days=5):
    if groww is None:
        return pd.DataFrame()
    end = now_ist()
    start = end - pd.Timedelta(days=days)
    try:
        fn = getattr(groww, "get_historical_candles", None)
        if fn is None:
            return pd.DataFrame()
        groww_symbol = f"{exchange}-{symbol}"
        interval_name = {1: "1minute", 5: "5minute", 10: "10minute", 15: "15minute", 30: "30minute", 60: "1hour", 1440: "1day"}[interval_minutes]
        raw = fn(
            exchange=exchange,
            segment=segment,
            groww_symbol=groww_symbol,
            start_time=start.strftime("%Y-%m-%d %H:%M:%S"),
            end_time=end.strftime("%Y-%m-%d %H:%M:%S"),
            candle_interval=interval_name,
        )
        payload = raw.get("payload", raw) if isinstance(raw, dict) else raw
        rows = payload.get("candles", []) if isinstance(payload, dict) else []
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        if df.shape[1] < 6:
            return pd.DataFrame()
        df = df.iloc[:, :6]
        df.columns = ["Timestamp", "Open", "High", "Low", "Close", "Volume"]
        df.index = pd.to_datetime(df["Timestamp"], errors="coerce")
        if df.index.isna().all():
            df.index = pd.to_datetime(pd.to_numeric(df["Timestamp"], errors="coerce"), unit="s", errors="coerce")
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["Open", "High", "Low", "Close"]).sort_index()
        return df[["Open", "High", "Low", "Close", "Volume"]]
    except Exception:
        return pd.DataFrame()

# -------------------------
# Indicators / SMC
# -------------------------
def add_indicators(df):
    out = df.copy()
    out["EMA20"] = ta.trend.ema_indicator(out["Close"], window=20)
    out["EMA50"] = ta.trend.ema_indicator(out["Close"], window=50)
    out["EMA200"] = ta.trend.ema_indicator(out["Close"], window=200)
    out["RSI"] = ta.momentum.rsi(out["Close"], window=14)
    out["MACD"] = ta.trend.macd(out["Close"])
    out["MACD_SIGNAL"] = ta.trend.macd_signal(out["Close"])
    out["ADX"] = ta.trend.adx(out["High"], out["Low"], out["Close"], window=14)
    out["ATR"] = ta.volatility.average_true_range(out["High"], out["Low"], out["Close"], window=14)
    typical = (out["High"] + out["Low"] + out["Close"]) / 3
    vol = out["Volume"].fillna(0)
    out["VWAP"] = (typical * vol).rolling(20).sum() / vol.rolling(20).sum().replace(0, np.nan)
    return out


def smc(df):
    x = df.copy()
    w = 5
    x["SwingHigh"] = x["High"].where(x["High"] == x["High"].rolling(w, center=True).max())
    x["SwingLow"] = x["Low"].where(x["Low"] == x["Low"].rolling(w, center=True).min())
    hs = x["SwingHigh"].dropna()
    ls = x["SwingLow"].dropna()
    sh = float(hs.iloc[-1]) if len(hs) else np.nan
    sl = float(ls.iloc[-1]) if len(ls) else np.nan
    structure = "MIXED"
    if len(hs) >= 2 and len(ls) >= 2:
        if hs.iloc[-1] > hs.iloc[-2] and ls.iloc[-1] > ls.iloc[-2]:
            structure = "BULLISH (HH/HL)"
        elif hs.iloc[-1] < hs.iloc[-2] and ls.iloc[-1] < ls.iloc[-2]:
            structure = "BEARISH (LH/LL)"
    price = float(x["Close"].iloc[-1])
    bos = "BULLISH BOS" if np.isfinite(sh) and price > sh else "BEARISH BOS" if np.isfinite(sl) and price < sl else "NO FRESH BOS"
    recent = x.tail(3)
    fvg = "NO FRESH FVG"
    if len(recent) == 3:
        a, b, c = recent.iloc[0], recent.iloc[1], recent.iloc[2]
        if c["Low"] > a["High"]:
            fvg = "BULLISH FVG"
        elif c["High"] < a["Low"]:
            fvg = "BEARISH FVG"
    return {"structure": structure, "bos": bos, "fvg": fvg, "swing_high": sh, "swing_low": sl}


def decision(x, s, live_price):
    r = x.iloc[-1]
    score = 0
    reasons = []
    p = live_price
    for condition, up, down in [
        (p > r.EMA20, 1, -1),
        (r.EMA20 > r.EMA50, 2, -2),
        (p > r.EMA200, 1, -1),
        (r.MACD > r.MACD_SIGNAL, 1, -1),
    ]:
        score += up if condition else down
    reasons += [
        "Price vs EMA20",
        "EMA20/EMA50 trend",
        "Price vs EMA200",
        "MACD direction",
    ]
    if np.isfinite(r.VWAP):
        score += 1 if p > r.VWAP else -1
        reasons.append("VWAP alignment")
    if 55 <= r.RSI <= 70:
        score += 1
    elif 30 <= r.RSI < 45:
        score -= 1
    if r.ADX >= 20:
        reasons.append(f"ADX {r.ADX:.1f}: trend strength usable")
    else:
        reasons.append(f"ADX {r.ADX:.1f}: trend weak")
    if "BULLISH" in s["structure"] or s["bos"] == "BULLISH BOS":
        score += 2
    elif "BEARISH" in s["structure"] or s["bos"] == "BEARISH BOS":
        score -= 2
    bias = "BULLISH" if score >= 5 else "BEARISH" if score <= -5 else "NEUTRAL"
    atr = float(r.ATR) if np.isfinite(r.ATR) and r.ATR > 0 else p * 0.005
    if bias == "BULLISH":
        risk = max(p - s["swing_low"], 1.2 * atr) if np.isfinite(s["swing_low"]) else 1.2 * atr
        sl = p - risk; t1 = p + 1.5 * risk; t2 = p + 2.5 * risk
    elif bias == "BEARISH":
        risk = max(s["swing_high"] - p, 1.2 * atr) if np.isfinite(s["swing_high"]) else 1.2 * atr
        sl = p + risk; t1 = p - 1.5 * risk; t2 = p - 2.5 * risk
    else:
        sl = t1 = t2 = np.nan
    return {"score": score, "bias": bias, "confidence": int(np.clip(50 + abs(score)*6, 50, 92)), "sl": sl, "t1": t1, "t2": t2, "reasons": reasons}

# -------------------------
# Browser voice assistant
# -------------------------
def voice_box(answer_text):
    safe = json.dumps(answer_text)
    html = f"""
    <div style='padding:10px;border:1px solid #ddd;border-radius:12px'>
      <button onclick='startVoice()' style='font-size:18px;padding:8px 14px'>🎙️ Speak</button>
      <span id='heard' style='margin-left:10px;color:#555'>Tap and speak</span>
    </div>
    <script>
    function startVoice() {{
      const R = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!R) {{ document.getElementById('heard').innerText='Voice input is not supported in this browser.'; return; }}
      const r = new R(); r.lang='en-IN'; r.interimResults=false;
      r.onresult = e => {{
        const q=e.results[0][0].transcript;
        document.getElementById('heard').innerText='You: '+q;
        const ans={safe};
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(new SpeechSynthesisUtterance(ans));
      }};
      r.start();
    }}
    </script>
    """
    st.components.v1.html(html, height=80)

# ============================================================
# UI
# ============================================================
st.title("📊 Sovendra Market AI")
st.caption("Groww live-data decision support • Read-only market connection • No order placement")

if groww is None:
    st.error("🔴 Groww live feed is not connected.")
    st.info("Add GROWW_ACCESS_TOKEN to Streamlit Secrets. Do not paste your token into chat or the code file.")
    st.stop()

with st.sidebar:
    name = st.selectbox("Instrument", list(QUICK_SYMBOLS.keys()))
    exchange, segment, symbol = QUICK_SYMBOLS[name]
    timeframe = st.selectbox("Analysis timeframe", ["1m", "5m", "15m", "30m", "1h", "1d"], index=2)
    if st.button("🔄 Refresh now", use_container_width=True):
        st.rerun()
    st.caption("Safety mode: READ ONLY")
    st.caption("Order APIs are intentionally not called by this application.")

# Live snapshot
live = read_ltp(exchange, segment, symbol)
if not live or not np.isfinite(live.get("ltp", np.nan)):
    st.error("🔴 Live market data unavailable — TRADE DISABLED")
    st.stop()

live_price = float(live["ltp"])
# REST LTP response is stamped at receipt; quote may include exchange timestamp.
feed_age = age_seconds(live.get("ts_ms"))

# Historical data is still required for technical calculations.
interval_map = {"1m":1, "5m":5, "15m":15, "30m":30, "1h":60, "1d":1440}
days_map = {"1m":2, "5m":5, "15m":15, "30m":30, "1h":120, "1d":730}
hist = historical_candles(exchange, segment, symbol, interval_map[timeframe], days_map[timeframe])

if hist.empty or len(hist) < 30:
    st.error("🔴 Historical candles unavailable — technical trade calculation disabled.")
    st.stop()

x = add_indicators(hist)
s = smc(x)
d = decision(x, s, live_price)

# Data-integrity gate.
valid = (
    market_open()
    and feed_age <= STALE_SECONDS
    and np.isfinite(live_price)
    and np.isfinite(float(x["RSI"].iloc[-1]))
    and np.isfinite(float(x["ATR"].iloc[-1]))
)

st.info(
    f"📡 **Groww LIVE** | {symbol} | LTP **₹{live_price:,.2f}** | "
    f"Tick age **{feed_age:.1f}s** | {live.get('source', 'Groww')} | {now_ist().strftime('%H:%M:%S IST')}"
)

if valid:
    st.success("🟢 DATA INTEGRITY CHECK PASSED — live price is fresh")
else:
    st.error("🔴 DATA INTEGRITY CHECK FAILED — NO TRADE")

# Single-screen decision panel
st.subheader("🎯 Trade Decision")
if not valid:
    st.error("NO TRADE — market/data condition is not reliable enough.")
elif d["bias"] == "BULLISH":
    st.success("🟢 BULLISH SETUP — wait for entry confirmation")
elif d["bias"] == "BEARISH":
    st.error("🔴 BEARISH SETUP — wait for entry confirmation")
else:
    st.warning("🟡 WAIT — trend confirmation is not strong enough")

c = st.columns(6)
c[0].metric("Live LTP", f"₹{live_price:,.2f}")
c[1].metric("RSI", f"{x.RSI.iloc[-1]:.2f}")
c[2].metric("ADX", f"{x.ADX.iloc[-1]:.2f}")
c[3].metric("VWAP", f"₹{x.VWAP.iloc[-1]:,.2f}" if np.isfinite(x.VWAP.iloc[-1]) else "N/A")
c[4].metric("ATR", f"₹{x.ATR.iloc[-1]:,.2f}")
c[5].metric("AI Strength", f"{d['confidence']}%")

if valid and d["bias"] != "NEUTRAL":
    p = live_price
    c = st.columns(4)
    c[0].metric("Entry", f"₹{p:,.2f}")
    c[1].metric("Stop Loss", f"₹{d['sl']:,.2f}")
    c[2].metric("Target 1", f"₹{d['t1']:,.2f}")
    c[3].metric("Target 2", f"₹{d['t2']:,.2f}")

c1, c2 = st.columns(2)
with c1:
    st.subheader("🧠 SMC Snapshot")
    st.write(f"**Structure:** {s['structure']}")
    st.write(f"**BOS:** {s['bos']}")
    st.write(f"**FVG:** {s['fvg']}")
with c2:
    st.subheader("📐 Risk / Context")
    st.write("**Mode:** Read-only")
    st.write("**Order placement:** Disabled in code")
    st.write("**Data source:** Groww API")

with st.expander("🤖 AI Voice Assistant"):
    if d["bias"] == "BULLISH":
        answer = f"Live price is {live_price:.2f}. The dashboard sees a bullish setup, but this is decision support, not a guaranteed call. Wait for your entry confirmation. Stop loss is around {d['sl']:.2f}, target one {d['t1']:.2f}."
    elif d["bias"] == "BEARISH":
        answer = f"Live price is {live_price:.2f}. The dashboard sees a bearish setup, but this is decision support, not a guaranteed call. Wait for confirmation. Stop loss is around {d['sl']:.2f}, target one {d['t1']:.2f}."
    else:
        answer = f"Live price is {live_price:.2f}. I would wait because the trend is not strong enough. Do not force a trade."
    st.write(answer)
    voice_box(answer)
    st.caption("Voice input uses your browser. The assistant does not place orders.")

st.subheader("📈 Price Action Chart")
chart = x.tail(180).copy()
fig = go.Figure(go.Candlestick(x=chart.index, open=chart.Open, high=chart.High, low=chart.Low, close=chart.Close, name="Price"))
for col, label in [("EMA20","EMA 20"),("EMA50","EMA 50"),("EMA200","EMA 200"),("VWAP","VWAP")]:
    if col in chart:
        fig.add_trace(go.Scatter(x=chart.index, y=chart[col], mode="lines", name=label))
fig.update_layout(height=600, xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=20,b=10), uirevision=symbol)
st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "responsive": True})

with st.expander("Latest indicator values"):
    st.dataframe(x.tail(10).round(2), use_container_width=True)

st.caption("⚠️ This dashboard is analytical decision-support only. Live market feeds can still fail at the broker/network/exchange level. If the integrity gate fails, the dashboard deliberately refuses to issue a trade setup.")

# Auto refresh without using an order API.
time.sleep(REFRESH_SECONDS)
st.rerun()
    score += 1 if r.RSI>=60 else -1 if r.RSI<=40 else 0
    score += 1 if r.MACD>r.MACDSignal else -1
    score += 2 if s["structure"]=="BULLISH" else -2 if s["structure"]=="BEARISH" else 0
    score += 2 if s["bos"]=="Bullish BOS" else -2 if s["bos"]=="Bearish BOS" else 0
    bias="BULLISH" if score>=4 else "BEARISH" if score<=-4 else "NEUTRAL"
    atr=float(r.ATR) if pd.notna(r.ATR) and r.ATR>0 else p*.005
    if bias=="BULLISH":
        risk=max(p-s["low"],1.2*atr); sl=p-risk; t1=p+1.5*risk; t2=p+2.5*risk; el=p-.25*atr; eh=p+.1*atr
    elif bias=="BEARISH":
        risk=max(s["high"]-p,1.2*atr); sl=p+risk; t1=p-1.5*risk; t2=p-2.5*risk; el=p-.1*atr; eh=p+.25*atr
    else: el=eh=sl=t1=t2=np.nan
    return {"score":score,"bias":bias,"confidence":int(np.clip(50+score*7,5,95)),"el":el,"eh":eh,"sl":sl,"t1":t1,"t2":t2}

st.title("📊 Sovendra Market AI")
st.caption("AI-assisted SMC + Price Action + Technical Analysis + Risk Engine")

with st.sidebar:
    name=st.selectbox("Instrument",list(SYMBOLS))
    symbol=SYMBOLS[name]
    tf=st.selectbox("Timeframe",["5m","15m","30m","1h","1d"],index=1)
    st.caption("Auto refresh: 15 seconds")
    if st.button("🔄 Refresh Now"):
        st.cache_data.clear(); st.rerun()

period = "1y" if tf=="1d" else ("730d" if tf=="1h" else "60d")
df=load_data(symbol, period, tf)
if df.empty or len(df)<30:
    st.error("Market data unavailable. Yahoo Finance may be temporarily rate-limited or the selected interval may not be available.")
    st.info("Try Refresh Now, or switch timeframe to 1h/1d. Intraday Yahoo data has provider-side limits and delays.")
    st.stop()
x=indicators(df); s=smc(x); d=decision(x,s)
status=market_status(); age=data_age(x.index[-1])
st.info(f"📡 NSE Status: **{status}** | Latest candle: **{ist(x.index[-1]).strftime('%d-%b-%Y %H:%M IST')}** | Data age: **~{age} min** | Feed: **Yahoo/yfinance (delayed)**")
if status=="CLOSED" or age>30: st.warning("⛔ TRADE DISABLED — market closed or data is stale.")
else: st.success("🟢 Data freshness check passed.")

r=x.iloc[-1]
c=st.columns(5)
c[0].metric("Price",f"{r.Close:,.2f}"); c[1].metric("RSI",f"{r.RSI:.2f}"); c[2].metric("ADX",f"{r.ADX:.2f}" if pd.notna(r.ADX) else "N/A")
c[3].metric("VWAP",f"{r.VWAP:,.2f}" if pd.notna(r.VWAP) else "N/A"); c[4].metric("Volume",f"{r.Volume:,.0f}" if r.Volume>0 else "N/A")

st.subheader("🤖 AI Decision Engine")
c=st.columns(4); c[0].metric("Bias",d["bias"]); c[1].metric("Confidence",f"{d['confidence']}%"); c[2].metric("AI Score",f"{d['score']:+d}")
trade_ok=status=="OPEN" and age<=30 and d["bias"]!="NEUTRAL"
c[3].metric("Decision","TRADE SETUP" if trade_ok else "WAIT / NO TRADE")
if trade_ok:
    st.success(f"🟢 {d['bias']} setup — wait for entry-zone confirmation.")
    z=st.columns(5); z[0].metric("Entry",f"{d['el']:.2f}–{d['eh']:.2f}"); z[1].metric("SL",f"{d['sl']:.2f}"); z[2].metric("T1",f"{d['t1']:.2f}"); z[3].metric("T2",f"{d['t2']:.2f}")
    z[4].metric("R:R T1",f"1:{abs(d['t1']-d['eh'])/abs(d['eh']-d['sl']):.2f}")
else: st.warning("🟡 WAIT / NO TRADE — conditions are not sufficiently aligned.")

st.subheader("🧠 SMC Snapshot")
c=st.columns(4); c[0].metric("Structure",s["structure"]); c[1].metric("BOS",s["bos"]); c[2].metric("FVG",s["fvg"]); c[3].metric("Liquidity","Not confirmed")

with st.expander("📈 Price Action Chart — Minimize / Maximize",expanded=True):
    q=x.tail(180); fig=go.Figure(go.Candlestick(x=q.index,open=q.Open,high=q.High,low=q.Low,close=q.Close,name="Price"))
    for col,label in [("EMA20","EMA20"),("EMA50","EMA50"),("EMA200","EMA200"),("VWAP","VWAP")]:
        if q[col].notna().any(): fig.add_trace(go.Scatter(x=q.index,y=q[col],mode="lines",name=label))
    for val,label in [(s["high"],"Swing High"),(s["low"],"Swing Low"),(d["sl"],"SL"),(d["t1"],"T1"),(d["t2"],"T2")]:
        if pd.notna(val): fig.add_hline(y=val,line_dash="dot",annotation_text=label)
    if trade_ok: fig.add_hrect(y0=d["el"],y1=d["eh"],line_width=0,annotation_text="Entry Zone")
    fig.update_layout(height=620,xaxis_rangeslider_visible=False,margin=dict(l=10,r=10,t=20,b=10),legend=dict(orientation="h"))
    st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False,"responsive":True,"scrollZoom":True})

st.caption("⚠️ Decision-support only. Yahoo/yfinance is delayed and is not an exchange-grade real-time feed. Verify live broker price before any order.")
st.markdown('<meta http-equiv="refresh" content="15">',unsafe_allow_html=True)
