import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Sovendra Market AI", page_icon="📊", layout="wide")

REFRESH_SECONDS = 15
SYMBOLS = {
    "NIFTY 50":"^NSEI","BANKNIFTY":"^NSEBANK","SENSEX":"^BSESN",
    "INDIA VIX":"^INDIAVIX","RELIANCE":"RELIANCE.NS","TCS":"TCS.NS",
    "HDFCBANK":"HDFCBANK.NS","SBIN":"SBIN.NS","ICICIBANK":"ICICIBANK.NS","ITC":"ITC.NS"
}

@st.cache_data(ttl=15, show_spinner=False)
def load_data(symbol, period, interval):
    try:
        df=yf.download(symbol,period=period,interval=interval,auto_adjust=False,progress=False,threads=False)
        if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
        cols=[c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
        return df[cols].dropna(subset=["Open","High","Low","Close"])
    except Exception:
        return pd.DataFrame()

def ist(ts):
    t=pd.Timestamp(ts)
    return t.tz_localize("Asia/Kolkata") if t.tzinfo is None else t.tz_convert("Asia/Kolkata")

def data_age(ts):
    return max(0,int((pd.Timestamp.now(tz="Asia/Kolkata")-ist(ts)).total_seconds()/60))

def market_status():
    n=pd.Timestamp.now(tz="Asia/Kolkata")
    if n.weekday()>=5:return "CLOSED"
    mins=n.hour*60+n.minute
    return "OPEN" if 555<=mins<=930 else "CLOSED"

def indicators(x):
    x=x.copy()
    x["EMA20"]=x.Close.ewm(span=20,adjust=False).mean()
    x["EMA50"]=x.Close.ewm(span=50,adjust=False).mean()
    x["EMA200"]=x.Close.ewm(span=200,adjust=False).mean()
    d=x.Close.diff(); gain=d.clip(lower=0).rolling(14).mean(); loss=(-d.clip(upper=0)).rolling(14).mean()
    x["RSI"]=100-100/(1+gain/loss.replace(0,np.nan))
    e12=x.Close.ewm(span=12,adjust=False).mean(); e26=x.Close.ewm(span=26,adjust=False).mean()
    x["MACD"]=e12-e26; x["MACDSignal"]=x.MACD.ewm(span=9,adjust=False).mean()
    tr=pd.concat([x.High-x.Low,(x.High-x.Close.shift()).abs(),(x.Low-x.Close.shift()).abs()],axis=1).max(axis=1)
    x["ATR"]=tr.rolling(14).mean()
    p=x.High.diff().clip(lower=0).rolling(14).mean(); m=(-x.Low.diff()).clip(lower=0).rolling(14).mean()
    dx=100*(p-m).abs()/(p+m).replace(0,np.nan)/x.ATR.replace(0,np.nan)*x.ATR.replace(0,np.nan)
    x["ADX"]=dx.rolling(14).mean()
    v=x.Volume.fillna(0)
    if v.sum()>0:
        typ=(x.High+x.Low+x.Close)/3
        day=x.index.tz_convert("Asia/Kolkata").date if getattr(x.index,"tz",None) else x.index.date
        x["VWAP"]=(typ*v).groupby(day).cumsum()/v.groupby(day).cumsum().replace(0,np.nan)
    else:x["VWAP"]=np.nan
    return x

def smc(x):
    w=x.tail(100); n=5
    hi=w.High.rolling(11,center=True).max(); lo=w.Low.rolling(11,center=True).min()
    hs=w[w.High.eq(hi)].High.dropna(); ls=w[w.Low.eq(lo)].Low.dropna()
    h=float(hs.iloc[-1]) if len(hs) else float(w.High.iloc[-1]); l=float(ls.iloc[-1]) if len(ls) else float(w.Low.iloc[-1])
    ph=float(hs.iloc[-2]) if len(hs)>1 else h; pl=float(ls.iloc[-2]) if len(ls)>1 else l
    price=float(w.Close.iloc[-1])
    bull=price>ph; bear=price<pl
    structure="BULLISH" if bull or (h>=ph and l>=pl) else "BEARISH" if bear or (h<=ph and l<=pl) else "RANGE"
    fvg="None"
    if len(w)>=3:
        a,c=w.iloc[-3],w.iloc[-1]
        if c.Low>a.High:fvg="Bullish FVG"
        elif c.High<a.Low:fvg="Bearish FVG"
    return {"structure":structure,"bos":"Bullish BOS" if bull else "Bearish BOS" if bear else "No fresh BOS","fvg":fvg,"high":h,"low":l}

def decision(x,s):
    r=x.iloc[-1]; p=float(r.Close); score=0; why=[]
    if p>r.EMA20>r.EMA50:score+=2;why.append("Price above EMA20/50")
    elif p<r.EMA20<r.EMA50:score-=2;why.append("Price below EMA20/50")
    else:why.append("EMA trend mixed")
    score += 1 if p>r.EMA200 else -1
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
if df.empty or len(df)<30:\n    st.error("Market data unavailable. Yahoo Finance may be temporarily rate-limited or the selected interval may not be available.")\n    st.info("Try Refresh Now, or switch timeframe to 1h/1d. Intraday Yahoo data has provider-side limits and delays.")\n    st.stop()
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
if df.empty or len(df)<30:\n    st.error("Market data unavailable. Yahoo Finance may be temporarily rate-limited or the selected interval may not be available.")\n    st.info("Try Refresh Now, or switch timeframe to 1h/1d. Intraday Yahoo data has provider-side limits and delays.")\n    st.stop()
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

df=load_data(symbol,"1y" if tf=="1d" else "3mo",tf)
if df.empty or len(df)<60: st.error("Market data unavailable."); st.stop()
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
