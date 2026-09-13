import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta

st.set_page_config(page_title="Market Pulse Dashboard", layout="wide")

st.title("📈 Market Pulse Live Dashboard")

# Top Indices Tickers
tickers = {
    "NIFTY 50": "^NSEI",
    "INDIA VIX": "^INDIAVIX",
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC"
}

st.subheader("Global & Domestic Indices")
cols = st.columns(len(tickers))

for col, (name, sym) in zip(cols, tickers.items()):
    try:
        t = yf.Ticker(sym)
        hist = t.history(period="2d")
        if len(hist) >= 2:
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2]
            change = current_price - prev_close
            pct_change = (change / prev_close) * 100
            col.metric(name, f"{current_price:,.2f}", f"{change:+,.2f} ({pct_change:+.2f}%)")
        elif len(hist) == 1:
            current_price = hist['Close'].iloc[-1]
            col.metric(name, f"{current_price:,.2f}")
    except Exception:
        col.metric(name, "N/A")

st.markdown("---")

# Stock / Index Analysis Engine
st.subheader("Technical Signal Engine")
symbol = st.text_input("Enter NSE/Global Ticker (e.g. ^NSEI, RELIANCE.NS, TCS.NS, AAPL):", value="^NSEI")

if symbol:
    try:
        df = yf.download(symbol, period="3mo", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if not df.empty and len(df) > 50:
            # Indicator Calculations
            df['EMA20'] = ta.trend.ema_indicator(df['Close'], window=20)
            df['EMA50'] = ta.trend.ema_indicator(df['Close'], window=50)
            df['RSI'] = ta.momentum.rsi(df['Close'], window=14)

            latest_close = df['Close'].iloc[-1]
            ema20 = df['EMA20'].iloc[-1]
            ema50 = df['EMA50'].iloc[-1]
            rsi = df['RSI'].iloc[-1]

            # Pivot Points
            high = df['High'].iloc[-2]
            low = df['Low'].iloc[-2]
            close = df['Close'].iloc[-2]
            pivot = (high + low + close) / 3
            r1 = (2 * pivot) - low
            s1 = (2 * pivot) - high

            # Signal Logic
            if latest_close > ema20 > ema50 and rsi > 55:
                signal = "BULLISH 🚀"
                entry = latest_close
                sl = s1
                target = r1
                badge_type = "success"
            elif latest_close < ema20 < ema50 and rsi < 45:
                signal = "BEARISH 🔻"
                entry = latest_close
                sl = r1
                target = s1
                badge_type = "error"
            else:
                signal = "SIDEWAYS / AVOID ⚖️"
                entry = latest_close
                sl = s1
                target = r1
                badge_type = "warning"

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Current Price", f"{latest_close:.2f}")
            c2.metric("20 EMA", f"{ema20:.2f}")
            c3.metric("50 EMA", f"{ema50:.2f}")
            c4.metric("RSI (14)", f"{rsi:.2f}")

            if badge_type == "success":
                st.success(f"**Signal: {signal}**")
            elif badge_type == "error":
                st.error(f"**Signal: {signal}**")
            else:
                st.warning(f"**Signal: {signal}**")

            st.write(f"- **Suggested Entry Level:** `{entry:.2f}`")
            st.write(f"- **Key Support / Stop-Loss:** `{sl:.2f}`")
            st.write(f"- **Key Resistance / Target:** `{target:.2f}`")

            st.line_chart(df[['Close', 'EMA20', 'EMA50']].tail(30))
        else:
            st.warning("Insufficient data available for this ticker.")
    except Exception as e:
        st.error(f"Error fetching data: {e}")
