import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go

# ============================================================
# SOVENDRA MARKET AI V3
# Upgraded from the original Market Pulse Dashboard:
# - Cached market data
# - Data freshness display
# - EMA20/50/200, RSI, MACD, ADX, ATR, VWAP, Volume
# - Multi-factor signal score
# - Support / resistance
# - Basic SMC: swing structure, BOS, FVG
# - Risk/reward trade setup
# - Candlestick chart
# - Mobile-friendly layout
# ============================================================

st.set_page_config(
    page_title="Sovendra Market AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------
# Configuration
# ---------------------------
MARKETS = {
    "NIFTY 50": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "INDIA VIX": "^INDIAVIX",
    "SENSEX": "^BSESN",
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
    "DOW JONES": "^DJI",
}

QUICK_SYMBOLS = {
    "NIFTY 50": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "SENSEX": "^BSESN",
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "SBIN": "SBIN.NS",
    "TATA POWER": "TATAPOWER.NS",
    "ITC": "ITC.NS",
}

INDEX_SYMBOLS = {
    "^NSEI", "^NSEBANK", "^BSESN", "^INDIAVIX",
    "^GSPC", "^IXIC", "^DJI"
}

TIMEFRAMES = {
    "5m": {"period": "5d", "interval": "5m"},
    "15m": {"period": "30d", "interval": "15m"},
    "30m": {"period": "60d", "interval": "30m"},
    "1H": {"period": "730d", "interval": "1h"},
    "1D": {"period": "2y", "interval": "1d"},
}

# ---------------------------
# Styling
# ---------------------------
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    .decision-card {
        padding: 1rem 1.2rem;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,.25);
        margin: .5rem 0 1rem 0;
    }

    .small-muted {
        font-size: .85rem;
        opacity: .75;
    }

    @media (max-width: 768px) {
        .block-container {
            padding-left: .75rem;
            padding-right: .75rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# DATA
# ============================================================
def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close"]
    if any(col not in out.columns for col in required):
        return pd.DataFrame()

    if "Volume" not in out.columns:
        out["Volume"] = 0

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["Open", "High", "Low", "Close"])

    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")

    out = out[~out.index.isna()].sort_index()
    return out


@st.cache_data(ttl=60, show_spinner=False)
def download_data(symbol: str, period: str, interval: str) -> pd.DataFrame:
    try:
        df = yf.download(
            symbol,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        return clean_ohlcv(df)
    except Exception:
        return pd.DataFrame()


def format_timestamp(ts):
    if ts is None:
        return "N/A"

    try:
        if getattr(ts, "tzinfo", None) is not None:
            ts = ts.tz_convert("Asia/Kolkata")
        return ts.strftime("%d-%b-%Y %H:%M IST")
    except Exception:
        return str(ts)


# ============================================================
# INDICATORS
# ============================================================
def add_indicators(df: pd.DataFrame, intraday: bool = False) -> pd.DataFrame:
    out = df.copy()

    out["EMA20"] = ta.trend.ema_indicator(out["Close"], window=20)
    out["EMA50"] = ta.trend.ema_indicator(out["Close"], window=50)
    out["EMA200"] = ta.trend.ema_indicator(out["Close"], window=200)

    out["RSI"] = ta.momentum.rsi(out["Close"], window=14)

    out["MACD"] = ta.trend.macd(out["Close"])
    out["MACD_SIGNAL"] = ta.trend.macd_signal(out["Close"])
    out["MACD_HIST"] = ta.trend.macd_diff(out["Close"])

    out["ADX"] = ta.trend.adx(
        out["High"],
        out["Low"],
        out["Close"],
        window=14,
    )

    out["ATR"] = ta.volatility.average_true_range(
        out["High"],
        out["Low"],
        out["Close"],
        window=14,
    )

    out["VOL_MA20"] = out["Volume"].rolling(20).mean()

    typical = (out["High"] + out["Low"] + out["Close"]) / 3

    if out["Volume"].fillna(0).sum() <= 0:
        out["VWAP"] = np.nan
    elif intraday:
        try:
            session_key = out.index.tz_convert("Asia/Kolkata").date
        except Exception:
            session_key = out.index.date

        pv = typical * out["Volume"]
        volume_cum = out["Volume"].groupby(session_key).cumsum()
        pv_cum = pv.groupby(session_key).cumsum()
        out["VWAP"] = pv_cum / volume_cum.replace(0, np.nan)
    else:
        rolling_volume = out["Volume"].rolling(20).sum()
        rolling_pv = (typical * out["Volume"]).rolling(20).sum()
        out["VWAP"] = rolling_pv / rolling_volume.replace(0, np.nan)

    return out


# ============================================================
# SUPPORT / RESISTANCE
# ============================================================
def nearest_levels(df: pd.DataFrame, lookback: int = 60):
    recent = df.tail(lookback).copy()
    price = float(df["Close"].iloc[-1])

    below = recent.loc[recent["Low"] < price, "Low"]
    above = recent.loc[recent["High"] > price, "High"]

    support = (
        float(below.max())
        if not below.empty
        else float(recent["Low"].min())
    )

    resistance = (
        float(above.min())
        if not above.empty
        else float(recent["High"].max())
    )

    if support >= price:
        support = price

    if resistance <= price:
        resistance = price

    return support, resistance


# ============================================================
# BASIC SMC ENGINE
# ============================================================
def smc_analysis(df: pd.DataFrame, swing_window: int = 2):
    out = df.copy()

    window = swing_window * 2 + 1

    out["SwingHigh"] = out["High"].where(
        out["High"] == out["High"].rolling(window, center=True).max()
    )

    out["SwingLow"] = out["Low"].where(
        out["Low"] == out["Low"].rolling(window, center=True).min()
    )

    swing_highs = out["SwingHigh"].dropna()
    swing_lows = out["SwingLow"].dropna()

    last_swing_high = (
        float(swing_highs.iloc[-1])
        if not swing_highs.empty
        else np.nan
    )

    last_swing_low = (
        float(swing_lows.iloc[-1])
        if not swing_lows.empty
        else np.nan
    )

    previous_high = (
        float(swing_highs.iloc[-2])
        if len(swing_highs) >= 2
        else last_swing_high
    )

    previous_low = (
        float(swing_lows.iloc[-2])
        if len(swing_lows) >= 2
        else last_swing_low
    )

    price = float(out["Close"].iloc[-1])

    if not np.isnan(previous_high) and price > previous_high:
        bos = "BULLISH BOS"
    elif not np.isnan(previous_low) and price < previous_low:
        bos = "BEARISH BOS"
    else:
        bos = "NO FRESH BOS"

    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        hh = swing_highs.iloc[-1] > swing_highs.iloc[-2]
        hl = swing_lows.iloc[-1] > swing_lows.iloc[-2]
        lh = swing_highs.iloc[-1] < swing_highs.iloc[-2]
        ll = swing_lows.iloc[-1] < swing_lows.iloc[-2]

        if hh and hl:
            structure = "BULLISH (HH/HL)"
        elif lh and ll:
            structure = "BEARISH (LH/LL)"
        else:
            structure = "MIXED"
    else:
        structure = "INSUFFICIENT SWINGS"

    # Basic three-candle FVG.
    recent = out.tail(30)

    bullish_fvg = False
    bearish_fvg = False

    if len(recent) >= 3:
        high_two_back = recent["High"].shift(2)
        low_two_back = recent["Low"].shift(2)

        bullish_fvg = bool(
            (recent["Low"] > high_two_back).iloc[-1]
        )

        bearish_fvg = bool(
            (recent["High"] < low_two_back).iloc[-1]
        )

    if bullish_fvg:
        fvg = "BULLISH FVG"
    elif bearish_fvg:
        fvg = "BEARISH FVG"
    else:
        fvg = "NO FRESH FVG"

    return {
        "structure": structure,
        "bos": bos,
        "fvg": fvg,
        "last_swing_high": last_swing_high,
        "last_swing_low": last_swing_low,
        "swing_highs": swing_highs,
        "swing_lows": swing_lows,
    }


# ============================================================
# MARKET SNAPSHOT + MOOD
# ============================================================
def market_snapshot(symbols):
    rows = []

    for name, symbol in symbols.items():
        df = download_data(symbol, "5d", "1d")

        if df.empty:
            rows.append({
                "name": name,
                "price": np.nan,
                "pct": np.nan,
            })
            continue

        price = float(df["Close"].iloc[-1])

        previous = (
            float(df["Close"].iloc[-2])
            if len(df) >= 2
            else np.nan
        )

        pct = (
            (price - previous) / previous * 100
            if np.isfinite(previous) and previous != 0
            else np.nan
        )

        rows.append({
            "name": name,
            "price": price,
            "pct": pct,
        })

    return pd.DataFrame(rows)


def calculate_mood(snapshot: pd.DataFrame, technical_signal="NEUTRAL"):
    score = 50.0

    def get_pct(name):
        row = snapshot.loc[
            snapshot["name"] == name,
            "pct",
        ]

        if len(row) and pd.notna(row.iloc[0]):
            return float(row.iloc[0])

        return 0.0

    nifty = get_pct("NIFTY 50")
    vix = get_pct("INDIA VIX")
    sp500 = get_pct("S&P 500")
    nasdaq = get_pct("NASDAQ")

    score += np.clip(nifty * 8, -15, 15)

    # Rising VIX generally increases risk-off pressure.
    score += np.clip(-vix * 4, -10, 10)

    score += np.clip(sp500 * 4, -8, 8)
    score += np.clip(nasdaq * 3, -6, 6)

    if technical_signal == "BULLISH":
        score += 6
    elif technical_signal == "BEARISH":
        score -= 6

    score = float(np.clip(score, 0, 100))

    if score >= 80:
        label = "EXTREME BULLISH"
    elif score >= 60:
        label = "BULLISH"
    elif score >= 40:
        label = "NEUTRAL"
    elif score >= 20:
        label = "BEARISH"
    else:
        label = "EXTREME BEARISH"

    return score, label


# ============================================================
# MULTI-FACTOR SIGNAL ENGINE
# ============================================================
def generate_signal(df: pd.DataFrame, smc: dict, mood_label: str):
    latest = df.iloc[-1]

    price = float(latest["Close"])
    ema20 = float(latest["EMA20"])
    ema50 = float(latest["EMA50"])

    ema200 = (
        float(latest["EMA200"])
        if pd.notna(latest["EMA200"])
        else ema50
    )

    rsi = float(latest["RSI"])
    macd = float(latest["MACD"])
    macd_signal = float(latest["MACD_SIGNAL"])
    adx = float(latest["ADX"])

    vwap = (
        float(latest["VWAP"])
        if pd.notna(latest["VWAP"])
        else np.nan
    )

    volume = float(latest["Volume"])

    vol_ma = (
        float(latest["VOL_MA20"])
        if pd.notna(latest["VOL_MA20"])
        else volume
    )

    score = 0
    reasons = []

    # Trend
    if price > ema20:
        score += 1
        reasons.append("Price above EMA20")
    else:
        score -= 1
        reasons.append("Price below EMA20")

    if ema20 > ema50:
        score += 2
        reasons.append("EMA20 above EMA50")
    else:
        score -= 2
        reasons.append("EMA20 below EMA50")

    if price > ema200:
        score += 1
        reasons.append("Price above EMA200")
    else:
        score -= 1
        reasons.append("Price below EMA200")

    # VWAP: score only when the data feed provides usable volume.
    if np.isfinite(vwap):
        if price > vwap:
            score += 1
            reasons.append("Price above VWAP")
        else:
            score -= 1
            reasons.append("Price below VWAP")
    else:
        reasons.append("VWAP unavailable for this data feed")

    # RSI
    # Oversold is NOT treated as an automatic short.
    if 55 <= rsi <= 70:
        score += 1
        reasons.append("RSI bullish")
    elif 30 <= rsi < 45:
        score -= 1
        reasons.append("RSI bearish")
    elif rsi < 30:
        reasons.append(
            "RSI oversold — short confirmation required"
        )
    elif rsi > 70:
        reasons.append(
            "RSI overbought — long confirmation required"
        )

    # MACD
    if macd > macd_signal:
        score += 1
        reasons.append("MACD bullish")
    else:
        score -= 1
        reasons.append("MACD bearish")

    # ADX
    if adx >= 20:
        reasons.append(
            f"Trend strength active (ADX {adx:.1f})"
        )
    else:
        reasons.append(
            f"Trend strength weak (ADX {adx:.1f})"
        )

    # Volume
    if pd.notna(vol_ma) and vol_ma > 0 and volume > 1.2 * vol_ma:
        candle_bullish = (
            float(latest["Close"]) >= float(latest["Open"])
        )

        if candle_bullish:
            score += 1
        else:
            score -= 1

        reasons.append("Volume above 20-period average")
    else:
        reasons.append(
            "Volume unavailable for this index/data feed"
            if volume == 0
            else "Volume confirmation weak"
        )

    # SMC
    if (
        "BULLISH" in smc["bos"]
        or "BULLISH" in smc["structure"]
    ):
        score += 2
        reasons.append("SMC bullish structure")

    elif (
        "BEARISH" in smc["bos"]
        or "BEARISH" in smc["structure"]
    ):
        score -= 2
        reasons.append("SMC bearish structure")

    # Market mood
    if mood_label in ("BULLISH", "EXTREME BULLISH"):
        score += 1
        reasons.append("Market mood bullish")

    elif mood_label in ("BEARISH", "EXTREME BEARISH"):
        score -= 1
        reasons.append("Market mood bearish")

    if score >= 6:
        signal = "BULLISH"
    elif score <= -6:
        signal = "BEARISH"
    else:
        signal = "NEUTRAL"

    # This is signal-strength, not probability of profit.
    confidence = int(
        np.clip(50 + abs(score) * 4, 50, 90)
    )

    # Confirmation gate.
    confirmation = True
    confirmation_reasons = []

    if signal == "BULLISH":
        if np.isfinite(vwap) and price <= vwap:
            confirmation = False
            confirmation_reasons.append(
                "Price is below VWAP"
            )

        if rsi > 75:
            confirmation = False
            confirmation_reasons.append(
                "RSI excessively overbought"
            )

    elif signal == "BEARISH":
        if np.isfinite(vwap) and price >= vwap:
            confirmation = False
            confirmation_reasons.append(
                "Price is above VWAP"
            )

        if rsi < 25:
            confirmation = False
            confirmation_reasons.append(
                "RSI deeply oversold"
            )

    return {
        "signal": signal,
        "score": score,
        "confidence": confidence,
        "reasons": reasons,
        "confirmation": confirmation,
        "confirmation_reasons": confirmation_reasons,
    }


# ============================================================
# TRADE SETUP / RISK
# ============================================================
def trade_setup(df: pd.DataFrame, signal: str):
    price = float(df["Close"].iloc[-1])
    atr = float(df["ATR"].iloc[-1])

    if not np.isfinite(atr) or atr <= 0:
        atr = price * 0.01

    support, resistance = nearest_levels(df)

    if signal == "BULLISH":
        structure_sl = (
            support
            if support < price
            else price - 1.2 * atr
        )

        sl = min(
            structure_sl,
            price - 0.8 * atr,
        )

        risk = price - sl

        target1 = price + 1.5 * risk
        target2 = price + 2.5 * risk

        entry_low = price - 0.25 * atr
        entry_high = price + 0.10 * atr

    elif signal == "BEARISH":
        structure_sl = (
            resistance
            if resistance > price
            else price + 1.2 * atr
        )

        sl = max(
            structure_sl,
            price + 0.8 * atr,
        )

        risk = sl - price

        target1 = price - 1.5 * risk
        target2 = price - 2.5 * risk

        entry_low = price - 0.10 * atr
        entry_high = price + 0.25 * atr

    else:
        return {
            "entry_low": price,
            "entry_high": price,
            "sl": np.nan,
            "target1": np.nan,
            "target2": np.nan,
            "rr": np.nan,
            "support": support,
            "resistance": resistance,
        }

    rr = (
        abs(target1 - price) / abs(price - sl)
        if sl != price
        else np.nan
    )

    return {
        "entry_low": entry_low,
        "entry_high": entry_high,
        "sl": sl,
        "target1": target1,
        "target2": target2,
        "rr": rr,
        "support": support,
        "resistance": resistance,
    }


# ============================================================
# CHART
# ============================================================
def make_chart(df: pd.DataFrame, smc: dict, setup: dict, signal: str):
    chart_df = df.tail(150).copy()
    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=chart_df.index,
            open=chart_df["Open"],
            high=chart_df["High"],
            low=chart_df["Low"],
            close=chart_df["Close"],
            name="Price",
        )
    )

    for column, name in [
        ("EMA20", "EMA 20"),
        ("EMA50", "EMA 50"),
        ("EMA200", "EMA 200"),
        ("VWAP", "VWAP"),
    ]:
        if column in chart_df.columns and chart_df[column].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=chart_df.index,
                    y=chart_df[column],
                    mode="lines",
                    name=name,
                )
            )

    for level, label in [
        (setup["support"], "Support"),
        (setup["resistance"], "Resistance"),
        (setup["sl"], "Stop Loss"),
        (setup["target1"], "Target 1"),
        (setup["target2"], "Target 2"),
    ]:
        if np.isfinite(level):
            fig.add_hline(
                y=level,
                line_dash="dot",
                annotation_text=label,
            )

    if signal != "NEUTRAL":
        fig.add_hrect(
            y0=setup["entry_low"],
            y1=setup["entry_high"],
            line_width=0,
            annotation_text="Entry Zone",
        )

    if np.isfinite(smc["last_swing_high"]):
        fig.add_hline(
            y=smc["last_swing_high"],
            line_dash="dash",
            annotation_text="Swing High",
        )

    if np.isfinite(smc["last_swing_low"]):
        fig.add_hline(
            y=smc["last_swing_low"],
            line_dash="dash",
            annotation_text="Swing Low",
        )

    fig.update_layout(
        height=620,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h"),
    )
    return fig


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("⚙️ Dashboard Controls")

    selected_quick = st.selectbox(
        "Quick Select",
        list(QUICK_SYMBOLS.keys()),
        index=0,
    )

    custom_symbol = st.text_input(
        "Or enter ticker",
        placeholder="RELIANCE.NS / TCS.NS / AAPL",
    ).strip()

    symbol = (
        custom_symbol
        if custom_symbol
        else QUICK_SYMBOLS[selected_quick]
    )

    timeframe = st.selectbox(
        "Chart timeframe",
        list(TIMEFRAMES.keys()),
        index=1,
    )

    if st.button(
        "🔄 Refresh data",
        use_container_width=True,
    ):
        st.cache_data.clear()
        st.rerun()

    st.caption(
        "Data is sourced from Yahoo Finance through yfinance. "
        "Availability and freshness can vary."
    )

# ============================================================
# HEADER
# ============================================================
st.title("📊 Sovendra Market AI")

st.caption(
    "Technical + market mood + basic SMC decision-support. "
    "Signals are analytical and are not guaranteed trade calls."
)

# ============================================================
# MARKET PULSE
# ============================================================
st.subheader("🌍 Global & Domestic Market Pulse")

pulse_columns = st.columns(4)

for i, (name, market_symbol) in enumerate(MARKETS.items()):
    df_market = download_data(
        market_symbol,
        "5d",
        "1d",
    )

    with pulse_columns[i % 4]:
        if df_market.empty:
            st.metric(name, "N/A")
            continue

        current = float(
            df_market["Close"].iloc[-1]
        )

        previous = (
            float(df_market["Close"].iloc[-2])
            if len(df_market) >= 2
            else np.nan
        )

        if np.isfinite(previous):
            delta = current - previous
            pct = delta / previous * 100

            st.metric(
                name,
                f"{current:,.2f}",
                f"{delta:+,.2f} ({pct:+.2f}%)",
            )
        else:
            st.metric(
                name,
                f"{current:,.2f}",
            )

st.divider()

# ============================================================
# ANALYZER DATA
# ============================================================
tf = TIMEFRAMES[timeframe]

raw_df = download_data(
    symbol,
    tf["period"],
    tf["interval"],
)

if raw_df.empty:
    st.error(
        f"Unable to fetch data for `{symbol}`. "
        "Check the ticker symbol or choose another timeframe."
    )
    st.stop()

intraday = timeframe != "1D"

df = add_indicators(
    raw_df,
    intraday=intraday,
)

analysis_df = df.dropna(
    subset=["EMA20", "EMA50", "RSI", "ATR"]
).copy()

if len(analysis_df) < 30:
    st.warning(
        "Not enough clean data for technical analysis."
    )
    st.stop()

smc = smc_analysis(analysis_df)

# ============================================================
# MARKET MOOD
# ============================================================
snapshot = market_snapshot(MARKETS)

_, initial_mood_label = calculate_mood(snapshot)

initial_signal = generate_signal(
    analysis_df,
    smc,
    initial_mood_label,
)

mood_score, mood_label = calculate_mood(
    snapshot,
    initial_signal["signal"],
)

signal_result = generate_signal(
    analysis_df,
    smc,
    mood_label,
)

setup = trade_setup(
    analysis_df,
    signal_result["signal"],
)

latest = analysis_df.iloc[-1]

# ============================================================
# DATA FRESHNESS
# ============================================================
latest_timestamp = raw_df.index[-1]

try:
    now_ist = pd.Timestamp.now(tz="Asia/Kolkata")
    latest_ist = (
        latest_timestamp.tz_convert("Asia/Kolkata")
        if getattr(latest_timestamp, "tzinfo", None) is not None
        else latest_timestamp.tz_localize("Asia/Kolkata")
    )
    age_minutes = max(
        0,
        int((now_ist - latest_ist).total_seconds() / 60)
    )
except Exception:
    age_minutes = 0

st.info(
    f"📡 **Latest available candle:** {format_timestamp(latest_timestamp)}  | "
    f"**Timeframe:** {timeframe}  | **Ticker:** {symbol}  | "
    f"**Data age:** ~{age_minutes} min"
)

st.caption(
    "⚠️ Data source: Yahoo Finance/yfinance. NSE quotes in this feed are delayed; "
    "use your broker/exchange feed to verify the live price before placing an order."
)

# ============================================================
# TECHNICAL METRICS
# ============================================================
st.subheader(
    f"🔍 Technical Analysis — {symbol}"
)

m1, m2, m3, m4, m5, m6 = st.columns(6)

m1.metric(
    "Price",
    f"{float(latest['Close']):,.2f}",
)

m2.metric(
    "EMA 20",
    f"{float(latest['EMA20']):,.2f}",
)

m3.metric(
    "EMA 50",
    f"{float(latest['EMA50']):,.2f}",
)

m4.metric(
    "EMA 200",
    f"{float(latest['EMA200']):,.2f}"
    if pd.notna(latest["EMA200"])
    else "N/A",
)

m5.metric(
    "RSI 14",
    f"{float(latest['RSI']):.2f}",
)

m6.metric(
    "ADX",
    f"{float(latest['ADX']):.2f}",
)

m7, m8, m9 = st.columns(3)

m7.metric(
    "VWAP",
    f"{float(latest['VWAP']):,.2f}"
    if pd.notna(latest["VWAP"])
    else "N/A",
)

m8.metric(
    "ATR",
    f"{float(latest['ATR']):,.2f}",
)

m9.metric(
    "Volume",
    f"{float(latest['Volume']):,.0f}"
    if float(latest["Volume"]) > 0
    else "N/A",
)

# ============================================================
# MARKET MOOD + SIGNAL
# ============================================================
left, right = st.columns(2)

with left:
    st.subheader("🧠 Market Mood")

    st.metric(
        "Mood Score",
        f"{mood_score:.0f}/100",
        mood_label,
    )

    st.progress(
        int(mood_score)
    )

    st.caption(
        "Higher score = stronger bullish environment; "
        "lower score = stronger bearish environment."
    )

with right:
    st.subheader("🎯 Signal Engine")

    signal = signal_result["signal"]
    score = signal_result["score"]
    confidence = signal_result["confidence"]

    if signal == "BULLISH":
        st.success(
            f"🟢 **BULLISH** — Score {score}"
        )

    elif signal == "BEARISH":
        st.error(
            f"🔴 **BEARISH** — Score {score}"
        )

    else:
        st.warning(
            f"🟡 **NEUTRAL / WAIT** — Score {score}"
        )

    st.metric(
        "Signal Strength",
        f"{confidence}%",
    )

    st.caption(
        "Signal strength measures indicator agreement; "
        "it is not a probability of profit."
    )

# ============================================================
# SIGNAL BREAKDOWN
# ============================================================
with st.expander("🧠 Signal Breakdown"):
    st.write(f"**Bias:** {signal}")
    st.write(f"**Score:** {signal_result['score']}")
    st.write(f"**SMC Structure:** {smc['structure']}")
    st.write(f"**BOS:** {smc['bos']}")
    st.write(f"**FVG:** {smc['fvg']}")
    for reason in signal_result["reasons"]:
        st.write(f"• {reason}")

# ============================================================
# TRADE DECISION
# ============================================================
st.subheader("🧭 Trade Decision")

if signal == "NEUTRAL":
    st.warning(
        "🟡 **NO TRADE / WAIT** — trend confirmation "
        "is not strong enough."
    )

else:
    if signal_result["confirmation"]:
        st.success(
            "**SETUP VALID — WAIT FOR PRICE CONFIRMATION**"
        )
    else:
        st.warning(
            "**WAIT — CONFIRMATION MISSING**"
        )

    t1, t2, t3, t4, t5 = st.columns(5)

    t1.metric(
        "Entry Zone",
        f"{setup['entry_low']:,.2f} – "
        f"{setup['entry_high']:,.2f}",
    )

    t2.metric(
        "Stop Loss",
        f"{setup['sl']:,.2f}",
    )

    t3.metric(
        "Target 1",
        f"{setup['target1']:,.2f}",
    )

    t4.metric(
        "Target 2",
        f"{setup['target2']:,.2f}",
    )

    t5.metric(
        "Risk : Reward",
        f"1 : {setup['rr']:.2f}"
        if np.isfinite(setup["rr"])
        else "N/A",
    )

    if signal_result["confirmation_reasons"]:
        st.caption(
            "⚠️ "
            + " | ".join(
                signal_result["confirmation_reasons"]
            )
        )

# ============================================================
# SUPPORT / RESISTANCE + SMC
# ============================================================
c1, c2 = st.columns(2)

with c1:
    st.subheader("📐 Support & Resistance")

    a, b = st.columns(2)

    a.metric(
        "Support",
        f"{setup['support']:,.2f}",
    )

    b.metric(
        "Resistance",
        f"{setup['resistance']:,.2f}",
    )

with c2:
    st.subheader("🧠 SMC Snapshot")

    st.write(
        f"**Market Structure:** {smc['structure']}"
    )

    st.write(
        f"**BOS:** {smc['bos']}"
    )

    st.write(
        f"**FVG:** {smc['fvg']}"
    )

# ============================================================
# WHY THIS SIGNAL?
# ============================================================
with st.expander(
    "Why is the dashboard giving this signal?"
):
    for reason in signal_result["reasons"]:
        st.write(f"• {reason}")

# ============================================================
# CHART
# ============================================================
st.subheader("📈 Price Action Chart")

fig = make_chart(
    analysis_df,
    smc,
    setup,
    signal,
)

st.plotly_chart(
    fig,
    use_container_width=True,
)

# ============================================================
# LATEST DATA TABLE
# ============================================================
with st.expander(
    "Latest indicator values"
):
    columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "EMA20",
        "EMA50",
        "EMA200",
        "RSI",
        "MACD",
        "MACD_SIGNAL",
        "ADX",
        "ATR",
        "VWAP",
    ]

    available_columns = [
        col
        for col in columns
        if col in analysis_df.columns
    ]

    st.dataframe(
        analysis_df[
            available_columns
        ].tail(10).round(2),
        use_container_width=True,
    )

# ============================================================
# DISCLAIMER
# ============================================================
st.caption(
    "⚠️ Educational/decision-support dashboard only. "
    "Market data may be delayed, incomplete or unavailable. "
    "Always independently verify price, liquidity, contract details "
    "and risk before placing an order."
)
