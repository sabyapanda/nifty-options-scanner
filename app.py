import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as stats
from datetime import datetime, timedelta

# Set up page configurations
st.set_page_config(page_title="Nifty 50 High-PoP Options Scanner", layout="wide")
st.title("📊 Nifty 50 Technical & High-Probability Option Scanner")
st.caption("Live Local Analysis Engine for Indian Stock Derivatives")

# Top 25 Nifty 50 Tickers (Mapped to Yahoo Finance extensions)
NIFTY_25 = {
    "RELIANCE": "RELIANCE.NS", "TCS": "TCS.NS", "HDFCBANK": "HDFCBANK.NS",
    "INFY": "INFY.NS", "ICICIBANK": "ICICIBANK.NS", "BHARTIARTL": "BHARTIARTL.NS",
    "SBI": "SBIN.NS", "LTIM": "LTIM.NS", "ITC": "ITC.NS", "HINDUNILVR": "HINDUNILVR.NS",
    "LT": "LT.NS", "BAJFINANCE": "BAJFINANCE.NS", "AXISBANK": "AXISBANK.NS",
    "HCLTECH": "HCLTECH.NS", "MARUTI": "MARUTI.NS", "SUNPHARMA": "SUNPHARMA.NS",
    "KOTAKBANK": "KOTAKBANK.NS", "TITAN": "TITAN.NS", "ULTRACEMCO": "ULTRACEMCO.NS",
    "ASIANPAINT": "ASIANPAINT.NS", "NESTLEIND": "NESTLEIND.NS", "M&M": "M&M.NS",
    "POWERGRID": "POWERGRID.NS", "NTPC": "NTPC.NS", "TATASTEEL": "TATASTEEL.NS"
}


# Generate next 12 months last-Tuesday monthly expiries dynamically
# Nifty 50 monthly options contracts in India typically expire on the last Tuesday of the month.
def get_monthly_expiries():
    expiries = []
    current_date = datetime.now()

    for month_offset in range(12):
        year = current_date.year + ((current_date.month - 1 + month_offset) // 12)
        month = (current_date.month - 1 + month_offset) % 12 + 1

        if month == 12:
            next_month_first_day = datetime(year + 1, 1, 1)
        else:
            next_month_first_day = datetime(year, month + 1, 1)

        last_day = next_month_first_day - timedelta(days=1)

        # Python weekday: Monday=0, Tuesday=1, ..., Sunday=6
        offset = (last_day.weekday() - 1) % 7
        last_tuesday = last_day - timedelta(days=offset)
        expiries.append(last_tuesday.date())

    return sorted(list(set(expiries)))


# Math calculations for RSI
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = delta.clip(lower=0).rolling(window=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window=window).mean()
    rs = gain / loss.replace(0, np.nan)
    rs = rs.fillna(0)
    return 100 - (100 / (1 + rs))


# Weekly scan helpers

def detect_bollinger_squeeze(df, threshold=0.08):
    if len(df) < 20:
        return {"exists": False, "status": "Insufficient weekly data", "bandwidth": None}

    df = df.copy()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["STD20"] = df["Close"].rolling(20).std()
    df["UpperBB"] = df["MA20"] + (2 * df["STD20"])
    df["LowerBB"] = df["MA20"] - (2 * df["STD20"])

    if pd.isna(df["MA20"].iloc[-1]):
        return {"exists": False, "status": "Insufficient weekly data", "bandwidth": None}

    bandwidth = (df["UpperBB"].iloc[-1] - df["LowerBB"].iloc[-1]) / df["MA20"].iloc[-1]
    exists = bool(bandwidth < threshold)
    return {
        "exists": exists,
        "status": "Squeeze Active" if exists else "Expansion Active",
        "bandwidth": bandwidth
    }


def detect_rsi_divergence(df, lookback=8):
    if len(df) < lookback + 2:
        return {"exists": False, "status": "No RSI Divergence"}

    closes = df["Close"].tail(lookback + 2).to_numpy()
    rsi_values = calculate_rsi(df["Close"]).tail(lookback + 2).to_numpy()

    prev_price_high = np.max(closes[:-1])
    prev_price_low = np.min(closes[:-1])
    prev_rsi_high = np.max(rsi_values[:-1])
    prev_rsi_low = np.min(rsi_values[:-1])

    bearish_divergence = (closes[-1] > prev_price_high) and (rsi_values[-1] < prev_rsi_high)
    bullish_divergence = (closes[-1] < prev_price_low) and (rsi_values[-1] > prev_rsi_low)

    if bearish_divergence:
        return {"exists": True, "status": "Bearish RSI Divergence"}
    if bullish_divergence:
        return {"exists": True, "status": "Bullish RSI Divergence"}

    return {"exists": False, "status": "No RSI Divergence"}


# Candlestick helpers

def candle_body(candle):
    return abs(candle["Close"] - candle["Open"])


def upper_wick(candle):
    return candle["High"] - max(candle["Open"], candle["Close"])


def lower_wick(candle):
    return min(candle["Open"], candle["Close"]) - candle["Low"]


def is_bearish_engulfing(current, previous):
    curr_body = candle_body(current)
    prev_body = candle_body(previous)
    return (
        current["Close"] < current["Open"]
        and previous["Close"] > previous["Open"]
        and current["Open"] >= previous["Close"]
        and current["Close"] <= previous["Open"]
        and curr_body > prev_body
    )


def is_bullish_engulfing(current, previous):
    curr_body = candle_body(current)
    prev_body = candle_body(previous)
    return (
        current["Close"] > current["Open"]
        and previous["Close"] < previous["Open"]
        and current["Open"] <= previous["Close"]
        and current["Close"] >= previous["Open"]
        and curr_body > prev_body
    )


def is_evening_star(current, previous, previous_2):
    return (
        current["Close"] < current["Open"]
        and previous["Close"] > previous["Open"]
        and previous_2["Close"] > previous_2["Open"]
        and current["Open"] > previous["Close"]
        and current["Close"] < ((previous["Open"] + previous["Close"]) / 2)
    )


def is_morning_star(current, previous, previous_2):
    return (
        current["Close"] > current["Open"]
        and previous["Close"] < previous["Open"]
        and previous_2["Close"] < previous_2["Open"]
        and current["Open"] < previous["Close"]
        and current["Close"] > ((previous["Open"] + previous["Close"]) / 2)
    )


def is_shooting_star(current):
    body = candle_body(current)
    if body <= 0:
        return False
    return upper_wick(current) > (2 * body) and lower_wick(current) < body and current["Close"] < current["Open"]


def is_hammer(current):
    body = candle_body(current)
    if body <= 0:
        return False
    return lower_wick(current) > (2 * body) and upper_wick(current) < body and current["Close"] > current["Open"]


def detect_weekly_patterns(df):
    results = {
        "bearish_engulfing": False,
        "evening_star": False,
        "shooting_star": False,
        "bullish_engulfing": False,
        "morning_star": False,
        "hammer": False,
    }

    if len(df) < 3:
        return results

    current = df.iloc[-1].to_dict()
    previous = df.iloc[-2].to_dict()
    previous_2 = df.iloc[-3].to_dict()

    results["bearish_engulfing"] = bool(is_bearish_engulfing(current, previous))
    results["evening_star"] = bool(is_evening_star(current, previous, previous_2))
    results["shooting_star"] = bool(is_shooting_star(current))
    results["bullish_engulfing"] = bool(is_bullish_engulfing(current, previous))
    results["morning_star"] = bool(is_morning_star(current, previous, previous_2))
    results["hammer"] = bool(is_hammer(current))

    return results


def format_yes_no(value):
    if value == "YES":
        return "color: green; font-weight: bold"
    if value == "NO":
        return "color: red; font-weight: bold"
    return ""


# Layout UI Filters
col1, col2 = st.columns(2)
with col1:
    selected_stock = st.selectbox("Select Underlying Nifty 50 Stock:", list(NIFTY_25.keys()))
with col2:
    available_expiries = get_monthly_expiries()
    selected_expiry = st.selectbox("Select Contract Expiry Date:", available_expiries)

# Fetching Data Live
ticker_symbol = NIFTY_25[selected_stock]
stock = yf.Ticker(ticker_symbol)

with st.spinner(f"Pulling real-time market matrix for {selected_stock}..."):
    hist_daily = stock.history(period="1y")
    hist_weekly = stock.history(period="2y").resample('W').last()

    if len(hist_daily) > 30:
        spot_price = hist_daily['Close'].iloc[-1]

        # Technical Indicator Calculations
        # 1. RSI calculations
        hist_daily['RSI'] = calculate_rsi(hist_daily['Close'])
        hist_weekly['RSI'] = calculate_rsi(hist_weekly['Close'])
        daily_rsi = hist_daily['RSI'].iloc[-1]
        weekly_rsi = hist_weekly['RSI'].iloc[-1]

        # 2. Bollinger Squeeze Calculation
        hist_daily['MA20'] = hist_daily['Close'].rolling(20).mean()
        hist_daily['STD20'] = hist_daily['Close'].rolling(20).std()
        hist_daily['UpperBB'] = hist_daily['MA20'] + (2 * hist_daily['STD20'])
        hist_daily['LowerBB'] = hist_daily['MA20'] - (2 * hist_daily['STD20'])
        bb_bandwidth = (hist_daily['UpperBB'].iloc[-1] - hist_daily['LowerBB'].iloc[-1]) / hist_daily['MA20'].iloc[-1]
        is_squeeze = "Active Squeeze (Low Volatility Consolidation)" if bb_bandwidth < 0.08 else "Bandwidth Expanded (High Volatility Trend)"

        # 3. Standard Pivots (Classic Daily Setup)
        high = hist_daily['High'].iloc[-2]
        low = hist_daily['Low'].iloc[-2]
        close = hist_daily['Close'].iloc[-2]

        pivot = (high + low + close) / 3
        r1 = (2 * pivot) - low
        s1 = (2 * pivot) - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        r3 = high + 2 * (pivot - low)
        s3 = low - 2 * (high - pivot)

        # 4. Candlestick Scanning
        open_c, high_c, low_c, close_c = hist_daily['Open'].iloc[-1], hist_daily['High'].iloc[-1], hist_daily['Low'].iloc[-1], hist_daily['Close'].iloc[-1]
        pattern = "Neutral Market Churn"
        if close_c > open_c and (open_c - low_c) > 2 * (close_c - open_c):
            pattern = "Bullish Hammer Candlestick detected"
        elif open_c > close_c and (high_c - open_c) > 2 * (open_c - close_c):
            pattern = "Bearish Shooting Star detected"

        # Weekly structural check for the selected stock
        weekly_bollinger = detect_bollinger_squeeze(hist_weekly)
        weekly_rsi_divergence = detect_rsi_divergence(hist_weekly)
        weekly_pattern_results = detect_weekly_patterns(hist_weekly)

        bullish_rsi_divergence = "YES" if weekly_rsi_divergence["exists"] and weekly_rsi_divergence["status"] == "Bullish RSI Divergence" else "NO"
        bearish_rsi_divergence = "YES" if weekly_rsi_divergence["exists"] and weekly_rsi_divergence["status"] == "Bearish RSI Divergence" else "NO"

        weekly_summary = pd.DataFrame({
            "Pattern": [
                "Bullish RSI divergence",
                "Bullish engulfing",
                "Morning star",
                "Hammer",
                "Bollinger squeeze",
                "Bearish RSI divergence",
                "Bearish engulfing",
                "Evening star",
                "Shooting star",
            ],
            "Status": [
                bullish_rsi_divergence,
                "YES" if weekly_pattern_results["bullish_engulfing"] else "NO",
                "YES" if weekly_pattern_results["morning_star"] else "NO",
                "YES" if weekly_pattern_results["hammer"] else "NO",
                "YES" if weekly_bollinger["exists"] else "NO",
                bearish_rsi_divergence,
                "YES" if weekly_pattern_results["bearish_engulfing"] else "NO",
                "YES" if weekly_pattern_results["evening_star"] else "NO",
                "YES" if weekly_pattern_results["shooting_star"] else "NO",
            ],
            "Sentiment": [
                "Bullish",
                "Bullish",
                "Bullish",
                "Bullish",
                "Neutral",
                "Bearish",
                "Bearish",
                "Bearish",
                "Bearish",
            ]
        })

        sentiment_rank = {"Bullish": 0, "Neutral": 1, "Bearish": 2}
        weekly_summary["sentiment_order"] = weekly_summary["Sentiment"].map(sentiment_rank)
        weekly_summary = weekly_summary.sort_values(["sentiment_order", "Pattern"]).drop(columns=["sentiment_order"]).reset_index(drop=True)

        def status_style(value):
            if value == "YES":
                return "color: green; font-weight: bold"
            if value == "NO":
                return "color: red; font-weight: bold"
            return ""

        weekly_summary_styler = (
            weekly_summary.style
            .apply(lambda row: [status_style(v) for v in row], subset=["Status"])
            .set_properties(**{"text-align": "center"})
        )

        # Math Modeling for 90-95% Profit Recommendation
        days_to_expiry = max((selected_expiry - datetime.now().date()).days, 1)
        # Approximate dynamic annualized Implied Volatility via standard deviation
        log_returns = np.log(hist_daily['Close'] / hist_daily['Close'].shift(1))
        historical_volatility = log_returns.std() * np.sqrt(252)

        # Target z-score mapping for 90% to 95% single-tail target probability (Delta 0.05 to 0.10)
        z_score_92 = stats.norm.ppf(0.925)
        price_std_dev = spot_price * historical_volatility * np.sqrt(days_to_expiry / 365)

        recommended_call_strike = round((spot_price + (z_score_92 * price_std_dev)) / 5) * 5
        recommended_put_strike = round((spot_price - (z_score_92 * price_std_dev)) / 5) * 5

        # Rendering Dashboard Display blocks
        st.subheader(f"⚡ Current Spot Price: ₹{spot_price:,.2f}")

        t1, t2, t3 = st.tabs(["🎯 Dynamic Option Selling Targets", "📈 Volatility & Momentum Matrix", "🧭 Floor Pivot Channels"])

        with t1:
            st.markdown(f"### **Optimal Premium Collection Strikes (90%–95% Probability of Profit)**")
            st.info(f"Calculated using an implied monthly annualized volatility model of **{historical_volatility*100:.1f}%** over an allocation window of **{days_to_expiry} days**.")

            c_col, p_col = st.columns(2)
            with c_col:
                st.metric(label="🛡️ RECOMMENDED CALL SELLING STRIKE (CE)", value=f"₹{recommended_call_strike:,.0f}")
                st.write(f"• **Positioning:** Short Call strategy (Bearish/Neutral premium collection)\n\n• **Margin Safety Buffer:** Needs underlying asset to surge **+{((recommended_call_strike - spot_price) / spot_price * 100):.1f}%** from spot to maintain a prudent cushion.")
            with p_col:
                st.metric(label="🛡️ RECOMMENDED PUT SELLING STRIKE (PE)", value=f"₹{recommended_put_strike:,.0f}")
                st.write(f"• **Positioning:** Short Put strategy (Bullish/Neutral premium collection)\n\n• **Margin Safety Buffer:** Protected against a market drop down to **-{((spot_price - recommended_put_strike) / spot_price * 100):.1f}%** from spot before losing the premium edge.")

        with t2:
            m1, m2, m3 = st.columns(3)
            m1.metric("Daily 14-Period RSI", f"{daily_rsi:.1f}", delta="Oversold" if daily_rsi < 30 else ("Overbought" if daily_rsi > 70 else "Neutral"))
            m2.metric("Weekly 14-Period RSI", f"{weekly_rsi:.1f}")
            m3.metric("Candlestick Trigger", "Scanned Signal", delta=pattern)
            st.text(f"Bollinger Structure Status: {is_squeeze}")

            st.subheader(f"📉 Weekly Formations: {selected_stock}")
            st.dataframe(weekly_summary_styler, use_container_width=True, hide_index=True)

            st.caption(
                f"Bollinger squeeze: {weekly_bollinger['status']} | RSI divergence: {weekly_rsi_divergence['status']}"
            )

        with t3:
            pivot_df = pd.DataFrame({
                "Technical Layer Boundary": ["Resistance 3 (R3)", "Resistance 2 (R2)", "Resistance 1 (R1)", "Central Pivot Point", "Support 1 (S1)", "Support 2 (S2)", "Support 3 (S3)"],
                "Calculated Strike Threshold (₹)": [r3, r2, r1, pivot, s1, s2, s3]
            })
            st.table(pivot_df.set_index("Technical Layer Boundary").style.format("₹{:#,.2f}"))
