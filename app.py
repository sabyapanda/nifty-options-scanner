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

# Generate next 12 months last-thursday monthly expiries dynamically
def get_monthly_expiries():
    expiries = []
    current_date = datetime.now()
    for i in range(12):
        target_month = current_date + timedelta(days=i*30)
        # Find last day of that month
        if target_month.month == 12:
            next_m = datetime(target_month.year + 1, 1, 1)
        else:
            next_m = datetime(target_month.year, target_month.month + 1, 1)
        last_day = next_m - timedelta(days=1)
        # Find last Thursday of that month
        offset = (last_day.weekday() - 3) % 7
        last_thursday = last_day - timedelta(days=offset)
        expiries.append(last_thursday.date())
    return sorted(list(set(expiries)))

# Math calculations for RSI
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

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
                st.write(f"• **Positioning:** Short Call strategy (Bearish/Neutral premium collection)\n\n• **Margin Safety Buffer:** Needs underlying asset to surge **+{((recommended_call_strike/spot_price)-1)*100:.1f}%** to breach.")
            with p_col:
                st.metric(label="🛡️ RECOMMENDED PUT SELLING STRIKE (PE)", value=f"₹{recommended_put_strike:,.0f}")
                st.write(f"• **Positioning:** Short Put strategy (Bullish/Neutral premium collection)\n\n• **Margin Safety Buffer:** Protected against market drops down to **-{((1-(recommended_put_strike/spot_price)))*100:.1f}%**.")

        with t2:
            m1, m2, m3 = st.columns(3)
            m1.metric("Daily 14-Period RSI", f"{daily_rsi:.1f}", delta="Oversold" if daily_rsi < 30 else ("Overbought" if daily_rsi > 70 else "Neutral"))
            m2.metric("Weekly 14-Period RSI", f"{weekly_rsi:.1f}")
            m3.metric("Candlestick Trigger", "Scanned Signal", delta=pattern)
            st.text(f"Bollinger Structure Status: {is_squeeze}")

        with t3:
            pivot_df = pd.DataFrame({
                "Technical Layer Boundary": ["Resistance 3 (R3)", "Resistance 2 (R2)", "Resistance 1 (R1)", "Central Pivot Point", "Support 1 (S1)", "Support 2 (S2)", "Support 3 (S3)"],
                "Calculated Strike Threshold (₹)": [r3, r2, r1, pivot, s1, s2, s3]
            })
            st.table(pivot_df.set_index("Technical Layer Boundary").style.format("₹{:#,.2f}"))

