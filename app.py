import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# -------------------------------
# Streamlit Config / Styles
# -------------------------------
st.set_page_config(page_title="AI Stock Dashboard", layout="wide")

# Minimal modern style
st.markdown("""
<style>
/* tighten top spacing */
.block-container { padding-top: 1rem; padding-bottom: 2rem; }
/* table font-size slightly smaller */
.css-1j7m5y7, .stDataFrame { font-size: 0.95rem; }
/* section headers */
h2, h3 { margin-top: 0.5rem; }
hr { margin: 0.8rem 0 1.2rem 0; }
.explainer { color: #444; font-size: 0.95rem; line-height: 1.4rem; }
.caption-quiet { color: #666; font-size: 0.9rem; }
.badge { display: inline-block; padding: 4px 10px; border-radius: 12px; background: #f2f2f2; margin-right: 6px; }
.good { background: #e6f4ea; color: #137333; }
.warn { background: #fff4e5; color: #a15c07; }
.risk { background: #fde7e9; color: #b3261e; }
</style>
""", unsafe_allow_html=True)

st.title("AI Stock Dashboard")

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("Inputs")
ticker = st.sidebar.text_input("Ticker", "AAPL").strip().upper()
period = st.sidebar.selectbox("History range", ["6mo", "1y", "2y", "5y"], index=1)
interval = st.sidebar.selectbox("Interval", ["1d", "1h", "30m"], index=0)

tabs = st.sidebar.radio("Section", [
    "Summary",
    "Technical",
    "Support/Resistance",
    "Options Ideas",
    "Fundamentals",
    "Forecast",
    "Risk",
    "News",
])

# -------------------------------
# Data
# -------------------------------
@st.cache_data(show_spinner=False)
def load_history(tkr: str, per: str, itv: str):
    df = yf.download(tkr, period=per, interval=itv, auto_adjust=True)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    return df

df_raw = load_history(ticker, period, interval)
if df_raw.empty:
    st.error(f"No data found for {ticker}. Try another symbol or range.")
    st.stop()

# -------------------------------
# Indicators / Helpers
# -------------------------------
def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # SMA/EMA (20/50)
    out["SMA20"] = out["Close"].rolling(20).mean()
    out["EMA20"] = out["Close"].ewm(span=20, adjust=False).mean()
    out["SMA50"] = out["Close"].rolling(50).mean()
    out["EMA50"] = out["Close"].ewm(span=50, adjust=False).mean()

    # Bollinger (20, 2)
    bb_win, bb_std = 20, 2
    out["BB_MID"] = out["Close"].rolling(bb_win).mean()
    out["BB_STD"] = out["Close"].rolling(bb_win).std()
    out["BB_UP"] = out["BB_MID"] + bb_std * out["BB_STD"]
    out["BB_DN"] = out["BB_MID"] - bb_std * out["BB_STD"]
    out["BB_WIDTH"] = (out["BB_UP"] - out["BB_DN"]) / out["BB_MID"]

    # RSI (14)
    delta = out["Close"].diff()
    gain = np.where(delta > 0, delta, 0).flatten()
    loss = np.where(delta < 0, -delta, 0).flatten()
    avg_gain = pd.Series(gain, index=out.index).rolling(14).mean()
    avg_loss = pd.Series(loss, index=out.index).rolling(14).mean()
    rs = avg_gain / avg_loss
    out["RSI"] = 100 - (100 / (1 + rs))

    # MACD (12,26,9)
    ema12 = out["Close"].ewm(span=12, adjust=False).mean()
    ema26 = out["Close"].ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_SIGNAL"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_HIST"] = out["MACD"] - out["MACD_SIGNAL"]

    # ATR(14)
    high_low = out["High"] - out["Low"]
    high_close = np.abs(out["High"] - out["Close"].shift())
    low_close = np.abs(out["Low"] - out["Close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    out["ATR14"] = tr.rolling(14).mean()

    return out

def find_support_resistance(df: pd.DataFrame, prominence_pct: float = 0.5):
    """
    Local-extrema clustering to detect S/R levels. prominence_pct is band width (% of price).
    """
    if df.empty:
        return [], []
    price = float(df["Close"].iloc[-1])
    x = df["Close"].values
    n = len(x)
    if n < 20:
        return [], []
    # local extrema
    local_max, local_min = [], []
    for i in range(1, n-1):
        if x[i] > x[i-1] and x[i] > x[i+1]:
            local_max.append(x[i])
        if x[i] < x[i-1] and x[i] < x[i+1]:
            local_min.append(x[i])
    levels = np.array(local_max + local_min)
    if levels.size == 0:
        return [], []

    band = price * (prominence_pct / 100.0)
    levels.sort()
    clusters = []
    cur = [levels[0]]
    for v in levels[1:]:
        if abs(v - np.mean(cur)) <= band:
            cur.append(v)
        else:
            clusters.append(cur)
            cur = [v]
    clusters.append(cur)
    centroids = [float(np.mean(c)) for c in clusters]
    supports = sorted([c for c in centroids if c < price], reverse=True)[:5]
    resistances = sorted([c for c in centroids if c > price])[:5]
    return supports, resistances

def nearest_expiry(expirations, min_days=7, max_days=35):
    today = datetime.date.today()
    best, best_days = None, None
    for e in expirations or []:
        try:
            d = datetime.datetime.strptime(e, "%Y-%m-%d").date()
            dd = (d - today).days
            if min_days <= dd <= max_days and (best is None or dd < best_days):
                best, best_days = e, dd
        except Exception:
            pass
    if best is None:
        for e in expirations or []:
            try:
                d = datetime.datetime.strptime(e, "%Y-%m-%d").date()
                dd = (d - today).days
                if dd > 0 and (best is None or dd < best_days):
                    best, best_days = e, dd
            except Exception:
                pass
    return best

def pick_option_row(chain_df: pd.DataFrame, target_strike: float):
    if chain_df is None or chain_df.empty:
        return None
    c = chain_df.copy()
    c["mid"] = (c["bid"].fillna(0) + c["ask"].fillna(0)) / 2.0
    c["dist"] = (c["strike"] - target_strike).abs()
    c.sort_values(["dist", "mid"], ascending=[True, False], inplace=True)
    return c.iloc[0].to_dict()

def bullish_bias(rsi, macd, macd_sig, close, sma20, sma50):
    score = 0
    # RSI mid-range is modest bullish vs extremes
    if pd.notna(rsi):
        if 40 <= rsi <= 60: score += 1
        if 50 <= rsi <= 65: score += 1
        if rsi < 30: score -= 2
        if rsi > 70: score -= 2
    if pd.notna(macd) and pd.notna(macd_sig):
        if macd > macd_sig: score += 2
        else: score -= 1
    if pd.notna(sma20) and pd.notna(close):
        if close > sma20: score += 1
        else: score -= 1
    if pd.notna(sma50) and pd.notna(close):
        if close > sma50: score += 1
        else: score -= 1
    return score

df = calc_indicators(df_raw)

last_close = float(df["Close"].iloc[-1])
last_rsi = float(df["RSI"].iloc[-1]) if pd.notna(df["RSI"].iloc[-1]) else np.nan
last_macd = float(df["MACD"].iloc[-1]) if pd.notna(df["MACD"].iloc[-1]) else np.nan
last_macd_sig = float(df["MACD_SIGNAL"].iloc[-1]) if pd.notna(df["MACD_SIGNAL"].iloc[-1]) else np.nan
last_sma20 = float(df["SMA20"].iloc[-1]) if pd.notna(df["SMA20"].iloc[-1]) else np.nan
last_sma50 = float(df["SMA50"].iloc[-1]) if pd.notna(df["SMA50"].iloc[-1]) else np.nan
last_bbw = float(df["BB_WIDTH"].iloc[-1]) if pd.notna(df["BB_WIDTH"].iloc[-1]) else np.nan

supports, resistances = find_support_resistance(df, prominence_pct=0.5)
nearest_support = supports[0] if supports else round(last_close * 0.95, 2)
nearest_resist = resistances[0] if resistances else round(last_close * 1.05, 2)

# -------------------------------
# Summary Tab (Overall verdict, events, quick facts)
# -------------------------------
if tabs == "Summary":
    st.header(f"{ticker} — Summary")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Last", f"${last_close:,.2f}")
    c2.metric("RSI(14)", f"{last_rsi:,.2f}" if pd.notna(last_rsi) else "—")
    c3.metric("MACD", f"{last_macd:,.2f}" if pd.notna(last_macd) else "—")
    c4.metric("SMA20", f"{last_sma20:,.2f}" if pd.notna(last_sma20) else "—")
    c5.metric("SMA50", f"{last_sma50:,.2f}" if pd.notna(last_sma50) else "—")

    st.markdown("<hr>", unsafe_allow_html=True)

    # Earnings & Dividends
    st.subheader("Key Dates")
    next_earn = None
    last_div = None
    try:
        tk = yf.Ticker(ticker)
        # Earnings dates (yfinance newer method)
        try:
            edf = tk.get_earnings_dates(limit=12)  # may not exist in older versions
            if edf is not None and not edf.empty:
                future = edf[edf.index >= pd.Timestamp.today()]
                if not future.empty:
                    next_earn = str(future.index[0].date())
        except Exception:
            pass
        # Dividends
        try:
            divs = tk.dividends
            if divs is not None and not divs.empty:
                last_div = f"{divs.iloc[-1]:.4f} on {divs.index[-1].date()}"
        except Exception:
            pass
    except Exception:
        pass

    colA, colB = st.columns(2)
    with colA:
        st.write("**Next earnings**:", next_earn if next_earn else "Not available")
    with colB:
        st.write("**Last dividend**:", last_div if last_div else "Not available")

    st.markdown("<hr>", unsafe_allow_html=True)

    # Final Verdict (plain English)
    score = bullish_bias(last_rsi, last_macd, last_macd_sig, last_close, last_sma20, last_sma50)
    verdict = "Neutral"
    color_fn = st.info
    if score >= 3:
        verdict = "Bullish"
        color_fn = st.success
    elif score <= -2:
        verdict = "Cautious / Bearish"
        color_fn = st.warning

    color_fn(f"**Overall View: {verdict}**")

    st.markdown(
        f"""
<div class="explainer">
<strong>What this means:</strong><br>
• RSI at {last_rsi:.1f} suggests {"overbought (risk of pullback)" if last_rsi>70 else ("oversold (rebound potential)" if last_rsi<30 else "balanced momentum")}.<br>
• MACD is {"above" if last_macd>last_macd_sig else "below"} its signal → {"bullish" if last_macd>last_macd_sig else "bearish"} momentum.<br>
• Price is {"above" if last_close>last_sma20 else "below"} its 20-day average and {"above" if last_close>last_sma50 else "below"} its 50-day average.<br><br>
<b>Practical guidance:</b><br>
• Nearest support: <span class="badge good">${nearest_support:,.2f}</span> — Safer entries or cash-secured puts near/below this level.<br>
• Nearest resistance: <span class="badge warn">${nearest_resist:,.2f}</span> — Good zone for covered calls to collect income.<br>
• Around earnings, expect higher volatility; size positions accordingly.
</div>
""", unsafe_allow_html=True)

# -------------------------------
# Technical
# -------------------------------
elif tabs == "Technical":
    st.header(f"{ticker} — Technical")

    # Top metrics and explanations
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Last", f"${last_close:,.2f}")
    c2.metric("RSI(14)", f"{last_rsi:,.2f}" if pd.notna(last_rsi) else "—")
    c3.metric("MACD", f"{last_macd:,.2f}" if pd.notna(last_macd) else "—")
    c4.metric("SMA20", f"{last_sma20:,.2f}" if pd.notna(last_sma20) else "—")
    c5.metric("SMA50", f"{last_sma50:,.2f}" if pd.notna(last_sma50) else "—")

    st.markdown(
        """
<div class="explainer">
<b>How to read:</b><br>
• <b>RSI</b> above 70 can mean overbought (pullback risk). Below 30 can mean oversold (rebound potential).<br>
• <b>MACD</b> above its signal shows bullish momentum; below shows bearish momentum.<br>
• <b>MAs</b> (SMA/EMA) show trend direction; price above them = uptrend strength.
</div>
""", unsafe_allow_html=True)

    # Candles + Indicators
    price_fig = go.Figure()
    price_fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"
    ))
    price_fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], mode="lines", name="SMA20"))
    price_fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], mode="lines", name="SMA50"))
    price_fig.add_trace(go.Scatter(x=df.index, y=df["BB_UP"], mode="lines", name="BB Upper", line=dict(width=1)))
    price_fig.add_trace(go.Scatter(x=df.index, y=df["BB_MID"], mode="lines", name="BB Mid", line=dict(width=1)))
    price_fig.add_trace(go.Scatter(x=df.index, y=df["BB_DN"], mode="lines", name="BB Lower", line=dict(width=1)))
    price_fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Price")

    vol_fig = go.Figure()
    vol_fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume"))
    vol_fig.update_layout(height=180, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Volume")

    rsi_fig = go.Figure()
    rsi_fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], mode="lines", name="RSI"))
    rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
    rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")
    rsi_fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="RSI")

    macd_fig = go.Figure()
    macd_fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], mode="lines", name="MACD"))
    macd_fig.add_trace(go.Scatter(x=df.index, y=df["MACD_SIGNAL"], mode="lines", name="Signal"))
    macd_fig.add_trace(go.Bar(x=df.index, y=df["MACD_HIST"], name="Hist"))
    macd_fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="MACD")

    st.plotly_chart(price_fig, use_container_width=True)
    st.plotly_chart(vol_fig, use_container_width=True)
    st.plotly_chart(rsi_fig, use_container_width=True)
    st.plotly_chart(macd_fig, use_container_width=True)

# -------------------------------
# Support/Resistance
# -------------------------------
elif tabs == "Support/Resistance":
    st.header(f"{ticker} — Support & Resistance")

    st.markdown(
        """
<div class="explainer">
<b>Support</b> is a price area where buyers previously stepped in. Prices often bounce here.<br>
<b>Resistance</b> is where sellers appeared. Prices often stall or pull back here.<br>
These levels are guides, not guarantees. Combine them with momentum (RSI/MACD) for better decisions.
</div>
""", unsafe_allow_html=True)

    sensitivity = st.slider("Sensitivity (percent band around clusters)", 0.2, 2.0, 0.5, 0.1)
    s_levels, r_levels = find_support_resistance(df, prominence_pct=sensitivity)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"
    ))
    for s in s_levels:
        fig.add_hline(y=s, line_color="green", line_dash="dot")
    for r in r_levels:
        fig.add_hline(y=r, line_color="red", line_dash="dot")
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Supports (nearest first)")
        st.table(pd.DataFrame({"Support": s_levels}) if s_levels else pd.DataFrame({"Support": []}))
    with c2:
        st.subheader("Resistances (nearest first)")
        st.table(pd.DataFrame({"Resistance": r_levels}) if r_levels else pd.DataFrame({"Resistance": []}))

    st.markdown(
        f"""
<div class="explainer">
<b>Practical use:</b><br>
• If you are <b>bullish</b>, consider entries or cash-secured puts near/under support (e.g., ~${(s_levels[0] if s_levels else nearest_support):,.2f}).<br>
• If you want <b>income</b> without selling shares, consider covered calls near resistance (e.g., ~${(r_levels[0] if r_levels else nearest_resist):,.2f}).
</div>
""", unsafe_allow_html=True)

# -------------------------------
# Options Ideas (CSP/CC)
# -------------------------------
elif tabs == "Options Ideas":
    st.header(f"{ticker} — Options Ideas")

    st.markdown(
        """
<div class="explainer">
We generate two simple, conservative income ideas:<br>
• <b>Cash-Secured Put (CSP)</b>: Get paid to potentially buy the stock at a lower price (near support).<br>
• <b>Covered Call (CC)</b>: Get paid while holding shares by agreeing to sell at a higher price (near resistance).<br>
These are suggestions, not advice. Check spreads, volume, earnings, and your risk tolerance.
</div>
""", unsafe_allow_html=True)

    stock = yf.Ticker(ticker)
    last = last_close
    s_levels, r_levels = supports, resistances
    s_price = s_levels[0] if s_levels else round(last * 0.95, 2)
    r_price = r_levels[0] if r_levels else round(last * 1.05, 2)

    # expirations
    try:
        expirations = stock.options or []
    except Exception:
        expirations = []
    exp_choice = nearest_expiry(expirations, min_days=7, max_days=35)
    st.write("Target Expiration:", exp_choice if exp_choice else "Not available")

    CC_row, CSP_row = None, None
    if exp_choice:
        try:
            chain = stock.option_chain(exp_choice)
            calls = chain.calls.copy()
            puts = chain.puts.copy()

            # target strikes by S/R; fallback to ~3% OTM
            target_cc = max(r_price, last * 1.03)
            target_csp = min(s_price, last * 0.97)

            CC_row = pick_option_row(calls, target_cc)
            CSP_row = pick_option_row(puts, target_csp)
        except Exception as e:
            st.error(f"Option chain error: {e}")

    ideas = []
    if CC_row:
        ideas.append({
            "Strategy": "Covered Call",
            "Strike": float(CC_row.get("strike", np.nan)),
            "Est. Premium (mid)": float(((CC_row.get("bid",0) + CC_row.get("ask",0))/2)),
            "Bid": float(CC_row.get("bid", 0.0)),
            "Ask": float(CC_row.get("ask", 0.0)),
            "Delta": float(CC_row.get("delta", np.nan)) if "delta" in CC_row else np.nan,
            "Rationale": "If you own shares, selling a call near resistance collects income; stock must trade above strike to be called away."
        })
    if CSP_row:
        ideas.append({
            "Strategy": "Cash-Secured Put",
            "Strike": float(CSP_row.get("strike", np.nan)),
            "Est. Premium (mid)": float(((CSP_row.get("bid",0) + CSP_row.get("ask",0))/2)),
            "Bid": float(CSP_row.get("bid", 0.0)),
            "Ask": float(CSP_row.get("ask", 0.0)),
            "Delta": float(CSP_row.get("delta", np.nan)) if "delta" in CSP_row else np.nan,
            "Rationale": "Get paid to potentially buy near support. If the stock stays above strike, you keep the premium and no shares are assigned."
        })

    if ideas:
        st.dataframe(pd.DataFrame(ideas), use_container_width=True)
        st.markdown(
            f"""
<div class="explainer">
<b>Interpretation:</b><br>
• Covered Call near resistance (~${r_price:,.2f}) is suitable when you are neutral-to-slightly-bullish and want income while holding shares.<br>
• Cash-Secured Put near support (~${s_price:,.2f}) is suitable when you are bullish and prefer to be paid to wait for a better entry.
</div>
""", unsafe_allow_html=True)
    else:
        st.info("No suitable ideas (missing chain or expiration). Try another stock or date window.")

# -------------------------------
# Fundamentals (with simple health markers)
# -------------------------------
elif tabs == "Fundamentals":
    st.header(f"{ticker} — Fundamentals")
    tk = yf.Ticker(ticker)
    try:
        fast = tk.fast_info
    except Exception as e:
        fast = {}

    # Build snapshot
    last_price = fast.get("lastPrice")
    eps_ttm = fast.get("epsTrailingTwelveMonths")
    pe_ttm = (last_price / eps_ttm) if last_price and eps_ttm else None
    mcap = fast.get("marketCap")
    dividend_yield = fast.get("dividendYield")
    year_high = fast.get("yearHigh")
    year_low = fast.get("yearLow")
    beta = fast.get("beta")
    shares = fast.get("shares")

    # Health markers (very simple illustrative rules)
    valuation = "Fair"
    if pe_ttm is not None:
        if pe_ttm < 15: valuation = "Attractive"
        elif pe_ttm > 30: valuation = "Expensive"

    income_quality = "Neutral"
    if dividend_yield and dividend_yield > 0.015:
        income_quality = "Income-friendly"

    risk_marker = "Typical"
    if beta and beta > 1.3:
        risk_marker = "High Beta (more volatile)"
    elif beta and beta < 0.8:
        risk_marker = "Low Beta (less volatile)"

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Snapshot")
        rows = {
            "Last Price": last_price,
            "Market Cap": mcap,
            "P/E (TTM)": pe_ttm,
            "EPS (TTM)": eps_ttm,
            "Dividend Yield": dividend_yield,
            "52-Week High": year_high,
            "52-Week Low": year_low,
            "Beta": beta,
            "Shares Outstanding": shares,
        }
        st.dataframe(pd.DataFrame.from_dict(rows, orient="index", columns=["Value"]), use_container_width=True)
    with c2:
        st.subheader("Health Markers")
        st.markdown(f'<span class="badge {"good" if valuation=="Attractive" else ("warn" if valuation=="Expensive" else "")}">Valuation: {valuation}</span>', unsafe_allow_html=True)
        st.markdown(f'<span class="badge {"good" if income_quality=="Income-friendly" else ""}">Income Profile: {income_quality}</span>', unsafe_allow_html=True)
        st.markdown(f'<span class="badge {"risk" if "High" in risk_marker else ("good" if "Low" in risk_marker else "")}">Risk: {risk_marker}</span>', unsafe_allow_html=True)
        st.markdown(
            """
<div class="explainer">
<b>How to read:</b><br>
• Attractive valuation means the price is reasonable vs earnings (lower P/E).<br>
• Income-friendly means dividends are meaningful (but verify sustainability).<br>
• High beta stocks can move more than the market (both up and down).
</div>
""", unsafe_allow_html=True)

# -------------------------------
if tabs == "Forecast":
    st.header(f"{ticker} — Machine Learning Forecast (Prophet)")

    from prophet import Prophet

    # Always load longer history
    df_hist = yf.download(ticker, period="5y", interval="1d", auto_adjust=True)

    # Ensure Close is 1D numeric
    close_series = df_hist["Close"]
    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.iloc[:, 0]

    df_prophet = df_hist.reset_index()[["Date"]].copy()
    df_prophet["y"] = pd.to_numeric(close_series.values, errors="coerce")
    df_prophet = df_prophet.rename(columns={"Date": "ds"}).dropna()

    # Train Prophet
    model = Prophet(daily_seasonality=True, changepoint_prior_scale=0.2)
    model.fit(df_prophet)

    # Forecast 1y ahead
    future = model.make_future_dataframe(periods=252, freq="B")
    forecast = model.predict(future)

    # Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_prophet["ds"], y=df_prophet["y"], name="Actual"))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="Forecast", line=dict(color="red", dash="dash")))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], name="Upper CI", line=dict(color="lightgrey", dash="dot")))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], name="Lower CI", line=dict(color="lightgrey", dash="dot")))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

    # Forecast table
    horizons = [7, 15, 21, 30, 252]
    rows = []
    last_close = float(close_series.iloc[-1])
    for h in horizons:
        row = forecast.iloc[len(df_prophet) + h - 1]
        pred_price = row["yhat"]
        pct = ((pred_price - last_close) / last_close) * 100
        rows.append({"Horizon": f"{h} days", "Predicted Price": f"${pred_price:,.2f}", "Change": f"{pct:+.2f}%"})
    st.subheader("Forecast Values")
    st.table(pd.DataFrame(rows))

    st.markdown("""
<div class="explainer">
<b>How to read:</b><br>
• Red dashed line = expected path.<br>
• Grey bands = uncertainty.<br>
• Short-term (7–30 days) more reliable than 1 year.<br><br>
<b>Practical ideas:</b><br>
• Upward forecasts → safer to sell cash-secured puts.<br>
• Flat forecasts near resistance → covered calls for income.<br>
• Always combine with RSI/MACD & S/R tabs.
</div>
""", unsafe_allow_html=True)


# -------------------------------
# Risk
# -------------------------------
elif tabs == "Risk":
    st.header(f"{ticker} — Risk Overview")

    msgs = []
    if pd.notna(last_rsi):
        if last_rsi > 70:
            msgs.append("RSI is overbought (>70): short-term pullback risk.")
        elif last_rsi < 30:
            msgs.append("RSI is oversold (<30): rebound potential.")
        else:
            msgs.append("RSI is neutral: momentum balanced.")
    if pd.notna(last_macd) and pd.notna(last_macd_sig):
        if last_macd > last_macd_sig:
            msgs.append("MACD above Signal: bullish momentum.")
        else:
            msgs.append("MACD below Signal: bearish momentum.")
    if pd.notna(last_bbw):
        if last_bbw < 0.05:
            msgs.append("Bollinger Band width is very narrow: potential breakout soon.")
        elif last_bbw > 0.20:
            msgs.append("Bollinger Band width is wide: elevated volatility.")

    if msgs:
        for m in msgs:
            if "overbought" in m or "bearish" in m:
                st.warning(m)
            elif "oversold" in m or "bullish" in m:
                st.success(m)
            else:
                st.info(m)
    else:
        st.info("No clear risk signals at the moment.")

    st.markdown(
        f"""
<div class="explainer">
<b>Positioning guidance:</b><br>
• If signals are mixed, smaller position sizes reduce risk.<br>
• Use support (~${nearest_support:,.2f}) for safer entries; consider stops under that area if it breaks.<br>
• If approaching resistance (~${nearest_resist:,.2f}), covered calls can generate income while limiting upside risk.
</div>
""", unsafe_allow_html=True)

# -------------------------------
# News
# -------------------------------
elif tabs == "News":
    st.header(f"{ticker} — Recent News")
    try:
        tk = yf.Ticker(ticker)
        news = tk.news
        if news:
            view_rows = []
            for n in news[:12]:
                title = n.get("title", "")
                link = n.get("link", "")
                provider = n.get("provider", "") or n.get("publisher", "")
                view_rows.append({"Title": title, "Source": provider, "Link": link})
            st.dataframe(pd.DataFrame(view_rows), use_container_width=True)
            st.caption("Tip: Click the Link cell to open in a new tab.")
        else:
            st.info("No recent news found.")
    except Exception as e:
        st.error(f"News fetch error: {e}")
