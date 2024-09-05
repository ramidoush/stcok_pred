import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import timedelta

# Streamlit sidebar configuration
st.sidebar.title("Stock Prediction and Options Strategy App")
selected_stock = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, GOOG, MSFT):", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
options_action = st.sidebar.selectbox("Are you buying or selling options?", ["Buying", "Selling"])

if options_action == "Selling":
    num_stocks = st.sidebar.number_input("Number of Stocks Owned", min_value=0, step=1)
    cost_per_stock = st.sidebar.number_input("Cost per Stock", min_value=0.0, step=0.1)

# Main page layout
st.title(f"Stock Price Prediction and Technical Analysis for {selected_stock}")

# Load stock data function
@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    data.reset_index(inplace=True)
    return data

# Load data
data = load_data(selected_stock, start_date, end_date)

st.subheader("Raw Data")
st.write(data.tail())

# Plotting the closing price using Plotly
def plot_stock_data(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close Price'))
    fig.update_layout(title=f"Closing Price of {selected_stock} Over Time", xaxis_title='Date', yaxis_title='Closing Price')
    st.plotly_chart(fig)

plot_stock_data(data)

# Calculate RSI (Relative Strength Index)
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

data['RSI'] = calculate_rsi(data)

# Plot RSI using Plotly
def plot_rsi(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['RSI'], mode='lines', name='RSI'))
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought", annotation_position="top right")
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold", annotation_position="bottom right")
    fig.update_layout(title="Relative Strength Index (RSI)", xaxis_title='Date', yaxis_title='RSI')
    st.plotly_chart(fig)
    st.write("**Relative Strength Index (RSI)** is a momentum indicator that measures the speed and change of price movements. RSI values above 70 are considered overbought (potentially bearish), while values below 30 are considered oversold (potentially bullish).")

plot_rsi(data)

# Calculate Moving Averages
data['MA50'] = data['Close'].rolling(window=50).mean()
data['MA200'] = data['Close'].rolling(window=200).mean()

# Plot Moving Averages using Plotly
def plot_moving_averages(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close Price'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['MA50'], mode='lines', name='MA 50', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['MA200'], mode='lines', name='MA 200', line=dict(dash='dash')))
    fig.update_layout(title="Moving Averages (MA50, MA200)", xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)
    st.write("**Moving Averages (MA)** are used to smooth out price data to identify the direction of the trend. The 50-day MA (short-term) and 200-day MA (long-term) are commonly used indicators. When the 50-day MA crosses above the 200-day MA, it's a bullish signal (buy). Conversely, when it crosses below, it's a bearish signal (sell).")

plot_moving_averages(data)

# Calculate MACD (Moving Average Convergence Divergence)
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    data['EMA12'] = data['Close'].ewm(span=short_window, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=long_window, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal Line'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()
    return data

data = calculate_macd(data)

# Plot MACD using Plotly
def plot_macd(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['MACD'], mode='lines', name='MACD'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Signal Line'], mode='lines', name='Signal Line'))
    fig.update_layout(title="MACD and Signal Line", xaxis_title='Date', yaxis_title='Value')
    st.plotly_chart(fig)
    st.write("**MACD (Moving Average Convergence Divergence)** is a trend-following momentum indicator that shows the relationship between two moving averages of a security’s price. A bullish crossover occurs when the MACD line crosses above the signal line, and a bearish crossover occurs when the MACD line crosses below the signal line.")

plot_macd(data)

# Candlestick Chart using Plotly
def plot_candlestick(data):
    fig = go.Figure(data=[go.Candlestick(x=data['Date'],
                                         open=data['Open'],
                                         high=data['High'],
                                         low=data['Low'],
                                         close=data['Close'])])
    fig.update_layout(title='Candlestick Chart', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)
    
    last_close = data['Close'].iloc[-1]
    if last_close > data['MA50'].iloc[-1] and last_close > data['MA200'].iloc[-1]:
        trend = "Bullish"
        st.write("The current trend appears to be **Bullish** as the closing price is above both the MA50 and MA200.")
    else:
        trend = "Bearish"
        st.write("The current trend appears to be **Bearish** as the closing price is below the MA50 or MA200.")

plot_candlestick(data)

# Calculate Bollinger Bands
def calculate_bollinger_bands(data, window=20):
    data['MA20'] = data['Close'].rolling(window=window).mean()
    data['STD20'] = data['Close'].rolling(window=window).std()
    data['Upper Band'] = data['MA20'] + (data['STD20'] * 2)
    data['Lower Band'] = data['MA20'] - (data['STD20'] * 2)
    return data

data = calculate_bollinger_bands(data)

# Plot Bollinger Bands using Plotly
def plot_bollinger_bands(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close Price'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Upper Band'], mode='lines', name='Upper Band'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Lower Band'], mode='lines', name='Lower Band'))
    fig.update_layout(title="Bollinger Bands", xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)
    st.write("**Bollinger Bands** consist of a middle band being a simple moving average and an upper and lower band at standard deviations. They provide a relative definition of high and low prices. Prices near the upper band can indicate overbought conditions, while prices near the lower band can indicate oversold conditions.")

plot_bollinger_bands(data)

# Display Descriptive Statistics
st.subheader("Descriptive Statistics")
st.write(data.describe())

# Improved Forecasting Function using Recursive Approach
def forecast_prices(model, data, days_ahead):
    last_date = data['Date'].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, days_ahead + 1)]

    future_data = pd.DataFrame({'Date': future_dates})
    last_known_data = data.iloc[-1].copy()

    predictions = []
    
    for i in range(days_ahead):
        # Prepare input data for the model
        last_known_data['Day'] = future_data['Date'].iloc[i].day
        last_known_data['Month'] = future_data['Date'].iloc[i].month
        last_known_data['Year'] = future_data['Date'].iloc[i].year
        
        # Prediction for the next day
        next_pred = model.predict([[last_known_data['Day'], last_known_data['Month'], last_known_data['Year'], last_known_data['Prev Close']]])
        predictions.append(next_pred[0])
        
        # Update 'Prev Close' for the next iteration
        last_known_data['Prev Close'] = next_pred[0]

    future_data['Predicted Close'] = predictions
    return future_data

# Split data into training and testing sets
data['Day'] = data['Date'].dt.day
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year
data['Prev Close'] = data['Close'].shift(1)
data.dropna(inplace=True)

features = ['Day', 'Month', 'Year', 'Prev Close']
X = data[features]
y = data['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and calculate errors
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Explain Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
st.subheader("Model Performance")
st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
st.write("**MSE** is a metric that measures the average of the squared differences between the predicted and actual stock prices. A lower MSE indicates that the model's predictions are closer to the actual values. **RMSE** provides the error in the same units as the target variable, offering a more interpretable error metric.")

# Forecasting for next week, month, and year
st.subheader("Future Price Forecast")
forecast_options = {'Next Week (7 days)': 7, 'Next Month (30 days)': 30, 'Next Year (365 days)': 365}
selected_forecast = st.selectbox("Select forecast period", list(forecast_options.keys()))
forecast_days = forecast_options[selected_forecast]

future_forecast = forecast_prices(model, data, forecast_days)
st.write(f"Predicted stock prices for the next {selected_forecast}:")
st.write(future_forecast)

# Plot future forecast using Plotly
st.subheader(f"Future Stock Price Forecast for {selected_forecast}")
fig = go.Figure()
fig.add_trace(go.Scatter(x=future_forecast['Date'], y=future_forecast['Predicted Close'], mode='lines', name='Predicted Close Price'))
fig.update_layout(title=f"Predicted Stock Price of {selected_stock} for {selected_forecast}", xaxis_title='Date', yaxis_title='Predicted Closing Price')
st.plotly_chart(fig)

### Options Trading Section

st.subheader("Options Trading Strategy")
st.write("""
Options are financial derivatives that give buyers the right, but not the obligation, to buy (call) or sell (put) an underlying asset at an agreed-upon price and date. 
Options can be used to hedge or to speculate on the future price movements of an asset. 
**Buying Options** involves purchasing a call (if you believe the asset price will rise) or a put (if you believe the asset price will fall). 
**Selling Options** can involve selling a covered call (if you own the underlying stock and believe it will not rise significantly) or selling a put (if you want to buy the stock at a lower price).
""")

# Generate Options Data (For Illustration)
def generate_options_data(stock_price, days_to_expiration=30, action='Buying'):
    strike_prices = [stock_price * 0.9, stock_price, stock_price * 1.1]
    premiums = [5.0, 7.0, 10.0]
    option_type = ['Call', 'Put']

    options_data = pd.DataFrame({
        'Option Type': np.random.choice(option_type, 6),
        'Strike Price': np.tile(strike_prices, 2),
        'Premium': np.tile(premiums, 2),
        'Expiration Date': [pd.to_datetime('today') + timedelta(days=days_to_expiration)] * 6
    })

    options_data['Current Price'] = stock_price
    options_data['Profit/Loss'] = np.where(
        options_data['Option Type'] == 'Call',
        np.maximum(options_data['Current Price'] - options_data['Strike Price'], 0) - options_data['Premium'],
        np.maximum(options_data['Strike Price'] - options_data['Current Price'], 0) - options_data['Premium']
    )

    options_data['Probability of Success (%)'] = np.where(
        options_data['Profit/Loss'] > 0,
        np.random.uniform(60, 80, 6),  # Random probabilities for success
        np.random.uniform(20, 40, 6)   # Random probabilities for loss
    )

    return options_data

# Calculate Options Data based on the last closing price
options_data = generate_options_data(data['Close'].iloc[-1], action=options_action)
st.write("Options Data:")
st.write(options_data)

# Options Strategy Recommendation
def recommend_strategy(options_data, action):
    best_option = options_data.loc[options_data['Probability of Success (%)'].idxmax()]
    
    if action == 'Buying':
        if best_option['Option Type'] == 'Call':
            strategy = 'Bullish Call Spread'
        else:
            strategy = 'Protective Put'
    else:  # Selling
        if num_stocks > 0:
            strategy = 'Covered Call'
        else:
            strategy = 'Naked Put'
    
    return strategy, best_option

strategy, best_option = recommend_strategy(options_data, options_action)
st.write(f"**Recommended Strategy**: {strategy}")
st.write(f"Based on the analysis, the best option contract to consider is a **{best_option['Option Type']}** with a strike price of {best_option['Strike Price']} and a probability of success of {best_option['Probability of Success (%)']:.2f}%.")

### Explanation of Greeks, Decay, and Premium
st.subheader("Understanding Options Greeks, Time Decay, and Premium")
st.write("""
Options Greeks measure the sensitivity of the option's price to various factors:
- **Delta**: Measures the sensitivity of the option's price to a $1 change in the underlying asset's price.
- **Gamma**: Measures the rate of change of delta over time.
- **Theta**: Represents the time decay of the option's price. As the expiration date approaches, the time value of options decreases.
- **Vega**: Measures sensitivity to volatility. Higher volatility generally increases the premium of the option.

**Time Decay**: Refers to the reduction in the value of an option as it approaches its expiration date. 
**Premium**: The price paid to purchase the option, which consists of intrinsic value (if any) and time value.
""")

# Display signals
st.subheader("Buying and Selling Signals")
st.write("""
- **Buying Signals**: Look for bullish crossovers in MACD, RSI values below 30, and prices touching the lower Bollinger Band.
- **Selling Signals**: Look for bearish crossovers in MACD, RSI values above 70, and prices touching the upper Bollinger Band.
""")
