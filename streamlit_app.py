import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
import requests

# User Input for Stock Ticker

STOCK = st.text_input('Enter the stock name', 'GOOG')
st.text("Enter(.bo) for indian stocks")
if False:
    print("error")
else:
    # Fetch data from Yahoo Finance
    try:
        stock_data = yf.download(STOCK, period="5y")  # Download last 5 years of data
        if stock_data.empty:
            st.error("No data found for the entered stock symbol.")
        else:
            stock_data.columns = ['open', 'high', 'low', 'close', 'adj close', 'volume']
            stock_data = stock_data.drop('adj close', axis=1)
            st.subheader("Stock Data")
            st.write(stock_data.iloc[-20:].iloc[::-1])  # Reverse the data to show the most recent first
    except Exception as e:
        st.error(f"Failed to retrieve data from Yahoo Finance: {e}")

# Only proceed if stock_data is not empty
if 'stock_data' in locals() and not stock_data.empty:
    # Streamlit App Title
    st.title("Enhanced Stock Analysis with Time Frame Selection")

    # Time Frame Selection
    time_frame = st.select_slider(
        "Select Time Frame for Candlestick Chart",
        options=["1 Week", "2 Weeks", "3 Weeks", "1 Month", "2 Months", "1 Year"],
        value="1 Week"
    )

    # Filter Data Based on Selected Time Frame
    if time_frame == "1 Week":
        filtered_data = stock_data[-7:]
    elif time_frame == "2 Weeks":
        filtered_data = stock_data[-14:]
    elif time_frame == "3 Weeks":
        filtered_data = stock_data[-21:]
    elif time_frame == "1 Month":
        filtered_data = stock_data[-30:]
    elif time_frame == "2 Months":
        filtered_data = stock_data[-60:]
    elif time_frame == "1 Year":
        filtered_data = stock_data[-365:]

    # Function to Plot Candlestick Chart
    def plot_candlestick(data):
        min_price = data['low'].min()
        max_price = data['high'].max()
        padding = (max_price - min_price) * 0.1  # 10% padding

        fig = go.Figure(data=[go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            increasing_line_color='green', decreasing_line_color='red'
        )])
        fig.update_layout(
            title=f'Candlestick Chart ({time_frame})',
            xaxis_title='Date',
            yaxis_title='Price',
            yaxis=dict(range=[min_price - padding, max_price + padding]),  # Add padding
            xaxis_rangeslider_visible=False,  # Hide range slider for cleaner view
            template='plotly_white'  # Use a white theme for better readability
        )
        return fig

    # Display Dynamic Candlestick Chart
    st.subheader(f'Candlestick Chart ({time_frame})')
    st.plotly_chart(plot_candlestick(filtered_data))

    # Function to Plot Moving Averages
    def plot_moving_averages(data):
        data['MA_50'] = data['close'].rolling(window=50).mean()
        data['MA_100'] = data['close'].rolling(window=100).mean()
        data['MA_200'] = data['close'].rolling(window=200).mean()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['close'], mode='lines', name='Close Price'))
        fig.add_trace(go.Scatter(x=data.index, y=data['MA_50'], mode='lines', name='50-Day MA'))
        fig.add_trace(go.Scatter(x=data.index, y=data['MA_100'], mode='lines', name='100-Day MA'))
        fig.add_trace(go.Scatter(x=data.index, y=data['MA_200'], mode='lines', name='200-Day MA'))
        
        fig.update_layout(title='Moving Averages', xaxis_title='Date', yaxis_title='Price')
        return fig

    # Display Moving Averages
    st.subheader('Moving Averages')
    st.plotly_chart(plot_moving_averages(stock_data))

    # Function to Plot Bollinger Bands
    def plot_bollinger_bands(data):
        data['SMA_20'] = data['close'].rolling(window=20).mean()
        data['STD_20'] = data['close'].rolling(window=20).std()
        data['Upper'] = data['SMA_20'] + (data['STD_20'] * 2)
        data['Lower'] = data['SMA_20'] - (data['STD_20'] * 2)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['close'], mode='lines', name='Close Price'))
        fig.add_trace(go.Scatter(x=data.index, y=data['Upper'], mode='lines', line=dict(color='rgba(0, 255, 0, 0.2)'), name='Upper Band'))
        fig.add_trace(go.Scatter(x=data.index, y=data['Lower'], mode='lines', line=dict(color='rgba(255, 0, 0, 0.2)'), name='Lower Band'))
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], mode='lines', line=dict(color='blue'), name='20-Day SMA'))
        
        fig.update_layout(title='Bollinger Bands', xaxis_title='Date', yaxis_title='Price')
        fig.update_traces(fill='tonexty', mode='lines')
        return fig

    # Display Bollinger Bands
    st.subheader('Bollinger Bands')
    st.plotly_chart(plot_bollinger_bands(stock_data))

    # Function to Plot Volume
    def plot_volume(data):
        fig = go.Figure(data=[go.Bar(x=data.index, y=data['volume'], marker_color='orange')])
        fig.update_layout(title='Trading Volume', xaxis_title='Date', yaxis_title='Volume')
        return fig

    # Display Trading Volume
    st.subheader('Trading Volume')
    st.plotly_chart(plot_volume(stock_data))
else:
    st.warning("Stock data is not available.")
