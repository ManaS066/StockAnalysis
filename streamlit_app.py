import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
import requests

# User Input for Stock Ticker
STOCK = st.text_input('Enter the stock name', 'GOOG')
st.text("Enter(.bo) for Indian stocks")

# Include Custom CSS
st.markdown("""
<style>
.custom-chart-container {
    width: 1000px; /* Set your desired width */
    margin: auto; /* Center the chart */
}
</style>
""", unsafe_allow_html=True)

if False:
    print("error")
else:
    # Fetch data from Yahoo Finance
    try:
        stock_data = yf.download(STOCK, period="3y")  # Download last 5 years of data
        if stock_data.empty:
            st.error("No data found for the entered stock symbol.")
        else:
            stock_data.columns = ['open', 'high', 'low', 'close', 'adj close', 'volume']
            stock_data = stock_data.drop('adj close', axis=1)
            # Ensure moving averages are calculated
            if 'MA_50' not in stock_data.columns:
                stock_data['MA_50'] = stock_data['close'].rolling(window=50).mean()

            if 'MA_200' not in stock_data.columns:
                stock_data['MA_200'] = stock_data['close'].rolling(window=200).mean()

            ticker = yf.Ticker(STOCK)
            company_info = ticker.info
            company_name = company_info.get('longName', 'Unknown Company Name')  # Fetch company name from info
            st.subheader(f"Company: {company_name}")
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
    st.markdown('<div class="custom-chart-container">', unsafe_allow_html=True)
    st.plotly_chart(plot_candlestick(filtered_data), use_container_width=False)  # Turn off container width
    st.markdown('</div>', unsafe_allow_html=True)
    ma_option = st.radio(
        "Select Moving Average to Display:",
        ('Short-Term (50-Day MA)', 'Long-Term (200-Day MA)')
    )
    # Function to Plot Moving Averages
    def plot_moving_average(data, ma_type):
        fig = go.Figure()

    # Plot the Close Price
        fig.add_trace(go.Scatter(x=data.index, y=data['close'], mode='lines', name='Close Price'))

    # Check if the required moving average column exists
        if ma_type == 'Short-Term (50-Day MA)' and 'MA_50' in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data['MA_50'], mode='lines', name='50-Day MA'))
        elif ma_type == 'Long-Term (200-Day MA)' and 'MA_200' in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data['MA_200'], mode='lines', name='200-Day MA'))
        else:
            st.error(f"{ma_type} data is not available.")

        fig.update_layout(title=f'Moving Averages - {ma_type}', xaxis_title='Date', yaxis_title='Price')
        return fig


    # Display Moving Averages
    st.subheader('Moving Averages')
    st.markdown('<div class="custom-chart-container">', unsafe_allow_html=True)
    st.plotly_chart(plot_moving_average(stock_data,ma_option), use_container_width=False)  # Turn off container width
    st.markdown('</div>', unsafe_allow_html=True)

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
    st.markdown('<div class="custom-chart-container">', unsafe_allow_html=True)
    st.plotly_chart(plot_bollinger_bands(stock_data), use_container_width=False)  # Turn off container width
    st.markdown('</div>', unsafe_allow_html=True)
    def plot_macd(data):
    # Calculate MACD
        short_ema = data['close'].ewm(span=12, adjust=False).mean()
        long_ema = data['close'].ewm(span=26, adjust=False).mean()
        macd_line = short_ema - long_ema
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line
    
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=macd_line, mode='lines', name='MACD Line'))
        fig.add_trace(go.Scatter(x=data.index, y=signal_line, mode='lines', name='Signal Line'))
        fig.add_trace(go.Bar(x=data.index, y=histogram, name='MACD Histogram', marker_color='gray'))
        fig.update_layout(title='MACD (Moving Average Convergence Divergence)', xaxis_title='Date', yaxis_title='MACD')
        return fig
    st.subheader('MACD (Moving Average Convergence Divergence)')
    st.plotly_chart(plot_macd(stock_data))


    # Function to Plot Volume
    def plot_volume(data):
        min_volume = data['volume'].min()
        max_volume = data['volume'].max()
        padding = (max_volume - min_volume) * 0.1  # 10% padding for better visualization

        fig = go.Figure(data=[go.Bar(x=data.index, y=data['volume'], marker_color='orange')])

    # Update layout with dynamic y-axis range
        fig.update_layout(
            title='Trading Volume',
            xaxis_title='Date',
            yaxis_title='Volume',
            yaxis=dict(range=[min_volume - padding, max_volume + padding]),  # Set y-axis range dynamically
            template='plotly_white'  # Optional: Set a white theme for better readability
        )
        return fig

    # Display Trading Volume
    st.subheader('Trading Volume')
    st.markdown('<div class="custom-chart-container">', unsafe_allow_html=True)
    st.plotly_chart(plot_volume(stock_data), use_container_width=False)  # Turn off container width
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.warning("Stock data is not available.")
