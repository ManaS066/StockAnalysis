import streamlit as st
import re
import numpy as np
import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from datetime import datetime, timedelta
# Function to validate email
def is_valid_email(email):
    # Basic email validation using regular expression
    pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
    return re.match(pattern, email)

# Function to validate phone number (10-digit number for this example)
def is_valid_phone(phone):
    # Basic phone validation (for 10-digit numbers)
    pattern = r"^\d{10}$"
    return re.match(pattern, phone)

def company_details():
    st.markdown(
        """
        <style>
        body {
            background-color: #f9f9f9;  /* Light background for better brightness */
            font-family: 'Arial', sans-serif;
        }
        h1 {
            color: #2e8b57;  /* Dark green color for title */
            text-align: center;
            font-size: 2.5em;
            margin-top: 20px;
        }
        h2 {
            color: #4682b4;  /* Steel blue for subtitles */
            font-size: 2em;
            margin-top: 15px;
        }
        h3 {
            color: #005f73;  /* Dark cyan for h3 headers */
            margin-top: 10px;
        }
        p {
            color: #333333;  /* Dark gray for text */
            font-size: 16px;
            line-height: 1.6;
            margin: 0 0;
        }
        .feature, .benefit {
            background-color: #ffffff;  /* White background for features and benefits */
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            transition: transform 0.2s;
        }
        .feature:hover, .benefit:hover {
            transform: scale(1.02);
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
        }
        .cta {
            background-color: #2e8b57;  /* Dark green for call to action */
            color: white;  /* White text */
            padding: 10px 20px;
            border-radius: 5px;
            text-align: center;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        .cta:hover {
            background-color: #005f73;  /* Darker green on hover */
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title('Company Overview and Financials')
    STOCK = st.text_input('Enter the stock name', 'GOOG')
    st.text("Enter(.bo) for Indian stocks")
    ticker = yf.Ticker(STOCK)
    stock_data = yf.download(STOCK, period="1y")
    company_info = ticker.info
    company_name = company_info.get('longName', 'Unknown Company Name')  # Fetch company name from info
    net_worth = company_info.get('networth')
    st.subheader(f"Company: {company_name}")
    st.table(stock_data.iloc[-10:].iloc[::-1])  # 
    income = ticker.income_stmt
    income_without_last_row = income.drop(income.index[-1])
    st.subheader(f'Income Statement of {company_name}')
    st.write(income_without_last_row)
    st.subheader(f'Major Holders of {company_name}')
    st.write(ticker.major_holders)
    st.subheader(f'Balance Sheet of {company_name}')
    st.write(ticker.balance_sheet)
    st.subheader(f'Recommendations of {company_name}')
    st.write(ticker.recommendations)
    st.subheader(f'Cash Flow Statement of {company_name}')
    st.write(ticker.quarterly_cashflow)
    st.subheader(f'Dividends of {company_name}')
    st.write(ticker.dividends)
    st.subheader(f'Financial Ratios of {company_name}')
    st.write(ticker.financials)
    st.subheader(f'Key Financials of {company_name}')
    st.write(ticker.institutional_holders)
    st.subheader(f'Sustainability Score of {company_name}')
    x=ticker.sustainability
    st.write(x.T)
    st.subheader(f'Earnings History of {company_name}')
    xx=ticker.calendar
    data = {
    "Event": ["Dividend Date", "Ex-Dividend Date", "Earnings Dates", "Earnings High", "Earnings Low", "Earnings Average", "Revenue High", "Revenue Low", "Revenue Average"],
    "Details": [
        xx["Dividend Date"],
        xx["Ex-Dividend Date"],
        f"{xx['Earnings Date'][0]} & {xx['Earnings Date'][1]}",
        xx["Earnings High"],
        xx["Earnings Low"],
        xx["Earnings Average"],
        f"{xx['Revenue High']:,}",
        f"{xx['Revenue Low']:,}",
        f"{xx['Revenue Average']:,}"
    ]
}

# Convert to DataFrame
    df = pd.DataFrame(data)

    # Display as a table in Streamlit
    st.subheader('Company Calendar Details (Tabular Format)')
    st.table(df)


    news_data = ticker.news  # Fetch the news dynamically from the ticker
    #Display the top 5 news articles
    st.subheader(f'Top 10 News Articles for {ticker.info["shortName"]}')  # Show company name

    for news_item in news_data[:10]:  # Limit to the top 5 news items
        title = news_item.get('title', 'No Title Available')
        publisher = news_item.get('publisher', 'Unknown Publisher')
        link = news_item.get('link', '#')  # Default to # if link is not available
        thumbnail = news_item.get('thumbnail', {}).get('resolutions', [{}])[0].get('url', '')

        # Display each news item with title, publisher, and link
        st.markdown(f"### [{title}]({link})")
        st.write(f"Published by: {publisher}")
        
        # Display the thumbnail if available
        if thumbnail:
            st.image(thumbnail, width=150)

        st.markdown("---")  # Divider between news articles


def prediction():
    st.title("Predictive Analysis")
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
    st.markdown(
        """
        <style>
        body {
            background-color: #f9f9f9;  /* Light background for better brightness */
            font-family: 'Arial', sans-serif;
        }
        h1 {
            color: #2e8b57;  /* Dark green color for title */
            text-align: center;
            font-size: 2.5em;
        }""")
   

    # Fetch stock data from yfinance (for the last 2 months)
    def get_stock_data(ticker):
        end_date = datetime.today().strftime('%Y-%m-%d')  # Current date
        start_date = (datetime.today() - timedelta(days=60)).strftime('%Y-%m-%d')  # Fetch last 60 days of data
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        return stock_data

    # Prepare the data for LSTM
    def prepare_data(data, time_step=30):  # Reduced time step to 30
        # Convert data into numpy arrays
        close_prices = data['Close'].values.reshape(-1, 1)
        open_prices = data['Open'].values.reshape(-1, 1)
        
        # Normalize the data
        scaler_close = MinMaxScaler(feature_range=(0, 1))
        scaler_open = MinMaxScaler(feature_range=(0, 1))
        scaled_close = scaler_close.fit_transform(close_prices)
        scaled_open = scaler_open.fit_transform(open_prices)
        
        # Prepare training data
        X, y_open, y_close = [], [], []
        for i in range(time_step, len(scaled_close)):
            X.append(scaled_close[i-time_step:i, 0])  # Using close prices as input
            y_open.append(scaled_open[i, 0])
            y_close.append(scaled_close[i, 0])
        
        X, y_open, y_close = np.array(X), np.array(y_open), np.array(y_close)
        
        # Reshape X for LSTM (samples, time steps, features)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        return X, y_open, y_close, scaler_open, scaler_close

    # Build the LSTM model
    def build_lstm_model():
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(30, 1)))  # Updated time step to 30
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dense(units=25))
        model.add(Dense(units=1))  # Output layer for price prediction
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    # Predict next 5 days' open and close prices
    def predict_next_days(model_open, model_close, last_30_days, scaler_open, scaler_close, n_days=5):
        predicted_open_prices = []
        predicted_close_prices = []
        
        last_30_days_scaled = scaler_close.transform(last_30_days)  # Normalize the last 30 days

        for _ in range(n_days):
            X_input = last_30_days_scaled.reshape(1, 30, 1)
            
            # Predict the next open and close prices
            pred_open_scaled = model_open.predict(X_input)
            pred_close_scaled = model_close.predict(X_input)
            
            pred_open = scaler_open.inverse_transform(pred_open_scaled)[0, 0]
            pred_close = scaler_close.inverse_transform(pred_close_scaled)[0, 0]
            
            predicted_open_prices.append(pred_open)
            predicted_close_prices.append(pred_close)
            
            # Update the last 30 days with predicted close price for next prediction
            last_30_days_scaled = np.append(last_30_days_scaled, pred_close_scaled)
            last_30_days_scaled = last_30_days_scaled[1:]  # Remove the first entry to maintain 30 time steps
        
        return predicted_open_prices, predicted_close_prices

    # Main function
    def main():
        # Step 1: Get stock data
        ticker = STOCK  # Change this to any ticker symbol
        stock_data = get_stock_data(ticker)
        
        # Ensure we have at least 30 days of data
        if len(stock_data) < 30:
            print("Not enough data to make predictions.")
            return
        
        # Step 2: Prepare the data for LSTM
        time_step = 30  # Reduced time step to 30 days
        X, y_open, y_close, scaler_open, scaler_close = prepare_data(stock_data, time_step)
        
        # Step 3: Build and train LSTM models for both open and close prices
        model_open = build_lstm_model()
        model_close = build_lstm_model()
        
        model_open.fit(X, y_open, batch_size=64, epochs=10, verbose=1)
        model_close.fit(X, y_close, batch_size=64, epochs=10, verbose=1)
        
        # Step 4: Get the last 30 days of data for prediction
        last_30_days = stock_data['Close'].values[-30:].reshape(-1, 1)
        
        # Step 5: Predict the next 5 days' open and close prices
        predicted_open, predicted_close = predict_next_days(model_open, model_close, last_30_days, scaler_open, scaler_close, n_days=5)
        
        # Step 6: Print the next 5 days' predicted prices
        print(f"Predicted stock prices for {ticker} for the next 5 days based on the last 30 days' data:")
        for i in range(5):
            st.write(f"Day {i+1}: Opening Price: {predicted_open[i]:.2f}, Closing Price: {predicted_close[i]:.2f}")

    # Run the program
    if __name__ == "__main__":
        main()

def analysis():
    st.title("Enhanced Stock Analysis with Time Frame Selection")
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
    st.markdown(
        """
        <style>
        body {
            background-color: #f9f9f9;  /* Light background for better brightness */
            font-family: 'Arial', sans-serif;
        }
        h1 {
            color: #2e8b57;  /* Dark green color for title */
            text-align: center;
            font-size: 2.5em;
            margin-top: 20px;
        }
        h2 {
            color: #4682b4;  /* Steel blue for subtitles */
            font-size: 2em;
            margin-top: 15px;
        }
        h3 {
            color: #005f73;  /* Dark cyan for h3 headers */
            margin-top: 10px;
        }
        p {
            color: #333333;  /* Dark gray for text */
            font-size: 16px;
            line-height: 1.6;
            margin: 0 0;
        }
        .feature, .benefit {
            background-color: #ffffff;  /* White background for features and benefits */
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            transition: transform 0.2s;
        }
        .feature:hover, .benefit:hover {
            transform: scale(1.02);
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
        }
        .cta {
            background-color: #2e8b57;  /* Dark green for call to action */
            color: white;  /* White text */
            padding: 10px 20px;
            border-radius: 5px;
            text-align: center;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        .cta:hover {
            background-color: #005f73;  /* Darker green on hover */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    if False:
        print("error")
    else:
        # Fetch data from Yahoo Finance
        try:
            stock_data = yf.download(STOCK, period="2y")  # Download last 2 years of data
            
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
                st.header(f"Company Name: {company_name}")
                #st.write(stock_data.iloc[-20:].iloc[::-1])  # Reverse the data to show the most recent first
        except Exception as e:
            st.error(f"Failed to retrieve data from Yahoo Finance: {e}")

    # Only proceed if stock_data is not empty
    if 'stock_data' in locals() and not stock_data.empty:
        # Streamlit App Title
       

        # Time Frame Selection
        

        # Function to Plot Candlestick Chart
        def plot_candlestick(data, time_frame):
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
                
                xaxis_title='Date',
                yaxis_title='Price',
                yaxis=dict(range=[min_price - padding, max_price + padding]),  # Add padding
                xaxis_rangeslider_visible=False,  # Hide range slider for cleaner view
                template='plotly_white'  # Use a white theme for better readability
            )
            return fig

# Initial time frame for chart display
        time_frame = "1 Week"

        # Filter data based on initial time frame
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

        time_frame = "1 Week"
        filtered_data = stock_data[-7:]

        # Display the candlestick chart
        st.subheader(f'Candlestick Chart')
        chart_placeholder = st.empty()  # Placeholder for the chart
        chart_placeholder.plotly_chart(plot_candlestick(filtered_data, time_frame), use_container_width=True)


        # Define the time frame with the slider
        time_frame = st.select_slider(
            "",
            options=["1 Week", "2 Weeks", "3 Weeks", "1 Month", "2 Months", "1 Year"],
            value="1 Week"  # Set "1 Week" as default
        )

        # Re-filter the data based on the selected time frame
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

        # Update the chart in the placeholder
        chart_placeholder.plotly_chart(plot_candlestick(filtered_data, time_frame), use_container_width=True)

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

            # Update layout for the figure
            fig.update_layout(
                
                xaxis_title='Date', 
                yaxis_title='Price',
                template='plotly_white'
            )
            
            return fig

        # Display the title of the section
        st.subheader('Moving Averages')

        # Create a placeholder for the moving average chart
        chart_placeholder = st.empty()

        # Create a radio button to select the moving average
        ma_option = st.radio(
           '',
            ('Short-Term (50-Day MA)', 'Long-Term (200-Day MA)')
        )

        # Display the initial moving average chart
        chart_placeholder.plotly_chart(plot_moving_average(stock_data, ma_option), use_container_width=True)

        # Update the chart in the placeholder based on the selection
        if ma_option:
            # Update the chart in the placeholder
            chart_placeholder.plotly_chart(plot_moving_average(stock_data, ma_option), use_container_width=True)

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
            
            fig.update_layout( xaxis_title='Date', yaxis_title='Price')
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
            fig.update_layout( xaxis_title='Date', yaxis_title='MACD')
            return fig
        st.subheader('MACD (Moving Average Convergence Divergence)')
        st.plotly_chart(plot_macd(stock_data))


        # Function to Plot Volume
        def plot_volume(data):
            min_volume = data['volume'].min()
            max_volume = data['volume'].max()
            padding = (max_volume - min_volume) * 0.1  # 10% padding for better visualization

            fig = go.Figure(data=[go.Bar(x=data.index, y=data['volume'], marker_color='orange')])
            fig.update_layout(
                
                xaxis_title='Date',
                yaxis_title='Volume',
                yaxis=dict(range=[min_volume - padding, max_volume + padding]),
                template='plotly_white'
            )
            return fig

        # Display Trading Volume
        st.subheader('Trading Volume')
        st.markdown('<div class="custom-chart-container">', unsafe_allow_html=True)
        st.plotly_chart(plot_volume(filtered_data), use_container_width=False)  # Turn off container width
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.warning("Stock data is not available.")

def home_page():
    st.title("Welcome to the Stock Prediction App!", anchor="header")
    
    # Custom CSS for styling
    st.markdown(
        """
        <style>
        body {
            background-color: #f9f9f9;  /* Light background for better brightness */
            font-family: 'Arial', sans-serif;
        }
        h1 {
            color: #2e8b57;  /* Dark green color for title */
            text-align: center;
            font-size: 2.5em;
            margin-top: 20px;
        }
        h2 {
            color: #4682b4;  /* Steel blue for subtitles */
            font-size: 2em;
            margin-top: 15px;
        }
        h3 {
            color: #005f73;  /* Dark cyan for h3 headers */
            margin-top: 10px;
        }
        p {
            color: #333333;  /* Dark gray for text */
            font-size: 16px;
            line-height: 1.6;
            margin: 0 0;
        }
        .feature, .benefit {
            background-color: #ffffff;  /* White background for features and benefits */
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            transition: transform 0.2s;
        }
        .feature:hover, .benefit:hover {
            transform: scale(1.02);
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
        }
        .cta {
            background-color: #2e8b57;  /* Dark green for call to action */
            color: white;  /* White text */
            padding: 10px 20px;
            border-radius: 5px;
            text-align: center;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        .cta:hover {
            background-color: #005f73;  /* Darker green on hover */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Introduction
    st.subheader("Empower Your Trading Decisions with Data-Driven Insights")
    st.write("""
    This app is designed to provide traders and investors with real-time analysis and predictions on stock market trends. 
    Using advanced machine learning models and technical analysis tools, we aim to help you make informed trading decisions.
    """)

    # Add images
    st.image("buysell.jpg", use_column_width=True)

    # Key features
    st.markdown("### Key Features")
    st.write("""
    <div class="feature">- <strong>Stock Price Prediction</strong>: Use machine learning algorithms to predict future stock prices based on historical data.</div>
    <div class="feature">- <strong>Technical Indicators</strong>: Analyze popular indicators like Moving Averages, MACD, Bollinger Bands, and more.</div>
    <div class="feature">- <strong>Candlestick Patterns</strong>: Identify key candlestick patterns for buy/sell signals.</div>
    <div class="feature">- <strong>Real-Time Market Data</strong>: Get the latest stock prices, trends, and market sentiment.</div>
    <div class="feature">- <strong>Custom Analysis</strong>: Perform in-depth technical analysis using charts and indicators of your choice.</div>
    """, unsafe_allow_html=True)

    # User benefits
    st.markdown("### Why Use This App?")
    st.write("""
    <div class="benefit">- <strong>Accurate Predictions</strong>: Leverage cutting-edge models trained on historical data to forecast future stock movements.</div>
    <div class="benefit">- <strong>Comprehensive Analysis</strong>: Get access to a variety of technical analysis tools to evaluate stocks thoroughly.</div>
    <div class="benefit">- <strong>User-Friendly Interface</strong>: The app is designed to be simple yet powerful, allowing traders of all experience levels to benefit.</div>
    <div class="benefit">- <strong>Customizable</strong>: Choose the stocks, indicators, and timeframes that matter to you.</div>
    """, unsafe_allow_html=True)

    # Stock overview
    st.markdown("### Live Stock Overview")
    st.write("Enter a stock ticker to get a quick snapshot of the current market trends, prices, and key metrics.")

    # Add a relevant image for live stock overview
    st.image("img.jpg", use_column_width=True)

    # Call to action
    st.markdown("""
    <div class="cta" onclick="alert('Redirecting to Analysis section...')">Ready to explore deeper analysis? Click here to go to the Analysis section!</div>
    """, unsafe_allow_html=True)
# Initialize session state for login if not already set
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Show login form if the user is not logged in
if not st.session_state.logged_in:
    st.title("Login")
    st.write("Please enter your email and phone number to access the app.")
    
    # Create a form for email and phone number input
    with st.form("login_form"):
        email = st.text_input("Email")
        phone = st.text_input("Phone Number")
        
        # Submit button
        submitted = st.form_submit_button("Login")
       
        if submitted:
            # Validate email and phone
            if is_valid_email(email) and is_valid_phone(phone):
                st.session_state.logged_in = True
                st.success("Login successful! You now have access to the app.")
            else:
                st.error("Invalid email or phone number. Please try again.")
else:
    # The main app content after successful login
    st.sidebar.title("STOCK PREDICTION")

    # Create individual buttons for each page
    if st.sidebar.button("Home"):
        st.session_state.selected_page = "Home"
    if st.sidebar.button("Analysis"):
        st.session_state.selected_page = "Analysis"
    if st.sidebar.button("Predict"):
        st.session_state.selected_page = "Predict"
    

    # Default page is Home if no button is clicked
    if 'selected_page' not in st.session_state:
        st.session_state.selected_page = "Home"

    # Home Page
    if st.session_state.selected_page == "Home":
        home_page()

    # Analysis Page
    elif st.session_state.selected_page == "Analysis":
        sp = st.sidebar.radio(
                    "Add Information",
                    ("Home","Company details")
                )
        if sp == "Company details":
            company_details()
        else:
            analysis()

    # Predict Page
    elif st.session_state.selected_page == "Predict":
        prediction()

    # Login Page
    