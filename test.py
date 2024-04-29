import streamlit as st
import pandas as pd
import yfinance as yf
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
import datetime
from datetime import date
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import sqlite3


def fetch_all_stock_symbols():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('SELECT DISTINCT ticker FROM stock_data')
    tickers = [row[0] for row in cursor.fetchall()]
    conn.close()
    return tickers


# Define the top 10 stocks from the S&P 100
top_10_stocks = ['AAPL', 'MSFT', 'AMZN', 'FB', 'GOOGL', 'GOOG', 'BRK.B', 'JNJ', 'JPM', 'V']
# Assuming you have a list of all stock symbols
all_stock_symbols = fetch_all_stock_symbols()


def app_page():
    st.title('Stock Price Predictions')
    st.sidebar.info('Welcome to the Stock Price Prediction App. Choose your options below')
    st.sidebar.info("Created and designed by DT")

    # Define widgets for portfolio selection
    portfolio_symbols = st.sidebar.multiselect('Select Portfolio Stocks', all_stock_symbols)

    today = datetime.date.today()
    duration = st.sidebar.number_input('Enter the duration', value=3000)
    before = today - datetime.timedelta(days=duration)
    start_date = st.sidebar.date_input('Start Date', value=before)
    end_date = st.sidebar.date_input('End date', today)

    # Default data for visualization
    selected_symbol = st.sidebar.selectbox('Select a Stock Symbol', all_stock_symbols, index=0)
    data = download_data(selected_symbol, start_date, end_date)
    scaler = StandardScaler()

    if st.sidebar.button('Send'):
        if start_date < end_date:
            st.sidebar.success(f'Start date: {start_date}\n\nEnd date: {end_date}')
            st.write(data)
        else:
            st.sidebar.error('Error: End date must fall after start date')


    option = st.sidebar.selectbox(
        'Make a choice',
        ['Visualize', 'Recent Data', 'Predict'],
        key='main_choice'
    )

    if st.sidebar.button('Analyze Portfolio'):
        portfolio_data = download_portfolio_data(portfolio_symbols, start_date, end_date)
        portfolio_performance = calculate_portfolio_performance(portfolio_data)
        st.line_chart(portfolio_performance)
        st.sidebar.text(f'Portfolio Performance: {calculate_portfolio_stats(portfolio_performance)}')

    if option == 'Visualize':
        st.sidebar.text(f'Statistics for Visualize Option: {calculate_statistics(data)}')
        tech_indicators(data)  # Pass the data variable as an argument
    elif option == 'Recent Data':
        dataframe(data)  # Pass the data variable as an argument
    else:
        predict(data)  # Pass the data variable as an argument

# ... (unchanged code)



def calculate_portfolio_stats(portfolio_performance):
    # Calculate portfolio performance as a number (e.g., sum of closing prices)
    return portfolio_performance.sum(axis=1).mean()

def calculate_statistics(data):
    # Calculate statistics for the Visualize option
    return f'Mean Close Price: {data["Close"].mean()}, Max Close Price: {data["Close"].max()}, Min Close Price: {data["Close"].min()}'

def calculate_portfolio_performance(data):
    # Pivot the data to have closing prices of each stock in separate columns
    portfolio_performance = data.pivot(columns='Symbol', values='Close')
    return portfolio_performance

@st.cache_resource
def download_data(symbols, start_date, end_date):
    all_data = pd.DataFrame()

    for symbol in symbols:
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)
        df['Symbol'] = symbol  # Add a column for the symbol
        all_data = pd.concat([all_data, df], axis=0)

    return all_data
def download_portfolio_data(symbols, start_date, end_date):
    all_data = pd.DataFrame()

    for symbol in symbols:
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)
        df['Symbol'] = symbol  # Add a column for the symbol
        all_data = pd.concat([all_data, df], axis=0)

    return all_data

def tech_indicators(data):  # Accept data as an argument
    st.header('Technical Indicators')
    option = st.radio('Choose a Technical Indicator to Visualize', ['Close', 'BB', 'MACD', 'RSI', 'SMA', 'EMA'])

    bb_indicator = BollingerBands(data.Close)
    bb = data
    bb['bb_h'] = bb_indicator.bollinger_hband()
    bb['bb_l'] = bb_indicator.bollinger_lband()
    bb = bb[['Close', 'bb_h', 'bb_l']]

    macd = MACD(data.Close).macd()
    rsi = RSIIndicator(data.Close).rsi()
    sma = SMAIndicator(data.Close, window=14).sma_indicator()
    ema = EMAIndicator(data.Close).ema_indicator()

    if option == 'Close':
        st.write('Close Price')
        st.line_chart(data.Close)
    elif option == 'BB':
        st.write('BollingerBands')
        st.line_chart(bb)
    elif option == 'MACD':
        st.write('Moving Average Convergence Divergence')
        st.line_chart(macd)
    elif option == 'RSI':
        st.write('Relative Strength Indicator')
        st.line_chart(rsi)
    elif option == 'SMA':
        st.write('Simple Moving Average')
        st.line_chart(sma)
    else:
        st.write('Exponential Moving Average')
        st.line_chart(ema)

def dataframe(data):  # Accept data as an argument
    st.header('Recent Data')
    st.dataframe(data.tail(10))

def predict(data):  # Accept data as an argument
    model = st.radio('Choose a model',
                     ['LinearRegression', 'RandomForestRegressor', 'ExtraTreesRegressor', 'KNeighborsRegressor',
                      'XGBoostRegressor'])
    num = st.number_input('How many days forecast?', value=5)
    num = int(num)
    if st.button('Predict'):
        if model == 'LinearRegression':
            engine = LinearRegression()
            model_engine(engine, num, data)  # Pass the data variable as an argument
        elif model == 'RandomForestRegressor':
            engine = RandomForestRegressor()
            model_engine(engine, num, data)  # Pass the data variable as an argument
        elif model == 'ExtraTreesRegressor':
            engine = ExtraTreesRegressor()
            model_engine(engine, num, data)  # Pass the data variable as an argument
        elif model == 'KNeighborsRegressor':
            engine = KNeighborsRegressor()
            model_engine(engine, num, data)  # Pass the data variable as an argument
        else:
            engine = XGBRegressor()
            model_engine(engine, num, data)  # Pass the data variable as an argument

def model_engine(model, num, data):
    scaler = StandardScaler()  # Define the scaler here
    df = data[['Close']]
    df['preds'] = data.Close.shift(-num)
    x = df.drop(['preds'], axis=1).values
    x = scaler.fit_transform(x)
    x_forecast = x[-num:]
    x = x[:-num]
    y = df.preds.values
    y = y[:-num]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    st.text(f'r2_score: {r2_score(y_test, preds)} \nMAE: {mean_absolute_error(y_test, preds)}')

    # Predicting stock prices based on the number of days
    forecast_pred = model.predict(x_forecast)
    day = 1
    for i in forecast_pred:
        st.text(f'Day {day}: {i}')
        day += 1


if __name__ == '__main__':
            app_page()