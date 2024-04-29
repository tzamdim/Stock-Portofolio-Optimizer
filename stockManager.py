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
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import sqlite3


# Function to connect to the SQLite database and retrieve stock symbols
def retrieve_stock_symbols():
    # Connect to the SQLite database
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()

    # Retrieve the list of stock symbols from the database
    cursor.execute("SELECT ticker FROM stock_data")
    stocks = [row[0] for row in cursor.fetchall()]

    conn.close()
    return stocks


# Streamlit app page
def app_page():
    st.title('Stock Price Visualization/Recent Data/Prediction')
    st.sidebar.info('Welcome to the Stock Price Prediction App. Choose your options below')
    st.sidebar.info("Created and designed by DT")

    # Retrieve stock symbols from the database
    stocks = retrieve_stock_symbols()

    # Stock symbol selection
    option = st.sidebar.selectbox('Enter a Stock Symbol', options=stocks, index=0)
    option = option.upper()

    today = datetime.date.today()
    duration = st.sidebar.number_input('Enter the duration', value=3000)
    before = today - datetime.timedelta(days=duration)
    start_date = st.sidebar.date_input('Start Date', value=before)
    end_date = st.sidebar.date_input('End date', today)

    if st.sidebar.button('Send'):
        if start_date < end_date:
            st.sidebar.success(f'Start date: {start_date}\n\nEnd date: {end_date}')
            data = download_data(option, start_date, end_date)
            st.write(data)
        else:
            st.sidebar.error('Error: End date must fall after start date')

    data = download_data(option, start_date, end_date)
    scaler = StandardScaler()

    option = st.sidebar.selectbox(
        'Make a choice',
        ['Visualize', 'Recent Data', 'Predict'],
        key='main_choice'
    )

    if option == 'Visualize':
        visualize_stock_data(data)  # Pass the data variable
    elif option == 'Recent Data':
        dataframe(data)  # Pass the data variable
    else:
        predict(data)  # Pass the data variable


# Function to download stock data
@st.cache_data
def download_data(op, start_date, end_date):
    df = yf.download(op, start=start_date, end=end_date, progress=False)
    return df


# Function to visualize stock data
def visualize_stock_data(data):
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['Close'], label='Close Price')

    # Calculate Bollinger Bands
    indicator_bb = BollingerBands(close=data['Close'], window=20, window_dev=2)
    data['bb_high'] = indicator_bb.bollinger_hband()
    data['bb_mid'] = indicator_bb.bollinger_mavg()
    data['bb_low'] = indicator_bb.bollinger_lband()

    plt.plot(data.index, data['bb_high'], label='Upper Band', color='red', alpha=0.3)
    plt.plot(data.index, data['bb_mid'], label='Middle Band', color='blue', alpha=0.3)
    plt.plot(data.index, data['bb_low'], label='Lower Band', color='green', alpha=0.3)

    plt.title("Stock Performance with Bollinger Bands")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(plt)


# Function to display recent data
def dataframe(data):
    st.header('Recent Data')
    st.dataframe(data.tail(10))


# Function to predict stock prices
def predict(data):
    model = st.radio('Choose a model',
                     ['LinearRegression', 'RandomForestRegressor', 'ExtraTreesRegressor', 'KNeighborsRegressor',
                      'XGBoostRegressor'])
    num = st.number_input('How many days forecast?', value=5)
    num = int(num)
    if st.button('Predict'):
        if model == 'LinearRegression':
            engine = LinearRegression()
            model_engine(engine, num, data)
        elif model == 'RandomForestRegressor':
            engine = RandomForestRegressor()
            model_engine(engine, num, data)
        elif model == 'ExtraTreesRegressor':
            engine = ExtraTreesRegressor()
            model_engine(engine, num, data)
        elif model == 'KNeighborsRegressor':
            engine = KNeighborsRegressor()
            model_engine(engine, num, data)
        else:
            engine = XGBRegressor()
            model_engine(engine, num, data)


# Function to train and predict with selected model
def model_engine(model, num, data):
    scaler = StandardScaler()
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
