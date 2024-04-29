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
import matplotlib.pyplot as plt

def app_page():
    st.title('Stock Price Predictions')
    st.sidebar.info('Welcome to the Stock Price Prediction App. Choose your options below')
    st.sidebar.info("Created and designed by DT")

    option = st.sidebar.text_input('Enter a Stock Symbol', value='AAPL')
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
        ['Visualize', 'Recent Data', 'Technical Indicators', 'Predict'],  # 'Technical Indicators' option
        key='main_choice'
    )

    if option == 'Visualize':
        visualize_stock_data(data)  # Pass the data variable
    elif option == 'Recent Data':
        dataframe(data)  # Pass the data variable
    elif option == 'Technical Indicators':  # Call the function for displaying technical indicators
        tech_indicators(data)  # Pass the data variable
    else:
        predict(data)  # Pass the data variable


def visualize_stock_data(data):
    plt.figure(figsize=(10, 6))

    # Add technical indicators selection
    option = st.selectbox(
        'Choose a Technical Indicator to Visualize',
        ['Close', 'Bollinger Bands', 'MACD', 'RSI', 'SMA', 'EMA']
    )

    if option == 'Close':
        plt.plot(data.index, data['Close'])
        plt.title("Close Price")
        plt.xlabel("Date")
        plt.ylabel("Price")
    elif option == 'Bollinger Bands':
        bb_indicator = BollingerBands(data.Close)
        bb = data
        bb['bb_h'] = bb_indicator.bollinger_hband()
        bb['bb_l'] = bb_indicator.bollinger_lband()
        plt.plot(data.index, data['Close'], label='Close')
        plt.plot(data.index, bb['bb_h'], label='Upper Band')
        plt.plot(data.index, bb['bb_l'], label='Lower Band')
        plt.title("Bollinger Bands")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
    elif option == 'MACD':
        macd = MACD(data.Close).macd()
        plt.plot(data.index, macd, label='MACD')
        plt.title("MACD")
        plt.xlabel("Date")
        plt.ylabel("MACD")
        plt.legend()
    elif option == 'RSI':
        rsi = RSIIndicator(data.Close).rsi()
        plt.plot(data.index, rsi, label='RSI')
        plt.title("Relative Strength Index (RSI)")
        plt.xlabel("Date")
        plt.ylabel("RSI")
        plt.legend()
    elif option == 'SMA':
        sma = SMAIndicator(data.Close, window=14).sma_indicator()
        plt.plot(data.index, sma, label='SMA')
        plt.title("Simple Moving Average (SMA)")
        plt.xlabel("Date")
        plt.ylabel("SMA")
        plt.legend()
    elif option == 'EMA':
        ema = EMAIndicator(data.Close).ema_indicator()
        plt.plot(data.index, ema, label='EMA')
        plt.title("Exponential Moving Average (EMA)")
        plt.xlabel("Date")
        plt.ylabel("EMA")
        plt.legend()

    st.pyplot(plt)


def calculate_portfolio_performance(data):
    portfolio_performance = data.pivot(columns='Symbol', values='Close')
    return portfolio_performance

@st.cache_resource
def download_data(op, start_date, end_date):
    df = yf.download(op, start=start_date, end=end_date, progress=False)
    return df


def tech_indicators(data):
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


def dataframe(data):
    st.header('Recent Data')
    st.dataframe(data.tail(10))


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

    forecast_pred = model.predict(x_forecast)
    day = 1
    for i in forecast_pred:
        st.text(f'Day {day}: {i}')
        day += 1


if __name__ == '__main__':
    app_page()
