import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
import uuid
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np


# Initialize portfolio with example data if not already in session state
def initialize_portfolio():
    if 'portfolio' not in st.session_state:
        st.session_state['portfolio'] = pd.DataFrame({
            'TransactionID': [str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())],
            'Stock': ['AAPL', 'MSFT', 'GOOGL'],
            'Quantity': [10, 15, 5],
            'Price': [150.0, 200.0, 2500.0],
            'Date': [datetime.date(2023, 1, 1), datetime.date(2023, 1, 15), datetime.date(2023, 1, 20)],
            'Latest Price': [150.0, 200.0, 2500.0],
            'Is Up': ['Up', 'Up', 'Up'],
        })


# Function to plot cumulative portfolio data
def plot_cumulative_portfolio_data(start_date, end_date):
    plt.figure(figsize=(10, 6))
    cumulative_data = pd.DataFrame()

    for ticker in st.session_state['portfolio']['Stock'].unique():
        data = yf.download(ticker, start=start_date, end=end_date)
        if not data.empty:
            quantity = st.session_state['portfolio'].loc[st.session_state['portfolio']['Stock'] == ticker, 'Quantity'].sum()
            data['Cumulative'] = data['Close'] * quantity
            cumulative_data = pd.concat([cumulative_data, data['Cumulative']], axis=1)

    if not cumulative_data.empty:
        cumulative_data['Total'] = cumulative_data.sum(axis=1)
        plt.plot(cumulative_data.index, cumulative_data['Total'], label='Total Portfolio Value', linewidth=2, linestyle='--')
        plt.title("Cumulative Portfolio Value Over Time")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        st.pyplot(plt)
    else:
        st.error("No data available for the portfolio in the selected date range.")

# Function to plot individual stock data
def plot_individual_stock_data(tickers, start_date, end_date):
    plt.figure(figsize=(10, 6))

    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date)
        if not data.empty:
            plt.plot(data.index, data['Close'], label=f"{ticker} Close")
        else:
            st.error(f"No data available for {ticker} in the selected date range.")

    plt.title("Individual Stock Performance")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(plt)


# Function to perform stock price prediction
def perform_prediction(stock_symbol, model_choice, start_date, end_date, prediction_days):
    # Download historical data for the selected stock
    data = yf.download(stock_symbol, start=start_date, end=end_date, progress=False)
    df = data[['Close']]
    df['preds'] = df['Close'].shift(-prediction_days)
    df.dropna(inplace=True)

    # Prepare data for prediction
    scaler = StandardScaler()
    X = scaler.fit_transform(df[['Close']])
    y = df['preds']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Select model
    if model_choice == 'Linear Regression':
        model = LinearRegression()
    elif model_choice == 'Random Forest':
        model = RandomForestRegressor()
    else:
        st.error("Model not supported")
        return

    # Train and predict
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    # Display metrics
    st.sidebar.write(f"R2 Score: {r2:.2f}")
    st.sidebar.write(f"Mean Absolute Error: {mae:.2f}")

    # Predict future prices
    last_data = scaler.transform(df[['Close']].tail(prediction_days))
    future_predictions = model.predict(last_data)
    future_dates = [end_date + datetime.timedelta(days=i+1) for i in range(prediction_days)]
    predicted_prices = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_predictions})

    # Display predicted prices
    st.write(f"Predicted Prices for {stock_symbol}")
    st.table(predicted_prices)


# Function to fetch the latest closing price for a stock
def get_latest_price(ticker):
    try:
        stock_data = yf.Ticker(ticker).history(period='5d')
        if stock_data.empty:
            return None
        return stock_data['Close'].dropna().iloc[-1]
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

# Function to add stocks to the portfolio
def add_stock(ticker, quantity):
    try:
        stock_data = yf.Ticker(ticker).history(period='1mo')
        if stock_data.empty:
            st.error(f"No data available for {ticker}. Please check the ticker symbol.")
            return
        price = stock_data['Close'].dropna().iloc[-1]
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return

    transaction_id = str(uuid.uuid4())
    date = datetime.date.today()
    new_stock = pd.DataFrame({'TransactionID': [transaction_id], 'Stock': [ticker], 'Quantity': [quantity], 'Price': [price], 'Date': [date]})
    st.session_state['portfolio'] = pd.concat([st.session_state['portfolio'], new_stock], ignore_index=True)

# Function to sell stocks from the portfolio
def sell_stock(transaction_id, quantity):
    if transaction_id not in st.session_state['portfolio']['TransactionID'].values:
        st.error('Transaction ID not found in portfolio.')
        return

    portfolio = st.session_state['portfolio']
    transaction = portfolio[portfolio['TransactionID'] == transaction_id].iloc[0]

    if quantity >= transaction['Quantity']:
        # Remove the transaction if all or more shares are sold
        st.session_state['portfolio'] = portfolio[portfolio['TransactionID'] != transaction_id]
    else:
        # Reduce the quantity of the stock
        portfolio.loc[portfolio['TransactionID'] == transaction_id, 'Quantity'] -= quantity

# Function to display each row with colored Up/Down indicator
def display_row(row):
    # Define color based on the Up/Down status
    color = "green" if row['Is Up'] == 'Up' else "red"
    # Display each column with markdown
    cols = st.columns(len(row))
    for i, col in enumerate(cols):
        value = row.iloc[i]
        # Apply color only to the 'Is Up' column
        if i == len(row) - 1:
            col.markdown(f"<span style='color: {color};'>{value}</span>", unsafe_allow_html=True)
        else:
            col.write(value)

# Function to create an HTML table with colored "Is Up" cells
def create_html_table(df):
    # Start the HTML table
    html = "<table style='width:100%;border-collapse: collapse;'>"

    # Add header row
    html += "<tr>"
    for col in df.columns:
        html += f"<th style='border: 1px solid black;padding: 8px;'>{col}</th>"
    html += "</tr>"

    # Add data rows
    for _, row in df.iterrows():
        html += "<tr>"
        for i, value in enumerate(row):
            # Apply color styling to the 'Is Up' column
            if i == len(row) - 1:
                color = "green" if value == 'Up' else "red"
                html += f"<td style='border: 1px solid black;padding: 8px;color: {color};'>{value}</td>"
            else:
                html += f"<td style='border: 1px solid black;padding: 8px;'>{value}</td>"
        html += "</tr>"

    # Close the table
    html += "</table>"
    return html

# Function to plot stock data
def plot_stock_data(tickers, start_date, end_date):
    plt.figure(figsize=(10, 6))
    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date)
        plt.plot(data.index, data['Close'], label=ticker)
        # Simple moving average as a prediction metric
        plt.plot(data.index, data['Close'].rolling(window=20).mean(), label=f"{ticker} 20-Day SMA")
    plt.title("Stock Performance")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(plt)

# Main application
def test3_page():
    # Initialize the portfolio at the beginning of the script
    initialize_portfolio()
    st.title("Simple Stock Portfolio")

    # Calculate and display overall total
    initial_investment = 100000
    current_value = (st.session_state['portfolio']['Quantity'] * st.session_state['portfolio']['Price']).sum()
    st.metric(label="Portfolio Value", value=f"${current_value:,.2f}",
                  delta=f"${current_value - initial_investment:,.2f}")


    # Update portfolio with the latest price information
    for i, row in st.session_state['portfolio'].iterrows():
        latest_price = get_latest_price(row['Stock'])
        if latest_price is not None:
            st.session_state['portfolio'].at[i, 'Latest Price'] = latest_price
            st.session_state['portfolio'].at[i, 'Is Up'] = 'Up' if latest_price > row['Price'] else 'Down'

    # Add stock to portfolio
    st.sidebar.header("Add Stock to Portfolio")
    ticker = st.sidebar.text_input("Stock Ticker (e.g., AAPL)")
    quantity = st.sidebar.number_input("Quantity", min_value=1, step=1)
    if st.sidebar.button("Add Stock"):
        add_stock(ticker.upper(), quantity)

    # Sell stock from portfolio
    st.sidebar.header("Sell Stock from Portfolio")
    transaction_id = st.sidebar.text_input("Transaction ID")
    sell_quantity = st.sidebar.number_input("Sell Quantity", min_value=1, step=1)
    if st.sidebar.button("Sell Stock"):
        sell_stock(transaction_id, sell_quantity)

    # Prediction sidebar
    st.sidebar.header("Stock Price Prediction")
    selected_stock = st.sidebar.selectbox("Select a stock for prediction",
                                          st.session_state['portfolio']['Stock'].unique())
    model_choice = st.sidebar.selectbox("Select Prediction Model", ["Linear Regression", "Random Forest"])
    prediction_start_date = st.sidebar.date_input("Prediction Start Date",
                                                  datetime.date.today() - datetime.timedelta(days=365))
    prediction_end_date = st.sidebar.date_input("Prediction End Date", datetime.date.today())
    prediction_days = st.sidebar.number_input("Days to Predict Into Future", 1, 30, 5)
    if st.sidebar.button("Predict"):
        perform_prediction(selected_stock, model_choice, prediction_start_date, prediction_end_date, prediction_days)

    # Display portfolio with colored Up/Down indicators
    st.header("Your Portfolio")
    html_table = create_html_table(st.session_state['portfolio'])
    st.markdown(html_table, unsafe_allow_html=True)

    # Display total shares and monetary value for each stock
    st.header("Portfolio Summary")
    summary = st.session_state['portfolio'].groupby('Stock').agg(Total_Quantity=('Quantity', 'sum'), Total_Value=('Quantity', 'sum')).reset_index()
    summary['Total_Value'] *= st.session_state['portfolio'].groupby('Stock')['Price'].mean().values
    st.table(summary)

    # Stock selection for visualization
    st.header("Stock Visualization")
    all_stocks = st.session_state['portfolio']['Stock'].unique()
    selected_stocks = st.multiselect("Select stocks to visualize", all_stocks, default=all_stocks[0])
    start_date = st.date_input("Start Date", datetime.date.today() - datetime.timedelta(days=365))
    end_date = st.date_input("End Date", datetime.date.today())

    if st.button("Show Portfolio Graph"):
        plot_cumulative_portfolio_data(start_date, end_date)

    if st.button("Show Individual Stock Graph"):
        plot_individual_stock_data(selected_stocks, start_date, end_date)


if __name__ == "__main__":
    test3_page()
