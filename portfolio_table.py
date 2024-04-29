import streamlit as st
import pandas as pd
import yfinance as yf
import sqlite3
from datetime import datetime, timedelta


class Node:
    def __init__(self, transaction_id, ticker, stock_data, quantity, price):
        self.transaction_id = transaction_id
        self.ticker = ticker
        self.stock_data = stock_data
        self.quantity = quantity
        self.price = price
        self.next = None


class TransactionLinkedList:
    def __init__(self):
        self.head = None

    def append(self, transaction_id, ticker, stock_data, quantity, price):
        new_node = Node(transaction_id, ticker, stock_data, quantity, price)
        if not self.head:
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node

    def remove(self, transaction_id):
        current = self.head
        if current and current.transaction_id == transaction_id:
            self.head = current.next
            return
        prev = None
        while current and current.transaction_id != transaction_id:
            prev = current
            current = current.next
        if current:
            prev.next = current.next

    def update(self, transaction_id, quantity):
        current = self.head
        while current and current.transaction_id != transaction_id:
            current = current.next
        if current:
            current.quantity = quantity

    def retrieve_transactions(self):
        transactions = []
        current = self.head
        while current:
            transactions.append({
                'transaction_id': current.transaction_id,
                'ticker': current.ticker,
                'Stock': current.stock_data['longName'],
                'Sector': current.stock_data['sector'],
                'Price': current.price,
                'Quantity': current.quantity
            })
            current = current.next
        return transactions

    def calculate_portfolio_value(self):
        total_value = 0
        current = self.head
        while current:
            total_value += current.price * current.quantity
            current = current.next
        return total_value


# Initialize the linked list
transaction_list = TransactionLinkedList()

# Function to insert a new transaction into the linked list
def insert_transaction(ticker, stock_data, quantity, price):
    global transaction_list
    transaction_id = len(transaction_list.retrieve_transactions()) + 1
    transaction_list.append(transaction_id, ticker, stock_data, quantity, price)

# Function to remove a transaction from the linked list
def remove_transaction(transaction_id):
    global transaction_list
    transaction_list.remove(transaction_id)

# Function to update a transaction in the linked list
def update_transaction(transaction_id, quantity):
    global transaction_list
    transaction_list.update(transaction_id, quantity)

# Function to connect to the SQLite database and retrieve transaction data
@st.cache_data
def retrieve_transactions():
    global transaction_list
    return transaction_list.retrieve_transactions()

# Function to get current or last close price for a stock
@st.cache_data
def get_current_or_last_close_price(ticker):
    # Get current date and time
    now = datetime.now()
    # Get today's date in YYYY-MM-DD format
    today_date = now.strftime("%Y-%m-%d")

    # Check if today is Sunday
    if now.strftime("%A") == "Sunday":
        # If today is Sunday, fetch historical data for the previous trading day (Friday)
        previous_trading_day = now - timedelta(days=2 if now.strftime("%A") == "Monday" else 1)
        previous_trading_day_date = previous_trading_day.strftime("%Y-%m-%d")
        data = yf.download(ticker, start=previous_trading_day_date, end=previous_trading_day_date)
    else:
        # Fetch historical data for the past day (including today)
        data = yf.download(ticker, start=today_date, end=today_date)

    if len(data) > 0:
        # If data is available for today or the previous trading day, return the close price
        return data['Close'].iloc[-1]
    else:
        # If data is not available for today or the previous trading day, fetch historical data for the previous day
        data = yf.download(ticker, period="2d")
        # Return the close price from the last available trading day
        return data['Close'].iloc[-1]


# Streamlit app page
def app_page():
    st.markdown("<h1 style='text-align: center; color: blue;'>Stock Portfolio Manager</h1>", unsafe_allow_html=True)
    st.sidebar.info('Welcome to the Stock Portfolio Manager. Choose your options below')

    # Retrieve transaction data
    transactions_df = retrieve_transactions()

    if transactions_df:
        # Display transaction data table
        st.dataframe(transactions_df)

    # Display current investment
    current_investment = transaction_list.calculate_portfolio_value()
    st.write(f'Current Investment: ${current_investment:.2f}')

    # Input fields for buying stocks
    st.sidebar.subheader('Buy Stocks from S&P 500')
    ticker = st.sidebar.text_input('Enter Ticker')
    quantity = st.sidebar.number_input('Enter Quantity', min_value=1, step=1)

    if st.sidebar.button('Buy'):
        stock_data = yf.Ticker(ticker).info
        price = get_current_or_last_close_price(ticker)
        insert_transaction(ticker, stock_data, quantity, price)
        st.sidebar.success('Transaction added successfully!')

    st.sidebar.subheader('Sell Stocks')
    transaction_id = st.sidebar.number_input('Enter Transaction ID', value=0, step=1)
    sell_quantity = st.sidebar.number_input('Enter Quantity to Sell', min_value=1, step=1)

    if st.sidebar.button('Sell'):
        if isinstance(transaction_id, int):
            if transaction_id in transactions_df['transaction_id'].values:
                current_quantity = \
                transactions_df.loc[transactions_df['transaction_id'] == transaction_id, 'Quantity'].values[0]
                if sell_quantity >= current_quantity:
                    remove_transaction(transaction_id)
                else:
                    update_transaction(transaction_id, current_quantity - sell_quantity)
                st.sidebar.success('Transaction completed successfully!')
            else:
                st.sidebar.error('Invalid Transaction ID')
        else:
            st.sidebar.error('Transaction ID must be an integer')

    # Calculate net profits
    net_profits = current_investment - transaction_list.calculate_portfolio_value()
    net_profits_color = 'green' if net_profits >= 0 else 'red'
    st.write(f'Net Profits: ${net_profits:.2f}', unsafe_allow_html=True, key='net_profits')

    # Refresh the page
    st.rerun()


if __name__ == '__main__':
    app_page()
