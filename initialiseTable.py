import streamlit as st
import sqlite3
import time


# Function to authenticate password
def authenticate_password(password):
    return password == 'DT'


# Function to initialize portfolio
def initialize_portfolio():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()

    # Drop existing stock data table if it exists
    cursor.execute("DROP TABLE IF EXISTS stock_data")

    # Create new stock data table
    cursor.execute('''CREATE TABLE stock_data (
                        transaction_id INTEGER PRIMARY KEY,
                        ticker TEXT,
                        Stock TEXT,
                        Sector TEXT,
                        Price REAL,
                        Quantity INTEGER)''')

    conn.commit()
    conn.close()


# Streamlit initialization page
def initialize_page():
    st.markdown("<h1 style='text-align: center; color: blue;'>Portfolio Initialization</h1>", unsafe_allow_html=True)
    st.sidebar.header("Authentication")
    password = st.sidebar.text_input("Enter Password", type="password")

    if authenticate_password(password):
        st.sidebar.success("Authenticated")

        if st.sidebar.button("Initialize Portfolio"):
            initialize_portfolio()
            st.success("Portfolio initialized successfully!")

            # Redirect to the homepage after 3 seconds
            st.experimental_set_query_params()
            time.sleep(3)  # Wait for 3 seconds
            st.experimental_set_query_params(page="home")  # Redirect to the homepage
    else:
        st.sidebar.error("Incorrect Password")


if __name__ == "__main__":
    initialize_page()
