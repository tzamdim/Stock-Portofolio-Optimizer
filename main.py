import streamlit as st
import stockManager  # Assuming stockManager.py is in the same directory
import stocklookup
import portfolio_table
import initialiseTable

def home_page():
    st.title('Welcome to the Stock Analysis Simulator by DT')
    st.write('This app provides tools for stock price prediction and portfolio management.')
    st.markdown('''
    - Use the *Stock Manager* page to manage your stock portfolio.
    - Use the *App Page* to view stock predictions and technical indicators.
    ''')
    st.image("stock-trading image.jpeg", caption="Stock Market Trading", use_column_width=True)

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Home", "Stock Manager", "Stock Lookup", "Portfolio Table", "Initialise Table"])

    if page == "Home":
        home_page()
    elif page == "Stock Lookup":
        stocklookup.app_page()
    elif page == "Stock Manager":
        stockManager.app_page()
    elif page == "Portfolio Table":
        portfolio_table.app_page()
    elif page == "Initialise Table":
        initialiseTable.initialize_page()

if __name__ == "__main__":
    main()
