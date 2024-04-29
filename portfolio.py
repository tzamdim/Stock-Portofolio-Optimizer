import streamlit as st
import pandas as pd

# Main application
def portfolio_management_page():
    st.title("Portfolio Management")

    def portfolio_management_page():
        # Initialize the portfolio if it doesn't exist
        if 'portfolio' not in st.session_state:
            st.session_state['portfolio'] = 'value'

        # Session State also supports the attribute based syntax
        if 'portfolio' not in st.session_state:
            st.session_state.key = 'value'

        # Rest of the code for buy and sell functionality goes here

    # Display total value and money invested
    total_value = (st.session_state['portfolio']['Quantity'] * st.session_state['portfolio']['Price']).sum()
    money_invested = (st.session_state['portfolio']['Quantity'] * st.session_state['portfolio']['Price']).sum()
    st.markdown(f"**Total Portfolio Value:** ${total_value:.2f}")
    st.markdown(f"**Money Invested:** ${money_invested:.2f}")

    # Display profits or losses
    if total_value > money_invested:
        st.markdown(f"**Profits:** +${total_value - money_invested:.2f}", unsafe_allow_html=True)
    elif total_value < money_invested:
        st.markdown(f"**Losses:** -${money_invested - total_value:.2f}", unsafe_allow_html=True)
    else:
        st.markdown("**No Profits or Losses**")

    # Display portfolio table
    st.write("## Portfolio")
    st.table(st.session_state['portfolio'])

    # Buy stocks
    st.write("## Buy Stocks")
    buy_ticker = st.text_input("Enter Stock Ticker (e.g., AAPL)")
    buy_quantity = st.number_input("Enter Quantity", min_value=1, step=1)
    if st.button("Buy"):
        st.session_state['portfolio'] = add_stock_to_portfolio(st.session_state['portfolio'], buy_ticker.upper(), buy_quantity)

    # Sell stocks
    st.write("## Sell Stocks")
    if not st.session_state['portfolio'].empty:
        sell_index = st.selectbox("Select Stock to Sell", st.session_state['portfolio']['Stock'])
        sell_quantity = st.number_input("Enter Quantity to Sell", min_value=1, max_value=st.session_state['portfolio'].loc[sell_index, 'Quantity'], step=1)
        if st.button("Sell"):
            st.session_state['portfolio'] = sell_stock_from_portfolio(st.session_state['portfolio'], sell_index, sell_quantity)

if __name__ == "__main__":
    portfolio_management_page()
