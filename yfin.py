import yfinance as yf


# Function to get stock information from Yahoo Finance
def get_stock_info(ticker):
    # Create a Ticker object for the specified stock
    stock = yf.Ticker(ticker)

    # Get information about the stock
    info = stock.info

    return info


# Get information about the stock AAPL (Apple Inc.)
stock_info = get_stock_info("AAPL")
print(stock_info)