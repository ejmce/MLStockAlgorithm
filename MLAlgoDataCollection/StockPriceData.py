import yfinance as yf

def get_stock_prices(tickers, period='1y'):
    stock_data = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        info = stock.info
        current_price = info.get('currentPrice', None)  # Get the current price, default to None if not available
        if current_price is None:  # Fallback to regular price
            current_price = info.get('Close', None)
        stock_data[ticker] = current_price
    return stock_data

# Example usage
# tickers = politician_stock_df['Ticker'].unique()
# stock_prices = get_stock_prices(tickers)
