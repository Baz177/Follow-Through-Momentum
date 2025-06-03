from finvizfinance.quote import finvizfinance
from finvizfinance.screener.overview import Overview

# Get fundamental data for a single stock (e.g., TSLA)
stock = finvizfinance('TSLA')
fundamentals = stock.ticker_fundament()
print(fundamentals)  # Returns a dictionary with metrics like P/E, EPS, etc.

# Screen stocks (e.g., S&P 500 stocks with high dividend yield)
foverview = Overview()
filters_dict = {'Index': 'S&P 500', 'Dividend Yield': 'High (>5%)'}
foverview.set_filter(filters_dict=filters_dict)
df = foverview.screener_view()
print(df)  # Returns a DataFrame with filtered

stock = finvizfinance('BULL')
news = stock.ticker_news()
print(news)  # Returns a list of news articles related to the stock