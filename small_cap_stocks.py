import pandas as pd
import yfinance as yf
from finvizfinance.screener.overview import Overview


def get_small_cap_stocks():
    """
    Fetch a list of small cap stocks.
    """
    try:
        screener = Overview()
        filters = {'Price': 'Under $20', 'Market Cap.': 'Small ($300mln to $2bln)'}
        screener.set_filter(filters_dict=filters)
        df_stocks = screener.screener_view()
        df_stocks.to_csv('small_cap_stocks.csv', index=False)  # Display the first few rows of the DataFrame for debugging
        return df_stocks
    except Exception as e:
        print(f"Error fetching stocks from Finviz: {str(e)}")
        return []
    
def get_micro_cap_stocks():
    """
    Fetch a list of microcap stocks.
    """
    try:
        screener = Overview()
        filters = {'Price': 'Under $20', 'Market Cap.': 'Micro ($50mln to $300mln)'}
        screener.set_filter(filters_dict=filters)
        df_stocks = screener.screener_view()
        df_stocks.to_csv('micro_cap_stocks.csv', index=False)  # Save to CSV for further analysis
        return df_stocks
    except Exception as e:
        print(f"Error fetching stocks from Finviz: {str(e)}")
        return []
    
