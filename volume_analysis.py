import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.dates import DayLocator, DateFormatter
import os
from finvizfinance.screener.overview import Overview
from finvizfinance.quote import finvizfinance
from transformers import pipeline


def analyze_sentiment_finbert(text):
    """Deciphers sentiment from text"""
    classifier = pipeline('sentiment-analysis', model='ProsusAI/finbert')
    result = classifier(text)[0]
    sentiment = result['label'].capitalize()  # e.g., 'Positive', 'Negative', 'Neutral'
    confidence = result['score']
    return sentiment, confidence

def news_sentiment(ticker): 
    """ Function fetches the news and decipher sentiment"""
    stock = finvizfinance(ticker)
    news = stock.ticker_news() 
    # Change date to Datetime
    news['Date'] = pd.to_datetime(news['Date'])
    today = datetime.now().date()
    yesterday = today - timedelta(days = 1)
    
    # Making date_only column
    news['Date_only'] = news['Date'].dt.date

    # Getting only recent news
    recent_news_df = news[(news['Date_only'] == today) | (news['Date_only'] == yesterday)]
    if recent_news_df.empty: # This is the correct way to check if the DataFrame is empty
        return 'No recent News'

    # Apply sentiment analysis 
    titles_sentiment = recent_news_df['Title'].apply(lambda x: analyze_sentiment_finbert(x))
    recent_news_df['Confidence'] = titles_sentiment.apply(lambda x: x[1])
    recent_news_df['Sentiment'] = titles_sentiment.apply(lambda x: x[0])

    # Selecting only positive and negative news 
    sentiment_news_df = recent_news_df[(recent_news_df['Sentiment'] == 'Positive') | (recent_news_df['Sentiment'] == 'Negative')]
    representative_article = sentiment_news_df.sort_values(
            by=['Confidence'], ascending=[False]).iloc[0]
    news_date = representative_article['Date']
    news_title = representative_article['Title']
    news_source = representative_article['Link']
    news_sentimemnt_last = representative_article['Sentiment']
    return news_date, news_title, news_source, news_sentimemnt_last


def stock_with_positive_change(tickers):
    """
    Identify stocks with a positive price change on the target date.
    Returns a list of tickers with positive price changes.
    """
    start_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    tickers_dic = []
    company_dic = [] 
    change_dic = []
    data = {}
    dates_dic = []
    volumes_dic = []
    prices_dic = []
    news_dic = []
    news_date = []
    news_title = [] 
    news_source = []
    news_sentimemnt = [] 
    for ticker in tickers:
        if not pd.notna(ticker):  # Skip NaN values
            print(f"Skipping invalid ticker: {ticker}")
            continue

        print(f"Processing ticker: {ticker}")     
        try:
            # Fetch historical data
            print(f"Loading data for {ticker} from {start_date} to {end_date}...")
            raw_data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
            dates = raw_data.index.strftime('%Y-%m-%d').tolist()


            df = pd.DataFrame(raw_data.values, columns=['Close', 'High', 'Low', 'Open', 'Volume'])
    
            df.insert(0, 'Date', dates)

            df['Date'] = pd.to_datetime(df['Date'])
            print(df)
            
            if df.empty or len(df) < 2:
                print(f"Not sufficient data for {ticker}.")
                continue

            # Ensure price is above $0.70
            if df['Close'].iloc[-1] < 0.70:
                print(f"Skipping {ticker} due to low price: {df['Close'].iloc[-1]}")
                continue

            # Check if volume is sufficient
            volumes = list(df['Volume'])
            if volumes[-1] < 1000000:
                continue
            
            # Get close prices
            close_prices = list(df['Close'])
            print(f"Close prices for {ticker}: {close_prices}")

            # Calculate average volume for the first 4 days
            avg_vol = df['Volume'][:4].mean()
            if len(volumes) >= 2 and pd.notna(volumes[-1]) and pd.notna(volumes[-2]) and pd.notna(close_prices[-1]) and pd.notna(close_prices[-2]):
                if (volumes[-1] > avg_vol) and (close_prices[-1] > close_prices[-2]):
                    print(f"Volume increased for {ticker}: {volumes[-1]} > {avg_vol}")
                    print(f"Price increased for {ticker}: {volumes[-1]} > {volumes[-2]}")
                    news_result = news_sentiment(ticker)
                    if news_result == 'No recent News':
                        print(f"No recent news for {ticker}. Skipping news sentiment data.")
                        continue
                    else:
                        one, two, three, four = news_result

                    news_date.append(one) 
                    news_title.append(two)
                    news_source.append(three)
                    news_sentimemnt.append(four)
                    dates_dic.append(df['Date'].iloc[-1])
                    tickers_dic.append(ticker)
                    company_dic.append(yf.Ticker(ticker).info.get('longName', 'Unknown'))
                    change_dic.append(close_prices[-1] - close_prices[-2])
                    volumes_dic.append(volumes[-1])
                    prices_dic.append(close_prices[-1])
                else:
                    print(f"{ticker} did not meet criteria.")
        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")
            continue
    # Put data in dictionary
    data = {
            'Date': dates_dic,
            'Ticker': tickers_dic,
            'Company': company_dic,
            'Price' : prices_dic,
            'Change': change_dic,
            'Volume': volumes_dic,
            'News_date' : news_date,
            'News_title' : news_title, 
            'News_source' : news_source, 
            'News_sentiment' : news_sentimemnt,
        }
    return pd.DataFrame(data)

def plot_volume(ticker):
    """
    Generate data for plotting trading volume and close price of a stock over a specified date range.
    Returns JSON-compatible data for Chart.js.
    """
    try:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=20)).strftime('%Y-%m-%d')

        # Fetch historical data
        print(f"Loading data for {ticker} from {start_date} to {end_date}...")
        raw_data = yf.download(ticker, start=start_date, end=end_date, interval='1d', progress=False)
        dates = raw_data.index.strftime('%Y-%m-%d').tolist()


        df_raw = pd.DataFrame(raw_data.values, columns=['Close', 'High', 'Low', 'Open', 'Volume'])
    
        df_raw.insert(0, 'Date', dates)

        df_raw['Date'] = pd.to_datetime(df_raw['Date'])

        if df_raw.empty:
            raise ValueError(f"No data found for ticker {ticker} between {start_date} and {end_date}.")

        # Use yfinance DataFrame directly
        df = df_raw[['Close', 'Volume']].copy()
        df['Date'] = pd.to_datetime(df_raw['Date'])

        # --- Potential problem area: Data types and NaNs ---
        # Ensure 'Close' and 'Volume' are numeric and handle missing values
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        # Drop rows where 'Close' or 'Volume' became NaN after conversion
        df = df.dropna(subset=['Close', 'Volume'])

        df = df.dropna(subset=['Close', 'Volume'])
        if len(df) < 2:
            print(f"Not enough data points for {ticker}: {len(df)} rows")
            return None

        # Create a complete date range for daily ticks
        full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        df = df.set_index('Date').reindex(full_date_range).reset_index()
        df['Close'] = df['Close'].interpolate(method='linear').fillna(method='ffill')
        df['Volume'] = df['Volume'].fillna(0)
        df['Date'] = df['index']

        # Plot the volume and close price with compressed layout
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True,
                                       gridspec_kw={'height_ratios': [2, 1]},  # Adjusted for compression
                                       figsize=(11, 5))  # Reduced size for compression

        ax1.plot(df['Date'], df['Close'], label='Close Price', color='#4b6cb7')  # Muted blue
        ax1.set_title(f'{ticker} Price and Volume', fontsize=12, fontweight='bold', color='#f8f9fa')
        ax1.set_ylabel('Price', color='#f8f9fa')
        ax1.legend(loc='upper left')
        ax1.grid(True, linestyle='--', alpha=0.7, color='#6c757d')
        ax1.yaxis.tick_right()
        ax1.yaxis.set_label_position("right")
        ax1.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))  # Fewer bins for compression
        ax1.tick_params(axis='y', colors='#f8f9fa')
        ax1.set_facecolor('#3c3f58')  # Lighter background for better contrast
        fig.set_facecolor('#1e1e3f')

        # Plot Volume
        ax2.bar(df['Date'], df['Volume'], label='Volume', color='#28a745', alpha=0.6, width=0.8)  # Muted green
        ax2.set_ylabel('Volume', color='#f8f9fa')
        ax2.grid(True, linestyle='--', alpha=0.7, color='#6c757d')
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4))  # Fewer bins for compression
        ax2.tick_params(axis='y', colors='#f8f9fa')
        ax2.tick_params(axis='x', colors='#f8f9fa')
        ax2.set_facecolor('#3c3f58')  # Lighter background for better contrast

        # Set daily date ticks
        ax2.xaxis.set_major_locator(DayLocator())
        ax2.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate(rotation=45)
        plt.xlabel('Date', color='#f8f9fa')
        plt.tight_layout()
        plt.subplots_adjust(right=0.92, top=0.9, bottom=0.15, hspace=0.2)  # Tightened layout

        # Save the plot
        if not os.path.exists('static'):
            os.makedirs('static')
        plot_path = os.path.join('static', f'volume_{ticker}.png')
        plt.savefig(plot_path, bbox_inches='tight', dpi=100, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"Volume chart for {ticker} saved successfully at {plot_path}.")
        return plot_path
    except Exception as e:
        print(f"Error generating chart for {ticker}: {str(e)}")
        return None
    
def plot_volume_1(ticker):
    """
    Plot the trading volume and close price of a stock over a specified date range.
    """
    try:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=20)).strftime('%Y-%m-%d')

        # Fetch historical data
        print(f"Loading data for {ticker} from {start_date} to {end_date}...")
        raw_data = yf.download(ticker, start=start_date, end=end_date, interval='1d', progress=False)
        dates = raw_data.index.strftime('%Y-%m-%d').tolist()


        df_raw = pd.DataFrame(raw_data.values, columns=['Close', 'High', 'Low', 'Open', 'Volume'])
    
        df_raw.insert(0, 'Date', dates)

        df_raw['Date'] = pd.to_datetime(df_raw['Date'])

        if df_raw.empty:
            raise ValueError(f"No data found for ticker {ticker} between {start_date} and {end_date}.")

        # Use yfinance DataFrame directly
        df = df_raw[['Close', 'Volume']].copy()
        df['Date'] = pd.to_datetime(df_raw['Date'])

        # --- Potential problem area: Data types and NaNs ---
        # Ensure 'Close' and 'Volume' are numeric and handle missing values
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        # Drop rows where 'Close' or 'Volume' became NaN after conversion
        df = df.dropna(subset=['Close', 'Volume'])

        # --- New check: Ensure enough data points remain after cleaning ---
        if len(df) < 2: # At least 2 points for a meaningful line or bar plot comparison
            raise ValueError(f"Not enough clean data points ({len(df)} rows) for {ticker} to plot after cleaning.")

        # Plot the volume and close price
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True,
                                         gridspec_kw={'height_ratios': [3, 1]},
                                         figsize=(14, 9))
        
        ax1.plot(df['Date'], df['Close'], label='Close Price', color='blue')
        ax1.set_title(f'{ticker} Price and Volume', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price')
        ax1.legend(loc='upper left')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.yaxis.tick_right()
        ax1.yaxis.set_label_position("right")
        ax1.yaxis.set_major_locator(mticker.MaxNLocator(nbins=10))

        # Plot Volume
        ax2.bar(df['Date'], df['Volume'], label='Volume', color='green', alpha=0.6, width=0.8)
        ax2.set_ylabel('Volume')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))

        # Format x-axis
        fig.autofmt_xdate(rotation=45)
        plt.xlabel('Date')
        plt.tight_layout()
        plt.subplots_adjust(right=0.92)

        # Save the plot
        if not os.path.exists('static'):
            os.makedirs('static')
        plot_path = os.path.join('static', f'volume_{ticker}.png')
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"Volume chart for {ticker} saved successfully at {plot_path}.")
        return plot_path

    except ValueError as ve: # Catch explicit ValueErrors we raise
        print(f"Error plotting volume for {ticker}: {str(ve)}")
        return None
    except Exception as e: # Catch any other unexpected errors
        print(f"An unexpected error occurred while plotting volume for {ticker}: {str(e)}")
        return None

def color_sentiment(row):
    ''' Colors the entire row based on the 'News_sentiment' column. '''
    color = ''
    # Safely get the sentiment value, default to empty string if not present or None
    # Use .get() to avoid KeyError if 'News_sentiment' is missing (though it should be fixed)
    sentiment_val = str(row.get('News_sentiment', '')).lower()

    if sentiment_val == 'positive':
        color = 'green'
    elif sentiment_val == 'negative':
        color = 'red'
    # For 'N/A', 'neutral', or other, 'color' remains empty string, so no styling is applied

    # Return a Series of CSS style strings for each column in the row
    # This correctly applies the same style to all cells in the row.
    return pd.Series([f'color: {color}'] * len(row), index=row.index)



#print(news_sentiment('AAPL'))