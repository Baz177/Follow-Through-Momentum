import pandas as pd
from volume_analysis import stock_with_positive_change, color_sentiment, plot_volume
from small_cap_stocks import get_small_cap_stocks, get_micro_cap_stocks
from matplotlib import style
import dataframe_image as dfi
import yfinance as yf
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify
import os
from waitress import serve

# --- Flask Application Setup ---
app = Flask(__name__)

# --- Trading Strategy Information ---
STRATEGY_NAME = "Follow Through Momentum"
STRATEGY_DESCRIPTION = """
This strategy identifies stocks that have experienced a recent surge in trading volume (above their average)
combined with a positive change in their closing price. For these identified stocks, relevant news sentiment
is then retrieved and displayed, allowing for a quick overview of market momentum backed by recent news events.
The strategy focuses on small to micro-cap stocks, aiming to capture early movements.
"""

def clean_static_folder():
    static_path = os.path.join(os.path.dirname(__file__), 'static')
    for filename in os.listdir(static_path):
        file_path = os.path.join(static_path, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {str(e)}")

def process_data():
    try:
        print("--- Starting Data Processing ---")
        small_df = get_small_cap_stocks()
        if small_df.empty:
            print("No small cap stocks found.")
            return pd.DataFrame().style, [], "No small cap stocks found."

        micro_df = get_micro_cap_stocks()
        if micro_df.empty:
            print("No micro cap stocks found.")
            return pd.DataFrame().style, [], "No micro cap stocks found."

        df = pd.concat([small_df, micro_df], ignore_index=True)
        df = df.drop_duplicates(subset='Ticker', keep='first')
        df = df.dropna(subset=['Ticker', 'Company'])
        print(f"Combined DataFrame contains {len(df)} unique stocks.")

        tickers = df['Ticker'].to_list()
        positive_df = stock_with_positive_change(tickers)
        print("Positive stocks DataFrame:", positive_df)
        print(f"Positive stocks columns: {positive_df.columns.tolist()}")

        styled_df = pd.DataFrame().style
        ticker_plot_paths = []
        error = None

        if not positive_df.empty:
            expected_cols = ['Date', 'Ticker', 'Company', 'Price', 'Change', 'Volume',
                            'News_date', 'News_title', 'News_source', 'News_sentiment', 'News_url']
            for col in expected_cols:
                if col not in positive_df.columns:
                    positive_df[col] = None

            positive_df['News_title'] = positive_df.apply(
                lambda row: f'<a href="{row["News_url"]}" class="news-link">{row["News_title"]}</a>'
                if row['News_url'] else row['News_title'], axis=1)

            styled_df = positive_df[['Ticker', 'Company', 'News_date', 'News_sentiment', 'News_title', 'News_source']].style.apply(color_sentiment, axis=1)
            output_image_path = os.path.join('static', 'styled_df.png')
            os.makedirs('static', exist_ok=True)
            dfi.export(styled_df, output_image_path, max_rows=-1)

            ticker_plot_paths = []
            for ticker in positive_df['Ticker']:
                plot_path = plot_volume(ticker)
                if plot_path and os.path.exists(plot_path):
                    ticker_plot_paths.append({'ticker': ticker, 'plot_path': plot_path})
                else:
                    print(f"No chart generated for {ticker}")
        else:
            print("positive_df is empty.")
            error = "No stocks with positive change found."

        print(f"Tickers with charts: {len(ticker_plot_paths)}")
        print("--- Data Processing Complete ---")
        return styled_df, ticker_plot_paths, error
    except Exception as e:
        print(f"Error in process_data: {str(e)}")
        return pd.DataFrame().style, [], f"Error processing data: {str(e)}"

@app.route('/')
def dashboard():
    return render_template('index.html',
                         strategy_name=STRATEGY_NAME,
                         strategy_description=STRATEGY_DESCRIPTION,
                         table_html="<p>Click 'Run Strategy' to see results.</p>",
                         ticker_plot_paths=[])

@app.route('/run_strategy', methods=['POST'])
def run_strategy():
    styled_df, ticker_plot_paths, error = process_data()
    if error:
        return jsonify({'error': error})
    response = jsonify({
        'table_html': styled_df.to_html(classes='data-table'),
        'ticker_plot_paths': ticker_plot_paths
    })
    return response

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting server on port {port}...")
    clean_static_folder()  # Clean static folder before starting server
    serve(app, host='0.0.0.0', port=port)