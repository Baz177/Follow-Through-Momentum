# User Guide and Documentation for Follow Through Momentum App

## Overview
Follow Through Momentum is a web application designed to identify small and micro-cap stocks exhibiting a recent surge in trading volume combined with a positive change in closing price. It integrates news sentiment analysis to provide insights into market momentum, helping users spot potential early movements in these stocks. This guide covers setup, usage, and technical details for end users and developers.

## 1. Purpose
This application targets traders and investors interested in momentum-based strategies, focusing on: Small to micro-cap stocks (typically lower market capitalization). Stocks with above-average trading volume and positive price changes. Recent news sentiment to contextualize market activity.

## 2. Features
Dashboard: Displays the strategy name, description, and a button to run the analysis. Stock Analysis: Identifies stocks with positive price changes and high volume from a list of small and micro-cap stocks. News Sentiment: Retrieves and displays news titles, sources, dates, and sentiment (color-coded for quick interpretation). Visualizations: Generates volume charts for qualifying stocks, saved as images in the static folder. Interactive Results: Click "Run Strategy" to view a table of results and corresponding charts.

## 3. Prerequisites

### For Users
- A modern web browser (e.g., Chrome, Firefox, Safari).
- Internet access to load the app and fetch stock/news data.

### For Developers/Administrators
- Software:
  - Python 3.9 or higher.
  - Git for version control.
- Dependencies (listed in requirements.txt):
  - pandas: Data manipulation.
  - yfinance: Stock data retrieval.
  - dataframe_image: Export styled tables to images.
  - flask: Web framework.
  - waitress: Production WSGI server.
  - matplotlib: Plotting (used in plot_volume).
  - Custom modules: volume_analysis.py, small_cap_stocks.py.
- Environment:
  - A hosting platform like Render for deployment.
  - Write access to a static folder for storing charts and table images.

## 4. Installation

### Local Setup
1. Clone the Repository
git clone <your-repo-url>
cd <your-repo-name>

Replace <your-repo-url> with your Git repository URL (e.g., from GitHub).

2. Install Dependencies
pip install -r requirements.txt

Ensure requirements.txt includes:
- pandas
- yfinance
- dataframe_image
- flask
- waitress
- matplotlib

3. Run the App Locally

python server.py
- The app starts on http://localhost:5000 by default.
- Access it in your browser to test the dashboard and functionality.

## 5. Usage
### Accessing the Dashboard
1. Open your browser and navigate to the app’s URL (e.g., http://localhost:5000 locally or Render’s URL).
2. You’ll see:
    - Strategy Name: "Follow Through Momentum"
    - Strategy Description: Explains the focus on volume surges, price changes, and news sentiment.
    - A “Run Strategy” button and placeholder text: “Click 'Run Strategy' to see results.”

### Running the Strategy
  - Click the “Run Strategy” button.
  - The app:
    - Fetches small and micro-cap stock data.
    - Identifies stocks with positive price changes and high volume.
    - Retrieves recent news and applies sentiment analysis.
    - Generates a styled table and volume charts.
  - Results display:
      - Table: Shows Ticker, Company, News Date, News Sentiment, News Title (clickable links), and News Source.
      - Sentiment Colors: Green for positive, red for negative, neutral otherwise (defined in volume_analysis.py).
      - Charts: Volume plots for each qualifying stock, displayed below the table.

### Interpreting Results
- Table Columns:
    - Ticker: Stock symbol (e.g., AAPL).
    - Company: Company name.
    - News Date: Date of the news article.
    - News Sentiment: Sentiment score or label (e.g., Positive, Negative).
    - News Title: Title of the article, hyperlinked to the source if available.
    - News Source: Origin of the news (e.g., Yahoo Finance).
- Charts: Visuals in the static folder show volume trends for each ticker, helping you assess momentum.

### Notes
- Results depend on data availability from yfinance and news sources.
- If no stocks meet the criteria, you’ll see: “No stocks with positive change found.”
- Errors (e.g., data fetch failures) are displayed if processing fails.

## 5. Limitations
- Depends on external data (e.g., yfinance), which may be unavailable or delayed.
- Processing time for large stock lists or news sentiment can slow the /run_strategy response.
- Static folder cleanup may fail if permissions are restricted on the host.
