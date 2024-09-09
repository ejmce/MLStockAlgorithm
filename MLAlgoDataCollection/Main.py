import yfinance as yf
from flask import Flask, jsonify, request
from yfinance import tickers
from NewsCatalysts import get_recent_news
from StockPriceData import get_stock_prices
from TickerData import get_combined_tickers
import pandas as pd
from ModelTraining import train_model as train
from RunModel import predict_tomorrow_stock_price as prediction
import numpy as np
from Random import get_modified_price as test3

app = Flask(__name__);

@app.route('/stockdata', methods=['GET'])
def get_stock_data():
    try:
        # Get query parameters with defaults
        ticker = request.args.get('tickers')  # Query param for tickers
        
        if not ticker:
            ticker = ['SPY']  # Default to SPY if no tickers provided

        stock_data = get_stock_prices(ticker)
        
        # Get whether it should be a buy
        prediction_value = prediction(ticker)
        
        # Extract values from the result dictionary
        latest_data = prediction_value.get('Latest Data', np.nan)
        lstm_pred = prediction_value.get('LSTM Prediction', np.nan)
    
        # Ensure LSTM Prediction is in correct format and rounded to 2 decimal places
        if not np.isnan(lstm_pred):
            lstm_prediction = round(lstm_pred, 2)
        else:
            lstm_prediction = 'N/A'
        
        predictions_list = []
        
        # Check if LSTM Prediction is greater than Latest Data
        if isinstance(latest_data, (int, float)) and isinstance(lstm_prediction, (int, float)) and lstm_prediction > latest_data:
            # Add to predictions_list if condition is met
            predictions_list.append({
                'Ticker': ticker,
                'LSTM Prediction': lstm_prediction,
                'Action': 'Buy'
            })
        else:
            predictions_list.append({
                'Ticker': ticker,
                'LSTM Prediction': lstm_prediction,
                'Action': 'Sell'
            }) 
        
        return jsonify(predictions_list)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/tickers', methods=['GET'])
def get_tickers():
    try:
        # 1. Scrape Dataroma for tickers
        potential_stocks = get_combined_tickers()
    
        # 2. Get stock prices for the tickers returned
        tickers = potential_stocks['Ticker'].unique()
        stock_prices = get_stock_prices(tickers)
        stock_prices_df = pd.DataFrame(list(stock_prices.items()), columns=['Ticker', 'Price'])
    
        # 3. Get news articles related to the tickers returned
        news_articles_sentiment_df = get_recent_news(tickers)
    
        # 4. Merge 2 and 3
        combined_df = pd.merge(stock_prices_df, news_articles_sentiment_df, on='Ticker', how='inner')
    
        # 5. Filter out tickers with an average sentiment below 0.05 or price less than 1 or null
        ticker_df = combined_df[(combined_df['Average Sentiment'] >= 0.2) & 
                              (combined_df['Price'].notnull()) & 
                              (combined_df['Price'] >= 1)]
        
        # 6. Generate predictions for each ticker
        predictions_list = []
        
        for ticker in ticker_df['Ticker']:
            # Call the prediction function and get the result directly
            result = prediction(ticker)

            # Extract values from the result dictionary
            latest_data = result.get('Latest Data', np.nan)
            lstm_pred = result.get('LSTM Prediction', np.nan)
    
            # Ensure LSTM Prediction is in correct format and rounded to 2 decimal places
            if not np.isnan(lstm_pred):
                lstm_prediction = round(lstm_pred, 2)
            else:
                lstm_prediction = 'N/A'
    
            # Check if LSTM Prediction is greater than Latest Data
            if isinstance(latest_data, (int, float)) and isinstance(lstm_prediction, (int, float)) and lstm_prediction > latest_data:
                # Add to predictions_list if condition is met
                predictions_list.append(ticker)           
        
        return jsonify(predictions_list)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/train', methods=['GET'])
def get_train():
    try:
        train();
        return jsonify("success")
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # You can adjust host and port as needed

