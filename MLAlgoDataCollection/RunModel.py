import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import joblib
from datetime import datetime, timedelta

# Load the trained models and scaler
linear_model = joblib.load('linear_model.pkl')
lstm_model = load_model('lstm_model.keras')
scaler = joblib.load('scaler.pkl')

# Define feature columns used for Linear Regression and LSTM
features = ['Lag1', 'Moving_Avg_5', 'Moving_Avg_20', 'Moving_Avg_50', 'Moving_Avg_200', 'Volatility', 'Volume']
features_lstm = ['Close', 'Volume', 'Moving_Avg_5', 'Moving_Avg_20', 'Moving_Avg_50', 'Moving_Avg_200', 'Volatility']

def prepare_data_for_prediction(ticker, scaler, time_step=10):
    # Fetch recent historical data
    today = datetime.now().strftime('%Y-%m-%d')
    data = yf.download(ticker, start='2020-01-01', end=today)
    
    if data.empty:
        raise ValueError(f"No data available for ticker: {ticker}")
    
    data = data[['Close', 'Volume']].ffill()
    
    # Compute features
    data['Moving_Avg_5'] = data['Close'].rolling(window=5).mean()
    data['Moving_Avg_20'] = data['Close'].rolling(window=20).mean()
    data['Moving_Avg_50'] = data['Close'].rolling(window=50).mean()
    data['Moving_Avg_200'] = data['Close'].rolling(window=200).mean()
    data['Volatility'] = data['Close'].rolling(window=50).std()
    
    # Drop rows with NaN values
    data.dropna(inplace=True)
    
    # Check if we have enough data for the time step
    if len(data) < time_step:
        raise ValueError(f"Not enough data to prepare input for LSTM. Required: {time_step}, Available: {len(data)}")
    
    # Create features for prediction
    latest_data = data[features_lstm].tail(time_step).copy()
    
    # Add Lag1 feature (previous day's Close price) for Linear Regression
    if len(data) > time_step:
        latest_data['Lag1'] = data['Close'].shift(1).tail(time_step).values
    
    # Ensure all required features are present
    for feature in features + ['Close']:
        if feature not in latest_data.columns:
            latest_data[feature] = np.nan
    
    # Fill NaN values in the latest_data for prediction
    latest_data.fillna(method='ffill', inplace=True)
    latest_data.fillna(method='bfill', inplace=True)
    
    # Scale data
    scaled_data = scaler.transform(latest_data[features_lstm])
    
    # Prepare data for LSTM
    X_lstm = np.array([scaled_data])
    
    return latest_data, X_lstm

def predict_tomorrow_stock_price(ticker):
    # Prepare data
    latest_data, X_lstm = prepare_data_for_prediction(ticker, scaler)
    
    # Make LSTM prediction
    lstm_pred = lstm_model.predict(X_lstm)
    
    # Inverse transform to original scale
    # Concatenate with zeros for features not used in prediction
    lstm_pred = scaler.inverse_transform(np.concatenate((lstm_pred, np.zeros((lstm_pred.shape[0], len(features_lstm) - 1))), axis=1))
    
    # Handle Linear Regression prediction
    linear_pred = np.nan
    if 'Lag1' in latest_data.columns and len(latest_data) > 0:
        # Ensure the correct feature list for Linear Regression prediction
        linear_features = ['Lag1'] + [f for f in features if f != 'Lag1']
        linear_pred = linear_model.predict(latest_data[linear_features].tail(1))
    
    # Calculate mean prediction
    mean_pred = (linear_pred[-1] if not np.isnan(linear_pred) else np.nan + lstm_pred[0][0]) / 2

    return {
        'Ticker': ticker,
        'Latest Data': latest_data,
        'Linear Prediction': linear_pred[-1] if not np.isnan(linear_pred) else 'N/A',
        'LSTM Prediction': lstm_pred[0][0],
        'Mean Prediction': mean_pred
    }
