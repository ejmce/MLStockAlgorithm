import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
from datetime import datetime, timedelta
import os
import joblib

def train_model():
    # Set random seeds for reproducibility
    np.random.seed(26)
    tf.random.set_seed(26)

    # Define the list of tickers
    tickers = ['SPY', 'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'ULTA', 'INTC', 'CRM', 'AMD', 'MCD', 'ESGV', 'QQQ']

    # Initialize a list to store combined data
    all_data = []

    # Function to fetch and prepare data for a given ticker
    def prepare_data(ticker):
        # Fetch historical data
        data = yf.download(ticker, start='2020-01-01', end=datetime.now().strftime('%Y-%m-%d'))
        if data.empty:
            return None
    
        # Use 'Close' price and forward fill to handle missing data
        data = data[['Close', 'Volume']].ffill()
    
        # Compute additional features: 5-day, 20-day, 50-day, and 200-day moving averages
        data['Moving_Avg_5'] = data['Close'].rolling(window=5).mean()
        data['Moving_Avg_20'] = data['Close'].rolling(window=20).mean()
        data['Moving_Avg_50'] = data['Close'].rolling(window=50).mean()
        data['Moving_Avg_200'] = data['Close'].rolling(window=200).mean()
        data['Volatility'] = data['Close'].rolling(window=50).std()
    
        data.dropna(inplace=True)  # Drop rows with NaN values
    
        # Create lagged feature for Linear Regression
        data['Lag1'] = data['Close'].shift(1)
        data.dropna(inplace=True)  # Drop rows with NaN values
    
        # Add ticker column for identification
        data['Ticker'] = ticker
    
        return data

    # Combine data for all tickers
    for ticker in tickers:
        data = prepare_data(ticker)
        if data is not None:
            all_data.append(data)

    # Combine all data into a single DataFrame
    combined_data = pd.concat(all_data, axis=0)

    # Define features and target
    features = ['Lag1', 'Moving_Avg_5', 'Moving_Avg_20', 'Moving_Avg_50', 'Moving_Avg_200', 'Volatility', 'Volume']
    X = combined_data[features]
    y = combined_data['Close']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Scale data for LSTM
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(combined_data[['Close', 'Volume', 'Moving_Avg_5', 'Moving_Avg_20', 
                                                     'Moving_Avg_50', 'Moving_Avg_200', 'Volatility']])

    # Define time step parameter
    time_step = 10  # Number of days to look back for LSTM

    # Prepare data for LSTM
    def create_dataset(data, time_step=1):
        X, y = [], []
        for i in range(len(data) - time_step):
            X.append(data[i:(i + time_step), :])
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)

    X_lstm, y_lstm = create_dataset(scaled_data, time_step)

    # Split LSTM data into training and testing sets
    X_lstm_train, X_lstm_test, y_lstm_train, y_lstm_test = train_test_split(X_lstm, y_lstm, test_size=0.2, shuffle=False)

    # Define model checkpoint and early stopping
    model_path = 'lstm_model.keras'
    if os.path.exists(model_path):
        print("Loading existing model...")
        model = load_model(model_path)
    else:
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, X_lstm.shape[2])))
        model.add(LSTM(units=50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss')
    
    # Train the LSTM model
    history = model.fit(X_lstm_train, y_lstm_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping, model_checkpoint])
    
    # Print training logs
    print("Final Training Loss:", history.history['loss'][-1])
    print("Final Validation Loss:", history.history['val_loss'][-1])

    # Predict using the LSTM model
    def predict_next_day(model, last_days):
        last_days = last_days.reshape((1, time_step, X_lstm.shape[2]))  # Reshape for LSTM input
        return model.predict(last_days)

    # Get the last `time_step` days of scaled data for prediction
    last_days = scaled_data[-time_step:]
    lstm_pred = predict_next_day(model, last_days)

    # Prepare data for inverse transformation
    lstm_pred_full = np.zeros((lstm_pred.shape[0], scaled_data.shape[1]))  # Initialize with zeros
    lstm_pred_full[:, 0] = lstm_pred[:, 0]  # Set 'Close' predictions
    lstm_pred = scaler.inverse_transform(lstm_pred_full)[:, 0]  # Inverse transform to original scale

    # Linear Regression Model
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    linear_pred_train = linear_model.predict(X_train)
    linear_pred_test = linear_model.predict(X_test)

    # Calculate the mean of Linear Regression and LSTM predictions
    mean_pred = (linear_pred_test[-1] + lstm_pred[0]) / 2

    # Fetch actual data for today
    today_date = datetime.now().strftime('%Y-%m-%d')
    actual_data = yf.download('SPY', start=today_date, end=(datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'))
    if actual_data.empty:
        actual_value = None
    else:
        actual_value = actual_data['Close'].iloc[0]

    # Calculate percentage differences
    if actual_value is not None:
        linear_diff = abs((actual_value - linear_pred_test[-1]) / actual_value) * 100
        lstm_diff = abs((actual_value - lstm_pred[0]) / actual_value) * 100
        mean_diff = abs((actual_value - mean_pred) / actual_value) * 100
    else:
        linear_diff = lstm_diff = mean_diff = None

    # Calculate RMSE and MAE for the training and testing sets

    lstm_rmse_train = np.sqrt(mean_squared_error(y_lstm_train, model.predict(X_lstm_train)))
    lstm_rmse_test = np.sqrt(mean_squared_error(y_lstm_test, model.predict(X_lstm_test)))

    lstm_mae_train = mean_absolute_error(y_lstm_train, model.predict(X_lstm_train))
    lstm_mae_test = mean_absolute_error(y_lstm_test, model.predict(X_lstm_test))
    
    # Save Linear Regression model
    joblib.dump(linear_model, 'linear_model.pkl')
    # Save scaler
    joblib.dump(scaler, 'scaler.pkl')

    # Collect results for export
    results = [{
        'Date': today_date,
        'Actual_Value': actual_value,
        'LSTM_Prediction': lstm_pred[0] if actual_value is not None else None,
        'LSTM_Percent_Difference': lstm_diff,
        'LSTM_RMSE_Train': lstm_rmse_train,
        'LSTM_RMSE_Test': lstm_rmse_test,
        'LSTM_MAE_Train': lstm_mae_train,
        'LSTM_MAE_Test': lstm_mae_test,
        'Linear_Regression_Prediction': linear_pred_test[-1] if actual_value is not None else None,
        'Linear_Percent_Difference': linear_diff,
        'Mean_Prediction': mean_pred if actual_value is not None else None,
        'Mean_Percent_Difference': mean_diff,
    }]

    # Convert results to DataFrame and export to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('model_results_all_tickers.csv', index=False)

    print("Results have been exported to model_results_all_tickers.csv")

