# Import Libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

from statsmodels.tsa.arima.model import ARIMA

#  Download Stock Data
df = yf.download("RELIANCE.BO", start="2015-01-01", end="2024-01-01")

#  Manually Calculating Technical Indicators
df['SMA_20'] = df['Close'].rolling(window=20).mean()

delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI_14'] = 100 - (100 / (1 + rs))

df.dropna(inplace=True)

# Create Lag Features and Rolling Stats
df['Close_t-1'] = df['Close'].shift(1)
df['rolling_mean_5'] = df['Close'].rolling(window=5).mean()
df['rolling_std_5'] = df['Close'].rolling(window=5).std()
df.dropna(inplace=True)

#  Prepare Data for LSTM
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['Close']])

X, y = [], []
for i in range(60, len(scaled_data)):
    X.append(scaled_data[i-60:i])  # Past 60 days
    y.append(scaled_data[i, 0])    # Next day

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Train/test split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build and Train LSTM Model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

# Predictions
lstm_pred = model.predict(X_test)
lstm_pred_rescaled = scaler.inverse_transform(lstm_pred)
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Baselines
# --- ARIMA ---
train_close = df['Close'][:train_size]
test_close = df['Close'][train_size:]

arima_model = ARIMA(train_close, order=(5,1,0))
arima_fit = arima_model.fit()
arima_pred = arima_fit.forecast(steps=len(test_close))

# --- Random Walk ---
rw_pred = test_close.shift(1).dropna()

# Evaluation Metrics
def evaluate(real, pred, name):
    rmse = np.sqrt(mean_squared_error(real, pred))
    mape = mean_absolute_percentage_error(real, pred)
    print(f"{name} -> RMSE: {rmse:.2f}, MAPE: {mape:.2f}, Accuracy: {100 - mape*100:.2f}%")

print("=== Model Comparison ===")
evaluate(y_test_rescaled, lstm_pred_rescaled, "LSTM")
evaluate(test_close.values, arima_pred.values, "ARIMA")
evaluate(test_close.iloc[1:].values, rw_pred.values, "Random Walk")

# Robustness under Volatility Regimes
df['volatility'] = df['Close'].pct_change().rolling(window=30).std()
median_vol = df['volatility'].median()

high_vol_idx = df['volatility'] > median_vol
low_vol_idx = df['volatility'] <= median_vol

print("\n=== Regime-Specific MAPE ===")
# High volatility
hv_real = y_test_rescaled[high_vol_idx[-len(y_test_rescaled):].values]
hv_pred = lstm_pred_rescaled[high_vol_idx[-len(y_test_rescaled):].values]
print("High Volatility Regime:")
evaluate(hv_real, hv_pred, "LSTM")

# Low volatility
lv_real = y_test_rescaled[low_vol_idx[-len(y_test_rescaled):].values]
lv_pred = lstm_pred_rescaled[low_vol_idx[-len(y_test_rescaled):].values]
print("Low Volatility Regime:")
evaluate(lv_real, lv_pred, "LSTM")

#  Visualization
plt.figure(figsize=(12,6))
plt.plot(y_test_rescaled, label='Actual')
plt.plot(lstm_pred_rescaled, label='LSTM Predicted')
plt.plot(arima_pred.values, label='ARIMA Predicted')
plt.plot(rw_pred.values, label='Random Walk Predicted')
plt.title("Stock Price Prediction: LSTM vs ARIMA vs Random Walk")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()
