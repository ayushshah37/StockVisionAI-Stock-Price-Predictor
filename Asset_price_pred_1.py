# 📦 Step 1: Import Libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 📥 Step 2: Download Stock Data
df = yf.download("RELIANCE.BO", start="2015-01-01", end="2024-01-01")

# 🧹 Step 3: Manually Calculate Technical Indicators
# 3.1 SMA (20-day Simple Moving Average)
df['SMA_20'] = df['Close'].rolling(window=20).mean()

# 3.2 RSI (14-day Relative Strength Index)
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI_14'] = 100 - (100 / (1 + rs))

df.dropna(inplace=True)  # Drop missing rows

# ⚙️ Step 4: Create Lag Features and Rolling Stats
df['Close_t-1'] = df['Close'].shift(1)
df['rolling_mean_5'] = df['Close'].rolling(window=5).mean()
df['rolling_std_5'] = df['Close'].rolling(window=5).std()
df.dropna(inplace=True)

# 🧠 Step 5: Prepare Data for LSTM
# We'll use only the 'Close' column for prediction
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['Close']])

X, y = [], []
for i in range(60, len(scaled_data)):
    X.append(scaled_data[i-60:i])  # Past 60 days
    y.append(scaled_data[i, 0])    # Next day

X, y = np.array(X), np.array(y)

# Reshape X for LSTM: (samples, timesteps, features)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# 🏗️ Step 6: Build and Train LSTM Model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=10, batch_size=32)

# 📉 Step 7: Evaluate the Model
predicted = model.predict(X)
predicted_prices = scaler.inverse_transform(predicted)
real_prices = scaler.inverse_transform(y.reshape(-1, 1))

rmse = np.sqrt(mean_squared_error(real_prices, predicted_prices))

mape = mean_absolute_percentage_error(real_prices, predicted_prices)

# Calculate accuracy
accuracy = 100 - mape * 100  # Percentage accuracy

# Display results
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}")
print(f"Accuracy: {accuracy:.2f}%")

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(real_prices, label='Actual Price')
plt.plot(predicted_prices, label='Predicted Price')
plt.legend()
plt.title(f"TCS (TCS.BO) Stock Price Prediction\nRMSE: {rmse:.2f} | MAPE: {mape:.2f}")
plt.xlabel("Time")
plt.ylabel("Price")
plt.grid(True)
plt.show()
