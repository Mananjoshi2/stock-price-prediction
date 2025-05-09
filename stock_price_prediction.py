import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Download historical data
ticker = 'AAPL'
df = yf.download(ticker, start='2022-01-01', end='2024-12-31')

# Create lag features
df['Lag1'] = df['Close'].shift(1)
df['Lag2'] = df['Close'].shift(2)

# Drop rows with NaN
df.dropna(inplace=True)

# Features and target
X = df[['Lag1', 'Lag2']]
y = df['Close']

# Train-test split
split_idx = int(len(df) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Root Mean Squared Error: {rmse:.2f}')

# Plot
plt.figure(figsize=(12, 6))
plt.plot(df.index[split_idx:], y_test, label='Actual')
plt.plot(df.index[split_idx:], y_pred, label='Predicted')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title(f'{ticker} Stock Price Prediction')
plt.legend()
plt.show()
