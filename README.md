# Stock Price Prediction 

This project builds a simple machine learning model to predict the next day's closing stock price using historical data.

I use Yahoo Finance data via the `yfinance` library to fetch stock prices and train a linear regression model on lagged price features.

## 📌 Problem Statement

Given past closing prices of a stock, predict the next day's closing price using machine learning.

## 📝 Approach

- Download historical stock prices
- Engineer lag features (e.g., yesterday's closing price, 2-day lag)
- Train/test split
- Train a linear regression model
- Evaluate performance using RMSE
- Visualize actual vs predicted prices

## ✅ Results

Example output (Apple stock): 
Root Mean Squared Error: 1.98


## 💻 How to Run

1. Clone this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the script:
```bash
python stock_price_prediction.py
```

