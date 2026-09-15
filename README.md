Vecaid-Patented is an advanced financial analytics platform that uses machine learning, technical indicators, and natural language sentiment analysis to predict short-term stock price movements with confidence scoring and backtesting capabilities.

This project combines quantitative finance, AI modeling, and data visualization to generate actionable insights for traders and researchers interested in financial forecasting.

📋 Overview
Vecaid-Beta analyzes real-world financial data — such as stock prices, volume, sentiment, and option activity — to make next-day stock predictions.
It integrates deep learning models, Bayesian optimization, and ensemble techniques for robust performance and interpretable outputs.

🔑 Key Features
📈 Real-Time Stock Prediction
Predicts next-day closing prices for any ticker symbol using hybrid AI models.

🧩 Technical & Statistical Indicators
Extracts over 20+ features including RSI, MACD, ADX, Bollinger Bands, ATR, and stochastic oscillators.

🧠 Deep Learning Ensemble
Trains multiple neural architectures — GRU, CNN-LSTM, BiLSTM, DNN, Transformer — optimized via Bayesian hyperparameter tuning.

💬 Sentiment Integration
Scrapes recent news from Yahoo Finance and uses VADER Sentiment Analysis to quantify market mood around each stock.

🪙 Options & Fundamental Analysis
Enhances predictions with live options chain data and fundamental ratios (like P/E and volume deltas).

🔁 Markov Chain & Mean Reversion Logic
Incorporates statistical techniques to identify market regime shifts and mean-reversion probabilities.

📊 Backtesting & Performance Metrics
Evaluates strategy accuracy, RMSE, MAE, and cumulative returns over historical data, generating performance visualizations automatically.

⚙️ How It Works
Data Collection

Gathers stock data from Yahoo Finance (daily, weekly, and intraday).
Scrapes financial news and sentiment headlines.
Retrieves options chain and fundamental data for signal enhancement.
Feature Engineering

Computes technical indicators and statistical metrics.
Aggregates intraday features such as volatility and volume profiles.
Model Training

Performs dimensionality reduction (PCA) and standard scaling.
Trains deep learning architectures (GRU, CNN-LSTM, Transformer, etc.).
Tunes hyperparameters via Bayesian Optimization on XGBoost models.
Ensemble Learning

Combines predictions from multiple models into a meta-model using stacking regression, improving overall accuracy and stability.
Prediction & Confidence Scoring

Generates next-day closing price forecasts, movement direction (up/down), and a confidence percentage based on model variance.
Backtesting

Simulates recent trading periods to evaluate model precision and cumulative return trends.
📊 Example Output
For a ticker such as AAPL:

Predicted Direction: Higher
Predicted Difference: $14.45
Predicted Price: $204.78
Confidence: 82.6%
Predicted Percent Move: 9.33%
