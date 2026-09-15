# Libraries:
# pip install yfinance pandas numpy pywavelets tensorflow scikit-learn xgboost nltk
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import datetime
import yfinance as yf
import pandas as pd
import numpy as np
import pywt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import math
import nltk
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment import SentimentIntensityAnalyzer

# Global parameters
FORECAST_HORIZON = 2

def safe_float(val):
    if np.isscalar(val):
        return float(val)
    if isinstance(val, np.ndarray):
        if val.size == 1:
            return float(val.item())
        else:
            return float(val[0])
    if hasattr(val, 'iloc'):
        return float(val.iloc[0])
    raise ValueError("Cannot convert input to float.")

def get_sentiment(ticker):
    try:
        t = yf.Ticker(ticker)
        news = t.news
        if not news or len(news) == 0:
            return 0.0
        sia = SentimentIntensityAnalyzer()
        scores = [sia.polarity_scores(article.get('title', ''))['compound']
                  for article in news if article.get('title', '')]
        return float(np.mean(scores)) if scores else 0.0
    except Exception:
        return 0.0

def get_fundamentals(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    fundamentals = {"trailingPE": info.get("trailingPE", np.nan),
                    "dividendYield": info.get("dividendYield", np.nan),
                    "priceToBook": info.get("priceToBook", np.nan),
                    "profitMargins": info.get("profitMargins", np.nan)}
    return {k: np.nan_to_num(v, nan=0) for k, v in fundamentals.items()}

def markov_chain_analysis(ticker, start_date="2021-01-01", end_date=datetime.date.today().strftime('%Y-%m-%d')):
    data = yf.download(ticker, start=start_date, end=end_date)
    price_col = "Adj Close" if "Adj Close" in data.columns else "Close"
    data["daily_return"] = data[price_col].pct_change()
    data["state"] = np.where(data["daily_return"] >= 0, "up", "down")
    numerator = len(data[(data["state"]=="up") & (data["state"].shift(-1)=="down") &
                         (data["state"].shift(-2)=="down") & (data["state"].shift(-3)=="down") &
                         (data["state"].shift(-4)=="down") & (data["state"].shift(-5)=="down")])
    denominator = len(data[(data["state"].shift(1)=="down") & (data["state"].shift(2)=="down") &
                           (data["state"].shift(3)=="down") & (data["state"].shift(4)=="down") &
                           (data["state"].shift(5)=="down")])
    return numerator / denominator if denominator != 0 else 0.0

def get_options_data(ticker, expiration):
    stock = yf.Ticker(ticker)
    chain = stock.option_chain(expiration)
    calls = chain.calls.copy()
    puts  = chain.puts.copy()
    calls['expiration'] = expiration
    puts['expiration']  = expiration
    return calls, puts

def filter_outliers(df, std_multiplier=2):
    mean_volume = df['volume'].mean()
    std_volume  = df['volume'].std()
    lower_threshold = max(mean_volume - std_multiplier * std_volume, 0)
    upper_threshold = mean_volume + std_multiplier * std_volume
    return df[(df['volume'] < lower_threshold) | (df['volume'] > upper_threshold)]

def compute_side_averages(calls_outliers, puts_outliers):
    avg_calls_volume = calls_outliers['volume'].mean() if not calls_outliers.empty else np.nan
    avg_puts_volume  = puts_outliers['volume'].mean() if not puts_outliers.empty else np.nan
    if np.isnan(avg_calls_volume) and np.isnan(avg_puts_volume):
        return None, None, None
    elif np.nan_to_num(avg_calls_volume) >= np.nan_to_num(avg_puts_volume):
        side = "calls"
        avg_strike = calls_outliers['strike'].mean() if not calls_outliers.empty else np.nan
    else:
        side = "puts"
        avg_strike = puts_outliers['strike'].mean() if not puts_outliers.empty else np.nan
    return side, (avg_calls_volume if side=="calls" else avg_puts_volume), avg_strike

def options_contracts_signal(ticker, std_multiplier=2):
    stock = yf.Ticker(ticker)
    expirations = stock.options
    if not expirations:
        return None, None, None
    expiration = expirations[0]
    calls, puts = get_options_data(ticker, expiration)
    calls_outliers = filter_outliers(calls, std_multiplier)
    puts_outliers  = filter_outliers(puts, std_multiplier)
    return compute_side_averages(calls_outliers, puts_outliers)

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def compute_vwap(df, period=14):
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    vol = df["Volume"].rolling(window=period, min_periods=1).sum()
    epsilon = 1e-8
    return (typical * df["Volume"]).rolling(window=period, min_periods=1).sum() / (vol + epsilon)

def compute_macd(series, fastperiod=12, slowperiod=26, signalperiod=9):
    ema_fast = series.ewm(span=fastperiod, adjust=False).mean()
    ema_slow = series.ewm(span=slowperiod, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signalperiod, adjust=False).mean()
    return macd_line, signal_line

def compute_stoch_osc(high_series, low_series, close_series, k_period=14, d_period=3):
    lowest_low = high_series.rolling(k_period, min_periods=1).min()
    highest_high = high_series.rolling(k_period, min_periods=1).max()
    stoch_k = 100 * (close_series - lowest_low) / (highest_high - lowest_low)
    stoch_d = stoch_k.rolling(d_period, min_periods=1).mean()
    return stoch_k, stoch_d

def compute_bollinger_bands(df, window=20, num_std=2):
    sma = df["Close"].rolling(window=window).mean()
    std = df["Close"].rolling(window=window).std()
    return sma + num_std * std, sma - num_std * std

def wavelet_denoise(series, wavelet='db4', level=1):
    coeff = pywt.wavedec(series, wavelet, mode="per")
    coeff[1:] = [np.zeros_like(detail) for detail in coeff[1:]]
    return pywt.waverec(coeff, wavelet, mode="per")[:len(series)]

def create_lag_features(df, n_lags=5):
    df = df.copy()
    df['Close_denoised'] = wavelet_denoise(df['Close'])
    for i in range(1, n_lags + 1):
        df[f'lag_{i}'] = df['Close_denoised'].shift(i)
    df.dropna(inplace=True)
    return df

def compute_linreg(s, window):
    x = np.arange(len(s))
    try:
        slope, _ = np.polyfit(x, s, 1)
    except np.linalg.LinAlgError:
        slope = np.nan
    return slope

def enhanced_lstm_forecast(df):
    data = df['Close'].values.ravel()
    window = 10
    if len(data) < window:
        return float(data[-1])
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i + window])
        y.append(data[i + window])
    X = np.array(X).reshape((len(X), window, 1))
    y = np.array(y)
    model = models.Sequential()
    model.add(layers.Input(shape=(window, 1)))
    model.add(layers.LSTM(64, activation='relu', return_sequences=True))
    model.add(layers.Dropout(0.3))
    model.add(layers.LSTM(32, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mse')
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
    model.fit(X, y, epochs=100, batch_size=16, callbacks=[early_stop], verbose=0)
    last_seq = data[-window:].reshape((1, window, 1))
    pred = model.predict(last_seq, verbose=0)
    return float(pred[0, 0])

def svr_forecast(df):
    data = df['Close'].values.ravel()
    window = 10
    if len(data) <= window:
        return float(data[-1])
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i + window])
        y.append(data[i + window])
    X = np.array(X)
    y = np.array(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    model.fit(X_scaled, y)
    last_seq = data[-window:]
    last_seq_scaled = scaler.transform(last_seq.reshape(1, -1))
    pred = model.predict(last_seq_scaled)
    return float(pred[0])

def final_combiner_model(df, base_model, options_strike, fundamentals, ticker, linreg_length=14):
    df = df.copy()
    if "Close_denoised" not in df.columns:
        df["Close_denoised"] = wavelet_denoise(df["Close"])
    if "target" not in df.columns:
        df["target"] = df["Close"].shift(-FORECAST_HORIZON)
    df["sma_200"] = df["Close"].rolling(200, min_periods=1).mean()
    df["linreg_14"] = df["Close"].rolling(linreg_length, min_periods=1).apply(lambda s: compute_linreg(s, linreg_length), raw=False)
    df["vwap_14"] = compute_vwap(df, period=14)
    df["rsi_14"] = compute_rsi(df["Close"], period=14)
    macd_line, signal_line = compute_macd(df["Close"])
    df["macd_line"] = macd_line
    df["macd_signal"] = signal_line
    stoch_k, stoch_d = compute_stoch_osc(df["High"], df["Low"], df["Close"])
    df["stoch_k"] = stoch_k
    df["stoch_d"] = stoch_d
    upper_bb, lower_bb = compute_bollinger_bands(df)
    df["upper_bb"] = upper_bb
    df["lower_bb"] = lower_bb
    # Use ticker for markov analysis and sentiment
    df["markov_prob"] = markov_chain_analysis(ticker, start_date="2021-01-01", 
                                                end_date=datetime.date.today().strftime('%Y-%m-%d'))
    sentiment = get_sentiment(ticker)
    df["sentiment_score"] = sentiment
    final_data = df[df["target"].notna()].copy()
    final_data.reset_index(drop=True, inplace=True)
    
    # Ensure "Adj Close" exists: if missing, hardcode it as "Close".
    if "Adj Close" not in final_data.columns:
        final_data["Adj Close"] = final_data["Close"]

    # Add fundamental and options features.
    final_data["opt_strike"] = options_strike if options_strike is not None else 0.0
    final_data["trailingPE"] = fundamentals.get("trailingPE", 0)
    final_data["dividendYield"] = fundamentals.get("dividendYield", 0)
    final_data["priceToBook"] = fundamentals.get("priceToBook", 0)
    final_data["profitMargins"] = fundamentals.get("profitMargins", 0)

    tech_features = ["sma_200", "linreg_14", "vwap_14", "rsi_14", "macd_line",
                     "macd_signal", "stoch_k", "stoch_d", "upper_bb", "lower_bb",
                     "markov_prob", "sentiment_score"]
    final_data[tech_features] = final_data[tech_features].ffill().fillna(final_data[tech_features].median()).fillna(0)
    final_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    final_data = final_data.fillna(0)

    # Build feature set for XGBoost.
    # We now ensure we always include "Adj Close" (either originally or hardcoded) to have 10 features.
    xgb_cols = ["Adj Close", "High", "Low", "Volume", "Close_denoised"] + [f'lag_{i}' for i in range(1, 6)]
    
    # Predict base model output.
    final_data["base_pred"] = base_model.predict(final_data[xgb_cols])
    enhanced_lstm_pred = float(enhanced_lstm_forecast(df))
    svr_pred = float(svr_forecast(df))
    final_data["enhanced_lstm_pred"] = enhanced_lstm_pred
    final_data["svr_pred"] = svr_pred

    features = final_data[["base_pred", "enhanced_lstm_pred", "svr_pred", "opt_strike", "linreg_14", "sma_200",
                           "vwap_14", "rsi_14", "macd_line", "macd_signal", "stoch_k", "stoch_d",
                           "upper_bb", "lower_bb", "trailingPE", "dividendYield", "priceToBook",
                           "profitMargins", "markov_prob", "sentiment_score"]]
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
    target = final_data["target"].replace([np.inf, -np.inf], np.nan).fillna(0)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features.values)
    meta_model = GradientBoostingRegressor(n_estimators=300, max_depth=6, random_state=42)
    meta_model.fit(scaled_features, target.values)

    latest_row = final_data.iloc[-1].fillna(0)
    meta_input = {
        "base_pred": safe_float(base_model.predict(np.array(latest_row[xgb_cols]).reshape(1, -1))),
        "enhanced_lstm_pred": safe_float(enhanced_lstm_forecast(df)),
        "svr_pred": safe_float(svr_forecast(df)),
        "opt_strike": safe_float(options_strike),
        "linreg_14": safe_float(latest_row["linreg_14"]),
        "sma_200": safe_float(latest_row["sma_200"]),
        "vwap_14": safe_float(latest_row["vwap_14"]),
        "rsi_14": safe_float(latest_row["rsi_14"]),
        "macd_line": safe_float(latest_row["macd_line"]),
        "macd_signal": safe_float(latest_row["macd_signal"]),
        "stoch_k": safe_float(latest_row["stoch_k"]),
        "stoch_d": safe_float(latest_row["stoch_d"]),
        "upper_bb": safe_float(latest_row["upper_bb"]),
        "lower_bb": safe_float(latest_row["lower_bb"]),
        "trailingPE": safe_float(fundamentals.get("trailingPE", 0)),
        "dividendYield": safe_float(fundamentals.get("dividendYield", 0)),
        "priceToBook": safe_float(fundamentals.get("priceToBook", 0)),
        "profitMargins": safe_float(fundamentals.get("profitMargins", 0)),
        "markov_prob": safe_float(latest_row["markov_prob"]),
        "sentiment_score": safe_float(get_sentiment(ticker))
    }
    feature_order = ["base_pred", "enhanced_lstm_pred", "svr_pred", "opt_strike", "linreg_14", "sma_200",
                     "vwap_14", "rsi_14", "macd_line", "macd_signal", "stoch_k", "stoch_d",
                     "upper_bb", "lower_bb", "trailingPE", "dividendYield", "priceToBook",
                     "profitMargins", "markov_prob", "sentiment_score"]
    meta_input_df = pd.DataFrame([meta_input], columns=feature_order).fillna(0)
    meta_scaled = scaler.transform(meta_input_df.values)
    meta_scaled = np.nan_to_num(meta_scaled)
    final_pred = float(meta_model.predict(meta_scaled)[0])
    return final_pred

def predict_forecast(df_train, options_strike, fundamentals, ticker, min_train_size=50):
    df_train = df_train.copy()
    df_train = create_lag_features(df_train)
    df_train["target"] = df_train["Close"].shift(-FORECAST_HORIZON)
    df_train.dropna(inplace=True)
    if len(df_train) < min_train_size:
        return None
    train_size = int(len(df_train) * 0.8)
    train_df = df_train.iloc[:train_size]
    # Drop only "Close" and "target" so that "Adj Close" remains if available.
    cols_to_drop = ["Close", "target"]
    X_train = train_df.drop(cols_to_drop, axis=1)
    y_train = train_df["target"]
    scaler = StandardScaler()
    scaled_X_train = scaler.fit_transform(X_train.values)
    xgb = XGBRegressor(random_state=42)
    param_grid = {'n_estimators': [50, 100, 150],
                  'learning_rate': [0.05, 0.1, 0.15],
                  'max_depth': [3, 5, 7]}
    grid_search = GridSearchCV(xgb, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(scaled_X_train, y_train)
    base_model = grid_search.best_estimator_
    forecast = final_combiner_model(df_train, base_model, options_strike, fundamentals, ticker, linreg_length=14)
    return forecast

def main():
    # For production or testing, define a ticker here.
    ticker = "AAPL"  # Example ticker; in practice, this should come from user input.
    # Download data including "Adj Close" for consistency.
    stock_data = yf.download(ticker, start="2021-01-01", 
                             end=datetime.date.today().strftime('%Y-%m-%d'))[['Adj Close','Close','High','Low','Volume']].copy()
    stock_data.to_csv("stock_data.csv")
    stock = yf.Ticker(ticker)
    options_list = stock.options
    if options_list:
        expiration = options_list[0]
        calls, puts = get_options_data(ticker, expiration)
        calls.to_csv("options_calls.csv", index=False)
        puts.to_csv("options_puts.csv", index=False)
    _, _, options_strike = options_contracts_signal(ticker, std_multiplier=2)
    fundamentals = get_fundamentals(ticker)
    print("Production forecast:")
    prod_data = yf.download(ticker, start="2021-01-01", 
                            end=datetime.date.today().strftime('%Y-%m-%d'))[['Adj Close','Close','High','Low','Volume']].copy()
    prod_data = create_lag_features(prod_data)
    prod_data["target"] = prod_data["Close"].shift(-FORECAST_HORIZON)
    prod_data.dropna(inplace=True)
    if len(prod_data) < 50:
        print("Not enough data for production forecast.")
    else:
        train_size = int(len(prod_data) * 0.8)
        train_df = prod_data.iloc[:train_size]
        cols_to_drop = ["Close", "target"]
        X_train = train_df.drop(cols_to_drop, axis=1)
        y_train = train_df["target"]
        scaler = StandardScaler()
        scaled_X_train = scaler.fit_transform(X_train.values)
        xgb = XGBRegressor(random_state=42)
        param_grid = {'n_estimators': [50, 100, 150],
                      'learning_rate': [0.05, 0.1, 0.15],
                      'max_depth': [3, 5, 7]}
        grid_search = GridSearchCV(xgb, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(scaled_X_train, y_train)
        base_model = grid_search.best_estimator_
        prod_forecast = final_combiner_model(prod_data, base_model, options_strike, fundamentals, ticker, linreg_length=14)
        print(f"FINAL PRICE PREDICTION: {prod_forecast:.2f}")

        # Retrieve the current (most recent) closing price.
        current_price = float(prod_data["Close"].iloc[-1])
        print(f"CURRENT PRICE: {current_price:.2f}")

        # Investment decision: Recommend investing if forecast > current price.
        recommendation = "Yes, invest in this stock." if prod_forecast > current_price else "No, do not invest in this stock."

        # Calculate a confidence rating based on the relative difference.
        diff = prod_forecast - current_price
        relative_diff = abs(diff) / current_price * 100  # percentage difference
        if relative_diff < 1:
            qualitative = "Low"
        elif relative_diff < 5:
            qualitative = "Medium"
        else:
            qualitative = "High"
        confidence = f"{qualitative} ({relative_diff:.1f}%)"

        print(f"Investment Recommendation: {recommendation}")
        print(f"Confidence Rating: {confidence}")

if __name__ == "__main__":
    main()