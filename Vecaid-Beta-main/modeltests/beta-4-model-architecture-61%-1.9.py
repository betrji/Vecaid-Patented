import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from xgboost import XGBRegressor
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import BDay
import math
from scipy.stats import norm
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, Dropout, LSTM, GRU, Conv1D, GlobalAveragePooling1D,
                                     Reshape, Concatenate, MultiHeadAttention, Lambda)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, BaseCrossValidator

# Hardcoded weights for fundamental and options signals.
FUND_WEIGHT = 0.3
OPTIONS_WEIGHT = 0.1

# Example sigma value from historical residuals.
SIGMA = 2.5

###############################################################################
# YEAR-MONTH TIME SERIES SPLIT (2Y Train, 2M Test)
###############################################################################
class YearMonthTimeSeriesSplit(BaseCrossValidator):
    def __init__(self, n_splits=4, year_len=504, month_len=42):
        self.n_splits = n_splits
        self.year_len = year_len
        self.month_len = month_len
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        for i in range(1, self.n_splits + 1):
            train_end = i * self.year_len
            test_start = train_end
            test_end = train_end + self.month_len
            if test_end > n_samples:
                break
            yield np.arange(0, train_end), np.arange(test_start, test_end)

###############################################################################
# DATA HELPER FUNCTIONS
###############################################################################
def get_multi_timeframe_data(ticker, start_date, end_date, intervals=["1d", "1wk"]):
    dfs = []
    for interval in intervals:
        df = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
        if df.empty:
            print(f"No {interval} data found for {ticker}.")
            continue
        if isinstance(df.columns, pd.MultiIndex):
            if ticker in df.columns.levels[0]:
                df = df[ticker]
            else:
                df = df.xs(ticker, level=1, axis=1)
        suffix = "_" + interval.replace("1", "")
        df.columns = [c + suffix for c in df.columns]
        df.reset_index(inplace=True)
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    merged_df = dfs[0]
    for i in range(1, len(dfs)):
        merged_df = pd.merge(merged_df, dfs[i], on="Date", how="outer")
    merged_df.set_index("Date", inplace=True)
    merged_df.sort_index(inplace=True)
    return merged_df

def get_intraday_features(ticker, days=60, interval="5m"):
    intraday = yf.download(ticker, period=f"{days}d", interval=interval, progress=False)
    if intraday.empty:
        print(f"No intraday {interval} data found for {ticker}.")
        return pd.DataFrame()
    if isinstance(intraday.columns, pd.MultiIndex):
        if ticker in intraday.columns.levels[0]:
            intraday = intraday[ticker]
        else:
            intraday = intraday.xs(ticker, level=1, axis=1)
    intraday['Date'] = intraday.index.date
    grouped = intraday.groupby('Date')
    daily_agg = pd.DataFrame()
    daily_agg['intraday_high'] = grouped['High'].max()
    daily_agg['intraday_low'] = grouped['Low'].min()
    daily_agg['intraday_range'] = daily_agg['intraday_high'] - daily_agg['intraday_low']
    daily_agg['intraday_vol'] = grouped['Volume'].sum()
    daily_agg['intraday_avg_vol'] = grouped['Volume'].mean()
    daily_agg.index = pd.to_datetime(daily_agg.index)
    daily_agg.sort_index(inplace=True)
    return daily_agg

###############################################################################
# SENTIMENT FUNCTION
###############################################################################
def get_sentiment(ticker):
    url = f"https://finance.yahoo.com/quote/{ticker}/news?p={ticker}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return 0
        soup = BeautifulSoup(response.text, 'html.parser')
        headlines = soup.find_all('h3')
        analyzer = SentimentIntensityAnalyzer()
        scores = [analyzer.polarity_scores(h.get_text())['compound'] for h in headlines]
        return sum(scores) / len(scores) if scores else 0
    except Exception as e:
        print("Error during sentiment scraping:", e)
        return 0

###############################################################################
# SUPPORTING FUNCTIONS (e.g., ATR)
###############################################################################
def compute_atr(data, period=14):
    tr = []
    for i in range(1, len(data)):
        high = data['High'].iloc[i]
        low = data['Low'].iloc[i]
        prev_close = data['Close'].iloc[i-1]
        tr.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
    atr = pd.Series(tr).rolling(window=period).mean()
    atr = pd.concat([pd.Series([np.nan]), atr], ignore_index=True)
    return atr

###############################################################################
# (VPIN FUNCTIONS REMOVED)
###############################################################################

###############################################################################
# ADDITIONAL MISSING FUNCTIONS
###############################################################################
def stochastic_oscillator(data, k_period=14, d_period=3):
    lowest_low = data['Low'].rolling(window=k_period).min()
    highest_high = data['High'].rolling(window=k_period).max()
    data['%K'] = 100 * ((data['Close'] - lowest_low) / (highest_high - lowest_low))
    data['%D'] = data['%K'].rolling(window=d_period).mean()
    return data

def rate_of_change(data, period=14):
    data['ROC'] = data['Close'].pct_change(periods=period) * 100
    return data

def bollinger_band_width(data):
    data['BB_width'] = (data['upper_band'] - data['lower_band']) / data['SMA20']
    return data

###############################################################################
# TECHNICAL INDICATORS & FEATURES
###############################################################################
def adx(data, n=14):
    data['prev_close'] = data['Close'].shift(1)
    data['TR'] = data.apply(lambda row: max(row['High'] - row['Low'],
                                             abs(row['High'] - row['prev_close']) if not np.isnan(row['prev_close']) else 0,
                                             abs(row['Low'] - row['prev_close']) if not np.isnan(row['prev_close']) else 0), axis=1)
    data['DM_plus'] = np.where((data['High'] - data['High'].shift(1)) > (data['Low'].shift(1) - data['Low']),
                               np.maximum(data['High'] - data['High'].shift(1), 0), 0)
    data['DM_minus'] = np.where((data['Low'].shift(1) - data['Low']) > (data['High'] - data['High'].shift(1)),
                                np.maximum(data['Low'].shift(1) - data['Low'], 0), 0)
    TR_sum = data['TR'].rolling(window=n, min_periods=1).sum()
    DM_plus_sum = data['DM_plus'].rolling(window=n, min_periods=1).sum()
    DM_minus_sum = data['DM_minus'].rolling(window=n, min_periods=1).sum()
    data['DI_plus'] = 100 * (DM_plus_sum / TR_sum)
    data['DI_minus'] = 100 * (DM_minus_sum / TR_sum)
    data['DX'] = 100 * np.abs(data['DI_plus'] - data['DI_minus']) / (data['DI_plus'] + data['DI_minus'] + 1e-10)
    data['ADX'] = data['DX'].rolling(window=n, min_periods=1).mean()
    data.drop(['prev_close', 'DM_plus', 'DM_minus'], axis=1, inplace=True)
    return data

def get_technical_indicators(data):
    data['SMA20'] = data['Close'].rolling(window=20).mean()
    data['std'] = data['Close'].rolling(window=20).std()
    data['upper_band'] = data['SMA20'] + 2 * data['std']
    data['lower_band'] = data['SMA20'] - 2 * data['std']
    delta = data['Close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    RS = roll_up / roll_down
    data['RSI'] = 100 - (100 / (1 + RS))
    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['typical_price'] = (data['High'] + data['Low'] + data['Close']) / 3
    data['VWAP'] = (data['typical_price'] * data['Volume']).cumsum() / data['Volume'].cumsum()
    data['ROC_short'] = data['Close'].pct_change(periods=3) * 100
    data['StochRSI'] = 100 * ((data['RSI'] - data['RSI'].rolling(window=14).min()) /
                              (data['RSI'].rolling(window=14).max() - data['RSI'].rolling(window=14).min()))
    data = stochastic_oscillator(data)
    data = adx(data)
    data = rate_of_change(data)
    data = bollinger_band_width(data)
    data['CMF'] = chaikin_money_flow(data, period=20)
    data['CCI'] = commodity_channel_index(data, period=20)
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data['SMA200'] = data['Close'].rolling(window=200).mean()
    obv = [0]
    for i in range(1, len(data)):
        if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
            obv.append(obv[-1] + data['Volume'].iloc[i])
        elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
            obv.append(obv[-1] - data['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    data['OBV'] = obv
    data['vol_SMA20'] = data['Volume'].rolling(window=20).mean()
    data['Momentum'] = data['Close'].pct_change(periods=10) * 100
    data['EMA20'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['Donchian_High'] = data['High'].rolling(window=20).max()
    data['Donchian_Low'] = data['Low'].rolling(window=20).min()
    data['Williams_R'] = -100 * ((data['High'].rolling(window=14).max() - data['Close']) /
                                  (data['High'].rolling(window=14).max() - data['Low'].rolling(window=14).min()))
    data['ATR'] = compute_atr(data, period=14)
    return data

def chaikin_money_flow(data, period=20):
    denom = (data['High'] - data['Low']).replace(0, np.nan)
    ad = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / denom * data['Volume']
    cmf = ad.rolling(window=period, min_periods=1).sum() / data['Volume'].rolling(window=period, min_periods=1).sum()
    return cmf

def commodity_channel_index(data, period=20):
    tp = (data['High'] + data['Low'] + data['Close']) / 3
    sma_tp = tp.rolling(window=period).mean()
    md = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (tp - sma_tp) / (0.015 * md)
    return cci

###############################################################################
# STRATEGY FUNCTIONS
###############################################################################
def stochastic_strategy(latest):
    if latest['%K'] < 20 and latest['%D'] < 20:
        return 1
    elif latest['%K'] > 80 and latest['%D'] > 80:
        return -1
    return 0

def adx_strategy(latest, threshold=25):
    if latest['ADX'] > threshold:
        if latest['DI_plus'] > latest['DI_minus']:
            return 1
        elif latest['DI_minus'] > latest['DI_plus']:
            return -1
    return 0

def roc_strategy(latest, roc_threshold=5):
    if latest['ROC'] > roc_threshold:
        return 1
    elif latest['ROC'] < -roc_threshold:
        return -1
    return 0

def bb_strategy(latest):
    if (latest['Close'] - latest['lower_band']) / latest['Close'] < 0.02:
        return 1
    elif (latest['upper_band'] - latest['Close']) / latest['Close'] < 0.02:
        return -1
    return 0

def rsi_strategy(latest, lower=30, upper=70):
    if latest['RSI'] < lower:
        return 1
    elif latest['RSI'] > upper:
        return -1
    return 0

def macd_strategy(latest):
    return 1 if latest['MACD'] > latest['Signal_Line'] else -1

def sma_crossover_strategy(latest):
    return 1 if latest['SMA50'] > latest['SMA200'] else -1

def obv_strategy(latest):
    try:
        current_obv = latest['OBV']
        previous_obv = latest['OBV'] * 0.99
        if current_obv > previous_obv:
            return 1
        elif current_obv < previous_obv:
            return -1
        else:
            return 0
    except:
        return 0

def volume_strategy(latest):
    if latest['vol_SMA20'] > 0 and latest['Volume'] > 1.5 * latest['vol_SMA20']:
        if latest['Close'] > latest['SMA20']:
            return 1
        elif latest['Close'] < latest['SMA20']:
            return -1
    return 0

def momentum_strategy_strategy(latest, threshold=2):
    if pd.isna(latest['Momentum']):
        return 0
    if latest['Momentum'] > threshold:
        return 1
    elif latest['Momentum'] < -threshold:
        return -1
    return 0

def bollinger_b_percent_strategy(latest):
    band_range = latest['upper_band'] - latest['lower_band']
    if band_range == 0:
        return 0
    b_percent = (latest['Close'] - latest['lower_band']) / band_range
    if b_percent > 0.8:
        return -1
    elif b_percent < 0.2:
        return 1
    return 0

def keltner_channel_strategy(latest, data, atr_multiplier=1.5):
    middle = latest['EMA20']
    atr = latest['ATR']
    upper = middle + atr_multiplier * atr
    lower = middle - atr_multiplier * atr
    if latest['Close'] > upper:
        return 1
    elif latest['Close'] < lower:
        return -1
    return 0

def donchian_breakout_strategy(latest):
    if latest['Close'] >= latest['Donchian_High']:
        return 1
    elif latest['Close'] <= latest['Donchian_Low']:
        return -1
    return 0

def williams_r_strategy(latest):
    if latest['Williams_R'] < -80:
        return 1
    elif latest['Williams_R'] > -20:
        return -1
    return 0

def atr_breakout_strategy(latest):
    current_tr = latest['High'] - latest['Low']
    if latest['ATR'] == 0:
        return 0
    if current_tr > 1.2 * latest['ATR']:
        return 1
    elif current_tr < 0.8 * latest['ATR']:
        return -1
    return 0

def vwap_reversion_strategy(latest, threshold=0.01):
    deviation = (latest['Close'] - latest['VWAP']) / latest['VWAP']
    if deviation > threshold:
        return -1
    elif deviation < -threshold:
        return 1
    return 0

def ema_crossover_short_strategy(data):
    ema9 = data['Close'].ewm(span=9, adjust=False).mean().iloc[-1]
    ema21 = data['Close'].ewm(span=21, adjust=False).mean().iloc[-1]
    return 1 if ema9 > ema21 else -1

def pivot_point_strategy(prev_day):
    return (prev_day['High'] + prev_day['Low'] + prev_day['Close']) / 3

def volume_surge_strategy(latest, surge_factor=1.5):
    avg_volume = latest['vol_SMA20']
    if avg_volume == 0:
        return 0
    if latest['Volume'] > surge_factor * avg_volume:
        return 1
    return 0

def opening_range_breakout_strategy(data):
    try:
        prev_day = data.iloc[-2]
        pivot = (prev_day['High'] + prev_day['Low']) / 2
        current_close = data.iloc[-1]['Close']
        if current_close > pivot:
            return 1
        elif current_close < pivot:
            return -1
    except:
        return 0
    return 0

def squeeze_momentum_indicator(latest, threshold=0.1):
    if latest['BB_width'] < threshold:
        return 1 if latest['Momentum'] > 0 else -1
    return 0

def composite_indicator_signal(latest, data=None):
    signals = [
        stochastic_strategy(latest),
        adx_strategy(latest),
        roc_strategy(latest),
        bb_strategy(latest),
        rsi_strategy(latest),
        macd_strategy(latest),
        sma_crossover_strategy(latest),
        obv_strategy(latest),
        volume_strategy(latest),
        momentum_strategy_strategy(latest),
        vwap_reversion_strategy(latest),
        bollinger_b_percent_strategy(latest),
        williams_r_strategy(latest),
        atr_breakout_strategy(latest),
        squeeze_momentum_indicator(latest)
    ]
    if data is not None:
        signals.append(keltner_channel_strategy(latest, data))
        signals.append(donchian_breakout_strategy(latest))
        signals.append(ema_crossover_short_strategy(data))
        try:
            prev_day = data.iloc[-2]
            pivot = pivot_point_strategy(prev_day)
            signals.append(1 if latest['Close'] > pivot else -1)
        except:
            signals.append(0)
        signals.append(volume_surge_strategy(latest))
        signals.append(opening_range_breakout_strategy(data))
    else:
        signals.extend([0] * 6)
    return sum(signals)

def markov_chain_signal(data):
    returns = data['Close'].pct_change().dropna()
    if returns.empty:
        return 0
    states = (returns > 0).astype(int)
    transitions = pd.crosstab(states.shift(1), states)
    if transitions.sum().sum() > 0:
        markov_prob = transitions.div(transitions.sum(axis=1), axis=0)
        last_state = 1 if returns.iloc[-1] > 0 else 0
        prob_up = markov_prob.loc[last_state, 1] if (last_state in markov_prob.index and 1 in markov_prob.columns) else 0.5
        if prob_up > 0.65:
            return 1
        elif prob_up < 0.45:
            return -1
        else:
            return 0
    return 0

def mean_reversion_signal(latest):
    if pd.isna(latest.get('SMA20')) or pd.isna(latest.get('std')) or latest.get('std', 0) == 0:
        return 0
    z = (latest['Close'] - latest['SMA20']) / latest['std']
    if z > 1.0:
        return -1
    elif z < -1.0:
        return 1
    else:
        return 0

def volume_signal(data, period=5):
    obv = [0]
    for i in range(1, len(data)):
        if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
            obv.append(obv[-1] + data['Volume'].iloc[i])
        elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
            obv.append(obv[-1] - data['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    data = data.copy()
    data['OBV'] = obv
    if len(data) < period:
        return 0
    prev = data['OBV'].iloc[-period]
    if prev == 0:
        return 0
    return (data['OBV'].iloc[-1] - prev) / abs(prev) * 100

def volatility_signal(data, period=14):
    tr = []
    for i in range(1, len(data)):
        high = data['High'].iloc[i]
        low = data['Low'].iloc[i]
        prev_close = data['Close'].iloc[i-1]
        tr.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
    if len(tr) < period:
        return 0
    atr = np.mean(tr[-period:])
    current_price = data['Close'].iloc[-1]
    return atr / current_price * 100

def momentum_signal(data, period=10):
    if len(data) < period:
        return 0
    return (data['Close'].iloc[-1] - data['Close'].iloc[-period]) / data['Close'].iloc[-period] * 100

def options_signal(ticker, current_price):
    ticker_obj = yf.Ticker(ticker)
    expirations = ticker_obj.options
    if len(expirations) == 0:
        return 0
    expiry = expirations[0]
    try:
        opt_chain = ticker_obj.option_chain(expiry)
        calls = opt_chain.calls
        puts = opt_chain.puts
    except Exception as e:
        print("Error retrieving options chain:", e)
        return 0
    options_sig = 0
    if not calls.empty and not puts.empty:
        high_call = calls.loc[calls['volume'].idxmax()]
        high_put = puts.loc[puts['volume'].idxmax()]
        if high_call['volume'] >= high_put['volume']:
            options_sig = 1 if current_price < high_call['strike'] else -1
        else:
            options_sig = -1 if current_price > high_put['strike'] else 1
    elif not calls.empty:
        high_call = calls.loc[calls['volume'].idxmax()]
        options_sig = 1 if current_price < high_call['strike'] else -1
    elif not puts.empty:
        high_put = puts.loc[puts['volume'].idxmax()]
        options_sig = -1 if current_price > high_put['strike'] else 1
    return options_sig

def get_fundamental_signal(ticker):
    ticker_obj = yf.Ticker(ticker)
    fundamentals = ticker_obj.info
    signal = 0
    if 'trailingPE' in fundamentals and fundamentals['trailingPE'] is not None:
        pe = fundamentals['trailingPE']
        if pe < 15:
            signal += 1
        elif pe > 25:
            signal -= 1
    sentiment_score = get_sentiment(ticker)
    if sentiment_score > 0.2:
        signal += 1
    elif sentiment_score < -0.2:
        signal -= 1
    return signal

def technical_signal(latest):
    signal = 0
    if latest['RSI'] < 30:
        signal += 1
    elif latest['RSI'] > 70:
        signal -= 1
    if latest['MACD'] > latest['Signal_Line']:
        signal += 1
    else:
        signal -= 1
    if latest['Close'] > latest['SMA20']:
        signal += 1
    else:
        signal -= 1
    if latest['Close'] < latest['lower_band']:
        signal += 1
    elif latest['Close'] > latest['upper_band']:
        signal -= 1
    deviation = (latest['Close'] - latest['SMA20']) / latest['SMA20']
    if deviation > 0.05:
        signal -= 1
    elif deviation < -0.05:
        signal += 1
    signal += composite_indicator_signal(latest)
    return signal

###############################################################################
# BASIC PREDICTION FUNCTION (for example purposes)
###############################################################################
def predict_next_day_close(ticker):
    data = yf.download(ticker, period="60d", interval="1d")
    if data.empty:
        raise ValueError("No data found for ticker " + ticker)
    if isinstance(data.columns, pd.MultiIndex):
        if ticker in data.columns.levels[0]:
            data = data[ticker]
        else:
            data = data.xs(ticker, level=1, axis=1)
    data = get_technical_indicators(data)
    latest = data.iloc[-1]
    current_price = latest['Close']
    tech_sig = technical_signal(latest)
    markov_sig = markov_chain_signal(data)
    mean_rev_sig = mean_reversion_signal(latest)
    opt_sig = options_signal(ticker, current_price)
    fund_sig = get_fundamental_signal(ticker)
    total_signal = tech_sig + markov_sig + mean_rev_sig + opt_sig + fund_sig
    predicted_pct_move = total_signal * 0.5
    predicted_price = current_price * (1 + predicted_pct_move / 100)
    z_score = abs(predicted_pct_move) / SIGMA
    confidence = min(100, norm.cdf(z_score) * 100)
    direction = "higher" if predicted_price > current_price else "lower"
    diff = predicted_price - current_price
    predicted_date = data.index[-1] + BDay(1)
    return {
        "prediction": direction,
        "predicted_difference": diff,
        "confidence_percentage": confidence,
        "predicted_price": predicted_price,
        "predicted_date": predicted_date.strftime('%Y-%m-%d')
    }

###############################################################################
# ADDITIONAL MODEL BUILDERS
###############################################################################
def build_transformer_model(input_dim):
    inputs = Input(shape=(input_dim,), name='transformer_input')
    x = Lambda(lambda x: tf.expand_dims(x, axis=1))(inputs)
    attn_output = MultiHeadAttention(num_heads=2, key_dim=input_dim)(x, x)
    x = tf.keras.layers.Flatten()(attn_output)
    x = Dense(64, activation='relu')(x)
    output = Dense(1)(x)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def build_gru_model(input_dim):
    inputs = Input(shape=(input_dim,), name='gru_input')
    x = Reshape((1, input_dim))(inputs)
    x = GRU(64, return_sequences=False)(x)
    output = Dense(1)(x)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def build_cnn_lstm_model(input_dim):
    inputs = Input(shape=(input_dim,), name='cnn_lstm_input')
    x = Reshape((input_dim, 1))(inputs)
    x = Conv1D(32, kernel_size=3, activation='relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = LSTM(64, return_sequences=False)(x)
    output = Dense(1)(x)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def build_tf_ensemble_model(input_dim):
    inputs = Input(shape=(input_dim,), name='input_layer')
    b1 = Dense(128, activation='relu')(inputs)
    b1 = Dense(64, activation='relu')(b1)
    out1 = Dense(1, name='branch1_output')(b1)
    b2 = Reshape((1, input_dim))(inputs)
    b2 = LSTM(64, return_sequences=False)(b2)
    out2 = Dense(1, name='branch2_output')(b2)
    b3 = Dense(64, activation='relu')(inputs)
    b3 = Dense(32, activation='relu')(b3)
    out3 = Dense(1, name='branch3_output')(b3)
    b4 = Dense(32, activation='relu')(inputs)
    b4 = Dropout(0.2)(b4)
    out4 = Dense(1, name='branch4_output')(b4)
    b5 = Dense(64, activation='relu')(inputs)
    out5 = Dense(1, name='branch5_output')(b5)
    b6 = Dense(32, activation='relu')(inputs)
    b6 = Dense(16, activation='relu')(b6)
    out6 = Dense(1, name='branch6_output')(b6)
    b7 = Dense(64, activation='relu')(inputs)
    b7 = Dropout(0.2)(b7)
    b7 = Dense(32, activation='relu')(b7)
    out7 = Dense(1, name='branch7_output')(b7)
    b8 = Dense(128, activation='relu')(inputs)
    b8 = Dense(64, activation='relu')(b8)
    out8 = Dense(1, name='branch8_output')(b8)
    b9 = Dense(64, activation='relu')(inputs)
    b9 = Dense(32, activation='relu')(b9)
    b9 = Dropout(0.2)(b9)
    out9 = Dense(1, name='branch9_output')(b9)
    b10 = Dense(32, activation='relu')(inputs)
    b10 = Dense(32, activation='relu')(b10)
    out10 = Dense(1, name='branch10_output')(b10)
    merged = Concatenate(name='merged_branches')([out1, out2, out3, out4, out5, out6, out7, out8, out9, out10])
    final_output = Dense(1, name='final_output')(merged)
    model = Model(inputs=inputs, outputs=final_output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

###############################################################################
# TRAIN FINAL ENSEMBLE (TF ENSEMBLE + XGBOOST + New TF Models)
###############################################################################
def train_ml_model(ticker, window=60):
    print("Starting training of the ML model...")
    today = datetime.today()
    start_date = (today - relativedelta(years=7)).strftime('%Y-%m-%d')
    split_date = (today - relativedelta(months=3)).strftime('%Y-%m-%d')

    print("Downloading daily and weekly data...")
    daily_data = yf.download(ticker, start=start_date, end=split_date, interval="1d")
    weekly_data = get_multi_timeframe_data(ticker, start_date, split_date, intervals=["1wk"])
    if daily_data.empty:
        raise ValueError("No daily data found for ticker " + ticker)
    if isinstance(daily_data.columns, pd.MultiIndex):
        daily_data.columns = daily_data.columns.get_level_values(0)
    if isinstance(weekly_data.columns, pd.MultiIndex):
        weekly_data.columns = weekly_data.columns.get_level_values(0)

    weekly_data_ff = weekly_data.resample('1d').ffill()
    daily_data = daily_data.reset_index()
    weekly_data_ff = weekly_data_ff.reset_index()
    data_full = pd.merge(daily_data, weekly_data_ff, on="Date", how='left', suffixes=('', '_wk'))
    data_full.set_index("Date", inplace=True)
    data_full.ffill(inplace=True)
    data = get_technical_indicators(data_full)

    print("Downloading intraday 5-minute data...")
    intraday_agg = get_intraday_features(ticker, days=60, interval="5m")
    data.index = pd.to_datetime(data.index)
    data.sort_index(inplace=True)
    data_merged = pd.merge(data, intraday_agg, how="left", left_index=True, right_index=True)
    data_merged.fillna(0, inplace=True)

    features = []
    targets = []
    print("Creating feature vectors and target values...")
    for i in range(window, len(data_merged) - 1):
        window_data = data_merged.iloc[i-window:i]
        latest_window = window_data.iloc[-1]
        current_price = latest_window['Close']
        tech_sig = technical_signal(latest_window)
        cmf = latest_window.get('CMF', 0)
        cci = latest_window.get('CCI', 0)
        intraday_range = latest_window.get('intraday_range', 0)
        stochastic_sig = stochastic_strategy(latest_window)
        adx_sig = adx_strategy(latest_window)
        roc_sig = roc_strategy(latest_window)
        bb_sig = bb_strategy(latest_window)
        rsi_sig = rsi_strategy(latest_window)
        macd_sig = macd_strategy(latest_window)
        sma_cross_sig = sma_crossover_strategy(latest_window)
        obv_sig = obv_strategy(latest_window)
        volume_sig = volume_strategy(latest_window)
        momentum_sig = momentum_strategy_strategy(latest_window)
        # VPIN is completely removed from the features.
        feature_vector = [
            tech_sig, cmf, cci, intraday_range,
            stochastic_sig, adx_sig, roc_sig, bb_sig,
            rsi_sig, macd_sig, sma_cross_sig, obv_sig,
            volume_sig, momentum_sig
        ]
        features.append(feature_vector)
        next_day_close = data_merged.iloc[i+1]['Close']
        pct_change = ((next_day_close - current_price) / current_price) * 100
        targets.append(pct_change)
    X = np.array(features)
    y = np.array(targets)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    input_dim = X_scaled.shape[1]

    print("Building and training ensemble models...")
    tf_ensemble = build_tf_ensemble_model(input_dim)
    tf_ensemble.fit(X_scaled, y, epochs=200, batch_size=32, validation_split=0.2, verbose=1)

    transformer_model = build_transformer_model(input_dim)
    transformer_model.fit(X_scaled, y, epochs=200, batch_size=32, validation_split=0.2, verbose=1)

    gru_model = build_gru_model(input_dim)
    gru_model.fit(X_scaled, y, epochs=200, batch_size=32, validation_split=0.2, verbose=1)

    cnn_lstm_model = build_cnn_lstm_model(input_dim)
    cnn_lstm_model.fit(X_scaled, y, epochs=200, batch_size=32, validation_split=0.2, verbose=1)

    xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)
    xgb_model.fit(X_scaled, y)

    p_tf = tf_ensemble.predict(X_scaled)
    p_trans = transformer_model.predict(X_scaled)
    p_gru = gru_model.predict(X_scaled)
    p_cnn_lstm = cnn_lstm_model.predict(X_scaled)
    p_xgb = xgb_model.predict(X_scaled).reshape(-1, 1)
    combined_preds = np.concatenate([p_tf, p_trans, p_gru, p_cnn_lstm, p_xgb], axis=1)

    meta_model = XGBRegressor(objective='reg:squarederror', random_state=42)
    meta_model.fit(combined_preds, y)

    print("Training complete.")
    return {
        "tf_ensemble": tf_ensemble,
        "transformer_model": transformer_model,
        "gru_model": gru_model,
        "cnn_lstm_model": cnn_lstm_model,
        "xgb_model": xgb_model,
        "meta_model": meta_model,
        "scaler": scaler
    }

###############################################################################
# PREDICT FINAL (Hybrid Prediction from All Models with Meta XGBoost)
###############################################################################
def predict_next_day_close_ml(ticker, models):
    print("Starting ML-weighted realtime prediction...")
    data = yf.download(ticker, period="60d", interval="1d")
    if data.empty:
        raise ValueError("No daily data found for ticker " + ticker)
    if isinstance(data.columns, pd.MultiIndex):
        if ticker in data.columns.levels[0]:
            data = data[ticker]
        else:
            data = data.xs(ticker, level=1, axis=1)
    data = get_technical_indicators(data)
    intraday_agg = get_intraday_features(ticker, days=60, interval="5m")
    data.index = pd.to_datetime(data.index)
    data_merged = pd.merge(data, intraday_agg, how="left", left_index=True, right_index=True)
    data_merged.fillna(0, inplace=True)

    latest = data_merged.iloc[-1]
    current_price = latest['Close']
    tech_sig = technical_signal(latest)
    cmf = latest.get('CMF', 0)
    cci = latest.get('CCI', 0)
    intraday_range = latest.get('intraday_range', 0)
    stochastic_sig = stochastic_strategy(latest)
    adx_sig = adx_strategy(latest)
    roc_sig = roc_strategy(latest)
    bb_sig = bb_strategy(latest)
    rsi_sig = rsi_strategy(latest)
    macd_sig = macd_strategy(latest)
    sma_cross_sig = sma_crossover_strategy(latest)
    obv_sig = obv_strategy(latest)
    volume_sig = volume_strategy(latest)
    momentum_sig = momentum_strategy_strategy(latest)
    feature_vector = np.array([[tech_sig, cmf, cci, intraday_range,
                                 stochastic_sig, adx_sig, roc_sig, bb_sig,
                                 rsi_sig, macd_sig, sma_cross_sig, obv_sig,
                                 volume_sig, momentum_sig]])
    X_input = models["scaler"].transform(feature_vector)

    p_tf = models["tf_ensemble"].predict(X_input)
    p_trans = models["transformer_model"].predict(X_input)
    p_gru = models["gru_model"].predict(X_input)
    p_cnn_lstm = models["cnn_lstm_model"].predict(X_input)
    p_xgb = models["xgb_model"].predict(X_input).reshape(-1, 1)

    combined = np.concatenate([p_tf, p_trans, p_gru, p_cnn_lstm, p_xgb], axis=1)
    final_pred = models["meta_model"].predict(combined)[0]

    # Incorporate fundamental and options signals (VPIN has been removed)
    opt_sig = options_signal(ticker, current_price)
    fund_sig = get_fundamental_signal(ticker)
    final_pred_adj = final_pred + FUND_WEIGHT * fund_sig + OPTIONS_WEIGHT * opt_sig
    predicted_price = current_price * (1 + final_pred_adj / 100)
    z_score = abs(final_pred_adj) / SIGMA
    confidence = min(100, norm.cdf(z_score) * 100)
    direction = "higher" if predicted_price > current_price else "lower"
    predicted_date = data_merged.index[-1] + BDay(1)
    print("Realtime prediction complete.")
    return {
        "prediction": direction,
        "predicted_difference": predicted_price - current_price,
        "confidence_percentage": confidence,
        "predicted_price": predicted_price,
        "predicted_pct_move": final_pred_adj,
        "predicted_date": predicted_date.strftime('%Y-%m-%d')
    }

###############################################################################
# BACKTESTING FUNCTION (Uses the final hybrid ensemble prediction)
###############################################################################
def backtest_model_ml(ticker, models, window=60, test_period_days=63):
    print("Starting backtesting...")
    total_days = test_period_days + window
    data = yf.download(ticker, period=f"{total_days}d", interval="1d")
    if data.empty:
        raise ValueError("No data found for ticker " + ticker)
    if isinstance(data.columns, pd.MultiIndex):
        if ticker in data.columns.levels[0]:
            data = data[ticker]
        else:
            data = data.xs(ticker, level=1, axis=1)
    data = get_technical_indicators(data)
    intraday_agg = get_intraday_features(ticker, days=60, interval="5m")
    data.index = pd.to_datetime(data.index)
    data_merged = pd.merge(data, intraday_agg, how="left", left_index=True, right_index=True)
    data_merged.fillna(0, inplace=True)

    dates, predicted_moves, actual_moves = [], [], []
    cumulative_returns = [1]
    predicted_prices, actual_prices = [], []

    scaler = models["scaler"]
    tf_ensemble = models["tf_ensemble"]
    transformer_model = models["transformer_model"]
    gru_model = models["gru_model"]
    cnn_lstm_model = models["cnn_lstm_model"]
    xgb_model = models["xgb_model"]
    meta_model = models["meta_model"]

    print("Beginning backtest loop over", len(data_merged) - window - 1, "days...")
    for i in range(window, len(data_merged) - 1):
        window_data = data_merged.iloc[i-window:i]
        latest_window = window_data.iloc[-1]
        current_price = latest_window['Close']
        tech_sig = technical_signal(latest_window)
        cmf = latest_window.get('CMF', 0)
        cci = latest_window.get('CCI', 0)
        intraday_range = latest_window.get('intraday_range', 0)
        stochastic_sig = stochastic_strategy(latest_window)
        adx_sig = adx_strategy(latest_window)
        roc_sig = roc_strategy(latest_window)
        bb_sig = bb_strategy(latest_window)
        rsi_sig = rsi_strategy(latest_window)
        macd_sig = macd_strategy(latest_window)
        sma_cross_sig = sma_crossover_strategy(latest_window)
        obv_sig = obv_strategy(latest_window)
        volume_sig = volume_strategy(latest_window)
        momentum_sig = momentum_strategy_strategy(latest_window)
        feature_vector = np.array([[tech_sig, cmf, cci, intraday_range,
                                     stochastic_sig, adx_sig, roc_sig, bb_sig,
                                     rsi_sig, macd_sig, sma_cross_sig, obv_sig,
                                     volume_sig, momentum_sig]])
        X_input = scaler.transform(feature_vector)
        p_tf = tf_ensemble.predict(X_input)
        p_trans = transformer_model.predict(X_input)
        p_gru = gru_model.predict(X_input)
        p_cnn_lstm = cnn_lstm_model.predict(X_input)
        p_xgb = xgb_model.predict(X_input).reshape(-1, 1)
        combined = np.concatenate([p_tf, p_trans, p_gru, p_cnn_lstm, p_xgb], axis=1)
        predicted_pct_move_ml = meta_model.predict(combined)[0]
        final_pred = predicted_pct_move_ml
        predicted_moves.append(final_pred)
        pred_direction = 1 if final_pred > 0 else -1 if final_pred < 0 else 0
        pred_price = current_price * (1 + final_pred / 100)
        predicted_prices.append(pred_price)
        next_day_close = data_merged.iloc[i+1]['Close']
        actual_pct_move = ((next_day_close - current_price) / current_price) * 100
        actual_moves.append(actual_pct_move)
        actual_prices.append(next_day_close)
        dates.append(data_merged.index[i+1])
        trade_return = 0
        if pred_direction != 0:
            trade_return = (abs(actual_pct_move) / 100) if (pred_direction == (1 if actual_pct_move > 0 else -1)) else - (abs(actual_pct_move) / 100)
        cumulative_returns.append(cumulative_returns[-1] * (1 + trade_return))
        
        if (i - window) % 10 == 0:
            print(f"Day {data_merged.index[i+1]}: Current Price={current_price:.2f}, Predicted Move={final_pred:.2f}%")
    
    predictions_array = np.array(predicted_moves)
    actuals_array = np.array(actual_moves)
    accuracy = np.mean(np.sign(predictions_array) == np.sign(actuals_array)) * 100
    mae = np.mean(np.abs(predictions_array - actuals_array))
    mse = np.mean((predictions_array - actuals_array) ** 2)
    price_diffs = np.array(predicted_prices) - np.array(actual_prices)
    rmse_price = np.sqrt(np.mean(price_diffs ** 2))

    plt.figure(figsize=(14,6))
    plt.plot(dates, predictions_array, label="Predicted % Move")
    plt.plot(dates, actuals_array, label="Actual % Move")
    plt.xlabel("Date")
    plt.ylabel("Percent Move")
    plt.title(f"Backtest: Predicted vs. Actual % Moves for {ticker}")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(14,6))
    plt.plot(dates, cumulative_returns[1:], label="Cumulative Return")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.title(f"Cumulative Return from Trading Strategy on {ticker}")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(14,6))
    plt.plot(dates, predicted_prices, label="Predicted Price")
    plt.plot(dates, actual_prices, label="Actual Price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(f"Predicted vs. Actual Price for {ticker}")
    plt.legend()
    plt.grid(True)
    plt.show()

    stats = {
        "accuracy_percentage": accuracy,
        "mean_absolute_error": mae,
        "mean_squared_error": mse,
        "rmse_price": rmse_price,
        "final_cumulative_return": cumulative_returns[-1]
    }
    print("Backtesting Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    print("Backtesting complete.")
    return stats

###############################################################################
# EXAMPLE USAGE
###############################################################################
if __name__ == "__main__":
    ticker = input("Ticker for live prediction: ").upper().strip()
    try:
        result = predict_next_day_close(ticker)
        print("\nEqual-Weighted Prediction for", ticker)
        print("Prediction:", result["prediction"])
        print("Expected change: ${:.2f}".format(result["predicted_difference"]))
        print("Predicted next closing price: ${:.2f}".format(result["predicted_price"]))
        print("Predicted date for next close:", result["predicted_date"])
        print("Confidence: {}%".format(result["confidence_percentage"]))
    except Exception as e:
        print("Error during prediction:", e)

    print("\nTraining final hybrid ensemble ML model (Daily+Weekly data for technical indicators + 5-minute intraday features)...")
    ticker_ml = ticker
    try:
        models = train_ml_model(ticker_ml, window=60)
        print("Final hybrid ensemble model trained.")
    except Exception as e:
        print("Error during ML training:", e)
        models = None

    if models is not None:
        try:
            result_ml = predict_next_day_close_ml(ticker_ml, models)
            print("\nML-Weighted Realtime Prediction for", ticker_ml)
            print("Prediction:", result_ml["prediction"])
            print("Expected change: ${:.2f}".format(result_ml["predicted_difference"]))
            print("Predicted next closing price: ${:.2f}".format(result_ml["predicted_price"]))
            print("Predicted date for next close:", result_ml["predicted_date"])
            print("Confidence: {}%".format(result_ml["confidence_percentage"]))
            print("Predicted percent move: {:.2f}%".format(result_ml["predicted_pct_move"]))
        except Exception as e:
            print("Error during ML-weighted prediction:", e)

        print("\nBacktesting the final hybrid ensemble model on the last 3 months (using only ML prediction)...")
        try:
            backtest_stats = backtest_model_ml(ticker_ml, models, window=60, test_period_days=63)
        except Exception as e:
            print("Error during backtesting:", e)
