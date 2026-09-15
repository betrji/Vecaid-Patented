import matplotlib
matplotlib.use('Agg')  # Use the Agg backend to prevent GUI issues on macOS
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, render_template, request
import yfinance as yf
import datetime
import io, base64
import pandas as pd
from vecaid_premium import predict_forecast, get_fundamentals, options_contracts_signal

app = Flask(__name__)

def generate_graph(ticker, forecast):
    # Download historical data for the ticker
    today = datetime.date.today().strftime('%Y-%m-%d')
    data = yf.download(ticker, start="2010-01-01", end=today)
    if data.empty:
        return ""
    plt.figure(figsize=(10,5))
    plt.plot(data.index, data['Close'], label="Close Price")
    # Plot forecast as a horizontal dashed line
    random_offset = np.random.uniform(-0.05 * forecast, 0.05 * forecast)
    plt.axhline(y=forecast + random_offset, color='r', linestyle='--', label=f"Forecast: {forecast + random_offset:.2f}")

    plt.title(f"{ticker.upper()} Historical Closing Prices")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close()  # Close the figure to free memory
    return f'<img src="data:image/png;base64,{encoded}" alt="Graph">'

def analyze_ticker(ticker):
    today = datetime.date.today().strftime('%Y-%m-%d')
    # Download historical data (ensure we get enough data)
    data = yf.download(ticker, start="2021-01-01", end=today)
    if data.empty or len(data) < 50:
        return "Not enough data", "N/A", ""
    
    # Get options signal and fundamentals
    try:
        options_signal = options_contracts_signal(ticker, std_multiplier=2)
    except Exception:
        options_signal = None
    options_strike = options_signal[2] if options_signal and options_signal[2] is not None else 0.0
    fundamentals = get_fundamentals(ticker)
    
    # Predict forecast using your model logic
    forecast = predict_forecast(data, options_strike, fundamentals, ticker, min_train_size=50)
    if forecast is None:
        return "Not enough data", "N/A", ""
    
    current_price = float(data["Close"].iloc[-1])
    decision = "Yes" if forecast > current_price else "No"
    
    # Calculate confidence rating based on percentage difference
    diff = forecast - current_price
    relative_diff = abs(diff) / current_price * 100
    if relative_diff < 1:
        qualitative = "Low"
    elif relative_diff < 5:
        qualitative = "Medium"
    else:
        qualitative = "High"
    confidence = f"{qualitative} ({relative_diff:.1f}%)"
    
    # Generate a historical price graph with forecast line
    graph_html = generate_graph(ticker, forecast)
    return decision, confidence, graph_html

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        ticker = request.form.get("ticker", "").strip()
        if ticker:
            try:
                decision, confidence, graph_html = analyze_ticker(ticker)
                return render_template("info.html", ticker=ticker.upper(), decision=decision, confidence=confidence, graph=graph_html)
            except Exception as e:
                error = f"Error processing ticker '{ticker}': {str(e)}"
                return render_template("index.html", error=error)
        else:
            error = "Please enter a ticker symbol."
            return render_template("index.html", error=error)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)