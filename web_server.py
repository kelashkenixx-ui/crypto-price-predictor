import os
import json
from flask import Flask, render_template, request, jsonify
from pathlib import Path
import plotly
import plotly.graph_objs as go
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.linear_model import LinearRegression
from accuracy import AccuracyChecker

app = Flask(__name__, template_folder='templates', static_folder='static')

# Store current prediction data
current_prediction = None
accuracy_results = None


def fetch_live_historical_data(symbol: str = "BTC-USD", years: int = 2) -> pd.DataFrame:
    """Fetch historical Bitcoin data from Yahoo Finance."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years)
    
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date)
    
    if df.empty:
        raise RuntimeError(f"Could not fetch data from Yahoo Finance for {symbol}.")
    
    df = df.reset_index()
    df = df.rename(columns={"Date": "Date", "Close": "Close"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[["Date", "Close"]].dropna().sort_values("Date").reset_index(drop=True)
    
    return df


def generate_prediction_chart(symbol: str, history_years: int = 2, predict_years: int = 4):
    """Generate prediction chart using Plotly."""
    global current_prediction
    
    df = fetch_live_historical_data(symbol, history_years)
    
    # Prepare training data
    df["Day_Number"] = np.arange(len(df))
    X = df[["Day_Number"]]
    y = df["Close"]
    
    # Split data
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    r2 = max(-100, min(100, float(np.corrcoef(y_test, y_pred)[0, 1]**2 * 100)))
    rmse = float(np.sqrt(np.mean((y_test - y_pred)**2)))
    
    # Fetch live price
    live_ticker = yf.Ticker(symbol)
    live_data = live_ticker.history(period="1d")
    live_price = float(live_data["Close"].iloc[-1])
    
    # Generate future predictions
    future_days = 365 * predict_years
    future_x = np.arange(len(df), len(df) + future_days).reshape(-1, 1)
    future_prices = model.predict(future_x)
    future_dates = pd.date_range(start=df["Date"].iloc[-1] + pd.Timedelta(days=1), periods=future_days, freq="D")
    
    # Create figure
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Close"],
        mode="lines", name="Historical Price",
        line=dict(color="#58a6ff", width=2)
    ))
    
    # Regression fit
    fig.add_trace(go.Scatter(
        x=df["Date"], y=model.predict(df[["Day_Number"]]),
        mode="lines", name=f"Regression Fit (R²={r2:.1f}%)",
        line=dict(color="#f0883e", width=2, dash="dash")
    ))
    
    # Future predictions
    fig.add_trace(go.Scatter(
        x=future_dates, y=future_prices,
        mode="lines+markers", name=f"{predict_years}-Year Forecast",
        line=dict(color="#3fb950", width=2),
        marker=dict(size=4)
    ))
    
    # Live price line
    fig.add_hline(y=live_price, line_dash="dot", line_color="#ff6b6b", 
                  annotation_text=f"Live: ${live_price:,.0f}")
    
    # Layout
    fig.update_layout(
        title=f"{symbol} Price Prediction",
        xaxis_title="Date",
        yaxis_title="Price (USD $)",
        hovermode="x unified",
        template="plotly_dark",
        height=500,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    current_prediction = {
        "symbol": symbol,
        "live_price": live_price,
        "predicted_price": float(future_prices[-1]),
        "r2_score": r2,
        "rmse": rmse,
        "trend": "UPTREND" if model.coef_[0] > 0 else "DOWNTREND",
        "slope": float(model.coef_[0]),
    }
    
    return json.loads(plotly.io.to_json(fig))


def generate_accuracy_chart(accuracy_data):
    """Generate accuracy metrics chart."""
    if not accuracy_data or "test_results" not in accuracy_data:
        return None
    
    results = accuracy_data["test_results"]
    
    fig = go.Figure()
    
    # Accuracy bars
    fig.add_trace(go.Bar(
        x=[r["test_date"] for r in results],
        y=[r["accuracy_percentage"] for r in results],
        name="Accuracy %",
        marker=dict(color="#3fb950"),
        yaxis="y1"
    ))
    
    # Error line
    fig.add_trace(go.Scatter(
        x=[r["test_date"] for r in results],
        y=[r["error_percentage"] for r in results],
        name="Error %",
        mode="lines+markers",
        line=dict(color="#ff6b6b", width=2),
        yaxis="y2"
    ))
    
    # Layout with dual y-axes
    fig.update_layout(
        title="Backtest Accuracy Results",
        xaxis_title="Test Date",
        yaxis=dict(title="Accuracy (%)", side="left"),
        yaxis2=dict(title="Error (%)", overlaying="y", side="right"),
        hovermode="x unified",
        template="plotly_dark",
        height=500,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return json.loads(plotly.io.to_json(fig))


@app.route("/")
def index():
    """Serve the dashboard."""
    return render_template("dashboard.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    """Run prediction for a cryptocurrency."""
    try:
        data = request.json
        symbol = data.get("symbol", "BTC-USD")
        history_years = int(data.get("history_years", 2))
        predict_years = int(data.get("predict_years", 4))
        
        chart = generate_prediction_chart(symbol, history_years, predict_years)
        
        return jsonify({
            "status": "success",
            "chart": chart,
            "prediction": current_prediction
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/backtest", methods=["POST"])
def backtest():
    """Run backtesting and accuracy check."""
    try:
        data = request.json
        symbol = data.get("symbol", "BTC-USD")
        num_tests = int(data.get("num_tests", 10))
        days_ahead = int(data.get("days_ahead", 30))
        years = int(data.get("years", 2))
        
        print(f"Starting backtest for {symbol}...")
        checker = AccuracyChecker(symbol)
        results = checker.run_multiple_tests(num_tests=num_tests, days_ahead=days_ahead, years=years)
        
        summary = checker.get_summary_metrics()
        
        accuracy_data = {
            "summary": summary,
            "test_results": results
        }
        
        chart = generate_accuracy_chart(accuracy_data)
        
        return jsonify({
            "status": "success",
            "summary": summary,
            "chart": chart,
            "test_results": results
        })
    except Exception as e:
        print(f"Backtest error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/live-price", methods=["GET"])
def live_price():
    """Get current live price."""
    try:
        symbol = request.args.get("symbol", "BTC-USD")
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1d")
        
        return jsonify({
            "symbol": symbol,
            "price": float(data["Close"].iloc[-1]),
            "change": float(data["Close"].pct_change().iloc[-1] * 100),
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    print("Starting Bitcoin Prediction Dashboard...")
    print("Visit: http://localhost:5000")
    app.run(debug=True, port=5000)
