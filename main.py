import argparse
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def fetch_live_historical_data(symbol: str = "BTC-USD", years: int = 2) -> pd.DataFrame:
    """Fetch historical Bitcoin data from Yahoo Finance."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years)
    
    print(f"Fetching {years}-year historical data for {symbol}...")
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date)
    
    if df.empty:
        raise RuntimeError(
            f"Could not fetch data from Yahoo Finance for {symbol}. "
            "Check internet connection and symbol validity."
        )
    
    # Reset index to have Date as a column
    df = df.reset_index()
    df = df.rename(columns={"Date": "Date", "Close": "Close"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[["Date", "Close"]].dropna().sort_values("Date").reset_index(drop=True)
    
    return df


def prepare_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    df["Day_Number"] = np.arange(len(df)).reshape(-1, 1)
    X = df[["Day_Number"]]
    y = df["Close"]

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[float, float, np.ndarray]:
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return r2, rmse, y_pred


def fetch_live_price(symbol: str = "BTC-USD") -> tuple[float, str]:
    btc = yf.Ticker(symbol)
    history = btc.history(period="1d")
    if history.empty:
        raise RuntimeError("Live price data could not be fetched from Yahoo Finance.")

    live_price = float(history["Close"].iloc[-1])
    fetched_at = datetime.now().strftime("%d %b %Y, %I:%M %p")
    return live_price, fetched_at


def create_plot(
    df: pd.DataFrame,
    model: LinearRegression,
    future_dates: pd.DatetimeIndex,
    future_prices: np.ndarray,
    live_price: float,
    output_path: Path,
    r2: float,
    symbol: str = "Crypto",
):
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    ax.plot(
        df["Date"],
        df["Close"],
        color="#58a6ff",
        linewidth=1.2,
        label="Actual Price (Historical)",
    )
    ax.plot(
        df["Date"],
        model.predict(df[["Day_Number"]]),
        color="#f0883e",
        linewidth=2,
        linestyle="--",
        label=f"Linear Regression  R2={r2:.3f}",
    )
    ax.plot(
        future_dates,
        future_prices,
        color="#3fb950",
        linewidth=2.5,
        marker="o",
        markersize=4,
        label=f"Future Prediction",
    )
    ax.axhline(
        y=live_price,
        color="#ff6b6b",
        linestyle=":",
        linewidth=1.2,
        alpha=0.7,
        label=f"Live Price Now: ${live_price:,.0f}",
    )
    ax.axvline(
        x=df["Date"].iloc[-1],
        color="white",
        linestyle=":",
        linewidth=1,
        alpha=0.5,
    )

    ax.set_title(
        f"{symbol}: Live Price vs Future Prediction",
        color="#58a6ff",
        fontsize=14,
    )
    ax.set_xlabel("Date", color="#8b949e")
    ax.set_ylabel("Price (USD $)", color="#8b949e")
    ax.tick_params(colors="#8b949e")
    ax.grid(True, color="#21262d", alpha=0.5)

    for spine in ax.spines.values():
        spine.set_edgecolor("#21262d")

    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white", fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.xticks(rotation=30)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bitcoin live price regression + future prediction using real-time Yahoo Finance data."
    )
    parser.add_argument(
        "--crypto",
        type=str,
        default="BTC-USD",
        help="Cryptocurrency symbol (e.g., BTC-USD, ETH-USD).",
    )
    parser.add_argument(
        "--history-years",
        type=int,
        default=2,
        help="Number of years of historical data to fetch for training.",
    )
    parser.add_argument(
        "--predict-years",
        type=int,
        default=4,
        help="Number of years to forecast into the future.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("bitcoin_price_prediction.png"),
        help="Output path for the generated plot image.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Fetch live historical data from Yahoo Finance
    df = fetch_live_historical_data(symbol=args.crypto, years=args.history_years)

    print("Data fetched successfully from Yahoo Finance!")
    print(f"  Total rows: {len(df)}")
    print(f"  Date range: {df['Date'].min().date()} to {df['Date'].max().date()}\n")

    X_train, X_test, y_train, y_test = prepare_training_data(df)
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows:  {len(X_test)}\n")

    model = train_model(X_train, y_train)
    print("Model trained!")
    print(f"  Slope     : {model.coef_[0]:+.2f} dollars per day")
    print(f"  Intercept : ${model.intercept_:,.2f}")
    print(f"  Trend     : {'UPTREND' if model.coef_[0] > 0 else 'DOWNTREND'}\n")

    r2, rmse, _ = evaluate_model(model, X_test, y_test)
    print("=" * 38)
    print("      MODEL ACCURACY RESULTS")
    print("=" * 38)
    print(f"  R2 Score : {r2:.4f}   (1.0 = perfect)")
    print(f"  RMSE     : ${rmse:,.2f}  (avg dollar error)")
    print(f"  Current  : ${df['Close'].iloc[-1]:,.2f}")
    print("=" * 38)

    live_price, fetched_at = fetch_live_price(args.crypto)
    print("\n" + "=" * 45)
    print("       LIVE CRYPTO PRICE")
    print("=" * 45)
    print(f"  Symbol    : {args.crypto}")
    print(f"  Price     : ${live_price:,.2f}")
    print(f"  Fetched at: {fetched_at}")
    print("=" * 45)

    future_days = 365 * args.predict_years
    future_x = np.arange(len(df), len(df) + future_days).reshape(-1, 1)
    future_prices = model.predict(future_x)
    future_dates = pd.date_range(
        start=df["Date"].iloc[-1] + pd.Timedelta(days=1),
        periods=future_days,
        freq="D",
    )

    create_plot(
        df=df,
        model=model,
        future_dates=future_dates,
        future_prices=future_prices,
        live_price=live_price,
        output_path=args.output,
        r2=r2,
        symbol=args.crypto,
    )

    predicted = float(future_prices[-1])
    change_from_live = ((predicted - live_price) / live_price) * 100
    change_from_dataset = ((predicted - df["Close"].iloc[-1]) / df["Close"].iloc[-1]) * 100

    print("\n" + "=" * 50)
    print(f"    LIVE vs {args.predict_years}-YEAR PREDICTION SUMMARY")
    print("=" * 50)
    print(f"  Live Price Right Now : ${live_price:,.2f}")
    print(f"  Dataset Last Price   : ${df['Close'].iloc[-1]:,.2f}")
    print(f"  Predicted ({args.predict_years} yrs) : ${predicted:,.2f}")
    print(f"  Change from Live     : {change_from_live:+.2f}%")
    print(f"  Change from Dataset  : {change_from_dataset:+.2f}%")
    print("=" * 50)
    print("\nNote: Straight-line prediction only. Real crypto is much more volatile!")
    print(f"Saved chart to: {args.output}")


if __name__ == "__main__":
    main()
