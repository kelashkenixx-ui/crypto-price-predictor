import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import json


class AccuracyChecker:
    """Backtesting and accuracy calculation for crypto predictions."""
    
    def __init__(self, symbol: str = "BTC-USD"):
        self.symbol = symbol
        self.test_results = []
    
    def fetch_historical_data(self, years: int = 3) -> pd.DataFrame:
        """Fetch historical data from Yahoo Finance."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * years)
        
        ticker = yf.Ticker(self.symbol)
        df = ticker.history(start=start_date, end=end_date)
        
        if df.empty:
            raise RuntimeError(f"Could not fetch data for {self.symbol}")
        
        df = df.reset_index()
        df = df.rename(columns={"Date": "Date", "Close": "Close"})
        df["Date"] = pd.to_datetime(df["Date"])
        df = df[["Date", "Close"]].dropna().sort_values("Date").reset_index(drop=True)
        
        return df
    
    def backtest_single(self, df: pd.DataFrame, test_date_idx: int, days_ahead: int = 30) -> dict:
        """
        Backtest prediction at a specific date.
        
        Args:
            df: Historical dataframe
            test_date_idx: Index in dataframe to use as 'today' for prediction
            days_ahead: How many days ahead to predict
            
        Returns:
            Dictionary with prediction accuracy metrics
        """
        if test_date_idx + days_ahead >= len(df):
            return None
        
        # Use data up to test_date_idx for training
        train_df = df.iloc[:test_date_idx].copy()
        train_df["Day_Number"] = np.arange(len(train_df))
        
        X_train = train_df[["Day_Number"]].values
        y_train = train_df["Close"].values
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make prediction for days_ahead
        future_x = np.array([[test_date_idx + days_ahead]])
        predicted_price = model.predict(future_x)[0]
        
        # Get actual price at that date
        actual_price = df.iloc[test_date_idx + days_ahead]["Close"]
        
        # Calculate accuracy metrics
        prediction_error = abs(predicted_price - actual_price)
        error_percentage = (prediction_error / actual_price) * 100
        accuracy_percentage = max(0, 100 - error_percentage)
        
        return {
            "test_date": df.iloc[test_date_idx]["Date"].strftime("%Y-%m-%d"),
            "test_date_idx": test_date_idx,
            "days_ahead": days_ahead,
            "predicted_price": float(predicted_price),
            "actual_price": float(actual_price),
            "error": float(prediction_error),
            "error_percentage": float(error_percentage),
            "accuracy_percentage": float(accuracy_percentage),
        }
    
    def run_multiple_tests(self, symbol: str = None, num_tests: int = 10, days_ahead: int = 30, years: int = 3):
        """
        Run multiple backtests at different historical points.
        
        Args:
            symbol: Cryptocurrency symbol to test
            num_tests: Number of tests to run
            days_ahead: Days ahead for each prediction
            years: Years of historical data to fetch
            
        Returns:
            List of test results with accuracy metrics
        """
        if symbol:
            self.symbol = symbol
        
        print(f"Fetching data for {self.symbol}...")
        df = self.fetch_historical_data(years=years)
        
        print(f"Running {num_tests} backtests with {days_ahead}-day predictions...")
        
        # Calculate evenly spaced test points
        step = len(df) // (num_tests + 1)
        test_results = []
        
        for i in range(1, num_tests + 1):
            test_idx = i * step
            if test_idx + days_ahead < len(df):
                result = self.backtest_single(df, test_idx, days_ahead)
                if result:
                    test_results.append(result)
                    print(f"  Test {len(test_results)}/{num_tests}: {result['test_date']} - Accuracy: {result['accuracy_percentage']:.2f}%")
        
        self.test_results = test_results
        return test_results
    
    def get_summary_metrics(self) -> dict:
        """Calculate summary accuracy metrics across all tests."""
        if not self.test_results:
            return None
        
        accuracies = [t["accuracy_percentage"] for t in self.test_results]
        errors = [t["error_percentage"] for t in self.test_results]
        
        return {
            "symbol": self.symbol,
            "total_tests": len(self.test_results),
            "average_accuracy": float(np.mean(accuracies)),
            "min_accuracy": float(np.min(accuracies)),
            "max_accuracy": float(np.max(accuracies)),
            "accuracy_std": float(np.std(accuracies)),
            "average_error_percentage": float(np.mean(errors)),
            "min_error_percentage": float(np.min(errors)),
            "max_error_percentage": float(np.max(errors)),
            "error_std": float(np.std(errors)),
            "prediction_range": {
                "min": float(np.min([t["predicted_price"] for t in self.test_results])),
                "max": float(np.max([t["predicted_price"] for t in self.test_results])),
                "avg": float(np.mean([t["predicted_price"] for t in self.test_results])),
            },
            "actual_range": {
                "min": float(np.min([t["actual_price"] for t in self.test_results])),
                "max": float(np.max([t["actual_price"] for t in self.test_results])),
                "avg": float(np.mean([t["actual_price"] for t in self.test_results])),
            },
        }
    
    def to_json(self) -> str:
        """Convert test results to JSON."""
        summary = self.get_summary_metrics()
        return json.dumps({
            "summary": summary,
            "test_results": self.test_results,
        }, indent=2)


if __name__ == "__main__":
    # Example usage
    checker = AccuracyChecker("BTC-USD")
    results = checker.run_multiple_tests(num_tests=10, days_ahead=30, years=2)
    
    print("\n" + "=" * 60)
    print("ACCURACY SUMMARY")
    print("=" * 60)
    
    summary = checker.get_summary_metrics()
    print(f"Symbol: {summary['symbol']}")
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Average Accuracy: {summary['average_accuracy']:.2f}%")
    print(f"Min Accuracy: {summary['min_accuracy']:.2f}%")
    print(f"Max Accuracy: {summary['max_accuracy']:.2f}%")
    print(f"Accuracy Std Dev: {summary['accuracy_std']:.2f}%")
    print(f"\nAverage Error: {summary['average_error_percentage']:.2f}%")
    print(f"Error Range: {summary['min_error_percentage']:.2f}% - {summary['max_error_percentage']:.2f}%")
