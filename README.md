# 🚀 Crypto Price Prediction Dashboard

A modern, AI-powered cryptocurrency price prediction dashboard with real-time backtesting and market analysis.

## Features

✨ **AI-Powered Predictions**
- Linear Regression model for price forecasting
- Real-time Bitcoin, Ethereum, Solana, Cardano, and Ripple predictions
- Configurable training data (1-5+ years)
- Flexible forecast periods

🔬 **Backtesting Lab**
- Historical accuracy validation
- Multiple test samples with configurable prediction windows
- Detailed performance metrics:
  - Accuracy percentage
  - Error analysis
  - Trend prediction accuracy
- Interactive results visualization

📊 **Modern Dashboard UI**
- Clean, professional financial interface
- Glassmorphism design with modern aesthetics
- Real-time price updates
- Interactive Plotly.js charts
- Responsive design for all devices

## Quick Start

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/crypto-price-predictor.git
   cd crypto-price-predictor
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the web server**
   ```bash
   .\.venv\Scripts\python.exe web_server.py  # Windows
   python web_server.py  # Linux/Mac
   ```

5. **Access the dashboard**
   Open your browser and navigate to: **http://localhost:5000**

## Usage

### AI Prediction Engine
1. Select a cryptocurrency (BTC, ETH, SOL, ADA, XRP)
2. Set training data period (years of historical data)
3. Set forecast period (years to predict ahead)
4. Click "Generate AI Prediction"
5. View results with:
   - Live current price
   - AI predicted price
   - Expected price change
   - Market trend (Bullish/Bearish)
   - Confidence level
   - Model accuracy (R² score)

### Backtesting Lab
1. Configure:
   - Test samples (number of historical tests)
   - Prediction window (days ahead to predict)
   - Historical data period
2. Click "Run Accuracy Tests"
3. View:
   - Average accuracy across all tests
   - Best and worst accuracy
   - Error margin analysis
   - Detailed test results table
   - Accuracy distribution chart

## Project Structure

```
crypto-price-predictor/
├── web_server.py          # Flask web server
├── accuracy.py            # Backtesting and validation
├── main.py                # CLI interface
├── requirements.txt       # Python dependencies
├── static/
│   └── style.css         # Modern dashboard styling
├── templates/
│   └── dashboard.html    # Main UI interface
├── ROADMAP.md            # Commercialization roadmap
└── README.md             # This file
```

## Technical Stack

- **Backend**: Flask 2.3+
- **ML/Data**: scikit-learn, pandas, numpy
- **Data Source**: yfinance (Yahoo Finance API)
- **Frontend**: HTML5, CSS3, Plotly.js, FontAwesome
- **Python**: 3.8+

## How It Works

### Prediction Model
1. Fetches historical cryptocurrency price data
2. Trains a Linear Regression model on past prices
3. Extrapolates future prices based on trends
4. Provides confidence metrics and accuracy estimates

### Backtesting Process
1. Selects historical "test dates"
2. Trains the model only on data before each test date
3. Makes predictions for N days ahead
4. Compares predictions with actual historical prices
5. Calculates accuracy metrics

## Performance Metrics

- **Average Accuracy**: 78.49% (across 10 historical backtests)
- **Best Accuracy**: 98.46%
- **Model R² Score**: Up to 28.93%
- **Error Margin**: Varies by market conditions

## Roadmap & Commercialization

See [ROADMAP.md](ROADMAP.md) for planned enhancements:
- Enhanced ML models (LSTM, XGBoost, Prophet)
- Multi-source market data integration
- User authentication and subscriptions
- Advanced portfolio analytics
- Production deployment infrastructure
- Compliance and risk management features

## Data Sources

- **Price Data**: Yahoo Finance API (`yfinance`)
- **Market Data**: Real-time cryptocurrency prices
- **Historical Data**: Up to 5+ years of past prices

## Disclaimer

⚠️ **This project is for educational purposes only.**
- Predictions are NOT financial advice
- Use at your own risk
- Always do your own research (DYOR)
- Past performance does not guarantee future results

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Author

**Your Name**
- GitHub: [@kelashkenixx-ui](https://github.com/kelashkenixx-ui)
- Email: kelashkenixx@gmail.com

---

## FAQ

**Q: How accurate are the predictions?**
A: The model achieves ~78% average accuracy in backtesting, but real-world performance varies based on market conditions.

**Q: Can I use this for real trading?**
A: This is an educational tool. Do not use for actual trading without thorough validation and risk management.

**Q: What data does it use?**
A: Real historical cryptocurrency prices from Yahoo Finance covering the past 1-5+ years.

**Q: Can I predict other cryptocurrencies?**
A: Yes! The dashboard supports Bitcoin, Ethereum, Solana, Cardano, and Ripple by default. You can add more in the code.

---

**Built with ❤️ using Python, Flask, and modern web technologies**
