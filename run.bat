@echo off
REM Run the Bitcoin/Crypto price prediction script with live API data
REM Default: Bitcoin with 2 years history and 4-year forecast
.\.venv\Scripts\python.exe main.py --crypto BTC-USD --history-years 2 --predict-years 4 --output bitcoin_price_prediction.png
pause
