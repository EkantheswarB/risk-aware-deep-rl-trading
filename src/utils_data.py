import yfinance as yf
import pandas as pd
from pathlib import Path

def download_price_data(tickers, start, end, out_csv):
    """
    Download daily close prices for a list of tickers and save to CSV.
    tickers: list like ["BTC-USD", "ETH-USD", "SPY"]
    """
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    data = yf.download(tickers, start=start, end=end)["Close"]
    data.to_csv(out_csv)
    return data

def load_prices(csv_path):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    return df.sort_index()
