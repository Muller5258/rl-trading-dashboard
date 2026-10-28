from pathlib import Path
import pandas as pd
import yfinance as yf

DATA_DIR = Path("data")

def download(ticker: str = "SPY", start: str = "2010-01-01") -> pd.DataFrame:
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    DATA_DIR.mkdir(exist_ok=True)
    df.to_parquet(DATA_DIR / f"{ticker}.parquet")
    return df

def load(ticker: str = "SPY") -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / f"{ticker}.parquet")

if __name__ == "__main__":
    print(download().tail())