# src/features.py
import numpy as np
import pandas as pd

def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).rolling(n).mean()
    down = (-delta.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / down)

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    c = df["Close"]
    f["ret_1"] = np.log(c).diff()
    f["ret_5"] = np.log(c).diff(5)
    f["ret_20"] = np.log(c).diff(20)
    f["ma_ratio"] = c / c.rolling(50).mean() - 1
    f["rsi"] = rsi(c) / 100
    f["vol_20"] = f["ret_1"].rolling(20).std()
    f["volume_chg"] = np.log(df["Volume"]).diff()
    # rolling z-score: uses ONLY the past 252 days
    mean = f.rolling(252).mean()
    std = f.rolling(252).std()
    f = ((f - mean) / std).clip(-5, 5)
    f["close"] = c  # kept for the env, not given to the agent
    return f.dropna()

def split(f: pd.DataFrame):
    train = f.loc[:"2019-12-31"]
    val = f.loc["2020-01-01":"2022-12-31"]
    test = f.loc["2023-01-01":]
    return train, val, test

if __name__ == "__main__":
    from src.data import load

    f = build_features(load())
    print(f.describe().T.round(2))
    print("missing values:", f.isna().sum().sum())

    train, val, test = split(f)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        print(f"{name}: {len(part)} days, {part.index[0].date()} -> {part.index[-1].date()}")