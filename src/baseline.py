# src/baseline.py
import numpy as np
import pandas as pd

COST = 0.001  # 0.1% per trade (commission + slippage)

def backtest(close: pd.Series, position: pd.Series) -> pd.Series:
    """position[t] = holding decided at end of day t, earns return of day t+1"""
    ret = close.pct_change().fillna(0)
    held = position.shift(1).fillna(0)
    trades = position.diff().fillna(position.iloc[0]).abs()  # entry counts as a trade
    strat = held * ret - trades * COST
    return (1 + strat).cumprod()

def metrics(equity: pd.Series) -> dict:
    daily = equity.pct_change().dropna()
    years = len(daily) / 252
    return {
        "total_return": equity.iloc[-1] - 1,
        "cagr": equity.iloc[-1] ** (1 / years) - 1,
        "sharpe": np.sqrt(252) * daily.mean() / daily.std(),
        "max_drawdown": (equity / equity.cummax() - 1).min(),
    }

def buy_and_hold(close):
    return pd.Series(1, index=close.index)

def ma_crossover(close):
    fast, slow = close.rolling(20).mean(), close.rolling(50).mean()
    return (fast > slow).astype(int)

if __name__ == "__main__":
    from src.data import load
    from src.features import build_features, split

    f = build_features(load())
    _, val, _ = split(f)

    results = {}
    for name, strategy in [("buy_hold", buy_and_hold), ("ma_cross", ma_crossover)]:
        # compute signals on full history, so the 50-day average is warm on day 1 of val
        position = strategy(f["close"]).loc[val.index]
        equity = backtest(val["close"], position)
        results[name] = metrics(equity)

    print(pd.DataFrame(results).T.round(3))