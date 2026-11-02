# tests/test_core.py
import numpy as np
import pandas as pd
from src.features import build_features
from src.baseline import backtest
from src.env import TradingEnv

def fake_prices(n=600):
    idx = pd.bdate_range("2015-01-01", periods=n)
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    volume = rng.lognormal(14, 0.3, n)  # must vary, or its z-score is NaN
    return pd.DataFrame({"Open": close, "High": close, "Low": close,
                         "Close": close, "Volume": volume}, index=idx)

def test_features_do_not_use_future():
    df = fake_prices()
    full = build_features(df)
    cut = build_features(df.iloc[:-50])
    common = cut.index
    pd.testing.assert_frame_equal(full.loc[common], cut)

def test_buy_and_hold_matches_price():
    df = fake_prices()
    eq = backtest(df["Close"], pd.Series(1, index=df.index))
    expected = df["Close"].iloc[-1] / df["Close"].iloc[0]
    assert np.isclose(eq.iloc[-1], expected * (1 - 0.001))  # one entry fee

def test_env_runs_full_episode():
    env = TradingEnv(build_features(fake_prices()), random_start=False)
    obs, _ = env.reset()
    done = False
    while not done:
        obs, r, done, _, _ = env.step(env.action_space.sample())
    assert np.isfinite(env.equity)