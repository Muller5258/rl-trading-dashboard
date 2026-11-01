# src/evaluate.py
import json
import pandas as pd
from stable_baselines3 import PPO

from src.data import load
from src.features import build_features, split
from src.env import TradingEnv
from src.baseline import backtest, metrics, buy_and_hold, ma_crossover

def run_agent(model, df) -> pd.DataFrame:
    env = TradingEnv(df, random_start=False)
    obs, _ = env.reset()
    rows, done = [], False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        rows.append(info)
    return pd.DataFrame(rows).set_index("date")

def main():
    _, _, test = split(build_features(load()))
    model = PPO.load("models/best_model.zip")
    agent = run_agent(model, test)

    close = test["close"].loc[agent.index[0]:]
    results = pd.DataFrame({
        "close": close,
        "agent": agent["equity"],
        "position": agent["position"],
        "buy_hold": backtest(close, buy_and_hold(close)),
        "ma_cross": backtest(close, ma_crossover(close)),
    }).dropna()
    results.to_parquet("data/results.parquet")

    summary = {name: metrics(results[name])
               for name in ["agent", "buy_hold", "ma_cross"]}
    json.dump(summary, open("data/metrics.json", "w"), indent=2, default=float)
    print(pd.DataFrame(summary).T.round(3))

if __name__ == "__main__":
    main()