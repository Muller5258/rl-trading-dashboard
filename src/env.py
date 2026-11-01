# src/env.py
import gymnasium as gym
import numpy as np
from gymnasium import spaces

class TradingEnv(gym.Env):
    def __init__(self, df, window=30, episode_len=252,
                 random_start=True, cost=0.001):
        super().__init__()
        self.feat = df.drop(columns="close").values.astype(np.float32)
        self.close = df["close"].values
        self.dates = df.index
        self.window, self.episode_len = window, episode_len
        self.random_start, self.cost = random_start, cost
        n_obs = window * self.feat.shape[1] + 1
        self.observation_space = spaces.Box(-np.inf, np.inf, (n_obs,), np.float32)
        self.action_space = spaces.Discrete(2)

    def _obs(self):
        w = self.feat[self.t - self.window + 1 : self.t + 1].ravel()
        return np.append(w, self.position).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        last = len(self.close) - 1
        if self.random_start:
            self.t = int(self.np_random.integers(self.window - 1,
                                                 last - self.episode_len))
            self.end = self.t + self.episode_len
        else:
            self.t, self.end = self.window - 1, last
        self.position, self.equity = 0, 1.0
        return self._obs(), {}

    def step(self, action):
        trade = abs(int(action) - self.position)
        self.position = int(action)
        market = np.log(self.close[self.t + 1] / self.close[self.t])
        reward = self.position * market - self.cost * trade
        self.equity *= np.exp(reward)
        self.t += 1
        done = self.t >= self.end
        info = {"date": self.dates[self.t], "position": self.position,
                "equity": self.equity}
        return self._obs(), float(reward), done, False, info

if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env
    from src.data import load
    from src.features import build_features, split

    train, _, _ = split(build_features(load()))
    env = TradingEnv(train)

    check_env(env)
    print("check_env passed")

    obs, _ = env.reset(seed=0)
    print("observation size:", obs.shape)

    done, steps, trades, last = False, 0, 0, 0
    while not done:
        obs, reward, done, _, info = env.step(env.action_space.sample())
        trades += info["position"] != last
        last = info["position"]
        steps += 1
    print(f"random agent: {steps} days, {trades} trades, final equity {env.equity:.3f}")