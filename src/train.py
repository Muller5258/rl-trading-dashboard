# src/train.py
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from src.data import load
from src.features import build_features, split
from src.env import TradingEnv

SEED = 42

def main():
    train, val, _ = split(build_features(load()))
    train_env = Monitor(TradingEnv(train, random_start=True))
    val_env = Monitor(TradingEnv(val, random_start=False))

    model = PPO("MlpPolicy", train_env, seed=SEED, verbose=1,
                learning_rate=3e-4, n_steps=2048, batch_size=64,
                gamma=0.99, ent_coef=0.01,
                tensorboard_log="runs/")

    eval_cb = EvalCallback(val_env, best_model_save_path="models/",
                           eval_freq=10_000, n_eval_episodes=1,
                           deterministic=True)
    model.learn(total_timesteps=300_000, callback=eval_cb)

if __name__ == "__main__":
    main()