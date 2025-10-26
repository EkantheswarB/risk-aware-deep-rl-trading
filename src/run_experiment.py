import sys
import os
PROJECT_ROOT = "/content/risk-aware-deep-rl-trading"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

from src.utils_data import download_price_data, load_prices
from src.feature_engineering import compute_features
from src.trading_env import TradingEnv
from src.build_agent import build_ppo_agent
from src.train_eval import train_agent, rollout_policy, buy_and_hold_baseline
from src.metrics import summarize, plot_equity, save_results_json

def run_experiment():

    Path("experiments/plots").mkdir(parents=True, exist_ok=True)

    # 1. Download data
    tickers = ["BTC-USD", "ETH-USD", "SPY"]
    download_price_data(
        tickers=tickers,
        start="2020-01-01",
        end="2024-12-31",
        out_csv="data/raw/market_prices.csv"
    )
    prices = load_prices("data/raw/market_prices.csv")

    # 2. Feature engineering
    features_df, btc_ret_series = compute_features(prices, window_vol=10)

    # We'll split indices into train and test by time, not random shuffle.
    # Let's define train up to 2023-12-31 and test in 2024.
    split_date = "2024-01-01"
    all_idx = features_df.index.sort_values()

    train_end_idx = np.where(all_idx < split_date)[0][-1]  # last index before 2024
    test_start_idx = train_end_idx + 1

    # 3. Build train environment
    env_train = TradingEnv(
        features_csv="data/processed/features.csv",
        btc_ret_csv="data/processed/btc_ret.csv",
        window_size=30,
        transaction_cost=0.0005,
        risk_penalty=0.1,
        start_index=30,               # warmup for window
        end_index=train_end_idx,
    )

    # 4. PPO agent
    model = build_ppo_agent(env_train)

    # 5. Train agent
    model = train_agent(model, timesteps=100_000)

    # 6. Test environment (out-of-sample 2024)
    env_test = TradingEnv(
        features_csv="data/processed/features.csv",
        btc_ret_csv="data/processed/btc_ret.csv",
        window_size=30,
        transaction_cost=0.0005,
        risk_penalty=0.1,
        start_index=test_start_idx,
        end_index=None,               # go to end
    )

    rl_nav, rl_rewards = rollout_policy(env_test, model, deterministic=True)
    bh_nav = buy_and_hold_baseline(env_test)

    # 7. Metrics
    rl_summary = summarize(rl_nav, label="RL")
    bh_summary = summarize(bh_nav, label="BuyHoldBTC")

    results_all = {
        "rl_summary": rl_summary,
        "buyhold_summary": bh_summary,
    }

    # 8. Plots
    plot_equity(rl_nav, bh_nav, out_prefix="experiments/plots")

    # 9. Save result json
    save_results_json(results_all, out_path="experiments/results_test.json")

    # 10. Print to console (so you can paste into your paper)
    print("\n=== RL STRATEGY ===")
    for k,v in rl_summary.items():
        if isinstance(v, (int, float)):
            print(f"{k:20s}: {v:.4f}")
        else:
            print(f"{k:20s}: {v}")


    print("\n=== BUY & HOLD BTC ===")
    for k,v in bh_summary.items():
        if isinstance(v, (int, float)):
            print(f"{k:20s}: {v:.4f}")
        else:
            print(f"{k:20s}: {v}")


    print("\nResults saved to experiments/results_test.json")
    print("Plots saved under experiments/plots/")

if __name__ == "__main__":
    run_experiment()
