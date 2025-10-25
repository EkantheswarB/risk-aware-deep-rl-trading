import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

def equity_to_returns(nav):
    nav = np.array(nav)
    rets = np.diff(nav) / nav[:-1]
    return rets

def sharpe_ratio(nav, eps=1e-9):
    rets = equity_to_returns(nav)
    mean_r = np.mean(rets)
    std_r = np.std(rets) + eps
    daily_sharpe = mean_r / std_r if std_r > 0 else 0.0
    return float(daily_sharpe * (252 ** 0.5))

def max_drawdown(nav):
    nav_arr = np.array(nav)
    peaks = np.maximum.accumulate(nav_arr)
    dd = (nav_arr - peaks) / peaks
    return float(dd.min())  # negative number

def annual_vol(nav):
    rets = equity_to_returns(nav)
    return float(np.std(rets) * (252 ** 0.5))

def summarize(nav, label="model"):
    summary = {
        "label": label,
        "final_nav": float(nav[-1]),
        "sharpe": sharpe_ratio(nav),
        "max_drawdown": max_drawdown(nav),
        "annual_volatility": annual_vol(nav),
    }
    return summary

def plot_equity(nav_rl, nav_bh, out_prefix="experiments/plots"):
    Path(out_prefix).mkdir(parents=True, exist_ok=True)

    # RL vs Buy-and-Hold NAV
    plt.figure(figsize=(8,4))
    plt.plot(nav_rl, label="RL strategy")
    plt.plot(nav_bh, label="Buy & Hold BTC")
    plt.title("Equity Curve")
    plt.xlabel("Step")
    plt.ylabel("Net Asset Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}/equity_curve_rl.png", dpi=300)
    plt.close()

    # RL drawdown
    nav_rl_arr = np.array(nav_rl)
    peaks = np.maximum.accumulate(nav_rl_arr)
    dd = (nav_rl_arr - peaks) / peaks

    plt.figure(figsize=(8,3))
    plt.plot(dd, label="RL drawdown")
    plt.title("RL Strategy Drawdown")
    plt.xlabel("Step")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}/drawdown_rl.png", dpi=300)
    plt.close()

def save_results_json(results_dict, out_path="experiments/results_test.json"):
    Path("experiments").mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results_dict, f, indent=2)
