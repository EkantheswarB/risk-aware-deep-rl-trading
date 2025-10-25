import numpy as np
from tqdm import trange

def train_agent(model, timesteps=100_000):
    model.learn(total_timesteps=timesteps, progress_bar=True)
    return model

def rollout_policy(env, model, deterministic=True):
    obs, info = env.reset()
    done = False
    nav_history = [info["nav"]]
    rewards = []

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, truncated, info = env.step(action)
        nav_history.append(info["nav"])
        rewards.append(reward)

    return nav_history, rewards

def buy_and_hold_baseline(env):
    """
    Baseline: always long BTC.
    No transaction cost, no penalty.
    """
    # same horizon as env
    nav = [1.0]
    # replicate env logic: we assume from start_index...end_index
    for idx in range(env.start_index, env.end_index):
        r_t = env.btc_ret[idx]
        nav.append(nav[-1] * (1.0 + r_t))
    return nav
