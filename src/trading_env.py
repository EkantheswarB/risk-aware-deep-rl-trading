import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

class TradingEnv(gym.Env):
    """
    Risk-aware trading environment for BTC using multi-asset context.
    State:
        window of past `window_size` rows of features
        (BTC return, ETH return, SPY return, rolling vol, etc.)
    Action:
        0 -> short (-1)
        1 -> flat (0)
        2 -> long (+1)
    Reward:
        position * btc_ret_t
        - transaction_cost_if_changed
        - risk_penalty * (btc_ret_t**2)
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        features_csv="data/processed/features.csv",
        btc_ret_csv="data/processed/btc_ret.csv",
        window_size=30,
        transaction_cost=0.0005,
        risk_penalty=0.1,
        start_index=0,
        end_index=None,
    ):
        super().__init__()

        # load data
        feat_df = pd.read_csv(features_csv, index_col=0)
        btc_ret_df = pd.read_csv(btc_ret_csv, index_col=0)

        # align
        feat_df = feat_df.sort_index()
        btc_ret_df = btc_ret_df.sort_index()

        # store arrays
        self.features = feat_df.values.astype(np.float32)
        self.btc_ret = btc_ret_df.values.astype(np.float32).reshape(-1)

        self.window_size = window_size
        self.transaction_cost = transaction_cost
        self.risk_penalty = risk_penalty

        # train/test slicing
        self.start_index = max(start_index, window_size)
        self.end_index = len(self.features) - 1 if end_index is None else end_index

        # spaces
        n_features = self.features.shape[1]
        # observation: window_size x n_features flattened
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size * n_features,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)

        # internal state
        self.current_step = None
        self.position = None
        self.prev_position = None
        self.nav_history = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = self.start_index
        self.position = 0
        self.prev_position = 0
        self.nav_history = [1.0]  # start NAV = 1.0

        obs = self._get_obs()
        info = {"nav": self.nav_history[-1], "step": self.current_step}
        return obs, info

    def _get_obs(self):
        start = self.current_step - self.window_size
        end = self.current_step
        window_feats = self.features[start:end]  # shape [window_size, n_features]
        return window_feats.flatten().astype(np.float32)

    def step(self, action):
        action_to_pos = {0: -1, 1: 0, 2: 1}
        self.prev_position = self.position
        self.position = action_to_pos[int(action) if not np.isscalar(action) else action]

        # today's BTC return
        r_t = self.btc_ret[self.current_step]

        pnl = self.position * r_t

        changed = (self.position != self.prev_position)
        cost = self.transaction_cost if changed else 0.0

        risk_pen = self.risk_penalty * (r_t ** 2)

        reward = pnl - cost - risk_pen

        # update NAV
        new_nav = self.nav_history[-1] * (1.0 + reward)
        self.nav_history.append(new_nav)

        # next step
        self.current_step += 1
        done = self.current_step >= self.end_index

        obs = self._get_obs()
        info = {
            "step": self.current_step,
            "btc_ret": float(r_t),
            "pnl": float(pnl),
            "cost": float(cost),
            "risk_pen": float(risk_pen),
            "reward": float(reward),
            "nav": float(new_nav),
        }

        return obs, reward, done, False, info

    def render(self):
        # optional live diagnostics later
        pass
