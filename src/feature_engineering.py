import pandas as pd
import numpy as np
from pathlib import Path

def compute_features(price_df, window_vol=10):
    """
    Input:
        price_df: dataframe with columns e.g. ["BTC-USD","ETH-USD","SPY"]
    Output:
        features_df: dataframe with engineered features
        returns_df: dataframe of daily returns (for reward calc)
    Features include:
      - asset returns
      - rolling volatility estimate
      - cross-asset returns for context
    We'll pick one 'trade_asset' (BTC-USD) but include ETH/SPY in state.
    """
    df = price_df.copy()

    # % returns for each asset
    rets = df.pct_change().fillna(0.0)
    rets.columns = [c + "_ret" for c in rets.columns]

    # rolling volatility (std of returns) for BTC as risk proxy
    vol = (
        df["BTC-USD"]
        .pct_change()
        .rolling(window_vol)
        .std()
        .fillna(0.0)
    )
    features = pd.concat([rets, vol.rename("btc_roll_vol")], axis=1)

    # we will TRAIN agent on BTC only for PnL calculation,
    # but state will include ETH/SPY returns etc.
    btc_ret = features["BTC-USD_ret"].copy()

    # Save processed
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    features.to_csv("data/processed/features.csv")
    btc_ret.to_csv("data/processed/btc_ret.csv")

    return features, btc_ret

def make_state_matrix(features_df, window_size=30):
    """
    Turn tabular features into rolling windows for RL state.
    We'll slice this in the environment, so here we just store raw matrix.
    """
    arr = features_df.values.astype("float32")
    cols = list(features_df.columns)
    return arr, cols
