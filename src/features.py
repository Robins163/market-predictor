"""
Build model features and the prediction target from data/spy_raw.csv.

Leakage rule: every feature on row t uses only rows <= t.
The ONLY forward-looking line in this file is the shift(-1) in add_target(),
because the target IS tomorrow's answer.
"""

import pandas as pd

INPUT_PATH = "data/spy_raw.csv"
OUTPUT_PATH = "data/spy_features.csv"

# All price-based features use the adjusted close (see README section 2).
PRICE = "Adj Close"

FEATURE_COLS = [
    "ret_1", "ret_5", "ret_20",
    "vol_20",
    "sma_ratio_20", "sma_ratio_50",
    "rsi_14",
    "macd",
    "vol_z_20",
    "range_pct",
]


def load_raw(path):
    """Read the frozen CSV with Date as the index."""
    return pd.read_csv(path, index_col="Date", parse_dates=True)


def add_return_features(df):
    """Lagged returns over 1, 5, 20 trading days. Scale-free momentum."""
    df["ret_1"] = df[PRICE].pct_change(1)
    df["ret_5"] = df[PRICE].pct_change(5)
    df["ret_20"] = df[PRICE].pct_change(20)
    return df


def add_volatility(df):
    """Rolling 20-day std of daily returns. How choppy the last month was."""
    df["vol_20"] = df["ret_1"].rolling(20).std()
    return df


def add_sma_ratios(df):
    """Distance from trend as a ratio, so it means the same at any price level."""
    sma_20 = df[PRICE].rolling(20).mean()
    sma_50 = df[PRICE].rolling(50).mean()
    df["sma_ratio_20"] = df[PRICE] / sma_20 - 1
    df["sma_ratio_50"] = df[PRICE] / sma_50 - 1
    return df


def add_rsi(df, window=14):
    """Relative Strength Index: average gain vs average loss, mapped to 0-100."""
    change = df[PRICE].diff()
    gain = change.clip(lower=0)      # keep positive moves, zero out negatives
    loss = -change.clip(upper=0)     # keep negative moves as positive numbers
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    df["rsi_14"] = 100 - 100 / (1 + rs)
    return df


def add_macd(df):
    """Fast EMA minus slow EMA, divided by price to make it scale-free."""
    ema_12 = df[PRICE].ewm(span=12, adjust=False).mean()
    ema_26 = df[PRICE].ewm(span=26, adjust=False).mean()
    df["macd"] = (ema_12 - ema_26) / df[PRICE]
    return df


def add_volume_zscore(df, window=20):
    """How unusual today's volume is vs the last 20 days, in std units."""
    vol_mean = df["Volume"].rolling(window).mean()
    vol_std = df["Volume"].rolling(window).std()
    df["vol_z_20"] = (df["Volume"] - vol_mean) / vol_std
    return df


def add_range_pct(df):
    """Intraday range as a fraction of close. Raw columns are fine here:
    High, Low, Close are all same-day, so the adjustment factor cancels."""
    df["range_pct"] = (df["High"] - df["Low"]) / df["Close"]
    return df


def add_target(df):
    """Target for row t: 1 if adj close at t+1 > adj close at t, else 0.

    shift(-1) pulls tomorrow's return up into today's row.
    This is the ONLY forward look in the pipeline.
    next_ret is kept for the backtest in Phase 5.
    """
    df["next_ret"] = df["ret_1"].shift(-1)
    df["target"] = (df["next_ret"] > 0).astype(int)
    return df


def build_features(df, drop_last=True):
    """Run every feature step in order, then drop rows that cannot be used."""
    df = add_return_features(df)
    df = add_volatility(df)
    df = add_sma_ratios(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_volume_zscore(df)
    df = add_range_pct(df)
    df = add_target(df)

    required = FEATURE_COLS + ["next_ret"] if drop_last else FEATURE_COLS
    df = df.dropna(subset=required)

    return df


def main():
    raw = load_raw(INPUT_PATH)
    df = build_features(raw)

    print("Rows after dropping warm-up and last row:", len(df))
    print("Share of up days (class prior):", round(df["target"].mean(), 4))
    print("\nLast 5 rows of features + target:")
    print(df[FEATURE_COLS + ["target"]].tail())

    df.to_csv(OUTPUT_PATH)
    print("\nSaved to", OUTPUT_PATH)


if __name__ == "__main__":
    main()
