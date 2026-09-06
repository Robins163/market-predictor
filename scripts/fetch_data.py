"""
Fetch raw daily OHLCV data for one ticker and save it to disk.

Run once. Every later step reads data/spy_raw.csv, never the network.
Why: yfinance data can change between fetches (Yahoo backfills corrections),
so a reviewer must be able to re-create the exact rows we trained on.
"""

import yfinance as yf
import pandas as pd
import sys

# Pinned constants so the dataset is frozen and reproducible.
TICKER = "SPY"          # SPDR S&P 500 ETF, liquid proxy for "the market"
START = "2015-01-01"    # ~10 years of daily data
END = "2025-08-31"      # fixed end date, not "today", so the CSV never changes
OUTPUT_PATH = "data/spy_raw.csv"


def fetch_ohlcv(ticker, start, end):
    """Download daily OHLCV for one ticker and return a clean DataFrame."""
    # auto_adjust=False keeps BOTH "Close" and "Adj Close" columns.
    # We need both: features use Adj Close, raw Close is kept for reference.
    df = yf.download(ticker, start=start, end=end, auto_adjust=False)

    # Newer yfinance versions return a two-level column header
    # like ("Close", "SPY"). Flatten it to a single level: "Close".
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df


def main():
    end = sys.argv[1] if len(sys.argv) > 1 else END
    df = fetch_ohlcv(TICKER, START, end)

    # Eyeball check before saving: shape, first rows, last rows.
    print("Shape (rows, columns):", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nLast 5 rows:")
    print(df.tail())

    # Save with the date index as a column so it survives the round trip.
    df.to_csv(OUTPUT_PATH)
    print("\nSaved to", OUTPUT_PATH)


if __name__ == "__main__":
    main()
