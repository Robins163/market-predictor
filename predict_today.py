"""
Print the model's prediction for the most recent available trading day.

Usage:
python predict_today.py            # uses the cached data/spy_raw.csv
python predict_today.py --refresh  # re-fetches data first, then predicts

The features are built with the SAME build_features() used in training,
so training and inference cannot drift apart.
"""

import argparse
from datetime import date
import subprocess
import sys

import joblib

from src.features import INPUT_PATH, build_features, load_raw

MODEL_PATH = "models/gbm.joblib"


def refresh_data():
    """Re-run the fetch script so the CSV has the latest rows."""
    today = date.today().isoformat()
    subprocess.run(
        [sys.executable, "scripts/fetch_data.py", today], check=True)


def predict_latest(model_path, raw_path):
    artifact = joblib.load(model_path)
    scaler = artifact["scaler"]
    model = artifact["model"]
    features = artifact["features"]

    df = build_features(load_raw(raw_path), drop_last=False)
    latest = df.iloc[[-1]]              # double bracket keeps it a DataFrame
    X = scaler.transform(latest[features])
    prob_up = model.predict_proba(X)[0][1]
    raw_last = load_raw(raw_path).index[-1]
    if latest.index[0] != raw_last:
        raise RuntimeError(
            f"Newest row {raw_last.date()} was dropped; features have NaN")

    return latest.index[0].date(), prob_up


def main():
    parser = argparse.ArgumentParser(
        description="Predict next-day SPY direction.")
    parser.add_argument("--refresh", action="store_true",
                        help="re-fetch data first")
    args = parser.parse_args()

    if args.refresh:
        refresh_data()

    as_of, prob_up = predict_latest(MODEL_PATH, INPUT_PATH)
    direction = "UP" if prob_up >= 0.5 else "DOWN"

    print(f"As of close on {as_of}")
    print(f"Probability next close is higher: {prob_up:.3f}")
    print(f"Prediction: {direction}")


if __name__ == "__main__":
    main()
