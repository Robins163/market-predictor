"""
Train baselines and two models with expanding-window walk-forward validation.

Fold = one test year. Train on every row before that year, test on that year.
Train never overlaps test. Scaler is fit inside each fold on train rows only.

Outputs:
data/walkforward_results.csv      per-fold metrics for every model
data/walkforward_predictions.csv  per-day predictions on test years (for backtest)
models/gbm.joblib                 final model trained on all data (for predict-today)
"""

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from src.features import FEATURE_COLS

INPUT_PATH = "data/spy_features.csv"
RESULTS_PATH = "data/walkforward_results.csv"
PREDICTIONS_PATH = "data/walkforward_predictions.csv"
MODEL_PATH = "models/gbm.joblib"

FIRST_TEST_YEAR = 2018   # first three years (2015-2017) are training only
LAST_TEST_YEAR = 2025


def load_features(path):
    return pd.read_csv(path, index_col="Date", parse_dates=True)


def make_models():
    """Return fresh, untrained models. Called once per fold so nothing
    learned in one fold can leak into the next."""
    return {
        "logreg": LogisticRegression(max_iter=1000),
        # Shallow trees + small learning rate: each tree is a small nudge.
        # Right setting for noisy data where deep trees would memorise noise.
        "gbm": GradientBoostingClassifier(
            n_estimators=100, max_depth=2, learning_rate=0.05, random_state=42
        ),
    }


def walk_forward_folds(df, first_test_year, last_test_year):
    """Expanding window: train = all rows before the test year, test = that year."""
    folds = []
    for year in range(first_test_year, last_test_year + 1):
        train = df[df.index.year < year]
        test = df[df.index.year == year]
        folds.append((year, train, test))
    return folds


def fit_and_predict(model, train, test):
    """Scale with TRAIN statistics only, fit on train, predict on test."""
    scaler = StandardScaler()
    # learns mean/std from train
    X_train = scaler.fit_transform(train[FEATURE_COLS])
    # applies them, does not re-learn
    X_test = scaler.transform(test[FEATURE_COLS])
    model.fit(X_train, train["target"])
    return model.predict(X_test)


def evaluate(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def run_walk_forward(df):
    """Loop over folds. Return per-fold metrics and per-day predictions."""
    metric_rows = []
    prediction_frames = []

    for year, train, test in walk_forward_folds(df, FIRST_TEST_YEAR, LAST_TEST_YEAR):
        y_true = test["target"]
        preds = {}

        # Baselines need no training.
        preds["always_up"] = pd.Series(1, index=test.index)
        preds["persistence"] = (test["ret_1"] > 0).astype(int)

        # Models: fresh instance per fold.
        for name, model in make_models().items():
            y_pred = fit_and_predict(model, train, test)
            preds[name] = pd.Series(y_pred, index=test.index)

        # Metrics for this fold.
        for name, y_pred in preds.items():
            row = {"year": year, "model": name,
                   "n_train": len(train), "n_test": len(test)}
            row.update(evaluate(y_true, y_pred))
            metric_rows.append(row)

        # Keep per-day predictions alongside next_ret for the backtest.
        fold_frame = test[["next_ret", "target"]].copy()
        for name, y_pred in preds.items():
            fold_frame["pred_" + name] = y_pred
        prediction_frames.append(fold_frame)

    results = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames)
    return results, predictions


def train_final_model(df):
    """Fit scaler + GBM on ALL rows for use in predict-today. Saved together."""
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURE_COLS])
    model = make_models()["gbm"]
    model.fit(X, df["target"])
    return {"scaler": scaler, "model": model, "features": FEATURE_COLS}


def main():
    df = load_features(INPUT_PATH)
    results, predictions = run_walk_forward(df)

    print("Per-year accuracy (rows = test year):")
    print(results.pivot(index="year", columns="model", values="accuracy").round(3))

    print("\nPooled accuracy over all test years:")
    for name in ["always_up", "persistence", "logreg", "gbm"]:
        pooled = accuracy_score(
            predictions["target"], predictions["pred_" + name])
        print(f"  {name:12s} {pooled:.4f}")

    results.to_csv(RESULTS_PATH, index=False)
    predictions.to_csv(PREDICTIONS_PATH)
    print("\nSaved", RESULTS_PATH, "and", PREDICTIONS_PATH)

    artifact = train_final_model(df)
    joblib.dump(artifact, MODEL_PATH)
    print("Saved final model to", MODEL_PATH)


if __name__ == "__main__":
    main()
