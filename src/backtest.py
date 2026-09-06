"""
Backtest walk-forward predictions with a long/flat strategy.

Rule: if the model predicted "up" for tomorrow, hold SPY tomorrow and earn
next_ret. Otherwise sit in cash and earn 0. Buy-and-hold is the benchmark.

next_ret comes from the predictions file, which was built with the single
shift(-1) in features.py, so the backtest inherits the leakage discipline.

Outputs:
data/backtest_summary.csv   one row per strategy, with and without costs
data/equity_curves.csv      daily equity for every strategy
reports/equity_curve.png    the chart
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('Agg')  # write PNG without needing a display

PREDICTIONS_PATH = "data/walkforward_predictions.csv"
SUMMARY_PATH = "data/backtest_summary.csv"
CURVES_PATH = "data/equity_curves.csv"
CHART_PATH = "reports/equity_curve.png"

COST_PER_SWITCH = 0.0005   # 0.05% each time the position changes
TRADING_DAYS = 252

STRATEGIES = ["always_up", "persistence", "logreg", "gbm"]


def load_predictions(path):
    return pd.read_csv(path, index_col="Date", parse_dates=True)


def strategy_returns(position, next_ret, cost_per_switch):
    """Daily return of a long/flat strategy.

    position: 1 = hold SPY tomorrow, 0 = cash.
    A switch is any day where position differs from the previous day.
    Cost is charged on the switch day.
    """
    switches = position.diff().abs().fillna(0)
    gross = position * next_ret
    net = gross - switches * cost_per_switch
    return gross, net


def equity_curve(daily_returns):
    """Growth of 1 unit of capital: cumulative product of (1 + r)."""
    return (1 + daily_returns).cumprod()


def sharpe_ratio(daily_returns):
    """Annualised return per unit of risk. 0 if the strategy never moved."""
    if daily_returns.std() == 0:
        return 0.0
    return daily_returns.mean() / daily_returns.std() * np.sqrt(TRADING_DAYS)


def max_drawdown(equity):
    """Largest peak-to-trough fall, as a negative fraction."""
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1
    return drawdown.min()


def summarise(name, daily_returns, position, cost_label):
    equity = equity_curve(daily_returns)
    return {
        "strategy": name,
        "costs": cost_label,
        "total_return": equity.iloc[-1] - 1,
        "sharpe": sharpe_ratio(daily_returns),
        "max_drawdown": max_drawdown(equity),
        "days_in_market": position.mean(),
        "switches": int(position.diff().abs().fillna(0).sum()),
    }


def run_backtest(df):
    summary_rows = []
    curves = pd.DataFrame(index=df.index)

    for name in STRATEGIES:
        position = df["pred_" + name]
        gross, net = strategy_returns(
            position, df["next_ret"], COST_PER_SWITCH)

        summary_rows.append(summarise(name, gross, position, "none"))
        summary_rows.append(summarise(name, net, position, "0.05%/switch"))

        curves[name + "_gross"] = equity_curve(gross)
        curves[name + "_net"] = equity_curve(net)

    return pd.DataFrame(summary_rows), curves


def plot_curves(curves, path):
    plt.figure(figsize=(11, 6))
    for name in STRATEGIES:
        plt.plot(curves.index, curves[name + "_net"],
                 label=name + " (net of costs)")
    plt.title("Long/flat strategy equity, walk-forward test years 2018-2025")
    plt.ylabel("Growth of 1 unit")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)


def main():
    df = load_predictions(PREDICTIONS_PATH)
    summary, curves = run_backtest(df)

    print(summary.round(4).to_string(index=False))

    summary.to_csv(SUMMARY_PATH, index=False)
    curves.to_csv(CURVES_PATH)
    plot_curves(curves, CHART_PATH)
    print("\nSaved", SUMMARY_PATH, CURVES_PATH, CHART_PATH)


if __name__ == "__main__":
    main()
