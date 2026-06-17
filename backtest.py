"""Betting-style backtest: does the model have *edge* over a naive market?

We build a naive 'market' probability from team ratings only (an Elo-only
logistic), then let the full model place flat bets at -110 odds whenever it
disagrees with the market by more than an edge threshold. We report ROI, hit
rate and a Brier-skill comparison, plus a cumulative-profit curve.

NOTE: synthetic data — this demonstrates backtest methodology, not real-world
profitability. Real sports betting is negative-EV after the vig for ~everyone.
"""
from __future__ import annotations
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPORTS = "reports"
DECIMAL_ODDS = 1.9091  # -110 American odds


def _market_prob(elo_diff: np.ndarray, slope: float = 1 / 130) -> np.ndarray:
    """Naive market that prices games off the rating gap alone."""
    return 1 / (1 + np.exp(-slope * elo_diff))


def run(preds_csv: str = f"{REPORTS}/holdout_predictions.csv",
        edge: float = 0.03) -> dict:
    df = pd.read_csv(preds_csv, parse_dates=["date"]).sort_values("date")
    if "p_market" not in df.columns:                 # fallback for older files
        df["p_market"] = _market_prob(df["elo_diff"].values)
    df["edge_home"] = df["p_model"] - df["p_market"]

    bets, profit = [], []
    for _, g in df.iterrows():
        if g.edge_home > edge:        # model likes the home side
            won = g.home_win == 1
        elif g.edge_home < -edge:     # model likes the away side
            won = g.home_win == 0
        else:
            continue
        bets.append(won)
        profit.append((DECIMAL_ODDS - 1) if won else -1)  # 1 unit risked

    bets = np.array(bets); profit = np.array(profit, dtype=float)
    n = len(bets)
    summary = dict(
        n_bets=int(n),
        bet_rate=round(n / len(df), 3),
        hit_rate=round(float(bets.mean()), 4) if n else None,
        roi=round(float(profit.sum() / n), 4) if n else None,
        total_units=round(float(profit.sum()), 2),
        breakeven_hit_rate=round(1 / DECIMAL_ODDS, 4),  # ~0.524 at -110
    )
    with open(f"{REPORTS}/backtest.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    if n:
        plt.figure(figsize=(8, 4))
        plt.plot(np.cumsum(profit), lw=1.5, color="#2f855a")
        plt.axhline(0, color="k", ls="--", alpha=.4)
        plt.xlabel("Bet #"); plt.ylabel("Cumulative units")
        plt.title(f"Backtest equity curve — ROI {summary['roi']:.1%} over {n} bets")
        plt.tight_layout(); plt.savefig(f"{REPORTS}/backtest_equity.png", dpi=120); plt.close()
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
