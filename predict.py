"""Illustrative inference CLI: win probability for a single matchup.

Feed each team's current Elo (and rest days) to the trained, calibrated model.
Example:
    python predict.py --home BOS --away LAL --home-elo 1620 --away-elo 1555
"""
from __future__ import annotations
import argparse
import joblib
import pandas as pd
from features import FEATURE_COLS

HOME_COURT_ELO = 65.0  # matches features.build_features default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", required=True)
    ap.add_argument("--away", required=True)
    ap.add_argument("--home-elo", type=float, default=1500.0)
    ap.add_argument("--away-elo", type=float, default=1500.0)
    ap.add_argument("--home-rest", type=int, default=2)
    ap.add_argument("--away-rest", type=int, default=2)
    args = ap.parse_args()

    model = joblib.load("reports/model.joblib")
    row = pd.DataFrame([{
        "elo_diff": (args.home_elo + HOME_COURT_ELO) - args.away_elo,
        "form_diff": 0.0,
        "netrtg_diff": 0.0,
        "rest_diff": args.home_rest - args.away_rest,
        "home_b2b": int(args.home_rest == 0),
        "away_b2b": int(args.away_rest == 0),
    }])[FEATURE_COLS]
    p = float(model.predict_proba(row)[0, 1])
    print(f"{args.home} (home) vs {args.away} (away)")
    print(f"  P({args.home} win) = {p:.1%}")


if __name__ == "__main__":
    main()
