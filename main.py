"""End-to-end pipeline: simulate -> features -> train -> evaluate.

Usage:
    python main.py                 # full run with defaults
    python main.py --seasons 6     # more data
"""
import argparse, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from simulate import simulate_seasons
from features import build_features
from model import train_and_evaluate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    os.makedirs("data", exist_ok=True)

    print("1/3  Simulating seasons ...")
    games = simulate_seasons(n_seasons=args.seasons, seed=args.seed)
    games.to_csv("data/games.csv", index=False)
    print(f"     {len(games):,} games | home win rate {games.home_win.mean():.3f}")

    print("2/3  Building leakage-free features ...")
    feats = build_features(games)
    feats.to_csv("data/features.csv", index=False)

    print("3/3  Training & evaluating ...")
    summary = train_and_evaluate()
    print("\nResults (held-out final season):")
    for name, m in summary["models"].items():
        print(f"  {name:22s} acc={m['accuracy']}  auc={m['roc_auc']}  "
              f"logloss={m['log_loss']}  brier={m['brier']}")
    print(f"  always-pick-home       acc={summary['baseline']['home_pick_accuracy']}")

if __name__ == "__main__":
    main()
