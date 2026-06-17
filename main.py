"""End-to-end: simulate -> features -> tune+train -> evaluate -> backtest."""
import argparse, logging, os
from simulate import simulate_seasons
from features import build_features
from model import train_and_evaluate
import backtest

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-tune", action="store_true", help="skip hyperparameter search")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    os.makedirs("data", exist_ok=True)

    games = simulate_seasons(n_seasons=args.seasons, seed=args.seed)
    games.to_csv("data/games.csv", index=False)
    feats = build_features(games)
    feats.to_csv("data/features.csv", index=False)
    summary = train_and_evaluate(tune=not args.no_tune)
    bt = backtest.run()

    print("\n== Model (held-out final season) ==")
    for n, m in summary["models"].items():
        print(f"  {n:20s} acc={m['accuracy']} auc={m['roc_auc']} "
              f"brier={m['brier']} skill={m['brier_skill']}")
    print(f"  baseline (home) acc={summary['baseline']['home_pick_accuracy']}")
    print(f"\n== Backtest vs naive market ==\n  {bt}")

if __name__ == "__main__":
    main()
