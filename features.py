"""Build leakage-free features from a chronological game log.

Every feature for a given game uses ONLY information available before tip-off:
- Elo ratings updated online after each game
- Rolling point differential and win% over each team's prior games
- Rest days and back-to-back flags
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from collections import defaultdict, deque

def _elo_expected(r_a: float, r_b: float) -> float:
    return 1 / (1 + 10 ** ((r_b - r_a) / 400))

def build_features(games: pd.DataFrame, k: float = 20.0, home_field: float = 65.0) -> pd.DataFrame:
    games = games.sort_values("date").reset_index(drop=True)
    elo = defaultdict(lambda: 1500.0)
    last10 = defaultdict(lambda: deque(maxlen=10))      # recent win/loss
    pts_for = defaultdict(lambda: deque(maxlen=10))     # recent points scored
    pts_against = defaultdict(lambda: deque(maxlen=10)) # recent points allowed
    feats = []
    for _, g in games.iterrows():
        h, a = g.home_team, g.away_team
        # ---- pre-game features (no leakage) ----
        elo_diff = (elo[h] + home_field) - elo[a]
        form_h = np.mean(last10[h]) if last10[h] else 0.5
        form_a = np.mean(last10[a]) if last10[a] else 0.5
        netrtg_h = (np.mean(pts_for[h]) - np.mean(pts_against[h])) if pts_for[h] else 0.0
        netrtg_a = (np.mean(pts_for[a]) - np.mean(pts_against[a])) if pts_for[a] else 0.0
        feats.append(dict(
            game_id=g.game_id, season=g.season, date=g.date, home_win=g.home_win,
            elo_diff=elo_diff,
            elo_home=elo[h], elo_away=elo[a],
            form_diff=form_h - form_a,
            netrtg_diff=netrtg_h - netrtg_a,
            rest_diff=g.home_rest - g.away_rest,
            home_b2b=int(g.home_rest == 0), away_b2b=int(g.away_rest == 0),
        ))
        # ---- post-game updates ----
        exp_h = _elo_expected(elo[h] + home_field, elo[a])
        s_h = float(g.home_win)
        margin = abs(g.home_pts - g.away_pts)
        mult = np.log(margin + 1)  # margin-of-victory weight
        elo[h] += k * mult * (s_h - exp_h)
        elo[a] += k * mult * ((1 - s_h) - (1 - exp_h))
        last10[h].append(s_h); last10[a].append(1 - s_h)
        pts_for[h].append(g.home_pts); pts_against[h].append(g.away_pts)
        pts_for[a].append(g.away_pts); pts_against[a].append(g.home_pts)
    return pd.DataFrame(feats)

FEATURE_COLS = ["elo_diff", "form_diff", "netrtg_diff", "rest_diff", "home_b2b", "away_b2b"]

if __name__ == "__main__":
    games = pd.read_csv("data/games.csv", parse_dates=["date"])
    f = build_features(games)
    f.to_csv("data/features.csv", index=False)
    print(f"Built features for {len(f):,} games -> data/features.csv")
