"""
Simulate multiple NBA-style regular seasons with realistic structure.

Why synthetic? It makes this repo 100% reproducible with no API keys or
network access. The generator encodes the real drivers of game outcomes
(team strength, home-court advantage, rest, randomness) so that the
downstream feature engineering and models behave like they would on real
data. See README for how to swap in real data from the `nba_api` package.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

TEAMS = [
    "ATL","BOS","BKN","CHA","CHI","CLE","DAL","DEN","DET","GSW",
    "HOU","IND","LAC","LAL","MEM","MIA","MIL","MIN","NOP","NYK",
    "OKC","ORL","PHI","PHX","POR","SAC","SAS","TOR","UTA","WAS",
]
HOME_ADV = 0.45          # in latent-strength units (~ real home-court edge)
STRENGTH_SCALE = 0.92    # maps strength gap -> win probability


def simulate_seasons(n_seasons: int = 5, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_teams = len(TEAMS)
    strength = rng.normal(0, 1.0, size=n_teams)   # latent true strength per team
    rows, game_id = [], 0
    start = pd.Timestamp("2021-10-19")

    for season in range(n_seasons):
        strength = 0.8 * strength + rng.normal(0, 0.6, size=n_teams)  # year-to-year drift
        matchups = [(h, a) for h in range(n_teams) for a in range(n_teams) if h != a]
        rng.shuffle(matchups)
        last_played = {t: None for t in range(n_teams)}
        cur = start + pd.Timedelta(days=season * 250)

        for i, (h, a) in enumerate(matchups):
            if i % 6 == 0:
                cur = cur + pd.Timedelta(days=1)
            home_rest = 3 if last_played[h] is None else max(0, min((cur - last_played[h]).days, 5))
            away_rest = 3 if last_played[a] is None else max(0, min((cur - last_played[a]).days, 5))

            rest_effect = 0.13 * (home_rest - away_rest)
            logit = STRENGTH_SCALE * (strength[h] - strength[a]) + HOME_ADV + rest_effect
            p_home = 1 / (1 + np.exp(-logit))
            home_win = int(rng.random() < p_home)

            # Generate scores consistent with the outcome.
            base = 112
            margin_mag = abs(rng.normal(9 * abs(2 * p_home - 1) + 4, 7)) + 1  # always >= 1
            margin = margin_mag if home_win else -margin_mag
            home_pts = int(round(base + margin / 2 + rng.normal(0, 3)))
            away_pts = int(round(base - margin / 2 + rng.normal(0, 3)))
            # guard against rounding flipping the sign
            if home_win and home_pts <= away_pts:
                home_pts = away_pts + int(rng.integers(1, 6))
            if not home_win and away_pts <= home_pts:
                away_pts = home_pts + int(rng.integers(1, 6))

            rows.append(dict(
                game_id=game_id, season=season, date=cur,
                home_team=TEAMS[h], away_team=TEAMS[a],
                home_rest=home_rest, away_rest=away_rest,
                home_pts=home_pts, away_pts=away_pts,
                home_win=int(home_pts > away_pts),
            ))
            last_played[h] = cur
            last_played[a] = cur
            game_id += 1

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


if __name__ == "__main__":
    df = simulate_seasons()
    df.to_csv("data/games.csv", index=False)
    print(f"Wrote {len(df):,} games to data/games.csv")
    print(f"Home win rate: {df.home_win.mean():.3f}")
