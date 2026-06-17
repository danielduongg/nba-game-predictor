import numpy as np
from simulate import simulate_seasons, TEAMS

def test_shape_and_columns():
    df = simulate_seasons(n_seasons=2, seed=1)
    assert len(df) == 2 * len(TEAMS) * (len(TEAMS) - 1)
    for col in ["home_team", "away_team", "home_pts", "away_pts", "home_win", "date"]:
        assert col in df.columns

def test_home_win_consistent_with_score():
    df = simulate_seasons(n_seasons=2, seed=1)
    assert (df.home_win == (df.home_pts > df.away_pts)).all()

def test_home_advantage_realistic():
    df = simulate_seasons(n_seasons=3, seed=2)
    assert 0.52 < df.home_win.mean() < 0.66  # realistic home edge

def test_deterministic():
    a = simulate_seasons(n_seasons=2, seed=5)
    b = simulate_seasons(n_seasons=2, seed=5)
    assert a.equals(b)
