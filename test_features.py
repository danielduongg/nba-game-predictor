import numpy as np
from simulate import simulate_seasons
from features import build_features, FEATURE_COLS

def test_no_nans_and_columns():
    f = build_features(simulate_seasons(n_seasons=2, seed=3))
    assert not f[FEATURE_COLS].isna().any().any()
    assert len(f) == len(simulate_seasons(n_seasons=2, seed=3))

def test_no_leakage_first_games_are_neutral():
    # Before any games, every team shares Elo 1500 and empty form, so the very
    # first chronological game must have neutral form/netrtg and elo_diff == home court.
    f = build_features(simulate_seasons(n_seasons=1, seed=3)).sort_values("date")
    first = f.iloc[0]
    assert abs(first.form_diff) < 1e-9
    assert abs(first.netrtg_diff) < 1e-9
    assert abs(first.elo_diff - 65.0) < 1e-6  # home-court bonus only

def test_feature_count():
    assert len(FEATURE_COLS) == 6
