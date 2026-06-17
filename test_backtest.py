import numpy as np
from backtest import _market_prob

def test_market_prob_monotonic_and_bounded():
    x = np.array([-300, -100, 0, 100, 300], dtype=float)
    p = _market_prob(x)
    assert np.all((p > 0) & (p < 1))
    assert np.all(np.diff(p) > 0)          # higher rating gap -> higher prob
    assert abs(p[2] - 0.5) < 1e-9          # zero gap -> 50%
