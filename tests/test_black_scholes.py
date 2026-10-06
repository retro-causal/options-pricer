import numpy as np
import pytest
from pricer.black_scholes import call_price, put_price

@pytest.mark.parametrize("S, K, T, r, sigma, q", [
    (100, 95, 0.5, 0.05, 0.25, 0.02),   # in the money call
    (100, 120, 2.0, 0.03, 0.40, 0.00),  # out of the money, long expiry
    (50, 50, 0.05, 0.01, 0.15, 0.01),   # at the money, near expiry
])

def test_put_call_parity(S, K, T, r, sigma, q):
    #S, K, T, r, sigma, q = 100, 95, 0.5, 0.05, 0.25, 0.02
    lhs = call_price(S, K, T, r, sigma, q) - put_price(S, K, T, r, sigma, q)
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)

    assert np.isclose(lhs, rhs)