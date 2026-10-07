import numpy as np
import pytest
from pricer.black_scholes import call_price, put_price

CASES = [
    dict(S=100, K=95, T=0.5, r=0.05, sigma=0.25, q=0.02),
    dict(S=100, K=120, T=2.0, r=0.03, sigma=0.40, q=0.00),
    dict(S=50, K=50, T=0.05, r=0.01, sigma=0.15, q=0.01)
]
CASE_IDS = ["itm_call", "otm_long_expiry", "atm_near_expiry"]


def test_call_price_matches_textbook_value():
    """Check taht the call price matches a known Black-Scholes value"""

    assert np.isclose(
        call_price(S=100, K=100, T=1, r=0.05, sigma=0.2),
        10.4506, atol=1e-4
        )


@pytest.mark.parametrize("params", CASES, ids=CASE_IDS)
def test_put_call_parity(params):
    """Check that calll and put prices satisfy put-call parity."""

    lhs = call_price(**params) - put_price(**params)
    rhs = (
        params["S"] * np.exp(-params["q"] * params["T"])
        - params["K"] * np.exp(-params["r"] * params["T"])
        )
    assert np.isclose(lhs, rhs)


