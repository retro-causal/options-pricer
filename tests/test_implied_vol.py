import numpy as np
import pytest
from scipy.optimize import brentq

from pricer.black_scholes import call_price
from pricer.implied_vol import implied_vol, _bs_price


CASES = {
    "atm_call": dict(S=100, K=100, T=1, r=0.05, q=0.0, sigma=0.35, option_type="call"),
    "otm_call_short": dict(S=100, K=130, T=0.1, r=0.05, q=0.0, sigma=0.40, option_type="call"),
    "itm_put_divs": dict(S=90, K=100, T=0.5, r=0.03, q=0.02, sigma=0.25, option_type="call"),
    "otm_put_long": dict(S=100, K=80, T=2, r=0.05, q=0.0, sigma=0.30, option_type="put")
}



@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_round_trip(case):
    inputs = dict(case)
    sigma=inputs.pop("sigma")
    price = _bs_price(**inputs, sigma=sigma)
    assert np.isclose(implied_vol(price, **inputs), sigma, atol=1e-6)



def test_atm_solved_by_newton_alone(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Brent fallback used for an easy ATM case")
    
    monkeypatch.setattr("pricer.implied_vol.brentq", fail)
    price = call_price(100, 100, 1, 0.05, 0.35)
    assert np.isclose(implied_vol(price, 100, 100, 1, 0.05), 0.35, atol=1e-6)


def test_hard_case_uses_fallback(monkeypatch):
    calls = []

    def spy(*args, **kwargs):
        calls.append(1)
        return brentq(*args, **kwargs)
    
    monkeypatch.setattr("pricer.implied_vol.brentq", spy)
    price = call_price(100, 130 ,0.1, 0.05, 0.4)
    assert np.isclose(implied_vol(price, 100, 130, 0.1, 0.05), 0.4, atol=1e-6)
    assert calls



@pytest.mark.parametrize(
    "price, option_type",
    [(150.0, "call"), (3.0, "call"), (96.0, "put")],
    ids=["call_above_stock", "call_below_intrinsic", "put_above_disc_strike"],
)
def test_price_outside_bounds_raises(price, option_type):
    with pytest.raises(ValueError, match="bounds"):
        implied_vol(price, 100, 100, 1, 0.05, option_type=option_type)



def test_invalid_option_type_raises():
    with pytest.raises(ValueError):
        implied_vol(10.0, 100, 100, 1, 0.05, option_type="Call")

