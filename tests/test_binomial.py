import numpy as np
import pytest
from pricer.black_scholes import call_price, put_price
from pricer.binomial import binomial_price

BASE = dict(S=100, K=100, T=1, r=0.05, sigma=0.2)


@pytest.mark.parametrize("option_type, bs_price", [("call", call_price), ("put", put_price)])
def test_european_converges_to_black_scholes(option_type, bs_price):
    tree = binomial_price(**BASE, option_type=option_type, steps=1000)
    assert np.isclose(tree, bs_price(**BASE), atol=0.01) 


def test_tree_satisfies_put_call_parity():
    c = binomial_price(**BASE, option_type="call")
    p = binomial_price(**BASE, option_type="put")
    S, K, T, r = BASE["S"], BASE["K"], BASE["T"], BASE["r"]
    assert np.isclose(c - p, S - K * np.exp(-r * T), rtol=1e-10)


def test_american_call_equals_european_without_dividends():
    euro = binomial_price(**BASE, option_type="call")
    amer = binomial_price(**BASE, option_type="call", american=True)
    assert np.isclose(amer, euro, rtol=1e-12)


def test_american_put_exceeds_european():
    euro = binomial_price(**BASE, option_type="put")
    amer = binomial_price(**BASE, option_type="put", american=True)
    assert amer > euro


def test_deep_itm_american_put_at_least_intrinsic():
    deep = dict(BASE, S=80)
    intrinsic = deep["K"] - deep["S"]
    euro = binomial_price(**deep, option_type="put")
    amer = binomial_price(**deep, option_type="put", american=True)
    assert euro < intrinsic
    assert amer >= intrinsic


def test_invalid_probability_raises():
    with pytest.raises(ValueError, match="Risk-neutral"):
        binomial_price(100, 100, 1, r=0.05, sigma=0.01, steps=1)


def test_invalid_option_type_raises():
    with pytest.raises(ValueError):
        binomial_price(**BASE, option_type="Call")

