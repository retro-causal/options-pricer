import numpy as np
import pytest
from pricer.black_scholes import call_price, put_price
from pricer.greeks import delta, gamma, vega, theta, rho

BASE = dict(S=100, K=95, T=0.5 ,r=0.05, sigma=0.25, q=0.02)
OPTION_TYPES = ["call", "put"]


def price(option_type, **params):
    """Price a call or put with the given keyword inputs"""
    pricer = call_price if option_type == "call" else put_price
    return pricer(**params)


def bumped(name, amount):
    """Return a copy of BASE with one input shifted by an amount"""
    return {**BASE, name: BASE[name] + amount}


def central_diff(option_type, name, h):
    """Numerical first derivative of the prie with respect to one input"""
    up = price(option_type, **bumped(name, h))
    down = price(option_type, **bumped(name, -h))
    return (up - down) / (2 * h)


@pytest.mark.parametrize("option_type", OPTION_TYPES)
def test_delta(option_type):
    numerical = central_diff(option_type, "S", h=0.01)
    assert np.isclose(delta(**BASE, option_type=option_type), numerical, rtol=1e-5)
    

@pytest.mark.parametrize("option_type", OPTION_TYPES)
def test_gamma(option_type):
    h = 0.01
    up = price(option_type, **bumped("S", h))
    mid = price(option_type, **BASE)
    down = price(option_type, **bumped("S", -h))
    numerical = (up - 2 * mid + down) / h**2
    assert np.isclose(gamma(**BASE), numerical, rtol=1e-4)


@pytest.mark.parametrize("option_type", OPTION_TYPES)
def test_vega(option_type):
    numerical = central_diff(option_type, "sigma", h=1e-4)
    assert np.isclose(vega(**BASE), numerical, rtol=1e-5)


@pytest.mark.parametrize("option_type", OPTION_TYPES)
def test_theta(option_type):
    # Time passing means T shrinks, so theta is minus the derivative with respect to T
    numerical = -central_diff(option_type, "T", h=1e-4)
    assert np.isclose(theta(**BASE, option_type=option_type), numerical, rtol=1e-5)


@pytest.mark.parametrize("option_type", OPTION_TYPES)
def test_rho(option_type):
    numerical = central_diff(option_type, "r", h=1e-4)
    assert np.isclose(rho(**BASE, option_type=option_type), numerical, rtol=1e-5)


def test_invalid_option_type_raises():
    with pytest.raises(ValueError):
        # "Call" (uppercase) is invalid; the guard must reject it
        delta(**BASE, option_type="Call")
