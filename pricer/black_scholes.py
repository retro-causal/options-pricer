import numpy as np
from scipy.stats import norm

def _d1_d2(S, K, T, r, q, sigma):
    """Calculate the d1 and d2 terms used in the Black-Scholes model."""

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def call_price(S, K, T, r, sigma, q=0.0):
    """Calculate the Black-Scholes price of a European call option."""

    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def put_price(S, K, T, r, sigma, q=0.0):
    """Calculate the Black-Scholes price of a European put option."""

    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)