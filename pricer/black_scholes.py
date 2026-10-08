import numpy as np
from scipy.stats import norm


def _check_option_type(option_type):
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")    


def _d1_d2(S, K, T, r, q, sigma):
    """
    Calculate the d1 and d2 terms used in the Black-Scholes model.
        Parameters:
        S: Current price of the underlying asset.
        K: Strike price.
        T: Time to expiry in years.
        r: Risk-free interest rate.
        sigma: Volatility of the underlying asset.
        q: Continuous dividend yield. Defaults to 0.0.
        option_type: Either "call" or "put".
    """
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def call_price(S, K, T, r, sigma, q=0.0):
    """
    Calculate the Black-Scholes price of a European call option.
    """
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def put_price(S, K, T, r, sigma, q=0.0):
    """
    Calculate the Black-Scholes price of a European put option.
    """
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)




