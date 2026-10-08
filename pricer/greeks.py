import numpy as np
from scipy.stats import norm
from pricer.black_scholes import _d1_d2, _check_option_type


def delta(S, K, T, r, sigma, q=0.0, option_type="call"):
    """
    Calculate the Black-Scholes delta of a European call or put option
    """
    _check_option_type(option_type)
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    if option_type == "call":
        return np.exp(-q * T) * norm.cdf(d1)
    return np.exp(-q * T) * (norm.cdf(d1) - 1)


def gamma(S, K, T, r, sigma, q=0.0):
    """
    Calculate the Black-Scholes gamma of a European call or put option
    """
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma, q=0.0):
    """
    Calculate the Black-Scholes vega of a European call or put option
    """
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def theta(S, K, T, r, sigma, q=0.0, option_type="call"):
    """
    Calculate the Black-Scholes theta of a European call or put option
    """
    _check_option_type(option_type)
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    if option_type == "call":
        return (
            -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
            - r * K * np.exp(-r * T) * norm.cdf(d2) 
            + q * S * np.exp(-q * T) * norm.cdf(d1)
            )
    
    return (-S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
            + r * K * np.exp(-r * T) * norm.cdf(-d2) 
            - q * S * np.exp(-q * T) * norm.cdf(-d1)
            )


def rho(S, K, T, r, sigma, q=0.0, option_type="call"):
    """
    Calculate the Black-Scholes rho of a European call or put option
    """
    _check_option_type(option_type)
    _, d2 = _d1_d2(S, K, T, r, q, sigma)
    if option_type == "call":
        return K * T * np.exp(-r * T) * norm.cdf(d2)
    return -K * T * np.exp(-r * T) * norm.cdf(-d2)