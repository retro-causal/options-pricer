import numpy as np
from scipy.optimize import brentq

from pricer.black_scholes import call_price, put_price, _check_option_type
from pricer.greeks import vega



def _bs_price(S, K, T, r, sigma, q, option_type):
    if option_type == "call":
        return call_price(S, K, T, r, sigma, q)
    return put_price(S, K, T, r, sigma, q)



def implied_vol(price, S, K, T, r, q=0.0, option_type ="call", tol=1e-8, max_iter=100):
    _check_option_type(option_type)
    disc_S = S * np.exp(-q * T)
    disc_K = K * np.exp(-r * T)

    if option_type == "call": 
        lower, upper = max(disc_S - disc_K, 0.0), disc_S
    else:
        lower, upper = max(disc_K - disc_S, 0.0), disc_K
    
    if not lower < price < upper:
        raise ValueError(
            f"Price {price:.4f} outside no arbitrage bounds "
            f"({lower:.4f}, {upper:.4f}); no implied volatility exists"
        )
    
    sigma = 0.2

    for _ in range(max_iter):
        diff = _bs_price(S, K, T, r, sigma, q, option_type) - price
        if abs(diff) < tol:
            return sigma
    
        v = vega(S=S, K=K, T=T, r=r, sigma=sigma, q=q)
        if v < 1e-8:
            break
        
        sigma -= diff / v

        if not 0.0 < sigma < 5.0:
            break

    def f(s):
        return _bs_price(S, K, T, r, s, q, option_type) - price
    
    print("falling back to Brent")
    return brentq(f, 1e-6, 5.0, xtol=tol)