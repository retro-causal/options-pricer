import numpy as np
from pricer.black_scholes import _check_option_type


def _payoff(ST, K, option_type):
    """"
    Calculate the payoff of an option at expiry.
    Parameters:
    ST - float or np.ndarray; underlying asset price at expiry
    K - float; strike price of the option
    
    Returns:
    float or np.ndarray; option payoff(s)
    """
    if option_type == "call":
        return np.maximum(ST - K, 0.0)
    return np.maximum(K - ST, 0.0)


def binomial_price(S, K, T, r, sigma, q=0.0, option_type="call", steps=500, american=False):
    """
    Price an option using the Cos-Ross-Rubinstein binomial model
    """
    _check_option_type(option_type)
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    if not 0.0 < p < 1.0:
        raise ValueError(f"Risk-neutral probability {p:.4f} outside (0, 1); increase steps")
    
    j = np.arange(steps + 1)
    ST = S * u**j * d**(steps - j)

    V = _payoff(ST, K, option_type)
    

    for i in range(steps - 1, -1, -1):
        V = disc * (p * V[1:] + (1 - p) * V[:-1])

        if american:
            # For American options, compare value from holding vs value from exercising now; choose the greater
            ST = S * u**j[:i + 1] * d**(i - j[:i + 1])
            V = np.maximum(V, _payoff(ST, K, option_type))
    
    return V[0]

