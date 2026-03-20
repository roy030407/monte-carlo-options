from __future__ import annotations

import math
from typing import Literal, Union

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

OptionType = Literal["call", "put"]


def _validate_option_type(option_type: str) -> OptionType:
    opt = str(option_type).strip().lower()
    if opt not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")
    return opt  # type: ignore[return-value]


Number = Union[int, float, np.floating]


def _d1_d2(S: Number, K: Number, T: Number, r: Number, sigma: Number) -> tuple[float, float]:
    S = float(S)
    K = float(K)
    T = float(T)
    r = float(r)
    sigma = float(sigma)

    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    vol_sqrt_T = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / vol_sqrt_T
    d2 = d1 - vol_sqrt_T
    return d1, d2


def black_scholes_price(
    S: Number,
    K: Number,
    T: Number,
    r: Number,
    sigma: Number,
    option_type: str,
) -> float:
    """
    Black-Scholes closed-form price for European options.

    Parameters
    ----------
    option_type : 'call' or 'put'
    """
    opt = _validate_option_type(option_type)
    d1, d2 = _d1_d2(S=S, K=K, T=T, r=r, sigma=sigma)
    S = float(S)
    K = float(K)
    T = float(T)
    r = float(r)

    disc = math.exp(-r * T)
    if opt == "call":
        return float(S * norm.cdf(d1) - K * disc * norm.cdf(d2))
    return float(K * disc * norm.cdf(-d2) - S * norm.cdf(-d1))


def implied_volatility(
    S: Number,
    K: Number,
    T: Number,
    r: Number,
    option_price: Number,
    option_type: str,
    vol_lower: float = 1e-6,
    vol_upper: float = 5.0,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> float:
    """
    Compute implied volatility using Brent's method.
    """
    opt = _validate_option_type(option_type)
    S = float(S)
    K = float(K)
    T = float(T)
    r = float(r)
    option_price = float(option_price)

    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if vol_lower <= 0 or vol_upper <= 0:
        raise ValueError("vol_lower and vol_upper must be positive.")
    if vol_lower >= vol_upper:
        raise ValueError("vol_lower must be less than vol_upper.")

    def f(vol: float) -> float:
        return black_scholes_price(S=S, K=K, T=T, r=r, sigma=vol, option_type=opt) - option_price

    a = float(vol_lower)
    b = float(vol_upper)
    fa = f(a)
    fb = f(b)

    # Expand upper bound if needed to bracket a root.
    # If the option price is outside the BS price range over [a,b], this will fail.
    upper = b
    expand_steps = 0
    while fa * fb > 0 and upper < 50.0 and expand_steps < 10:
        upper *= 2.0
        b = upper
        fb = f(b)
        expand_steps += 1

    if fa * fb > 0:
        raise ValueError(
            "Implied volatility root not bracketed. "
            "Check that option_price is within a feasible Black-Scholes range."
        )

    # brentq is deterministic and uses tolerance on interval size.
    # max_iter is passed as `maxiter`.
    root = brentq(f, a, b, xtol=tol, rtol=tol, maxiter=max_iter)
    return float(root)


# Backwards-compatible alias (in case you prefer an explicit name).
implied_volatility_brent = implied_volatility

