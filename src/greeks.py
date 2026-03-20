from __future__ import annotations

import math
from typing import List, Literal, Tuple, Union

import numpy as np
from scipy.stats import norm


OptionType = Literal["call", "put"]


def _validate_option_type(option_type: str) -> OptionType:
    opt = str(option_type).strip().lower()
    if opt not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")
    return opt  # type: ignore[return-value]


Number = Union[int, float, np.floating]


def _d1_d2(S: Number, K: Number, T: Number, r: Number, sigma: Number) -> Tuple[float, float]:
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


def delta(S: Number, K: Number, T: Number, r: Number, sigma: Number, option_type: str) -> float:
    opt = _validate_option_type(option_type)
    d1, _ = _d1_d2(S=S, K=K, T=T, r=r, sigma=sigma)
    if opt == "call":
        return float(norm.cdf(d1))
    return float(norm.cdf(d1) - 1.0)


def gamma(S: Number, K: Number, T: Number, r: Number, sigma: Number, option_type: str) -> float:
    # Gamma is the same for call and put.
    _, d2 = _d1_d2(S=S, K=K, T=T, r=r, sigma=sigma)
    # Use d1 in pdf expression: N'(d1) is pdf at d1. We recompute via d1:
    d1, _ = _d1_d2(S=S, K=K, T=T, r=r, sigma=sigma)
    return float(norm.pdf(d1) / (float(S) * float(sigma) * math.sqrt(float(T))))


def vega(S: Number, K: Number, T: Number, r: Number, sigma: Number, option_type: str) -> float:
    # Vega is the same for call and put; derivative w.r.t sigma (volatility).
    d1, _ = _d1_d2(S=S, K=K, T=T, r=r, sigma=sigma)
    return float(float(S) * norm.pdf(d1) * math.sqrt(float(T)))


def theta(S: Number, K: Number, T: Number, r: Number, sigma: Number, option_type: str) -> float:
    opt = _validate_option_type(option_type)
    d1, d2 = _d1_d2(S=S, K=K, T=T, r=r, sigma=sigma)
    S = float(S)
    K = float(K)
    T = float(T)
    r = float(r)
    sigma = float(sigma)

    first_term = -(S * norm.pdf(d1) * sigma) / (2.0 * math.sqrt(T))
    if opt == "call":
        second_term = -r * K * math.exp(-r * T) * norm.cdf(d2)
        return float(first_term + second_term)

    # Put
    second_term = r * K * math.exp(-r * T) * norm.cdf(-d2)
    return float(first_term + second_term)


def rho(S: Number, K: Number, T: Number, r: Number, sigma: Number, option_type: str) -> float:
    opt = _validate_option_type(option_type)
    _, d2 = _d1_d2(S=S, K=K, T=T, r=r, sigma=sigma)
    S = float(S)
    K = float(K)
    T = float(T)
    r = float(r)
    if opt == "call":
        return float(K * T * math.exp(-r * T) * norm.cdf(d2))
    return float(-K * T * math.exp(-r * T) * norm.cdf(-d2))


def delta_vs_spot(
    S: Number, K: Number, T: Number, r: Number, sigma: Number, option_type: str
) -> Tuple[List[float], List[float]]:
    opt = _validate_option_type(option_type)
    base = float(S)
    spot_prices = np.linspace(0.5 * base, 1.5 * base, 50)
    deltas = [delta(float(s), float(K), float(T), float(r), float(sigma), opt) for s in spot_prices]
    return [float(x) for x in spot_prices.tolist()], deltas

