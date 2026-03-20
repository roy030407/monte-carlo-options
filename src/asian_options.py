from __future__ import annotations

import math
from typing import Literal, Union

import numpy as np


OptionType = Literal["call", "put"]


def _validate_option_type(option_type: str) -> OptionType:
    opt = str(option_type).strip().lower()
    if opt not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")
    return opt  # type: ignore[return-value]


Number = Union[int, float, np.floating]


def asian_mc_price(
    S: Number,
    K: Number,
    T: Number,
    r: Number,
    sigma: Number,
    n_paths: int,
    n_steps: int,
    option_type: str,
) -> float:
    """
    Arithmetic Asian option price using Monte Carlo on GBM.

    Simulates n_steps time steps per path, computes the arithmetic average price
    along each path, and discounts the corresponding payoff.
    """
    opt = _validate_option_type(option_type)
    S = float(S)
    K = float(K)
    T = float(T)
    r = float(r)
    sigma = float(sigma)
    n_paths = int(n_paths)
    n_steps = int(n_steps)

    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    if n_paths <= 0:
        raise ValueError("n_paths must be positive.")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")

    rng = np.random.default_rng()

    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)

    z = rng.standard_normal((n_paths, n_steps))
    drift = (r - 0.5 * sigma * sigma) * dt
    diffusion = sigma * sqrt_dt
    log_increments = drift + diffusion * z

    log_paths = math.log(S) + np.cumsum(log_increments, axis=1)
    s_paths = np.exp(log_paths)
    avg_prices = s_paths.mean(axis=1)

    if opt == "call":
        payoff = np.maximum(avg_prices - K, 0.0)
    else:
        payoff = np.maximum(K - avg_prices, 0.0)

    discount = math.exp(-r * T)
    return float(discount * payoff.mean())

