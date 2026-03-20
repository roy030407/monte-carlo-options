from __future__ import annotations

from typing import List, Literal, Tuple, Union

import math

import numpy as np


OptionType = Literal["call", "put"]


def _validate_option_type(option_type: str) -> OptionType:
    opt = str(option_type).strip().lower()
    if opt not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")
    return opt  # type: ignore[return-value]


Number = Union[int, float, np.floating]


def mc_price(
    S: Number,
    K: Number,
    T: Number,
    r: Number,
    sigma: Number,
    n_paths: int,
    option_type: str,
) -> float:
    """
    Monte Carlo simulation price for European options under GBM.
    Uses discounted payoff under risk-neutral measure.
    """
    opt = _validate_option_type(option_type)
    S = float(S)
    K = float(K)
    T = float(T)
    r = float(r)
    sigma = float(sigma)
    n_paths = int(n_paths)

    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    if n_paths <= 0:
        raise ValueError("n_paths must be positive.")

    rng = np.random.default_rng()

    z = rng.standard_normal(n_paths)
    st = S * np.exp((r - 0.5 * sigma * sigma) * T + sigma * math.sqrt(T) * z)
    if opt == "call":
        payoff = np.maximum(st - K, 0.0)
    else:
        payoff = np.maximum(K - st, 0.0)

    discount = math.exp(-r * T)
    disc_payoff = discount * payoff
    return float(disc_payoff.mean())


def convergence_data(
    S: Number,
    K: Number,
    T: Number,
    r: Number,
    sigma: Number,
    max_paths: int,
    option_type: str,
) -> List[Tuple[int, float, float, float]]:
    """
    Run MC at 20 evenly spaced points from 100 to max_paths.
    Returns list of (n_paths, price, lower_ci, upper_ci) tuples for each point.
    Confidence intervals use a normal approximation (95%):
        mean +/- 1.96 * std / sqrt(n)
    with std computed on discounted payoffs.
    """
    opt = _validate_option_type(option_type)
    S = float(S)
    K = float(K)
    T = float(T)
    r = float(r)
    sigma = float(sigma)
    max_paths = int(max_paths)

    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    if max_paths < 100:
        raise ValueError("max_paths must be at least 100.")

    rng = np.random.default_rng()
    discount = math.exp(-r * T)

    n_values_raw = np.linspace(100, max_paths, 20)
    n_values = np.unique(n_values_raw.astype(int)).tolist()
    if n_values[0] != 100:
        n_values[0] = 100
    if n_values[-1] != max_paths:
        n_values.append(max_paths)

    results: List[Tuple[int, float, float, float]] = []
    sqrtT = math.sqrt(T)

    for n in n_values:
        z = rng.standard_normal(n)
        st = S * np.exp((r - 0.5 * sigma * sigma) * T + sigma * sqrtT * z)
        if opt == "call":
            payoff = np.maximum(st - K, 0.0)
        else:
            payoff = np.maximum(K - st, 0.0)

        disc_payoff = discount * payoff
        price = float(disc_payoff.mean())

        if n < 2:
            results.append((n, price, price, price))
            continue

        std = float(disc_payoff.std(ddof=1))
        se = std / math.sqrt(n)
        half_width = 1.96 * se
        lower = price - half_width
        upper = price + half_width
        results.append((n, price, lower, upper))

    # Ensure ordering
    results.sort(key=lambda x: x[0])
    return results

