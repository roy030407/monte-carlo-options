import math
from typing import Any, List, Literal, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.asian_options import asian_mc_price
from src.black_scholes import black_scholes_price
from src.greeks import delta, delta_vs_spot, gamma, rho, theta, vega
from src.monte_carlo import convergence_data, mc_price


st.set_page_config(
    page_title="Monte Carlo Option Pricer",
    layout="wide",
    page_icon="📈",
)


def _plotly_dark_transparent(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
    return fig


def _format_maybe_number(x: Any, digits: int = 6) -> str:
    if x is None:
        return "N/A"
    try:
        xf = float(x)
    except Exception:
        return "N/A"
    if math.isnan(xf):
        return "N/A"
    return f"${xf:,.{digits}f}"


def _format_percent(x: Any, digits: int = 3) -> str:
    if x is None:
        return "N/A"
    try:
        xf = float(x)
    except Exception:
        return "N/A"
    if math.isnan(xf):
        return "N/A"
    sign = "-" if xf < 0 else ""
    return f"{sign}{abs(xf):,.{digits}f}%"


def _asian_convergence_data(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    max_paths: int,
    option_type: Literal["call", "put"],
    n_steps: int,
) -> List[Tuple[int, float, float, float]]:
    """
    Mirrors the European convergence_data API but for arithmetic Asian options.
    """
    from src.asian_options import asian_mc_price  # local import to keep app import fast

    if max_paths < 100:
        raise ValueError("Simulation Paths must be at least 100.")

    rng = np.random.default_rng()
    n_values_raw = np.linspace(100, max_paths, 20)
    n_values = np.unique(n_values_raw.astype(int)).tolist()
    if n_values[0] != 100:
        n_values[0] = 100
    if n_values[-1] != max_paths:
        n_values.append(max_paths)
    n_values = sorted(n_values)

    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)
    discount = math.exp(-r * T)

    results: List[Tuple[int, float, float, float]] = []

    for n in n_values:
        z = rng.standard_normal((n, n_steps))
        drift = (r - 0.5 * sigma * sigma) * dt
        diffusion = sigma * sqrt_dt
        log_increments = drift + diffusion * z
        log_paths = math.log(S) + np.cumsum(log_increments, axis=1)
        s_paths = np.exp(log_paths)
        avg_prices = s_paths.mean(axis=1)

        if option_type == "call":
            payoff = np.maximum(avg_prices - K, 0.0)
        else:
            payoff = np.maximum(K - avg_prices, 0.0)

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

    return results


st.title("📈 Monte Carlo Option Pricer")
st.subheader("Interactive options pricing using Monte Carlo simulation and Black-Scholes analytical models")

with st.sidebar:
    st.header("Option Parameters")
    S = st.number_input("Stock Price S", value=100.0, min_value=1.0)
    K = st.number_input("Strike Price K", value=100.0, min_value=1.0)
    T = st.number_input("Time to Expiry T", value=1.0, min_value=0.01, step=0.1)
    r = st.number_input("Risk-free Rate r", value=0.05, step=0.01)
    sigma = st.number_input("Volatility sigma", value=0.20, step=0.01)

    n_paths = st.slider("Simulation Paths", min_value=1000, max_value=100000, value=10000, step=1000)

    option_type_label = st.radio("Option Type", ["Call", "Put"], index=0)
    exotic_type = st.radio("Exotic Type", ["European", "Asian (Avg Price)"], index=0)

    run = st.button("Run Simulation")


if run:
    option_type = "call" if option_type_label == "Call" else "put"
    n_steps_asian = 50

    with st.spinner("Running simulation..."):
        bs_price = black_scholes_price(S=S, K=K, T=T, r=r, sigma=sigma, option_type=option_type)

        # Pricing
        if exotic_type == "European":
            mc_price_value = mc_price(
                S=S, K=K, T=T, r=r, sigma=sigma, n_paths=n_paths, option_type=option_type
            )
            conv = convergence_data(
                S=S, K=K, T=T, r=r, sigma=sigma, max_paths=n_paths, option_type=option_type
            )
        else:
            mc_price_value = asian_mc_price(
                S=S,
                K=K,
                T=T,
                r=r,
                sigma=sigma,
                n_paths=n_paths,
                n_steps=n_steps_asian,
                option_type=option_type,
            )
            conv = _asian_convergence_data(
                S=float(S),
                K=float(K),
                T=float(T),
                r=float(r),
                sigma=float(sigma),
                max_paths=int(n_paths),
                option_type=option_type,  # type: ignore[arg-type]
                n_steps=n_steps_asian,
            )

        # Distribution simulation (for the selected exotic type)
        rng = np.random.default_rng()
        z_st = rng.standard_normal(int(n_paths))
        st_terminal = float(S) * np.exp(
            (float(r) - 0.5 * float(sigma) * float(sigma)) * float(T) + float(sigma) * math.sqrt(float(T)) * z_st
        )

        if exotic_type == "European":
            if option_type == "call":
                payoff_terminal = np.maximum(st_terminal - float(K), 0.0)
            else:
                payoff_terminal = np.maximum(float(K) - st_terminal, 0.0)
            disc_payoff_terminal = math.exp(-float(r) * float(T)) * payoff_terminal
        else:
            dt = float(T) / n_steps_asian
            sqrt_dt = math.sqrt(dt)
            z = rng.standard_normal((int(n_paths), n_steps_asian))
            drift = (float(r) - 0.5 * float(sigma) * float(sigma)) * dt
            diffusion = float(sigma) * sqrt_dt
            log_increments = drift + diffusion * z
            log_paths = math.log(float(S)) + np.cumsum(log_increments, axis=1)
            s_paths = np.exp(log_paths)
            avg_prices = s_paths.mean(axis=1)
            st_terminal = s_paths[:, -1]

            if option_type == "call":
                payoff_terminal = np.maximum(avg_prices - float(K), 0.0)
            else:
                payoff_terminal = np.maximum(float(K) - avg_prices, 0.0)
            disc_payoff_terminal = math.exp(-float(r) * float(T)) * payoff_terminal

        results = {
            "bs_price": bs_price,
            "mc_price": float(mc_price_value),
            "conv": conv,
            "st_terminal": st_terminal,
            "option_payoffs": payoff_terminal,
            "disc_payoff_terminal": disc_payoff_terminal,
        }

    st.session_state["mc_option_results"] = results


results = st.session_state.get("mc_option_results")

if not results:
    st.info("Adjust the parameters in the sidebar and click `Run Simulation` to see results.")
    st.stop()

bs_price_val = float(results["bs_price"])
mc_price_val = float(results["mc_price"])
conv = results["conv"]

diff_pct = (mc_price_val - bs_price_val) / bs_price_val * 100.0 if bs_price_val != 0 else float("nan")

tab1, tab2, tab3, tab4 = st.tabs(["💰 Pricing", "📊 Payoff Distribution", "🔢 Greeks", "🌋 Volatility Surface"])

with tab1:
    cols = st.columns(3)
    cols[0].metric(label="BS Price", value=_format_maybe_number(bs_price_val))
    cols[1].metric(label="MC Price", value=_format_maybe_number(mc_price_val))
    cols[2].metric(label="Difference (%)", value=_format_percent(diff_pct))

    xs = [int(t[0]) for t in conv]
    ys = [float(t[1]) for t in conv]
    lowers = [float(t[2]) for t in conv]
    uppers = [float(t[3]) for t in conv]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=lowers,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            name="Lower CI",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=uppers,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(0,255,255,0.12)",
            showlegend=False,
            name="Upper CI",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines+markers",
            name="MC Price",
            line=dict(color="rgba(0, 200, 255, 0.95)", width=2),
            marker=dict(size=6),
        )
    )

    fig.add_shape(
        type="line",
        x0=xs[0],
        x1=xs[-1],
        y0=bs_price_val,
        y1=bs_price_val,
        line=dict(color="rgba(255,255,255,0.7)", width=2, dash="dash"),
    )

    _plotly_dark_transparent(fig)
    fig.update_layout(
        title="Monte Carlo Convergence to Black-Scholes Price",
        xaxis_title="Number of simulated paths",
        yaxis_title="Option price (discounted)",
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📚 What does this mean?"):
        st.write(
            "The convergence chart shows how the Monte Carlo estimate approaches the "
            "analytical Black-Scholes price as we increase the number of simulated paths. "
            "The shaded band represents the 95% confidence interval."
        )

with tab2:
    option_type = "call" if option_type_label == "Call" else "put"
    st_terminal = np.asarray(results["st_terminal"])
    option_payoffs = np.asarray(results["option_payoffs"])

    st_fig = go.Figure()
    st_fig.add_trace(
        go.Histogram(
            x=st_terminal,
            nbinsx=50,
            name="Terminal Stock Price",
            marker=dict(color="rgba(0, 200, 255, 0.65)"),
        )
    )
    st_fig.add_vline(x=float(K), line_width=2, line_dash="solid", line_color="rgba(255,255,255,0.8)")
    st_fig.add_vline(x=float(S), line_width=2, line_dash="dash", line_color="rgba(255,255,255,0.8)")
    st_fig.add_annotation(
        x=float(K),
        y=1.0,
        xref="x",
        yref="paper",
        text="Strike Price",
        showarrow=False,
        yshift=10,
        font=dict(color="white"),
    )
    st_fig.add_annotation(
        x=float(S),
        y=1.0,
        xref="x",
        yref="paper",
        text="Current Price",
        showarrow=False,
        yshift=10,
        font=dict(color="white"),
    )
    st_fig.update_layout(
        title="Distribution of Terminal Stock Prices",
    )
    _plotly_dark_transparent(st_fig)
    st.plotly_chart(st_fig, use_container_width=True)

    payoff_fig = go.Figure()
    payoff_fig.add_trace(
        go.Histogram(
            x=option_payoffs,
            nbinsx=50,
            name="Discounted Payoff",
            marker=dict(color="rgba(255, 153, 51, 0.65)"),
        )
    )
    payoff_fig.update_layout(title="Distribution of Option Payoffs")
    _plotly_dark_transparent(payoff_fig)
    st.plotly_chart(payoff_fig, use_container_width=True)

    with st.expander("Distribution explanation"):
        st.write(
            "The first histogram shows possible terminal stock prices under the GBM model. "
            "The second histogram shows the resulting option payoffs (discounted) for the selected option type."
        )

with tab3:
    cols5 = st.columns(5)
    cols5[0].metric(label="Delta", value=f"{delta(S, K, T, r, sigma, option_type):.6f}")
    cols5[1].metric(label="Gamma", value=f"{gamma(S, K, T, r, sigma, option_type):.6f}")
    cols5[2].metric(label="Vega", value=f"{vega(S, K, T, r, sigma, option_type):.6f}")
    cols5[3].metric(label="Theta", value=f"{theta(S, K, T, r, sigma, option_type):.6f}")
    cols5[4].metric(label="Rho", value=f"{rho(S, K, T, r, sigma, option_type):.6f}")

    spot_prices, deltas = delta_vs_spot(S, K, T, r, sigma, option_type)
    delta_fig = go.Figure()
    delta_fig.add_trace(
        go.Scatter(
            x=spot_prices,
            y=deltas,
            mode="lines",
            name="Delta",
            line=dict(color="rgba(0, 200, 255, 0.95)", width=2),
        )
    )
    delta_fig.add_shape(
        type="line",
        x0=float(S),
        x1=float(S),
        y0=min(deltas),
        y1=max(deltas),
        line=dict(color="rgba(255,255,255,0.8)", width=2, dash="dash"),
    )
    delta_fig.update_layout(title="Delta Sensitivity to Stock Price")
    _plotly_dark_transparent(delta_fig)
    delta_fig.update_xaxes(title="Spot Price")
    delta_fig.update_yaxes(title="Delta")
    st.plotly_chart(delta_fig, use_container_width=True)

    with st.expander("Greek explanations"):
        st.write(
            "Delta: sensitivity of the option price to small changes in the stock price.\n\n"
            "Gamma: sensitivity of Delta to changes in the stock price (curvature).\n\n"
            "Vega: sensitivity of the option price to changes in volatility.\n\n"
            "Theta: sensitivity of the option price to the passage of time.\n\n"
            "Rho: sensitivity of the option price to changes in interest rates."
        )

with tab4:
    strikes = np.linspace(80, 120, 20)
    vols = np.linspace(0.1, 0.5, 20)
    X, Y = np.meshgrid(strikes, vols)
    Z = np.zeros_like(X, dtype=float)

    # Compute the surface from Black-Scholes prices.
    for i in range(len(vols)):
        for j in range(len(strikes)):
            Z[i, j] = black_scholes_price(
                S=float(S),
                K=float(strikes[j]),
                T=float(T),
                r=float(r),
                sigma=float(vols[i]),
                option_type=option_type,
            )

    surface_fig = go.Figure(
        data=go.Surface(
            x=X,
            y=Y,
            z=Z,
            colorscale="Viridis",
        )
    )
    surface_fig.update_layout(
        title="Volatility Surface",
        scene=dict(
            xaxis_title="Strike Price",
            yaxis_title="Volatility",
            zaxis_title="Option Price",
        ),
    )
    _plotly_dark_transparent(surface_fig)
    st.plotly_chart(surface_fig, use_container_width=True)

    with st.expander("What is a volatility surface?"):
        st.write(
            "A volatility surface describes how implied volatility varies with strike and maturity. "
            "Traders use it to price options consistently when the market-implied volatility is not constant."
        )

