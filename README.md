# Monte Carlo Option Pricer (Streamlit)

An interactive web app that prices European call/put options with both:
- Black-Scholes closed-form pricing (and implied volatility via Brent's method)
- Monte Carlo simulation under geometric Brownian motion (GBM)

It also supports arithmetic Asian options (Monte Carlo) and displays option Greeks and a Black-Scholes volatility surface.

## Requirements

This project uses Python packages listed in `requirements.txt`:
- `streamlit`
- `numpy`
- `scipy`
- `plotly`
- `pandas`

## Run the app

From this folder:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- Monte Carlo prices include a 95% confidence interval band in the convergence chart.
- Greeks are computed analytically from Black-Scholes (European options).
- The volatility surface uses the Black-Scholes model to compute option prices across strike and volatility ranges.

