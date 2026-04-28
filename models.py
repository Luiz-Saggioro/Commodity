#!/usr/bin/env python3
"""
models.py — Forecasting and statistical models for commodity prices.

Models implemented:
1. Log-Normal (mu=0, martingale): commodity-appropriate, no drift assumption
2. Historical Bootstrap (demeaned): preserves fat-tail structure
3. OLS Linear Regression: trend-based, short-term only
4. Ensemble: weighted combination of log-normal + bootstrap

All models follow commodity finance principles:
- No trend extrapolation (mu=0 for long-horizon)
- Mean reversion at long horizons
- Volatility scaling with sqrt(T)
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression


# ── CONSTANTS ─────────────────────────────────────────────────────────────────

HORIZON_MAP = {
    "7 days":   7,
    "14 days":  14,
    "30 days":  21,
    "60 days":  42,
    "90 days":  63,
}

VOL_CAP_ANNUAL = 0.80  # cap annualised vol at 80%


# ── RETURNS & VOLATILITY ──────────────────────────────────────────────────────

def compute_returns(prices: np.ndarray) -> np.ndarray:
    """Log returns from price series."""
    if len(prices) < 2:
        return np.array([])
    return np.diff(np.log(np.maximum(prices, 1e-8)))


def daily_vol(returns: np.ndarray) -> float:
    """Daily volatility, capped at annualised 80%."""
    if len(returns) < 5:
        return 0.012
    sig = float(np.std(returns, ddof=1))
    return min(sig, VOL_CAP_ANNUAL / np.sqrt(252))


def annualised_vol(returns: np.ndarray) -> float:
    return daily_vol(returns) * np.sqrt(252)


def rolling_vol(prices: pd.Series, window: int = 20) -> pd.Series:
    """Rolling annualised volatility (%)."""
    log_ret = np.log(prices / prices.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(252) * 100


def rolling_ma(prices: pd.Series, window: int) -> pd.Series:
    return prices.rolling(window).mean()


# ── OLS LINEAR REGRESSION ─────────────────────────────────────────────────────

def ols_regression(prices: np.ndarray):
    """
    Fit OLS on last N days. Returns (slope, alpha, r2, fitted).
    slope is in $/day. R2 indicates trend strength.
    """
    n = len(prices)
    t = np.arange(n).reshape(-1, 1)
    model = LinearRegression().fit(t, prices)
    fitted = model.predict(t)
    ss_res = np.sum((prices - fitted) ** 2)
    ss_tot = np.sum((prices - np.mean(prices)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {
        "slope": float(model.coef_[0]),
        "intercept": float(model.intercept_),
        "r2": float(r2),
        "fitted": fitted,
    }


# ── LOG-NORMAL FORECAST ───────────────────────────────────────────────────────

def lognormal_forecast(current: float, returns: np.ndarray, horizon_days: int,
                        ci_level: float = 0.90):
    """
    Log-normal forward distribution with mu=0 (commodity martingale).
    Returns point forecast, CI bands, and daily path for chart.
    """
    sig = daily_vol(returns)
    # Ito correction: ensures median = current price
    log_mu = np.log(current) - 0.5 * sig ** 2 * horizon_days
    log_sig = sig * np.sqrt(horizon_days)

    z = norm.ppf((1 + ci_level) / 2)
    mid = float(np.exp(log_mu))
    low = float(np.exp(log_mu - z * log_sig))
    high = float(np.exp(log_mu + z * log_sig))

    # Daily path (for chart: CI widens with sqrt(t))
    steps = np.arange(1, horizon_days + 1)
    path_mid  = current * np.exp(-0.5 * sig**2 * steps)
    path_low  = current * np.exp(-0.5 * sig**2 * steps - z * sig * np.sqrt(steps))
    path_high = current * np.exp(-0.5 * sig**2 * steps + z * sig * np.sqrt(steps))

    return {
        "mid": mid,
        "low": low,
        "high": high,
        "path_mid":  path_mid,
        "path_low":  path_low,
        "path_high": path_high,
        "annualised_vol_pct": round(sig * np.sqrt(252) * 100, 1),
    }


# ── BOOTSTRAP FORECAST ────────────────────────────────────────────────────────

def bootstrap_forecast(current: float, returns: np.ndarray, horizon_days: int,
                        n_sims: int = 4000, ci_level: float = 0.90):
    """
    Historical bootstrap with demeaned returns.
    Preserves fat-tail structure without trend bias.
    """
    r = np.array(returns) if len(returns) > 5 else np.random.normal(0, 0.012, 60)
    r = r - np.mean(r)  # demean: remove trend, keep vol structure
    rng = np.random.default_rng(42)
    drawn = rng.choice(r, size=(n_sims, horizon_days), replace=True)
    final = current * np.exp(np.sum(drawn, axis=1))

    lo_pct = (1 - ci_level) / 2 * 100
    hi_pct = (1 + ci_level) / 2 * 100
    return {
        "mid": float(np.median(final)),
        "low": float(np.percentile(final, lo_pct)),
        "high": float(np.percentile(final, hi_pct)),
        "simulations": final,
    }


# ── ENSEMBLE FORECAST ─────────────────────────────────────────────────────────

def ensemble_forecast(current: float, returns: np.ndarray, horizon_days: int,
                       w_lognormal: float = 0.55, w_bootstrap: float = 0.45,
                       ci_level: float = 0.90):
    """
    Weighted ensemble of log-normal and bootstrap models.
    Returns combined forecast with CI and per-model breakdown.
    """
    ln  = lognormal_forecast(current, returns, horizon_days, ci_level)
    bs  = bootstrap_forecast(current, returns, horizon_days, ci_level=ci_level)

    mid  = w_lognormal * ln["mid"]  + w_bootstrap * bs["mid"]
    low  = w_lognormal * ln["low"]  + w_bootstrap * bs["low"]
    high = w_lognormal * ln["high"] + w_bootstrap * bs["high"]

    change_pct = (mid - current) / current * 100

    return {
        "current": current,
        "mid": round(mid, 4),
        "low": round(low, 4),
        "high": round(high, 4),
        "change_pct": round(change_pct, 2),
        "annualised_vol_pct": ln["annualised_vol_pct"],
        "path_mid":  ln["path_mid"],
        "path_low":  ln["path_low"],
        "path_high": ln["path_high"],
        "simulations": bs["simulations"],
        "lognormal": ln,
        "bootstrap": bs,
        "model": f"{int(w_lognormal*100)}% log-normal + {int(w_bootstrap*100)}% bootstrap",
    }


# ── PROBABILITY DISTRIBUTION ──────────────────────────────────────────────────

def price_probability_bins(current: float, returns: np.ndarray, horizon_days: int,
                             n_bins: int = 20, n_sims: int = 5000):
    """
    Compute probability distribution over price bins via bootstrap simulation.
    Returns (bin_centers, probabilities, bin_edges).
    """
    r = np.array(returns) if len(returns) > 5 else np.random.normal(0, 0.012, 60)
    r = r - np.mean(r)
    rng = np.random.default_rng(42)
    drawn = rng.choice(r, size=(n_sims, horizon_days), replace=True)
    final = current * np.exp(np.sum(drawn, axis=1))

    sig = daily_vol(r)
    spread = sig * np.sqrt(horizon_days) * 3
    lo_edge = current * np.exp(-spread)
    hi_edge = current * np.exp(+spread)

    edges = np.linspace(lo_edge, hi_edge, n_bins + 1)
    counts, _ = np.histogram(final, bins=edges)
    probs = counts / n_sims
    centers = (edges[:-1] + edges[1:]) / 2

    return centers, probs, edges


# ── SCENARIO ANALYSIS ─────────────────────────────────────────────────────────

SCENARIOS = {
    "Base Case":      {"vol_mult": 1.0,  "drift": 0.0,    "label": "Normal market conditions"},
    "High Demand":    {"vol_mult": 0.9,  "drift": +0.002,  "label": "Strong demand, price pressure"},
    "Supply Shock":   {"vol_mult": 1.8,  "drift": +0.004,  "label": "Disruption (weather/geopolitical)"},
    "Recession":      {"vol_mult": 1.3,  "drift": -0.003,  "label": "Demand destruction, price drop"},
    "Bumper Harvest": {"vol_mult": 0.8,  "drift": -0.002,  "label": "Oversupply, price pressure down"},
}


def scenario_forecast(current: float, returns: np.ndarray, horizon_days: int,
                       n_sims: int = 3000):
    """
    Run all scenarios and return median paths + CIs for each.
    """
    results = {}
    r_base = np.array(returns) if len(returns) > 5 else np.random.normal(0, 0.012, 60)
    r_base = r_base - np.mean(r_base)
    sig = daily_vol(r_base)
    rng = np.random.default_rng(42)

    for name, params in SCENARIOS.items():
        r_mod = r_base * params["vol_mult"] + params["drift"]
        drawn = rng.choice(r_mod, size=(n_sims, horizon_days), replace=True)
        paths = current * np.exp(np.cumsum(drawn, axis=1))  # (n_sims, horizon_days)
        results[name] = {
            "median_path": np.median(paths, axis=0),
            "p10_path":    np.percentile(paths, 10, axis=0),
            "p90_path":    np.percentile(paths, 90, axis=0),
            "final_median": float(np.median(paths[:, -1])),
            "final_p10":    float(np.percentile(paths[:, -1], 10)),
            "final_p90":    float(np.percentile(paths[:, -1], 90)),
            "label": params["label"],
        }
    return results


# ── RISK METRICS ──────────────────────────────────────────────────────────────

def risk_metrics(current: float, returns: np.ndarray, horizon_days: int):
    """
    Compute VaR, CVaR, risk level classification.
    """
    bs = bootstrap_forecast(current, returns, horizon_days, n_sims=5000)
    sims = bs["simulations"]
    pnl_pct = (sims - current) / current * 100

    var_95 = float(np.percentile(pnl_pct, 5))
    var_99 = float(np.percentile(pnl_pct, 1))
    cvar_95 = float(np.mean(pnl_pct[pnl_pct <= var_95]))

    ann_vol = annualised_vol(returns) * 100
    if ann_vol < 15:
        risk_level = "LOW"
        risk_color = "#27e8a0"
    elif ann_vol < 30:
        risk_level = "MEDIUM"
        risk_color = "#ffd060"
    else:
        risk_level = "HIGH"
        risk_color = "#ff4d6a"

    return {
        "var_95": round(var_95, 2),
        "var_99": round(var_99, 2),
        "cvar_95": round(cvar_95, 2),
        "ann_vol_pct": round(ann_vol, 1),
        "risk_level": risk_level,
        "risk_color": risk_color,
        "prob_up": round(float(np.mean(sims > current) * 100), 1),
        "prob_down": round(float(np.mean(sims < current) * 100), 1),
    }


# ── DRIVER IMPORTANCE (SIMULATED) ────────────────────────────────────────────

DRIVER_TEMPLATES = {
    "Agriculture": {
        "Weather / Drought Risk":   0.28,
        "Global Demand":            0.22,
        "USD Strength":             0.16,
        "Freight / Logistics":      0.10,
        "Government Policy":        0.12,
        "Speculative Positioning":  0.07,
        "Competitor Supply":        0.05,
    },
    "Metals": {
        "Industrial Demand":        0.30,
        "USD / DXY Index":          0.22,
        "China Growth Proxy":       0.20,
        "Mining Output":            0.12,
        "Energy Costs":             0.08,
        "Speculative Positioning":  0.08,
    },
    "Livestock": {
        "Feed Costs (Corn)":        0.28,
        "Consumer Demand":          0.22,
        "Disease / Health Risk":    0.18,
        "Export Markets":           0.15,
        "Seasonal Demand":          0.10,
        "Fuel / Transport":         0.07,
    },
}


def driver_importance(category: str, current_vol: float, base_vol: float = 0.012):
    """
    Return simulated driver importance scores, perturbed by current vol regime.
    High vol → weather/supply drivers amplified.
    """
    base = DRIVER_TEMPLATES.get(category, DRIVER_TEMPLATES["Agriculture"]).copy()
    vol_ratio = current_vol / base_vol

    # Amplify top driver in high vol regime
    drivers = list(base.items())
    drivers[0] = (drivers[0][0], drivers[0][1] * min(vol_ratio, 2.0))

    # Re-normalise
    total = sum(v for _, v in drivers)
    return {k: round(v / total, 4) for k, v in drivers}
