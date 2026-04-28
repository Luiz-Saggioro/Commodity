#!/usr/bin/env python3
"""
generate_data.py — Generate realistic synthetic commodity price data.
Each commodity has its own volatility, seasonality, trend, and mean-reversion parameters.
"""

import numpy as np
import pandas as pd
import os
from datetime import date, timedelta

SEED = 42
np.random.seed(SEED)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Days of data to generate (3 years of daily data)
N_DAYS = 365 * 3 + 60
START_DATE = date.today() - timedelta(days=N_DAYS)

COMMODITIES = {
    # Agriculture
    "soybeans": {
        "name": "Soybeans",
        "unit": "$/bushel",
        "category": "Agriculture",
        "emoji": "🌱",
        "base_price": 13.50,
        "daily_vol": 0.012,
        "trend": 0.0001,
        "seasonality_amp": 0.08,
        "seasonality_peak_month": 8,  # Aug/Sep (harvest pressure)
        "mean_reversion": 0.03,
        "long_run_mean": 13.80,
        "description": "CBOT front-month continuous futures",
    },
    "corn": {
        "name": "Corn",
        "unit": "$/bushel",
        "category": "Agriculture",
        "emoji": "🌽",
        "base_price": 4.85,
        "daily_vol": 0.013,
        "trend": 0.00005,
        "seasonality_amp": 0.10,
        "seasonality_peak_month": 7,
        "mean_reversion": 0.04,
        "long_run_mean": 4.90,
        "description": "CBOT front-month continuous futures",
    },
    "wheat": {
        "name": "Wheat",
        "unit": "$/bushel",
        "category": "Agriculture",
        "emoji": "🌾",
        "base_price": 5.90,
        "daily_vol": 0.016,
        "trend": 0.0001,
        "seasonality_amp": 0.12,
        "seasonality_peak_month": 6,
        "mean_reversion": 0.03,
        "long_run_mean": 6.10,
        "description": "CBOT SRW Wheat front-month continuous futures",
    },
    "rice": {
        "name": "Rice",
        "unit": "$/cwt",
        "category": "Agriculture",
        "emoji": "🍚",
        "base_price": 18.40,
        "daily_vol": 0.009,
        "trend": 0.00015,
        "seasonality_amp": 0.06,
        "seasonality_peak_month": 10,
        "mean_reversion": 0.025,
        "long_run_mean": 18.80,
        "description": "CBOT Rough Rice front-month continuous futures",
    },
    "coffee": {
        "name": "Coffee",
        "unit": "¢/lb",
        "category": "Agriculture",
        "emoji": "☕",
        "base_price": 185.0,
        "daily_vol": 0.018,
        "trend": 0.0002,
        "seasonality_amp": 0.10,
        "seasonality_peak_month": 5,
        "mean_reversion": 0.02,
        "long_run_mean": 190.0,
        "description": "ICE Arabica Coffee front-month continuous futures",
    },
    "sugar": {
        "name": "Sugar #11",
        "unit": "¢/lb",
        "category": "Agriculture",
        "emoji": "🍬",
        "base_price": 22.5,
        "daily_vol": 0.017,
        "trend": 0.00005,
        "seasonality_amp": 0.09,
        "seasonality_peak_month": 4,
        "mean_reversion": 0.035,
        "long_run_mean": 22.0,
        "description": "ICE Sugar #11 front-month continuous futures",
    },
    "cocoa": {
        "name": "Cocoa",
        "unit": "$/MT",
        "category": "Agriculture",
        "emoji": "🍫",
        "base_price": 4200.0,
        "daily_vol": 0.014,
        "trend": 0.0003,
        "seasonality_amp": 0.08,
        "seasonality_peak_month": 10,
        "mean_reversion": 0.02,
        "long_run_mean": 4300.0,
        "description": "ICE Cocoa front-month continuous futures",
    },
    "cotton": {
        "name": "Cotton",
        "unit": "¢/lb",
        "category": "Agriculture",
        "emoji": "🧶",
        "base_price": 78.5,
        "daily_vol": 0.013,
        "trend": 0.0001,
        "seasonality_amp": 0.08,
        "seasonality_peak_month": 9,
        "mean_reversion": 0.03,
        "long_run_mean": 80.0,
        "description": "ICE Cotton #2 front-month continuous futures",
    },
    # Metals
    "gold": {
        "name": "Gold",
        "unit": "$/troy oz",
        "category": "Metals",
        "emoji": "🥇",
        "base_price": 2050.0,
        "daily_vol": 0.009,
        "trend": 0.0003,
        "seasonality_amp": 0.03,
        "seasonality_peak_month": 9,
        "mean_reversion": 0.01,
        "long_run_mean": 2100.0,
        "description": "COMEX Gold front-month continuous futures",
    },
    "silver": {
        "name": "Silver",
        "unit": "$/troy oz",
        "category": "Metals",
        "emoji": "🥈",
        "base_price": 24.5,
        "daily_vol": 0.016,
        "trend": 0.0002,
        "seasonality_amp": 0.04,
        "seasonality_peak_month": 9,
        "mean_reversion": 0.02,
        "long_run_mean": 25.5,
        "description": "COMEX Silver front-month continuous futures",
    },
    "copper": {
        "name": "Copper",
        "unit": "$/lb",
        "category": "Metals",
        "emoji": "🟤",
        "base_price": 3.90,
        "daily_vol": 0.013,
        "trend": 0.0001,
        "seasonality_amp": 0.04,
        "seasonality_peak_month": 3,
        "mean_reversion": 0.03,
        "long_run_mean": 4.00,
        "description": "COMEX Copper front-month continuous futures",
    },
    "aluminum": {
        "name": "Aluminum",
        "unit": "$/MT",
        "category": "Metals",
        "emoji": "⚙️",
        "base_price": 2280.0,
        "daily_vol": 0.011,
        "trend": 0.00005,
        "seasonality_amp": 0.03,
        "seasonality_peak_month": 4,
        "mean_reversion": 0.025,
        "long_run_mean": 2300.0,
        "description": "LME Aluminum front-month continuous futures",
    },
    "nickel": {
        "name": "Nickel",
        "unit": "$/MT",
        "category": "Metals",
        "emoji": "🔩",
        "base_price": 16800.0,
        "daily_vol": 0.020,
        "trend": -0.0001,
        "seasonality_amp": 0.04,
        "seasonality_peak_month": 5,
        "mean_reversion": 0.03,
        "long_run_mean": 16500.0,
        "description": "LME Nickel front-month continuous futures",
    },
    # Livestock
    "live_cattle": {
        "name": "Live Cattle",
        "unit": "¢/lb",
        "category": "Livestock",
        "emoji": "🐄",
        "base_price": 182.0,
        "daily_vol": 0.007,
        "trend": 0.0002,
        "seasonality_amp": 0.05,
        "seasonality_peak_month": 3,
        "mean_reversion": 0.04,
        "long_run_mean": 185.0,
        "description": "CME Live Cattle front-month continuous futures",
    },
    "feeder_cattle": {
        "name": "Feeder Cattle",
        "unit": "¢/lb",
        "category": "Livestock",
        "emoji": "🐂",
        "base_price": 248.0,
        "daily_vol": 0.008,
        "trend": 0.0002,
        "seasonality_amp": 0.06,
        "seasonality_peak_month": 9,
        "mean_reversion": 0.04,
        "long_run_mean": 252.0,
        "description": "CME Feeder Cattle front-month continuous futures",
    },
    "lean_hogs": {
        "name": "Lean Hogs",
        "unit": "¢/lb",
        "category": "Livestock",
        "emoji": "🐷",
        "base_price": 84.0,
        "daily_vol": 0.014,
        "trend": 0.0001,
        "seasonality_amp": 0.12,
        "seasonality_peak_month": 5,
        "mean_reversion": 0.05,
        "long_run_mean": 85.0,
        "description": "CME Lean Hogs front-month continuous futures",
    },
}


def generate_price_series(config, n_days=N_DAYS, seed_offset=0):
    """
    Generate realistic commodity price series using:
    - Log-OU (mean-reverting) base process
    - Sinusoidal seasonality
    - Fat-tail shocks (occasional jump events)
    - Momentum persistence
    """
    rng = np.random.default_rng(SEED + seed_offset)

    price = config["base_price"]
    kappa = config["mean_reversion"]
    theta = config["long_run_mean"]
    sigma = config["daily_vol"]
    trend = config["trend"]
    amp   = config["seasonality_amp"]
    peak  = config["seasonality_peak_month"]

    prices = []
    current_date = START_DATE
    log_price = np.log(price)
    log_theta = np.log(theta)
    momentum  = 0.0

    for i in range(n_days):
        if current_date.weekday() >= 5:  # skip weekends
            current_date += timedelta(days=1)
            continue

        # Seasonality: sinusoidal by day-of-year
        doy = current_date.timetuple().tm_yday
        peak_doy = (peak - 1) * 30 + 15
        season = amp * np.sin(2 * np.pi * (doy - peak_doy) / 365)

        # OU mean reversion in log-space
        log_theta_adj = log_theta + trend * i + season
        ou_drift = kappa * (log_theta_adj - log_price)

        # Random shock (fat-tailed via t-distribution)
        shock = rng.standard_t(df=5) * sigma
        # Rare large shock (geopolitical, weather event)
        if rng.random() < 0.005:
            shock += rng.choice([-1, 1]) * rng.uniform(3, 6) * sigma

        # Momentum (short-term persistence)
        momentum = 0.15 * momentum + shock
        log_price = log_price + ou_drift + momentum

        prices.append({
            "date": current_date.strftime("%Y-%m-%d"),
            "price": round(float(np.exp(log_price)), 4),
        })
        current_date += timedelta(days=1)

    df = pd.DataFrame(prices)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df


def generate_all():
    print("Generating commodity price data...")
    for key, cfg in COMMODITIES.items():
        seed_off = list(COMMODITIES.keys()).index(key) * 100
        df = generate_price_series(cfg, seed_offset=seed_off)
        out_path = os.path.join(OUTPUT_DIR, f"{key}.csv")
        df.to_csv(out_path)
        print(f"  ✓ {cfg['name']:20s}  {len(df):4d} rows  → {out_path}")
    print(f"Done. {len(COMMODITIES)} files written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    generate_all()
