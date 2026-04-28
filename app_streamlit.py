#!/usr/bin/env python3
"""
Commodity Intelligence Dashboard
Streamlit-based enterprise-grade commodity price analysis and forecasting tool.
Run with: streamlit run app_streamlit.py
"""

import os
import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# Local modules
from generate_data import COMMODITIES, generate_price_series, OUTPUT_DIR
import models as mdl

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Commodity Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── THEME / CSS ───────────────────────────────────────────────────────────────

DARK_BG      = "#080c12"
SURFACE      = "#0e1520"
SURFACE2     = "#141e2e"
BORDER       = "#1e2d45"
TEXT         = "#d8e4f0"
MUTED        = "#5a7a9a"
ACCENT_BLUE  = "#00c8f0"
ACCENT_GOLD  = "#f5a623"
GREEN        = "#27e8a0"
RED          = "#ff4d6a"
GOLD         = "#ffd060"
PURPLE       = "#9d7aff"

PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE2,
        font=dict(color=TEXT, family="IBM Plex Mono, monospace", size=11),
        xaxis=dict(gridcolor=BORDER, linecolor=BORDER, zerolinecolor=BORDER),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER, zerolinecolor=BORDER),
        legend=dict(bgcolor="rgba(14,21,32,0.8)", bordercolor=BORDER, borderwidth=1),
        margin=dict(l=10, r=10, t=36, b=10),
    )
)

CATEGORY_COLORS = {
    "Agriculture": ACCENT_GOLD,
    "Metals":      ACCENT_BLUE,
    "Livestock":   GREEN,
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #080c12;
    color: #d8e4f0;
}

/* Main app background */
.stApp {
    background: #080c12;
    background-image:
        radial-gradient(ellipse at 15% 0%, rgba(0,200,240,0.04) 0%, transparent 55%),
        radial-gradient(ellipse at 85% 100%, rgba(245,166,35,0.04) 0%, transparent 55%);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0e1520;
    border-right: 1px solid #1e2d45;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stSlider label {
    color: #5a7a9a !important;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-family: 'IBM Plex Mono', monospace;
}

/* Metrics */
[data-testid="metric-container"] {
    background: #0e1520;
    border: 1px solid #1e2d45;
    border-radius: 8px;
    padding: 12px 16px;
}
[data-testid="metric-container"] label {
    color: #5a7a9a !important;
    font-size: 9px !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-family: 'IBM Plex Mono', monospace !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 20px !important;
    font-weight: 700 !important;
    color: #d8e4f0 !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
}

/* Section headers */
.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    font-weight: 700;
    color: #5a7a9a;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    border-bottom: 1px solid #1e2d45;
    padding-bottom: 8px;
    margin-top: 28px;
    margin-bottom: 14px;
}

/* Risk badge */
.risk-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
}

/* Narrative box */
.narrative-box {
    background: #0e1520;
    border: 1px solid #1e2d45;
    border-radius: 8px;
    padding: 16px 20px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    line-height: 1.8;
    color: #8aa0bc;
    white-space: pre-wrap;
}

/* Commodity pill */
.commodity-pill {
    display: inline-block;
    background: #141e2e;
    border: 1px solid #1e2d45;
    border-radius: 20px;
    padding: 4px 14px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #5a7a9a;
    margin: 2px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #00c8f0, #0090c0);
    color: #000;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 700;
    font-size: 12px;
    border: none;
    border-radius: 6px;
    padding: 8px 20px;
    letter-spacing: 0.5px;
    transition: opacity 0.15s;
    width: 100%;
}
.stButton > button:hover { opacity: 0.88; }

/* Selectbox */
.stSelectbox > div > div {
    background: #0e1520 !important;
    border: 1px solid #1e2d45 !important;
    color: #d8e4f0 !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* Slider */
.stSlider > div > div > div > div {
    background: #00c8f0 !important;
}

/* Horizontal rule */
hr { border-color: #1e2d45 !important; }

/* Info/warning boxes */
.stInfo, .stWarning, .stSuccess, .stError {
    background: #0e1520 !important;
    border-color: #1e2d45 !important;
    color: #d8e4f0 !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0e1520; }
::-webkit-scrollbar-thumb { background: #1e2d45; border-radius: 3px; }

/* Header area */
.dashboard-header {
    border-bottom: 1px solid #1e2d45;
    padding-bottom: 16px;
    margin-bottom: 24px;
}
.dashboard-title {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 22px;
    font-weight: 700;
    color: #d8e4f0;
    letter-spacing: 0.3px;
}
.dashboard-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #5a7a9a;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)


# ── DATA LOADING ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_commodity(key: str) -> pd.DataFrame:
    """Load or generate commodity price data."""
    csv_path = os.path.join(OUTPUT_DIR, f"{key}.csv")
    if not os.path.exists(csv_path):
        cfg = COMMODITIES[key]
        seed_off = list(COMMODITIES.keys()).index(key) * 100
        df = generate_price_series(cfg, seed_offset=seed_off)
        df.to_csv(csv_path)
    else:
        df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
    df = df.sort_index()
    # Ensure we only use data up to today
    df = df[df.index <= pd.Timestamp.today()]
    return df


def prepare_data(df: pd.DataFrame, lookback_days: int = 365):
    """Slice history, compute returns, moving averages, rolling vol."""
    df = df.tail(lookback_days).copy()
    prices = df["price"].values
    returns = mdl.compute_returns(prices)
    df["ma7"]  = mdl.rolling_ma(df["price"], 7)
    df["ma30"] = mdl.rolling_ma(df["price"], 30)
    df["vol20"] = mdl.rolling_vol(df["price"], 20)
    return df, prices, returns


# ── CHART HELPERS ─────────────────────────────────────────────────────────────

def apply_theme(fig):
    fig.update_layout(
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE2,
        font=dict(color=TEXT, family="IBM Plex Mono, monospace", size=11),
        xaxis=dict(gridcolor=BORDER, linecolor=BORDER, zerolinecolor=BORDER, showgrid=True),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER, zerolinecolor=BORDER, showgrid=True),
        legend=dict(bgcolor="rgba(14,21,32,0.8)", bordercolor=BORDER, borderwidth=1,
                    font=dict(size=10)),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def future_dates(last_date: pd.Timestamp, n_days: int) -> list:
    """Generate n_days of weekday-only future dates from last_date."""
    dates = []
    d = last_date
    while len(dates) < n_days:
        d += datetime.timedelta(days=1)
        if d.weekday() < 5:
            dates.append(d)
    return dates


# ── CHART: PRICE HISTORY + FORECAST ──────────────────────────────────────────

def chart_price_forecast(df, forecast, cfg, horizon_days):
    last_date  = df.index[-1]
    current    = df["price"].iloc[-1]
    fcast_dates = future_dates(last_date, horizon_days)
    cat_color   = CATEGORY_COLORS.get(cfg["category"], ACCENT_BLUE)

    fig = go.Figure()

    # MA bands (fill between)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["ma30"],
        mode="lines", name="30d MA",
        line=dict(color=ACCENT_GOLD, width=1.5, dash="dot"),
        opacity=0.7,
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["ma7"],
        mode="lines", name="7d MA",
        line=dict(color=GOLD, width=1.2, dash="dash"),
        opacity=0.7,
    ))

    # Main price line
    fig.add_trace(go.Scatter(
        x=df.index, y=df["price"],
        mode="lines", name="Price",
        line=dict(color=cat_color, width=2.5),
        fill="tozeroy",
        fillcolor=f"rgba({int(cat_color[1:3],16)},{int(cat_color[3:5],16)},{int(cat_color[5:7],16)},0.06)",
    ))

    # Forecast CI band
    fig.add_trace(go.Scatter(
        x=fcast_dates, y=forecast["path_high"],
        mode="lines", name="90% CI High",
        line=dict(color=GREEN, width=0, dash="dot"),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=fcast_dates, y=forecast["path_low"],
        mode="lines", name="90% CI",
        fill="tonexty", fillcolor="rgba(39,232,160,0.10)",
        line=dict(color=GREEN, width=0.8, dash="dot"),
    ))

    # Forecast midline
    fig.add_trace(go.Scatter(
        x=[last_date] + fcast_dates,
        y=[current] + list(forecast["path_mid"]),
        mode="lines", name="Forecast (median)",
        line=dict(color=GREEN, width=2, dash="dash"),
    ))

    # Current price marker
    fig.add_trace(go.Scatter(
        x=[last_date], y=[current],
        mode="markers+text",
        marker=dict(color="white", size=8, line=dict(color=cat_color, width=2)),
        text=[f"  {cfg['unit'].split('/')[0]}{current:.2f}"],
        textfont=dict(color="white", size=10),
        textposition="middle right",
        name="Current",
        showlegend=False,
    ))

    # Vertical separator
    fig.add_vline(x=last_date, line_dash="dash", line_color=MUTED, line_width=1, opacity=0.5)

    fig.update_layout(
        title=dict(text=f"{cfg['emoji']} {cfg['name']} — Price History & {horizon_days}-Day Forecast",
                   font=dict(size=13, color=TEXT), x=0.01),
        xaxis_title="",
        yaxis_title=cfg["unit"],
        height=380,
        hovermode="x unified",
    )
    apply_theme(fig)
    return fig


# ── CHART: ROLLING VOLATILITY ─────────────────────────────────────────────────

def chart_volatility(df, cfg):
    cat_color = CATEGORY_COLORS.get(cfg["category"], ACCENT_BLUE)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["vol20"],
        mode="lines", name="20d Rolling Vol",
        line=dict(color=RED, width=2),
        fill="tozeroy",
        fillcolor="rgba(255,77,106,0.08)",
    ))
    # Horizontal level lines
    for level, label, color in [(15, "Low (<15%)", GREEN), (30, "High (>30%)", RED)]:
        fig.add_hline(y=level, line_dash="dot", line_color=color, line_width=1, opacity=0.5,
                      annotation_text=label, annotation_font_color=color, annotation_font_size=9)
    fig.update_layout(
        title=dict(text="📉 20-Day Rolling Annualised Volatility (%)", font=dict(size=12, color=TEXT), x=0.01),
        yaxis_title="Annualised Vol (%)",
        height=260,
        hovermode="x unified",
    )
    apply_theme(fig)
    return fig


# ── CHART: PROBABILITY DISTRIBUTION ──────────────────────────────────────────

def chart_prob_distribution(current, returns, horizon_days, cfg):
    centers, probs, edges = mdl.price_probability_bins(current, returns, horizon_days)
    cat_color = CATEGORY_COLORS.get(cfg["category"], ACCENT_BLUE)

    # Color bars: red below current, green above
    colors = [RED if c < current else GREEN for c in centers]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=centers, y=probs * 100,
        marker_color=colors,
        marker_line_color="rgba(0,0,0,0)",
        name="Probability",
        hovertemplate=f"Price: {cfg['unit'].split('/')[0]}%{{x:.2f}}<br>Prob: %{{y:.1f}}%<extra></extra>",
    ))
    fig.add_vline(x=current, line_dash="dash", line_color="white", line_width=2,
                  annotation_text="Current", annotation_font_color="white", annotation_font_size=9)
    fig.update_layout(
        title=dict(text=f"🎲 Price Probability Distribution ({horizon_days}-Day Horizon)",
                   font=dict(size=12, color=TEXT), x=0.01),
        xaxis_title=cfg["unit"],
        yaxis_title="Probability (%)",
        height=280,
        bargap=0.05,
        showlegend=False,
    )
    apply_theme(fig)
    return fig


# ── CHART: SCENARIO COMPARISON ────────────────────────────────────────────────

def chart_scenarios(current, returns, horizon_days, cfg):
    scen_results = mdl.scenario_forecast(current, returns, horizon_days)
    last_date    = pd.Timestamp.today().normalize()
    fcast_dates  = future_dates(last_date, horizon_days)

    colors = {
        "Base Case":      ACCENT_BLUE,
        "High Demand":    GREEN,
        "Supply Shock":   ACCENT_GOLD,
        "Recession":      RED,
        "Bumper Harvest": PURPLE,
    }
    dashes = {
        "Base Case":      "solid",
        "High Demand":    "dash",
        "Supply Shock":   "dot",
        "Recession":      "dashdot",
        "Bumper Harvest": "longdash",
    }

    fig = go.Figure()
    for name, res in scen_results.items():
        color = colors.get(name, MUTED)
        fig.add_trace(go.Scatter(
            x=fcast_dates, y=res["median_path"],
            mode="lines", name=name,
            line=dict(color=color, width=2, dash=dashes.get(name, "solid")),
            hovertemplate=f"{name}: {cfg['unit'].split('/')[0]}%{{y:.4f}}<extra></extra>",
        ))
    fig.add_hline(y=current, line_dash="dot", line_color="white", line_width=1, opacity=0.4,
                  annotation_text="Current Price", annotation_font_color="white", annotation_font_size=9)
    fig.update_layout(
        title=dict(text=f"🗺️ Scenario Comparison ({horizon_days}-Day Horizon)",
                   font=dict(size=12, color=TEXT), x=0.01),
        xaxis_title="",
        yaxis_title=cfg["unit"],
        height=300,
        hovermode="x unified",
    )
    apply_theme(fig)
    return fig


# ── CHART: DRIVER IMPORTANCE ──────────────────────────────────────────────────

def chart_drivers(category, current_vol):
    drivers = mdl.driver_importance(category, current_vol)
    items   = sorted(drivers.items(), key=lambda x: x[1])
    names   = [k for k, _ in items]
    vals    = [v * 100 for _, v in items]
    colors  = [CATEGORY_COLORS.get(category, ACCENT_BLUE)] * len(names)
    # Top driver highlighted
    colors[-1] = ACCENT_GOLD

    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker_color=colors,
        marker_line_color="rgba(0,0,0,0)",
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="⚡ Price Driver Importance (%)",
                   font=dict(size=12, color=TEXT), x=0.01),
        xaxis_title="Importance (%)",
        height=300,
        showlegend=False,
    )
    apply_theme(fig)
    return fig


# ── CHART: MULTI-COMMODITY COMPARISON ────────────────────────────────────────

def chart_multi_commodity(selected_keys):
    fig = go.Figure()
    for key in selected_keys:
        cfg = COMMODITIES[key]
        df  = load_commodity(key)
        df  = df.tail(90)
        # Normalise to 100 for comparison
        base = df["price"].iloc[0]
        normalised = df["price"] / base * 100
        color = CATEGORY_COLORS.get(cfg["category"], ACCENT_BLUE)
        fig.add_trace(go.Scatter(
            x=df.index, y=normalised,
            mode="lines", name=f"{cfg['emoji']} {cfg['name']}",
            line=dict(width=2),
        ))
    fig.add_hline(y=100, line_dash="dot", line_color=MUTED, line_width=1, opacity=0.5)
    fig.update_layout(
        title=dict(text="📊 90-Day Normalised Performance (Base = 100)",
                   font=dict(size=12, color=TEXT), x=0.01),
        yaxis_title="Indexed Price (Base=100)",
        height=320,
        hovermode="x unified",
    )
    apply_theme(fig)
    return fig


# ── MARKET SUMMARY TEXT ───────────────────────────────────────────────────────

def generate_summary(cfg, current, forecast, risk, ols, horizon_days):
    direction = "▲ HIGHER" if forecast["change_pct"] > 0.5 else \
                "▼ LOWER"  if forecast["change_pct"] < -0.5 else "→ FLAT"
    lines = [
        f"=== {cfg['name'].upper()} — MARKET ANALYSIS | {datetime.date.today()} ===",
        "",
        "PRICE SNAPSHOT",
        f"  Current price : {current:.4f} {cfg['unit']}",
        f"  {horizon_days}d forecast   : {forecast['low']:.4f} – {forecast['high']:.4f}",
        f"  Forecast mid  : {forecast['mid']:.4f}  ({direction}  {forecast['change_pct']:+.1f}%)",
        "",
        "VOLATILITY & RISK",
        f"  Annualised vol: {risk['ann_vol_pct']:.1f}%  [{risk['risk_level']}]",
        f"  VaR (95%, {horizon_days}d): {risk['var_95']:+.1f}%",
        f"  CVaR (95%):    {risk['cvar_95']:+.1f}%",
        f"  P(price up)  : {risk['prob_up']:.1f}%",
        f"  P(price down): {risk['prob_down']:.1f}%",
        "",
        "TREND",
        f"  OLS slope (90d): {ols['slope']:+.4f} per day",
        f"  Trend R²       : {ols['r2']:.4f}  {'(strong)' if ols['r2'] > 0.6 else '(weak/noisy)'}",
        "",
        "MODEL",
        f"  {forecast['model']}",
        f"  Horizon: {horizon_days} trading days | Category: {cfg['category']}",
        "",
        "NOTE: Probabilities are model outputs. Not financial advice.",
    ]
    return "\n".join(lines)


# ── SIDEBAR ───────────────────────────────────────────────────────────────────

def build_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding: 16px 0 20px 0; border-bottom: 1px solid #1e2d45; margin-bottom: 20px;">
            <div style="font-size:24px; margin-bottom:6px;">📊</div>
            <div style="font-family:'IBM Plex Sans',sans-serif; font-weight:700; font-size:14px; color:#d8e4f0;">Commodity Intelligence</div>
            <div style="font-family:'IBM Plex Mono',monospace; font-size:9px; color:#5a7a9a; margin-top:2px;">DETERMINISTIC PRICE ENGINE v1.0</div>
        </div>
        """, unsafe_allow_html=True)

        # Category filter
        st.markdown('<div style="font-family:IBM Plex Mono,monospace;font-size:9px;color:#5a7a9a;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">CATEGORY</div>', unsafe_allow_html=True)
        categories = ["All"] + sorted(set(c["category"] for c in COMMODITIES.values()))
        selected_category = st.selectbox("", categories, label_visibility="collapsed")

        # Filter commodities by category
        if selected_category == "All":
            filtered_keys = list(COMMODITIES.keys())
        else:
            filtered_keys = [k for k, v in COMMODITIES.items() if v["category"] == selected_category]

        # Commodity selector
        st.markdown('<div style="font-family:IBM Plex Mono,monospace;font-size:9px;color:#5a7a9a;text-transform:uppercase;letter-spacing:1px;margin-top:16px;margin-bottom:6px;">COMMODITY</div>', unsafe_allow_html=True)
        commodity_options = {k: f"{COMMODITIES[k]['emoji']} {COMMODITIES[k]['name']}" for k in filtered_keys}
        selected_key = st.selectbox("", list(commodity_options.keys()),
                                     format_func=lambda k: commodity_options[k],
                                     label_visibility="collapsed")

        st.markdown("---")

        # Horizon
        st.markdown('<div style="font-family:IBM Plex Mono,monospace;font-size:9px;color:#5a7a9a;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">FORECAST HORIZON</div>', unsafe_allow_html=True)
        horizon_label = st.selectbox("", list(mdl.HORIZON_MAP.keys()), index=1, label_visibility="collapsed")
        horizon_days  = mdl.HORIZON_MAP[horizon_label]

        # Lookback
        st.markdown('<div style="font-family:IBM Plex Mono,monospace;font-size:9px;color:#5a7a9a;text-transform:uppercase;letter-spacing:1px;margin-top:16px;margin-bottom:6px;">HISTORY WINDOW</div>', unsafe_allow_html=True)
        lookback_days = st.slider("", 90, 825, 365, step=30, label_visibility="collapsed")
        st.markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:9px;color:#00c8f0;text-align:right;margin-top:-8px;">{lookback_days} trading days</div>', unsafe_allow_html=True)

        st.markdown("---")

        # Run button
        run_btn = st.button("▶ RUN ANALYSIS", use_container_width=True)

        st.markdown("---")

        # Multi-compare
        st.markdown('<div style="font-family:IBM Plex Mono,monospace;font-size:9px;color:#5a7a9a;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">COMPARE COMMODITIES</div>', unsafe_allow_html=True)
        all_options = {k: f"{COMMODITIES[k]['emoji']} {COMMODITIES[k]['name']}" for k in COMMODITIES}
        compare_keys = st.multiselect(
            "", list(all_options.keys()),
            default=[selected_key],
            format_func=lambda k: all_options[k],
            label_visibility="collapsed",
            max_selections=5,
        )

        st.markdown("---")

        # Info
        st.markdown(f"""
        <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:#2e4560;line-height:1.8;">
        Models: 55% log-normal<br>+ 45% bootstrap<br><br>
        Data: simulated from<br>realistic OU process<br><br>
        ⚠ Not financial advice
        </div>
        """, unsafe_allow_html=True)

    return selected_key, selected_category, horizon_days, horizon_label, lookback_days, run_btn, compare_keys


# ── MAIN DASHBOARD ────────────────────────────────────────────────────────────

def main():
    selected_key, selected_category, horizon_days, horizon_label, lookback_days, run_btn, compare_keys = build_sidebar()

    cfg = COMMODITIES[selected_key]
    cat_color = CATEGORY_COLORS.get(cfg["category"], ACCENT_BLUE)

    # ── HEADER ──
    col_title, col_badge = st.columns([3, 1])
    with col_title:
        st.markdown(f"""
        <div class="dashboard-header">
            <div style="display:flex;align-items:center;gap:14px;">
                <div style="width:42px;height:42px;border-radius:10px;
                     background:linear-gradient(135deg,{cat_color},{ACCENT_GOLD});
                     display:flex;align-items:center;justify-content:center;font-size:22px;">
                    {cfg['emoji']}
                </div>
                <div>
                    <div class="dashboard-title">Commodity Intelligence Dashboard</div>
                    <div class="dashboard-sub">
                        {cfg['name'].upper()} · {cfg['category'].upper()} · {cfg['description']}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_badge:
        st.markdown(f"""
        <div style="text-align:right;padding-top:10px;">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:#5a7a9a;letter-spacing:1px;">HORIZON</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:20px;font-weight:700;color:{ACCENT_BLUE};">{horizon_label}</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:#5a7a9a;">{horizon_days} TRADING DAYS</div>
        </div>
        """, unsafe_allow_html=True)

    # ── LOAD & COMPUTE ──
    if "last_key" not in st.session_state or run_btn or st.session_state.get("last_key") != selected_key or \
       st.session_state.get("last_horizon") != horizon_days or st.session_state.get("last_lookback") != lookback_days:

        with st.spinner("Computing models..."):
            df = load_commodity(selected_key)
            df, prices, returns = prepare_data(df, lookback_days)
            current  = float(df["price"].iloc[-1])
            forecast = mdl.ensemble_forecast(current, returns, horizon_days)
            risk     = mdl.risk_metrics(current, returns, horizon_days)
            ols      = mdl.ols_regression(prices[-90:] if len(prices) >= 90 else prices)
            summary  = generate_summary(cfg, current, forecast, risk, ols, horizon_days)
            ann_vol  = mdl.daily_vol(returns)

        st.session_state.update({
            "last_key": selected_key,
            "last_horizon": horizon_days,
            "last_lookback": lookback_days,
            "df": df,
            "prices": prices,
            "returns": returns,
            "current": current,
            "forecast": forecast,
            "risk": risk,
            "ols": ols,
            "summary": summary,
            "ann_vol": ann_vol,
        })

    # Retrieve from state
    df       = st.session_state["df"]
    prices   = st.session_state["prices"]
    returns  = st.session_state["returns"]
    current  = st.session_state["current"]
    forecast = st.session_state["forecast"]
    risk     = st.session_state["risk"]
    ols      = st.session_state["ols"]
    summary  = st.session_state["summary"]
    ann_vol  = st.session_state["ann_vol"]

    # ── KPI METRICS ──
    st.markdown('<div class="section-header">KPI SNAPSHOT</div>', unsafe_allow_html=True)
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    unit_sym = cfg["unit"].split("/")[0].replace("$", "$").replace("¢", "¢")
    prev_price = float(df["price"].iloc[-2]) if len(df) > 1 else current
    chg_1d = (current - prev_price) / prev_price * 100

    with k1:
        st.metric("CURRENT PRICE", f"{unit_sym}{current:.4f}", f"{chg_1d:+.2f}% 1d")
    with k2:
        st.metric("FORECAST MID", f"{unit_sym}{forecast['mid']:.4f}", f"{forecast['change_pct']:+.1f}% vs now")
    with k3:
        st.metric("FORECAST LOW", f"{unit_sym}{forecast['low']:.4f}", "90% CI floor")
    with k4:
        st.metric("FORECAST HIGH", f"{unit_sym}{forecast['high']:.4f}", "90% CI ceiling")
    with k5:
        st.metric("ANN. VOL", f"{risk['ann_vol_pct']:.1f}%", risk["risk_level"])
    with k6:
        st.metric("P(PRICE UP)", f"{risk['prob_up']:.1f}%", f"VaR95: {risk['var_95']:+.1f}%")

    # ── RISK BADGE ──
    risk_bg = {"LOW": "rgba(39,232,160,0.15)", "MEDIUM": "rgba(255,208,96,0.15)", "HIGH": "rgba(255,77,106,0.15)"}
    risk_border = {"LOW": GREEN, "MEDIUM": GOLD, "HIGH": RED}
    rc = risk["risk_color"]
    rl = risk["risk_level"]
    st.markdown(f"""
    <div style="margin:12px 0;display:flex;align-items:center;gap:12px;">
        <div style="background:{risk_bg.get(rl,'#141e2e')};border:1px solid {risk_border.get(rl,BORDER)};
             border-radius:6px;padding:6px 18px;font-family:'IBM Plex Mono',monospace;font-size:12px;
             font-weight:700;color:{rc};letter-spacing:1px;">
            RISK LEVEL: {rl}
        </div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:{MUTED};">
            {cfg['category']} · {horizon_days}d horizon · Model: {forecast['model']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── MAIN PRICE CHART ──
    st.markdown('<div class="section-header">PRICE HISTORY & FORECAST<span style="font-weight:400;font-size:9px;color:#2e4560;margin-left:12px;">Scroll to zoom · Hover for details · Toggle overlays in legend</span></div>', unsafe_allow_html=True)
    st.plotly_chart(chart_price_forecast(df, forecast, cfg, horizon_days),
                    use_container_width=True, config={"displayModeBar": False})

    # ── VOLATILITY + PROBABILITY ──
    st.markdown('<div class="section-header">VOLATILITY & PROBABILITY ANALYSIS</div>', unsafe_allow_html=True)
    col_vol, col_prob = st.columns(2)
    with col_vol:
        st.plotly_chart(chart_volatility(df, cfg), use_container_width=True,
                        config={"displayModeBar": False})
    with col_prob:
        st.plotly_chart(chart_prob_distribution(current, returns, horizon_days, cfg),
                        use_container_width=True, config={"displayModeBar": False})

    # ── SCENARIOS + DRIVERS ──
    st.markdown('<div class="section-header">SCENARIO ANALYSIS & PRICE DRIVERS</div>', unsafe_allow_html=True)
    col_scen, col_drv = st.columns(2)
    with col_scen:
        st.plotly_chart(chart_scenarios(current, returns, horizon_days, cfg),
                        use_container_width=True, config={"displayModeBar": False})
    with col_drv:
        st.plotly_chart(chart_drivers(cfg["category"], ann_vol),
                        use_container_width=True, config={"displayModeBar": False})

    # ── SCENARIO TABLE ──
    st.markdown('<div class="section-header">SCENARIO OUTCOME TABLE</div>', unsafe_allow_html=True)
    scen_results = mdl.scenario_forecast(current, returns, horizon_days)
    scen_rows = []
    for name, res in scen_results.items():
        chg = (res["final_median"] - current) / current * 100
        scen_rows.append({
            "Scenario": name,
            "Description": res["label"],
            f"Median ({unit_sym})": f"{res['final_median']:.4f}",
            f"Low 10% ({unit_sym})": f"{res['final_p10']:.4f}",
            f"High 90% ({unit_sym})": f"{res['final_p90']:.4f}",
            "Change vs Now": f"{chg:+.1f}%",
        })
    scen_df = pd.DataFrame(scen_rows)
    st.dataframe(
        scen_df,
        use_container_width=True,
        hide_index=True,
    )

    # ── MULTI-COMMODITY COMPARISON ──
    if len(compare_keys) > 1:
        st.markdown('<div class="section-header">MULTI-COMMODITY PERFORMANCE COMPARISON</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_multi_commodity(compare_keys), use_container_width=True,
                        config={"displayModeBar": False})

    # ── OLS TABLE ──
    st.markdown('<div class="section-header">STATISTICAL SUMMARY</div>', unsafe_allow_html=True)
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        stat_data = {
            "Metric": ["Current Price", "7d MA", "30d MA", "90d OLS Slope", "OLS R²", "Ann. Volatility", "VaR 95%", "CVaR 95%", "P(Up)", "P(Down)"],
            "Value": [
                f"{unit_sym}{current:.4f}",
                f"{unit_sym}{df['ma7'].iloc[-1]:.4f}" if not pd.isna(df['ma7'].iloc[-1]) else "N/A",
                f"{unit_sym}{df['ma30'].iloc[-1]:.4f}" if not pd.isna(df['ma30'].iloc[-1]) else "N/A",
                f"{ols['slope']:+.4f}/day",
                f"{ols['r2']:.4f}",
                f"{risk['ann_vol_pct']:.1f}%",
                f"{risk['var_95']:+.1f}%",
                f"{risk['cvar_95']:+.1f}%",
                f"{risk['prob_up']:.1f}%",
                f"{risk['prob_down']:.1f}%",
            ],
        }
        st.dataframe(pd.DataFrame(stat_data), use_container_width=True, hide_index=True)

    with col_stat2:
        driver_data = mdl.driver_importance(cfg["category"], ann_vol)
        drv_df = pd.DataFrame([
            {"Driver": k, "Importance": f"{v*100:.1f}%"}
            for k, v in sorted(driver_data.items(), key=lambda x: -x[1])
        ])
        st.dataframe(drv_df, use_container_width=True, hide_index=True)

    # ── MARKET SUMMARY ──
    st.markdown('<div class="section-header">MARKET SUMMARY</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="narrative-box">{summary}</div>', unsafe_allow_html=True)

    # ── FOOTER ──
    st.markdown("""
    <div style="margin-top:48px;padding:16px 0;border-top:1px solid #1e2d45;
         font-family:'IBM Plex Mono',monospace;font-size:9px;color:#2e4560;text-align:center;">
        Commodity Intelligence Dashboard · Deterministic Price Engine ·
        Models: 55% Log-Normal + 45% Historical Bootstrap ·
        Data: Synthetic OU Process · Not financial advice.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
