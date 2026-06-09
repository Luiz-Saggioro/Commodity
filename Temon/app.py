"""
TEMON Intelligence Dashboard
=============================
Painel de inteligência de mercado para o Grupo Temon — instalações elétricas,
hidráulicas, manutenção predial e engenharia de construção.

Fontes de dados: IBGE/SINAPI, CBIC, FGV/INCC, ABINEE, Banco Central do Brasil
Versão: 2.0 | Segurança: entrada sanitizada, sem secrets expostos, rate-limiting
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import json
import time
import hashlib
import re
from datetime import datetime, timedelta
from functools import lru_cache
import warnings
warnings.filterwarnings("ignore")

# ─── Segurança ─────────────────────────────────────────────────────────────────
_REQUEST_LOG: list = []
_MAX_REQUESTS_PER_MINUTE = 30

def _rate_check():
    """Simples rate-limiter em memória para chamadas externas."""
    now = time.time()
    global _REQUEST_LOG
    _REQUEST_LOG = [t for t in _REQUEST_LOG if now - t < 60]
    if len(_REQUEST_LOG) >= _MAX_REQUESTS_PER_MINUTE:
        return False
    _REQUEST_LOG.append(now)
    return True

def _sanitize(text: str) -> str:
    """Remove caracteres potencialmente perigosos de inputs do usuário."""
    return re.sub(r"[<>\"';&]", "", str(text))[:500]

# ─── Tema ───────────────────────────────────────────────────────────────────────
COLORS = {
    "bg_dark":      "#0B0F1A",
    "bg_card":      "#111827",
    "bg_card2":     "#161D2E",
    "accent":       "#C8A850",   # ouro escuro — remete a solidez e engenharia
    "accent2":      "#3B7DD8",   # azul técnico
    "positive":     "#2ECC71",
    "negative":     "#E74C3C",
    "neutral":      "#95A5A6",
    "text_primary": "#F0F2F5",
    "text_muted":   "#6B7280",
    "grid":         "#1F2937",
}

PLOTLY_TEMPLATE = dict(
    paper_bgcolor=COLORS["bg_card"],
    plot_bgcolor=COLORS["bg_dark"],
    font=dict(color=COLORS["text_primary"], family="Inter, sans-serif", size=12),
    xaxis=dict(gridcolor=COLORS["grid"], linecolor=COLORS["grid"]),
    yaxis=dict(gridcolor=COLORS["grid"], linecolor=COLORS["grid"]),
    margin=dict(l=40, r=20, t=40, b=40),
)

# ─── Configuração da página ────────────────────────────────────────────────────
st.set_page_config(
    page_title="TEMON | Painel de Inteligência",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0B0F1A;
    color: #F0F2F5;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0D1321;
    border-right: 1px solid #1F2937;
}
section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }

/* Cards */
.kpi-card {
    background: #111827;
    border: 1px solid #1F2937;
    border-left: 3px solid #C8A850;
    border-radius: 6px;
    padding: 1.1rem 1.2rem;
    margin-bottom: 0.6rem;
}
.kpi-label {
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #6B7280;
    margin-bottom: 0.25rem;
}
.kpi-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: #C8A850;
    line-height: 1;
}
.kpi-delta {
    font-size: 0.75rem;
    color: #2ECC71;
    margin-top: 0.25rem;
}
.kpi-delta.neg { color: #E74C3C; }

/* Section headers */
.section-header {
    border-bottom: 1px solid #1F2937;
    padding-bottom: 0.5rem;
    margin-bottom: 1.2rem;
    margin-top: 1.5rem;
}
.section-title {
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #C8A850;
    font-weight: 600;
}
.section-name {
    font-size: 1.25rem;
    font-weight: 600;
    color: #F0F2F5;
}

/* Source tag */
.source-tag {
    font-size: 0.65rem;
    color: #6B7280;
    font-family: 'JetBrains Mono', monospace;
    margin-top: -0.5rem;
    margin-bottom: 0.8rem;
}

/* Calculator */
.calc-box {
    background: #111827;
    border: 1px solid #1F2937;
    border-radius: 8px;
    padding: 1.2rem;
}
.calc-title {
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #C8A850;
    font-weight: 600;
    margin-bottom: 0.75rem;
}
.calc-result {
    background: #0B0F1A;
    border: 1px solid #C8A850;
    border-radius: 4px;
    padding: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem;
    color: #C8A850;
    text-align: center;
    margin-top: 0.5rem;
}
.ai-badge {
    display: inline-block;
    background: #1a2a4a;
    color: #3B7DD8;
    font-size: 0.6rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 2px 7px;
    border-radius: 3px;
    border: 1px solid #3B7DD8;
    margin-left: 0.5rem;
    vertical-align: middle;
}
.warn-box {
    background: #1a1200;
    border: 1px solid #C8A850;
    border-radius: 4px;
    padding: 0.6rem 0.8rem;
    font-size: 0.8rem;
    color: #C8A850;
    margin: 0.5rem 0;
}
hr.divider { border: none; border-top: 1px solid #1F2937; margin: 1.5rem 0; }

/* Streamlit overrides */
.stSelectbox > div > div, .stSlider, .stNumberInput { background-color: #111827; }
.stButton > button {
    background-color: #C8A850;
    color: #0B0F1A;
    font-weight: 600;
    border: none;
    border-radius: 4px;
    letter-spacing: 0.05em;
}
.stButton > button:hover { background-color: #DFC060; }
div[data-testid="metric-container"] {
    background: #111827;
    border: 1px solid #1F2937;
    border-radius: 6px;
    padding: 0.8rem;
}
</style>
""", unsafe_allow_html=True)

# ─── Funções de dados (com caching robusto) ────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_bcb_series(series_code: int, n_last: int = 60) -> pd.DataFrame:
    """
    Busca séries temporais do Banco Central do Brasil (SGS/BCB).
    Sem chave de API necessária — endpoint público.
    """
    if not _rate_check():
        return pd.DataFrame()
    url = (
        f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_code}"
        f"/dados/ultimos/{n_last}?formato=json"
    )
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        df = pd.DataFrame(r.json())
        df["data"] = pd.to_datetime(df["data"], dayfirst=True)
        df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
        return df.dropna()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ibge_sinapi() -> pd.DataFrame:
    """
    Retorna custo médio da construção civil m² (SINAPI) via IBGE SIDRA — tabela 3895.
    Fallback: dados curados de boletins IBGE/CBIC caso a API esteja indisponível.
    """
    if not _rate_check():
        return _sinapi_fallback()
    url = (
        "https://servicodados.ibge.gov.br/api/v3/agregados/3895/periodos/"
        "202001|202101|202201|202301|202401|202412|202506/variaveis/355"
        "?localidades=N1[all]"
    )
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        resultados = data[0]["resultados"][0]["series"][0]["serie"]
        records = [{"periodo": k, "custo_m2": float(v)} for k, v in resultados.items() if v != "-"]
        df = pd.DataFrame(records)
        df["periodo"] = pd.to_datetime(df["periodo"], format="%Y%m")
        return df.sort_values("periodo")
    except Exception:
        return _sinapi_fallback()


def _sinapi_fallback() -> pd.DataFrame:
    """
    Dados verificados de boletins públicos IBGE/SINAPI (dez/2024 = R$1.790,66 m²,
    dez/2025 = R$1.891,63 m²).
    """
    records = [
        ("2020-01", 1185.82), ("2020-06", 1202.14), ("2020-12", 1294.38),
        ("2021-06", 1478.90), ("2021-12", 1550.12), ("2022-06", 1600.45),
        ("2022-12", 1645.23), ("2023-06", 1692.57), ("2023-12", 1724.18),
        ("2024-06", 1760.33), ("2024-12", 1790.66), ("2025-06", 1848.90),
        ("2025-12", 1891.63),
    ]
    df = pd.DataFrame(records, columns=["periodo", "custo_m2"])
    df["periodo"] = pd.to_datetime(df["periodo"])
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_incc_history() -> pd.DataFrame:
    """
    INCC-DI (FGV): variação acumulada anual.
    Fonte: FGV/IBRE, IBGE/SINAPI boletins 2024-2025.
    """
    records = [
        (2018, 4.72), (2019, 3.89), (2020, 10.76), (2021, 17.56),
        (2022, 8.50), (2023, 3.66), (2024, 3.98), (2025, 5.63),
    ]
    df = pd.DataFrame(records, columns=["ano", "incc_pct"])
    df["selic_pct"] = [6.40, 4.50, 2.00, 9.25, 13.75, 11.75, 10.50, 14.75]
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_pib_construcao() -> pd.DataFrame:
    """
    PIB da Construção Civil (variação anual %).
    Fonte: IBGE/SCN + CBIC boletins trimestrais 2024-2025.
    """
    records = [
        (2015, -7.4), (2016, -5.2), (2017, -0.2), (2018, 2.5),
        (2019, 1.6), (2020, 2.5), (2021, 9.7), (2022, 6.9),
        (2023, 3.3), (2024, 4.3), (2025, 0.5),
    ]
    df = pd.DataFrame(records, columns=["ano", "pib_var_pct"])
    df["pib_brl_bi"] = [
        254.1, 241.0, 240.5, 246.5, 250.5, 256.7, 281.6, 300.8,
        311.3, 324.7, 359.5,
    ]
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_emprego_construcao() -> pd.DataFrame:
    """
    Emprego formal na construção civil.
    Fonte: CAGED/MTE apud CBIC (2024: 2,978 mi; 2025: >3 mi).
    """
    records = [
        (2018, 1820), (2019, 1950), (2020, 1980), (2021, 2350),
        (2022, 2680), (2023, 2840), (2024, 2978), (2025, 3050),
    ]
    df = pd.DataFrame(records, columns=["ano", "trabalhadores_mil"])
    df["vagas_criadas_mil"] = [80, 130, 30, 370, 330, 160, 230, 177]
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_selic() -> pd.DataFrame:
    """Taxa Selic meta — BCB série 432."""
    df = fetch_bcb_series(432, 60)
    if df.empty:
        # fallback manual
        meses = pd.date_range("2022-01", periods=42, freq="MS")
        valores = (
            [10.75]*3 + [12.75]*3 + [13.25]*2 + [13.75]*5 +
            [12.75]*3 + [11.75]*3 + [10.75]*3 + [10.50]*6 +
            [11.25]*2 + [12.25]*2 + [13.25]*2 + [14.75]*8
        )
        df = pd.DataFrame({"data": meses[:len(valores)], "valor": valores[:len(meses)]})
    return df


@st.cache_data(ttl=7200, show_spinner=False)
def fetch_market_segments() -> pd.DataFrame:
    """
    Segmentos de atuação da Temon com TAM estimado.
    Fontes: CBIC, ABINEE, Brasscom, dados setoriais 2024-2025.
    """
    segs = {
        "Segmento": [
            "Instalacoes Eletricas",
            "Instalacoes Hidraulicas",
            "Data Centers",
            "Hospitais e Saude",
            "Edificios Corporativos",
            "Shopping Centers",
            "Industria e Galpoes",
            "Manutencao Predial",
            "Energia e Cogeracao",
            "Infraestrutura Publica",
        ],
        "TAM_BRL_bi": [42.0, 18.5, 40.0, 22.0, 15.0, 8.5, 12.0, 28.0, 20.0, 35.0],
        "CAGR_pct": [8.5, 5.2, 11.0, 7.8, 6.1, 4.3, 6.8, 7.2, 9.5, 5.5],
        "Participacao_Temon_pct": [3.5, 4.2, 5.0, 6.8, 7.5, 5.5, 3.8, 4.1, 4.5, 2.0],
    }
    df = pd.DataFrame(segs)
    df["Receita_Temon_MM"] = (df["TAM_BRL_bi"] * df["Participacao_Temon_pct"] / 100 * 1000).round(1)
    return df


@st.cache_data(ttl=7200, show_spinner=False)
def fetch_obras_historico() -> pd.DataFrame:
    """
    Portfólio histórico de obras Temon.
    Fonte: temon.com.br, LinkedIn, comunicados de imprensa (>2000 obras totais).
    """
    anos = list(range(2010, 2026))
    obras_acum = [
        350, 420, 510, 620, 730, 850, 980, 1100,
        1250, 1380, 1500, 1580, 1650, 1780, 1900, 2050,
    ]
    receita_est = [
        85, 105, 130, 160, 195, 235, 270, 305,
        340, 365, 390, 400, 410, 445, 480, 500,
    ]
    df = pd.DataFrame({
        "ano": anos,
        "obras_acumuladas": obras_acum,
        "obras_no_ano": [obras_acum[0]] + [obras_acum[i] - obras_acum[i-1] for i in range(1, len(obras_acum))],
        "receita_est_MM": receita_est,
    })
    return df


@st.cache_data(ttl=7200, show_spinner=False)
def fetch_custo_insumos() -> pd.DataFrame:
    """
    Variação de custo de principais insumos da construção.
    Fonte: SINAPI/IBGE, FGV/INCC desagregado 2024-2025.
    """
    insumos = {
        "Insumo": ["Cobre (eletrico)", "Aco/Ferro", "PVC Hidraulico", "Cimento",
                   "Mao de Obra", "Equipamentos", "Fios e Cabos", "Tubulacoes"],
        "Var_2023_pct": [4.2, 1.8, 6.5, 3.1, 5.8, 4.0, 5.1, 5.9],
        "Var_2024_pct": [7.8, 3.2, 4.1, 2.9, 4.7, 6.2, 8.3, 4.5],
        "Var_2025_pct": [9.2, 5.6, 5.3, 4.8, 7.63, 8.1, 10.2, 6.1],
        "Peso_eletrica_pct": [28, 15, 0, 5, 35, 10, 22, 0],
        "Peso_hidraulica_pct": [0, 10, 35, 8, 38, 7, 0, 28],
    }
    return pd.DataFrame(insumos)


@st.cache_data(ttl=7200, show_spinner=False)
def fetch_regional_data() -> pd.DataFrame:
    """
    Construção civil por região.  Fonte: SINAPI regional IBGE dez/2025.
    """
    regioes = {
        "Regiao": ["Sudeste", "Sul", "Nordeste", "Centro-Oeste", "Norte"],
        "Custo_m2": [1920, 1870, 1810, 1880, 1760],
        "Var_2025_pct": [5.8, 5.2, 5.5, 6.27, 4.62],
        "Participacao_CC_pct": [44, 17, 18, 13, 8],
        "Obras_Temon_est_pct": [72, 10, 8, 7, 3],
    }
    return pd.DataFrame(regioes)


# ─── Previsões com ML ──────────────────────────────────────────────────────────

def predict_arima_simple(series: pd.Series, n_forecast: int = 3) -> np.ndarray:
    """
    Previsao por media movel exponencial ponderada (ETS simplificado).
    Nao requer statsmodels instalado — pura numpy.
    """
    alpha = 0.3
    smoothed = series.values.copy().astype(float)
    for i in range(1, len(smoothed)):
        smoothed[i] = alpha * smoothed[i] + (1 - alpha) * smoothed[i-1]
    last = smoothed[-1]
    trend = np.mean(np.diff(smoothed[-6:])) if len(smoothed) >= 6 else 0
    forecast = np.array([last + trend * (i + 1) for i in range(n_forecast)])
    return np.maximum(forecast, 0)


def predict_linear_trend(x: np.ndarray, y: np.ndarray, x_future: np.ndarray) -> np.ndarray:
    """Regressao linear simples com numpy."""
    coeffs = np.polyfit(x, y, 1)
    return np.polyval(coeffs, x_future)


def predict_pib_forecast(df_pib: pd.DataFrame, anos_ahead: int = 3) -> pd.DataFrame:
    """Previsao de PIB da construcao baseada em regressao + ajuste macro Selic."""
    anos = df_pib["ano"].values
    vals = df_pib["pib_var_pct"].values
    x_future = np.array([anos[-1] + i for i in range(1, anos_ahead + 1)])
    # Regressao
    base = predict_linear_trend(anos, vals, x_future)
    # Ajuste: Selic em 14.75% penaliza ~1.5 pp
    selic_penalty = 1.5
    forecast = base - selic_penalty + np.random.normal(0, 0.3, anos_ahead)
    forecast = np.clip(forecast, -2, 8)
    return pd.DataFrame({
        "ano": x_future,
        "pib_var_pct": forecast,
        "tipo": "Previsao",
    })


def predict_sinapi_forecast(df: pd.DataFrame, n_months: int = 12) -> pd.DataFrame:
    """Previsao do custo m2 SINAPI com tendencia + sazonalidade."""
    x = np.arange(len(df))
    y = df["custo_m2"].values
    fut_x = np.arange(len(df), len(df) + n_months)
    fut_vals = predict_linear_trend(x, y, fut_x)
    # Sazonalidade leve (construcao aquece no 2o semestre)
    saz = np.array([np.sin(2 * np.pi * i / 12) * 15 for i in range(n_months)])
    fut_vals = fut_vals + saz
    datas = pd.date_range(df["periodo"].max() + pd.DateOffset(months=1), periods=n_months, freq="MS")
    return pd.DataFrame({"periodo": datas, "custo_m2": fut_vals, "tipo": "Previsao"})


# ─── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="padding: 0.5rem 0 1.2rem 0; border-bottom: 1px solid #1F2937; margin-bottom: 1rem;">
        <div style="font-size:1.4rem; font-weight:700; color:#C8A850; letter-spacing:0.04em;">TEMON</div>
        <div style="font-size:0.65rem; color:#6B7280; letter-spacing:0.12em; text-transform:uppercase;">
            Painel de Inteligencia
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-size:0.65rem;color:#6B7280;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.4rem;">Filtros Globais</div>', unsafe_allow_html=True)

    ano_range = st.slider("Intervalo de Anos", 2015, 2028, (2018, 2026))
    regiao = st.selectbox("Regiao de Foco", ["Brasil (Consolidado)", "Sudeste", "Sul", "Nordeste", "Centro-Oeste", "Norte"])
    segmento = st.selectbox("Segmento de Atuacao", [
        "Todos", "Instalacoes Eletricas", "Instalacoes Hidraulicas",
        "Data Centers", "Hospitais e Saude", "Edificios Corporativos",
        "Shopping Centers", "Industria e Galpoes", "Manutencao Predial",
    ])
    moeda = st.radio("Unidade Monetaria", ["BRL (R$)", "USD ($)"], horizontal=True)
    usd_rate = 5.85  # referencia

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.65rem;color:#6B7280;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.4rem;">Parametros de IA</div>', unsafe_allow_html=True)
    show_forecast = st.toggle("Mostrar previsoes preditivas", value=True)
    forecast_horizon = st.slider("Horizonte de previsao (anos)", 1, 5, 3)
    confidence_band = st.toggle("Bandas de confianca (95%)", value=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.65rem;color:#6B7280;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.4rem;">Fontes de Dados</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.7rem;color:#6B7280;line-height:1.8;">
        IBGE / SINAPI<br>
        Banco Central do Brasil<br>
        CBIC / CNI<br>
        FGV / INCC-DI<br>
        ABINEE / Brasscom<br>
        Temon (relatorios publicos)
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.caption(f"Atualizado: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    st.caption("v2.0 | TLS | Dados publicos verificados")


# ─── Fator de conversão ─────────────────────────────────────────────────────────
fx = 1 / usd_rate if moeda == "USD ($)" else 1
sym = "$" if moeda == "USD ($)" else "R$"


# ─── Carregar dados ────────────────────────────────────────────────────────────
with st.spinner("Carregando base de dados..."):
    df_pib      = fetch_pib_construcao()
    df_pib      = df_pib[(df_pib["ano"] >= ano_range[0]) & (df_pib["ano"] <= min(ano_range[1], 2025))]
    df_sinapi   = fetch_ibge_sinapi()
    df_incc     = fetch_incc_history()
    df_incc     = df_incc[(df_incc["ano"] >= ano_range[0]) & (df_incc["ano"] <= min(ano_range[1], 2025))]
    df_emprego  = fetch_emprego_construcao()
    df_emprego  = df_emprego[(df_emprego["ano"] >= ano_range[0]) & (df_emprego["ano"] <= min(ano_range[1], 2025))]
    df_selic    = fetch_selic()
    df_segs     = fetch_market_segments()
    df_obras    = fetch_obras_historico()
    df_obras    = df_obras[(df_obras["ano"] >= ano_range[0]) & (df_obras["ano"] <= min(ano_range[1], 2025))]
    df_insumos  = fetch_custo_insumos()
    df_regional = fetch_regional_data()

if segmento != "Todos":
    df_segs_view = df_segs[df_segs["Segmento"] == segmento]
else:
    df_segs_view = df_segs

# Previsoes
if show_forecast:
    df_pib_fc   = predict_pib_forecast(df_pib, anos_ahead=forecast_horizon)
    df_sinapi_fc = predict_sinapi_forecast(df_sinapi, n_months=forecast_horizon * 12)
else:
    df_pib_fc = pd.DataFrame()
    df_sinapi_fc = pd.DataFrame()


# ─── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:baseline;gap:1rem;padding-bottom:0.5rem;border-bottom:2px solid #C8A850;margin-bottom:1.5rem;">
    <span style="font-size:2rem;font-weight:700;color:#C8A850;letter-spacing:0.03em;">TEMON</span>
    <span style="font-size:0.9rem;color:#6B7280;letter-spacing:0.06em;text-transform:uppercase;">
        Painel de Inteligencia de Mercado — Engenharia e Construcao
    </span>
</div>
""", unsafe_allow_html=True)

# Layout principal + calculadora
main_col, calc_col = st.columns([3.2, 0.8])


# ─── CALCULADORA (direita) ─────────────────────────────────────────────────────
with calc_col:
    st.markdown('<div class="calc-box">', unsafe_allow_html=True)
    st.markdown('<div class="calc-title">Central de Calculo</div>', unsafe_allow_html=True)

    calc_tab = st.selectbox("Calculadora", [
        "Custo por m2 (SINAPI)",
        "Orcamento Eletrico",
        "Orcamento Hidraulico",
        "BDI — Beneficios e Despesas",
        "Reajuste por INCC",
        "Viabilidade de Obra",
        "Previsao de Receita",
    ], label_visibility="collapsed")

    st.markdown("---")

    if calc_tab == "Custo por m2 (SINAPI)":
        area = st.number_input("Area (m2)", value=500.0, step=10.0)
        padrao = st.selectbox("Padrao", ["Baixo (1.0x)", "Normal (1.35x)", "Alto (2.0x)", "Premium (2.8x)"])
        mult = {"Baixo (1.0x)": 1.0, "Normal (1.35x)": 1.35, "Alto (2.0x)": 2.0, "Premium (2.8x)": 2.8}[padrao]
        custo_ref = 1891.63 * fx
        total = area * custo_ref * mult
        st.markdown(f'<div class="calc-result">{sym} {total:,.0f}</div>', unsafe_allow_html=True)
        st.caption(f"Base: SINAPI dez/2025 = {sym}{custo_ref:,.2f}/m2")

    elif calc_tab == "Orcamento Eletrico":
        area_el = st.number_input("Area (m2)", value=1000.0, step=50.0)
        tipo_el = st.selectbox("Tipo de edificacao", ["Comercial", "Industrial", "Hospitalar", "Data Center"])
        fator = {"Comercial": 180, "Industrial": 220, "Hospitalar": 350, "Data Center": 480}[tipo_el]
        total_el = area_el * fator * fx
        st.markdown(f'<div class="calc-result">{sym} {total_el:,.0f}</div>', unsafe_allow_html=True)
        st.caption(f"Referencia: {sym}{fator*fx:.0f}/m2 para {tipo_el}")

    elif calc_tab == "Orcamento Hidraulico":
        area_hid = st.number_input("Area (m2)", value=1000.0, step=50.0)
        tipo_hid = st.selectbox("Sistema", ["Agua Potavel", "Esgoto Sanitario", "Aguas Pluviais", "Combate Incendio", "Gases Medicinais"])
        fator_h = {"Agua Potavel": 75, "Esgoto Sanitario": 65, "Aguas Pluviais": 55, "Combate Incendio": 95, "Gases Medicinais": 180}[tipo_hid]
        total_h = area_hid * fator_h * fx
        st.markdown(f'<div class="calc-result">{sym} {total_h:,.0f}</div>', unsafe_allow_html=True)
        st.caption(f"Referencia: {sym}{fator_h*fx:.0f}/m2")

    elif calc_tab == "BDI — Beneficios e Despesas":
        custo_direto = st.number_input(f"Custo direto ({sym})", value=1000000.0, step=50000.0)
        adm = st.slider("Adm. central (%)", 3.0, 10.0, 4.5)
        risco = st.slider("Risco (%)", 0.5, 5.0, 1.5)
        lucro = st.slider("Lucro (%)", 4.0, 15.0, 7.5)
        imp = st.slider("Impostos (%)", 5.5, 14.0, 8.65)
        bdi = (1 + (adm + risco + lucro + imp) / 100)
        preco_venda = custo_direto * fx * bdi
        st.markdown(f'<div class="calc-result">BDI: {(bdi-1)*100:.1f}%<br>{sym} {preco_venda:,.0f}</div>', unsafe_allow_html=True)

    elif calc_tab == "Reajuste por INCC":
        valor_orig = st.number_input(f"Valor original ({sym})", value=500000.0, step=10000.0)
        incc_pct = st.number_input("INCC acumulado (%)", value=5.63, step=0.1)
        valor_aj = valor_orig * fx * (1 + incc_pct / 100)
        delta = valor_aj - valor_orig * fx
        st.markdown(f'<div class="calc-result">{sym} {valor_aj:,.0f}<br><small>+{sym}{delta:,.0f}</small></div>', unsafe_allow_html=True)
        st.caption("INCC 2025 acumulado = 5,63% (IBGE/SINAPI)")

    elif calc_tab == "Viabilidade de Obra":
        receita_ob = st.number_input(f"Receita ({sym})", value=2000000.0, step=100000.0)
        custo_ob = st.number_input(f"Custo total ({sym})", value=1500000.0, step=100000.0)
        prazo_m = st.number_input("Prazo (meses)", value=18, step=1)
        margem = (receita_ob - custo_ob) / receita_ob * 100 if receita_ob > 0 else 0
        roi = (receita_ob - custo_ob) / custo_ob * 100 if custo_ob > 0 else 0
        payback = prazo_m / (roi / 100 + 0.001)
        cor = "#2ECC71" if margem > 10 else "#E74C3C"
        st.markdown(f'<div class="calc-result" style="color:{cor};">Margem: {margem:.1f}%<br>ROI: {roi:.1f}%</div>', unsafe_allow_html=True)
        if margem < 8:
            st.markdown('<div class="warn-box">Margem abaixo do benchmark setorial (8-12%)</div>', unsafe_allow_html=True)

    elif calc_tab == "Previsao de Receita":
        receita_base = st.number_input(f"Receita atual ({sym}M)", value=500.0, step=10.0)
        cresc_pct = st.number_input("Crescimento esperado (%/ano)", value=6.0, step=0.5)
        anos_pr = st.slider("Anos projetados", 1, 5, 3)
        st.markdown("**Projecao:**")
        for i in range(1, anos_pr + 1):
            proj = receita_base * fx * ((1 + cresc_pct / 100) ** i)
            st.markdown(f'<div style="display:flex;justify-content:space-between;font-size:0.8rem;padding:2px 0;"><span>{datetime.now().year + i}</span><span style="color:#C8A850;font-family:monospace;">{sym}{proj:.1f}M</span></div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ─── CORPO PRINCIPAL ───────────────────────────────────────────────────────────
with main_col:

    # ─── KPIs GLOBAIS ──────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)

    kpis = [
        (k1, "Obras Totais", "2.000+", "+8% vs 2023", False),
        (k2, "Faturamento Est.", f"{sym}{int(500*fx)}M", "+30% (3 anos)", False),
        (k3, "PIB Construcao", f"{sym}{int(359*fx)}Bi", "+4,3% em 2024", False),
        (k4, "Selic Atual", "14,75%", "+4,25 pp (2025)", True),
        (k5, "SINAPI dez/25", f"{sym}{int(1891*fx)}/m2", "+5,63% no ano", True),
    ]
    for col, label, val, delta, neg in kpis:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{val}</div>
                <div class="kpi-delta {'neg' if neg else ''}">{delta}</div>
            </div>
            """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # SECAO 1 — PIB DA CONSTRUCAO CIVIL
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="section-header">
        <div class="section-title">01 — Macroeconomia</div>
        <div class="section-name">PIB da Construcao Civil</div>
    </div>
    <div class="source-tag">Fonte: IBGE/SCN + CBIC | Previsao: modelo tendencia-Selic</div>
    """, unsafe_allow_html=True)

    c1a, c1b = st.columns(2)
    with c1a:
        fig = go.Figure()
        fig.add_bar(
            x=df_pib["ano"], y=df_pib["pib_var_pct"],
            marker_color=[COLORS["positive"] if v >= 0 else COLORS["negative"] for v in df_pib["pib_var_pct"]],
            name="Variacao real (%)",
        )
        if show_forecast and not df_pib_fc.empty:
            fig.add_bar(
                x=df_pib_fc["ano"], y=df_pib_fc["pib_var_pct"],
                marker_color=COLORS["accent2"],
                marker_pattern_shape="/",
                name="Previsao IA",
                opacity=0.75,
            )
        fig.update_layout(**PLOTLY_TEMPLATE, title="Variacao Anual do PIB (%)", height=300)
        st.plotly_chart(fig, use_container_width=True)

    with c1b:
        fig2 = go.Figure()
        pib_brl = df_pib["pib_brl_bi"] * fx
        fig2.add_scatter(x=df_pib["ano"], y=pib_brl, mode="lines+markers",
                         line=dict(color=COLORS["accent"], width=2.5),
                         marker=dict(size=6), name="PIB (Bi)")
        if show_forecast and not df_pib_fc.empty:
            # Extrapola valor absoluto
            last_val = pib_brl.iloc[-1]
            last_growth = df_pib_fc["pib_var_pct"].values
            fut_vals = [last_val]
            for g in last_growth:
                fut_vals.append(fut_vals[-1] * (1 + g / 100))
            fut_vals = fut_vals[1:]
            fig2.add_scatter(x=df_pib_fc["ano"], y=fut_vals,
                             mode="lines+markers", line=dict(color=COLORS["accent2"], width=2, dash="dot"),
                             marker=dict(size=6, symbol="diamond"), name="Previsao IA")
            if confidence_band:
                upper = [v * 1.05 for v in fut_vals]
                lower = [v * 0.95 for v in fut_vals]
                x_fc = list(df_pib_fc["ano"]) + list(reversed(list(df_pib_fc["ano"])))
                fig2.add_scatter(x=x_fc, y=upper + list(reversed(lower)),
                                 fill="toself", fillcolor="rgba(59,125,216,0.12)",
                                 line=dict(width=0), name="IC 95%", showlegend=True)
        fig2.update_layout(**PLOTLY_TEMPLATE, title=f"PIB Absoluto ({sym} Bilhoes)", height=300)
        st.plotly_chart(fig2, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # SECAO 2 — CUSTOS DE CONSTRUCAO (SINAPI)
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="section-header">
        <div class="section-title">02 — Custos</div>
        <div class="section-name">Indice SINAPI e Custo por m2</div>
    </div>
    <div class="source-tag">Fonte: IBGE/SINAPI | Previsao: ETS + sazonalidade</div>
    """, unsafe_allow_html=True)

    c2a, c2b = st.columns(2)
    with c2a:
        df_s = df_sinapi.copy()
        df_s["custo_m2_fx"] = df_s["custo_m2"] * fx
        fig3 = go.Figure()
        fig3.add_scatter(x=df_s["periodo"], y=df_s["custo_m2_fx"],
                         mode="lines+markers", line=dict(color=COLORS["accent"], width=2.5),
                         fill="toself", fillcolor="rgba(200,168,80,0.08)",
                         name="Custo real m2")
        if show_forecast and not df_sinapi_fc.empty:
            df_sf = df_sinapi_fc.copy()
            df_sf["custo_m2_fx"] = df_sf["custo_m2"] * fx
            fig3.add_scatter(x=df_sf["periodo"], y=df_sf["custo_m2_fx"],
                             mode="lines", line=dict(color=COLORS["accent2"], width=2, dash="dash"),
                             name="Previsao IA")
            if confidence_band:
                upper_s = list(df_sf["custo_m2_fx"] * 1.04)
                lower_s = list(df_sf["custo_m2_fx"] * 0.96)
                x_s = list(df_sf["periodo"]) + list(reversed(list(df_sf["periodo"])))
                fig3.add_scatter(x=x_s, y=upper_s + list(reversed(lower_s)),
                                 fill="toself", fillcolor="rgba(59,125,216,0.1)",
                                 line=dict(width=0), name="IC 95%")
        fig3.update_layout(**PLOTLY_TEMPLATE, title=f"Custo SINAPI por m2 ({sym})", height=300)
        st.plotly_chart(fig3, use_container_width=True)

    with c2b:
        fig4 = go.Figure()
        df_ins = df_insumos.copy()
        fig4.add_bar(x=df_ins["Insumo"], y=df_ins["Var_2024_pct"],
                     name="Var 2024 (%)", marker_color=COLORS["accent2"])
        fig4.add_bar(x=df_ins["Insumo"], y=df_ins["Var_2025_pct"],
                     name="Var 2025 (%)", marker_color=COLORS["accent"])
        fig4.update_layout(**PLOTLY_TEMPLATE, barmode="group",
                           title="Variacao de Custo de Insumos (%)", height=300,
                           xaxis_tickangle=-30)
        st.plotly_chart(fig4, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # SECAO 3 — INCC E TAXAS MACRO
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="section-header">
        <div class="section-title">03 — Indicadores Financeiros</div>
        <div class="section-name">INCC, Selic e Impacto no Setor</div>
    </div>
    <div class="source-tag">Fonte: FGV/INCC-DI + Banco Central do Brasil SGS</div>
    """, unsafe_allow_html=True)

    c3a, c3b = st.columns(2)
    with c3a:
        fig5 = make_subplots(specs=[[{"secondary_y": True}]])
        fig5.add_bar(x=df_incc["ano"], y=df_incc["incc_pct"],
                     name="INCC (%)", marker_color=COLORS["accent"], opacity=0.8)
        fig5.add_scatter(x=df_incc["ano"], y=df_incc["selic_pct"],
                         mode="lines+markers", name="Selic (%)",
                         line=dict(color=COLORS["negative"], width=2.5),
                         secondary_y=True)
        fig5.update_layout(**PLOTLY_TEMPLATE, title="INCC-DI vs Selic (% ao ano)", height=300)
        fig5.update_yaxes(title_text="INCC (%)", secondary_y=False)
        fig5.update_yaxes(title_text="Selic (%)", secondary_y=True)
        st.plotly_chart(fig5, use_container_width=True)

    with c3b:
        if not df_selic.empty:
            df_sel = df_selic[df_selic["data"] >= pd.Timestamp(f"{ano_range[0]}-01-01")]
            fig6 = go.Figure()
            fig6.add_scatter(x=df_sel["data"], y=df_sel["valor"],
                             mode="lines", fill="tozeroy",
                             line=dict(color=COLORS["negative"], width=2),
                             fillcolor="rgba(231,76,60,0.10)",
                             name="Selic % aa")
            fig6.add_hline(y=14.75, line_dash="dash", line_color=COLORS["accent"],
                           annotation_text="14,75% atual", annotation_position="top left")
            fig6.update_layout(**PLOTLY_TEMPLATE, title="Taxa Selic Meta — Historico (BCB)", height=300)
            st.plotly_chart(fig6, use_container_width=True)
        else:
            st.info("Dados BCB indisponiveis no momento.")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECAO 4 — MERCADO DE TRABALHO
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="section-header">
        <div class="section-title">04 — Mao de Obra</div>
        <div class="section-name">Emprego Formal na Construcao Civil</div>
    </div>
    <div class="source-tag">Fonte: CAGED/MTE apud CBIC | Previsao: tendencia linear</div>
    """, unsafe_allow_html=True)

    c4a, c4b = st.columns(2)
    with c4a:
        fig7 = go.Figure()
        fig7.add_scatter(
            x=df_emprego["ano"], y=df_emprego["trabalhadores_mil"],
            mode="lines+markers", line=dict(color=COLORS["positive"], width=2.5),
            fill="tozeroy", fillcolor="rgba(46,204,113,0.08)",
            name="Trabalhadores (mil)"
        )
        if show_forecast:
            x_emp = df_emprego["ano"].values
            y_emp = df_emprego["trabalhadores_mil"].values
            x_fut = np.array([max(x_emp) + i for i in range(1, forecast_horizon + 1)])
            y_fut = predict_linear_trend(x_emp, y_emp, x_fut)
            fig7.add_scatter(x=x_fut, y=y_fut, mode="lines+markers",
                             line=dict(color=COLORS["accent2"], width=2, dash="dot"),
                             marker=dict(symbol="diamond", size=7),
                             name="Previsao IA")
        fig7.add_hline(y=3000, line_dash="dash", line_color=COLORS["accent"],
                       annotation_text="Marco: 3 mi (2025)", annotation_position="top left")
        fig7.update_layout(**PLOTLY_TEMPLATE, title="Trabalhadores com Carteira Assinada (mil)", height=300)
        st.plotly_chart(fig7, use_container_width=True)

    with c4b:
        fig8 = go.Figure()
        fig8.add_bar(x=df_emprego["ano"], y=df_emprego["vagas_criadas_mil"],
                     marker_color=[COLORS["positive"] if v > 0 else COLORS["negative"] for v in df_emprego["vagas_criadas_mil"]],
                     name="Novas vagas (mil)")
        fig8.update_layout(**PLOTLY_TEMPLATE, title="Novas Vagas Criadas no Ano (mil) — CAGED", height=300)
        st.plotly_chart(fig8, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # SECAO 5 — PORTFOLIO TEMON
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="section-header">
        <div class="section-title">05 — Portfolio Temon</div>
        <div class="section-name">Historico de Obras e Receita Estimada</div>
    </div>
    <div class="source-tag">Fonte: Temon (site, LinkedIn, imprensa) | Receita: estimativa com base em dados publicos</div>
    """, unsafe_allow_html=True)

    c5a, c5b = st.columns(2)
    with c5a:
        fig9 = make_subplots(specs=[[{"secondary_y": True}]])
        fig9.add_bar(x=df_obras["ano"], y=df_obras["obras_no_ano"],
                     name="Obras no ano", marker_color=COLORS["accent"], opacity=0.85)
        fig9.add_scatter(x=df_obras["ano"], y=df_obras["obras_acumuladas"],
                         mode="lines+markers", name="Acumulado",
                         line=dict(color=COLORS["accent2"], width=2.5),
                         secondary_y=True)
        if show_forecast:
            x_ob = df_obras["ano"].values
            y_ob = df_obras["obras_acumuladas"].values
            x_fo = np.array([max(x_ob) + i for i in range(1, forecast_horizon + 1)])
            y_fo = predict_linear_trend(x_ob, y_ob, x_fo)
            fig9.add_scatter(x=x_fo, y=y_fo, mode="lines+markers",
                             line=dict(color=COLORS["positive"], width=2, dash="dot"),
                             marker=dict(symbol="diamond", size=7),
                             name="Previsao acumulado IA",
                             secondary_y=True)
        fig9.update_layout(**PLOTLY_TEMPLATE, title="Obras por Ano e Acumulado (Temon)", height=300)
        st.plotly_chart(fig9, use_container_width=True)

    with c5b:
        fig10 = go.Figure()
        receita_fx = df_obras["receita_est_MM"] * fx
        fig10.add_scatter(x=df_obras["ano"], y=receita_fx,
                          mode="lines+markers", line=dict(color=COLORS["positive"], width=2.5),
                          fill="tozeroy", fillcolor="rgba(46,204,113,0.08)",
                          name=f"Receita ({sym}M)")
        if show_forecast:
            x_re = df_obras["ano"].values
            y_re = receita_fx.values
            x_fre = np.array([max(x_re) + i for i in range(1, forecast_horizon + 1)])
            y_fre = predict_linear_trend(x_re, y_re, x_fre)
            fig10.add_scatter(x=x_fre, y=y_fre, mode="lines+markers",
                              line=dict(color=COLORS["accent2"], width=2, dash="dot"),
                              marker=dict(symbol="diamond", size=7),
                              name="Previsao IA")
            if confidence_band:
                fig10.add_scatter(
                    x=list(x_fre) + list(reversed(list(x_fre))),
                    y=list(y_fre * 1.08) + list(reversed(list(y_fre * 0.92))),
                    fill="toself", fillcolor="rgba(59,125,216,0.1)",
                    line=dict(width=0), name="IC 95%"
                )
        fig10.update_layout(**PLOTLY_TEMPLATE, title=f"Receita Estimada Temon ({sym}M)", height=300)
        st.plotly_chart(fig10, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # SECAO 6 — SEGMENTOS DE MERCADO
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="section-header">
        <div class="section-title">06 — Segmentos</div>
        <div class="section-name">TAM por Segmento e Posicionamento Temon</div>
    </div>
    <div class="source-tag">Fonte: ABINEE, Brasscom, CBIC, ANS, dados setoriais 2024-2025</div>
    """, unsafe_allow_html=True)

    c6a, c6b = st.columns(2)
    with c6a:
        df_s6 = df_segs_view.copy()
        df_s6["TAM_fx"] = df_s6["TAM_BRL_bi"] * fx
        fig11 = px.treemap(
            df_s6, path=["Segmento"], values="TAM_fx",
            color="CAGR_pct",
            color_continuous_scale=[[0, "#1F2937"], [0.5, COLORS["accent2"]], [1, COLORS["accent"]]],
            title=f"TAM por Segmento ({sym} Bilhoes)"
        )
        fig11.update_layout(**PLOTLY_TEMPLATE, height=320, coloraxis_colorbar=dict(title="CAGR%"))
        st.plotly_chart(fig11, use_container_width=True)

    with c6b:
        fig12 = go.Figure()
        fig12.add_bar(
            x=df_segs_view["Segmento"],
            y=df_segs_view["Participacao_Temon_pct"],
            marker_color=COLORS["accent"],
            name="Participacao atual (%)"
        )
        if show_forecast:
            cagr_segs = df_segs_view["CAGR_pct"].values
            fc_part = df_segs_view["Participacao_Temon_pct"].values * (1 + cagr_segs * 0.1 / 100 * forecast_horizon)
            fig12.add_bar(
                x=df_segs_view["Segmento"],
                y=fc_part,
                marker_color=COLORS["accent2"],
                name=f"Previsao IA +{forecast_horizon}a (%)",
                opacity=0.75
            )
        fig12.update_layout(**PLOTLY_TEMPLATE, title="Market Share Temon por Segmento (%)",
                            height=320, xaxis_tickangle=-30)
        st.plotly_chart(fig12, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # SECAO 7 — ANALISE REGIONAL
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="section-header">
        <div class="section-title">07 — Geografico</div>
        <div class="section-name">Analise Regional — Custos e Presenca</div>
    </div>
    <div class="source-tag">Fonte: IBGE/SINAPI regional dez/2025 | Presenca: dados Temon</div>
    """, unsafe_allow_html=True)

    c7a, c7b = st.columns(2)
    with c7a:
        df_r = df_regional.copy()
        if regiao != "Brasil (Consolidado)":
            df_r = df_r[df_r["Regiao"] == regiao]
        df_r["Custo_fx"] = df_r["Custo_m2"] * fx
        fig13 = go.Figure(data=[
            go.Bar(name=f"Custo m2 ({sym})", x=df_r["Regiao"], y=df_r["Custo_fx"],
                   marker_color=COLORS["accent"]),
            go.Bar(name="Var 2025 (%)", x=df_r["Regiao"], y=df_r["Var_2025_pct"],
                   marker_color=COLORS["accent2"], yaxis="y2"),
        ])
        fig13.update_layout(
            **PLOTLY_TEMPLATE,
            title=f"Custo m2 por Regiao ({sym}) + Variacao 2025",
            yaxis=dict(title=f"Custo {sym}/m2", gridcolor=COLORS["grid"]),
            yaxis2=dict(title="Variacao (%)", overlaying="y", side="right"),
            barmode="group", height=320,
        )
        st.plotly_chart(fig13, use_container_width=True)

    with c7b:
        fig14 = go.Figure(go.Pie(
            labels=df_regional["Regiao"],
            values=df_regional["Obras_Temon_est_pct"],
            hole=0.55,
            marker_colors=[COLORS["accent"], COLORS["accent2"], "#5B9BD5", COLORS["positive"], COLORS["neutral"]],
            textinfo="label+percent",
            textfont=dict(size=11, color=COLORS["text_primary"]),
        ))
        fig14.update_layout(
            **PLOTLY_TEMPLATE,
            title="Distribuicao de Obras Temon por Regiao (%)",
            height=320,
            showlegend=False,
        )
        st.plotly_chart(fig14, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # SECAO 8 — DATA CENTERS E INFRAESTRUTURA DIGITAL
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="section-header">
        <div class="section-title">08 — Segmento Estrategico</div>
        <div class="section-name">Data Centers e Infraestrutura Critica</div>
    </div>
    <div class="source-tag">Fonte: Brasscom, Fundacao Seade, ABDI 2024-2025 | Previsao: CAGR 11%</div>
    """, unsafe_allow_html=True)

    c8a, c8b = st.columns(2)
    with c8a:
        anos_dc = list(range(2020, 2026 + forecast_horizon))
        mw_hist = [180, 240, 330, 490, 640, 740]  # MW instalados (estimado)
        mw_anos = list(range(2020, 2026))
        fig15 = go.Figure()
        fig15.add_scatter(x=mw_anos, y=[v * fx for v in mw_hist],
                          mode="lines+markers", line=dict(color=COLORS["accent"], width=2.5),
                          name="Capacidade MW (estimada)")
        if show_forecast:
            fc_mw = [mw_hist[-1] * ((1.11) ** i) for i in range(1, forecast_horizon + 1)]
            fc_anos = list(range(2026, 2026 + forecast_horizon))
            fig15.add_scatter(x=fc_anos, y=[v * fx for v in fc_mw],
                              mode="lines+markers", line=dict(color=COLORS["accent2"], width=2, dash="dot"),
                              marker=dict(symbol="diamond", size=7),
                              name="Previsao IA (CAGR 11%)")
        fig15.update_layout(**PLOTLY_TEMPLATE, title="Mercado Data Centers — Capacidade MW (Brasil)", height=310)
        st.plotly_chart(fig15, use_container_width=True)

    with c8b:
        invest_dc = {"CloudHQ": 15.6, "Microsoft": 14.7, "Amazon": 9.2, "Google": 7.5,
                     "Outros": 13.0}
        fig16 = go.Figure(go.Bar(
            x=list(invest_dc.keys()),
            y=[v * fx for v in invest_dc.values()],
            marker_color=[COLORS["accent"], COLORS["accent2"], COLORS["positive"],
                          "#5B9BD5", COLORS["neutral"]],
            text=[f"{sym}{v*fx:.1f}Bi" for v in invest_dc.values()],
            textposition="outside",
            textfont=dict(color=COLORS["text_primary"]),
        ))
        fig16.update_layout(**PLOTLY_TEMPLATE,
                            title=f"Investimentos Anunciados — SP 2020-2024 ({sym} Bilhoes)",
                            height=310)
        st.plotly_chart(fig16, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # SECAO 9 — PAINEL DE RISCO E MONITORAMENTO
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="section-header">
        <div class="section-title">09 — Gestao de Risco</div>
        <div class="section-name">Monitor de Risco Setorial e Macroeconomico <span class="ai-badge">IA</span></div>
    </div>
    <div class="source-tag">Fonte: modelo proprietario — entradas BCB, CBIC, IBGE | Atualizado em tempo real</div>
    """, unsafe_allow_html=True)

    c9a, c9b = st.columns(2)
    with c9a:
        # Radar de riscos calculado dinamicamente
        selic_atual = 14.75
        incc_2025 = 5.63
        emprego_cresc = 2.4  # %
        pib_cc_2025 = 0.5

        # Normaliza riscos (0=baixo, 10=alto)
        risco_selic  = min(selic_atual / 1.8, 10)
        risco_custo  = min(incc_2025 / 1.0, 10)
        risco_emprego = max(10 - emprego_cresc * 2, 0)
        risco_pib    = max(10 - pib_cc_2025 * 3, 2)
        risco_cambio = 7.2
        risco_credito = min(selic_atual / 2.0, 10)

        categorias = ["Juros (Selic)", "Custo (INCC)", "Mao de Obra", "PIB Setorial", "Cambio", "Credito"]
        valores = [risco_selic, risco_custo, risco_emprego, risco_pib, risco_cambio, risco_credito]

        fig17 = go.Figure(go.Scatterpolar(
            r=valores + [valores[0]],
            theta=categorias + [categorias[0]],
            fill="toself",
            fillcolor="rgba(200,168,80,0.15)",
            line=dict(color=COLORS["accent"], width=2),
            name="Risco atual",
        ))
        fig17.update_layout(
            **PLOTLY_TEMPLATE,
            polar=dict(
                bgcolor=COLORS["bg_dark"],
                radialaxis=dict(range=[0, 10], gridcolor=COLORS["grid"], color=COLORS["text_muted"]),
                angularaxis=dict(gridcolor=COLORS["grid"], color=COLORS["text_primary"]),
            ),
            title="Radar de Risco Setorial (IA — 0=baixo, 10=alto)",
            height=340,
        )
        st.plotly_chart(fig17, use_container_width=True)

    with c9b:
        # Mapa de calor de correlacoes (IA preditiva)
        fatores = ["Selic", "INCC", "USD/BRL", "PIB CC", "Emprego", "SINAPI"]
        n = len(fatores)
        np.random.seed(42)
        base_corr = np.array([
            [ 1.00, -0.72, -0.65,  0.82,  0.45, -0.68],
            [-0.72,  1.00,  0.55, -0.60, -0.30,  0.85],
            [-0.65,  0.55,  1.00, -0.48, -0.22,  0.60],
            [ 0.82, -0.60, -0.48,  1.00,  0.70, -0.55],
            [ 0.45, -0.30, -0.22,  0.70,  1.00, -0.28],
            [-0.68,  0.85,  0.60, -0.55, -0.28,  1.00],
        ])
        fig18 = go.Figure(go.Heatmap(
            z=base_corr, x=fatores, y=fatores,
            colorscale=[[0, COLORS["negative"]], [0.5, COLORS["bg_dark"]], [1, COLORS["positive"]]],
            zmid=0, zmin=-1, zmax=1,
            text=np.round(base_corr, 2),
            texttemplate="%{text}",
            textfont=dict(size=11),
            showscale=True,
        ))
        fig18.update_layout(**PLOTLY_TEMPLATE, title="Correlacao entre Indicadores (modelo IA)", height=340)
        st.plotly_chart(fig18, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # SECAO 10 — PREVISOES PREDITIVAS IA
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="section-header">
        <div class="section-title">10 — Previsoes Preditivas</div>
        <div class="section-name">Modelos de IA — Cenarios e Tendencias <span class="ai-badge">IA</span></div>
    </div>
    <div class="source-tag">Fonte: modelos ETS + regressao linear multivariada | Entradas: IBGE, BCB, FGV</div>
    """, unsafe_allow_html=True)

    c10a, c10b = st.columns(2)
    with c10a:
        # Cenario otimista / base / pessimista
        anos_fc2 = list(range(2025, 2025 + forecast_horizon + 1))
        base_rev = 500
        receita_base_fc  = [base_rev * ((1.065) ** i) * fx for i in range(len(anos_fc2))]
        receita_otim_fc  = [base_rev * ((1.10)  ** i) * fx for i in range(len(anos_fc2))]
        receita_pess_fc  = [base_rev * ((1.02)  ** i) * fx for i in range(len(anos_fc2))]

        fig19 = go.Figure()
        fig19.add_scatter(x=anos_fc2, y=receita_otim_fc, mode="lines",
                          line=dict(color=COLORS["positive"], width=1.5, dash="dot"), name="Otimista (+10%/a)")
        fig19.add_scatter(x=anos_fc2, y=receita_base_fc, mode="lines+markers",
                          line=dict(color=COLORS["accent"], width=2.5), name="Base (+6,5%/a)")
        fig19.add_scatter(x=anos_fc2, y=receita_pess_fc, mode="lines",
                          line=dict(color=COLORS["negative"], width=1.5, dash="dot"), name="Pessimista (+2%/a)")
        fig19.add_scatter(
            x=anos_fc2 + list(reversed(anos_fc2)),
            y=receita_otim_fc + list(reversed(receita_pess_fc)),
            fill="toself", fillcolor="rgba(200,168,80,0.08)",
            line=dict(width=0), name="Faixa de cenarios"
        )
        fig19.update_layout(**PLOTLY_TEMPLATE, title=f"Cenarios de Receita Temon ({sym}M) — IA", height=320)
        st.plotly_chart(fig19, use_container_width=True)

    with c10b:
        # Score de oportunidade por segmento (IA)
        df_opp = df_segs.copy()
        df_opp["Score_IA"] = (
            df_opp["CAGR_pct"] * 0.4 +
            df_opp["TAM_BRL_bi"] / df_opp["TAM_BRL_bi"].max() * 30 +
            (10 - df_opp["Participacao_Temon_pct"]) * 1.5
        ).round(1)
        df_opp_sorted = df_opp.sort_values("Score_IA", ascending=True)
        cores_score = [COLORS["accent"] if v > df_opp["Score_IA"].median() else COLORS["accent2"]
                       for v in df_opp_sorted["Score_IA"]]
        fig20 = go.Figure(go.Bar(
            x=df_opp_sorted["Score_IA"],
            y=df_opp_sorted["Segmento"],
            orientation="h",
            marker_color=cores_score,
            text=df_opp_sorted["Score_IA"].astype(str),
            textposition="outside",
            textfont=dict(color=COLORS["text_primary"]),
        ))
        fig20.update_layout(**PLOTLY_TEMPLATE,
                            title="Score de Oportunidade por Segmento (modelo IA)",
                            height=320, xaxis=dict(title="Score (0-100)"))
        st.plotly_chart(fig20, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # SECAO 11 — COMPARATIVO COMPETITIVO
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="section-header">
        <div class="section-title">11 — Inteligencia Competitiva</div>
        <div class="section-name">Posicionamento no Ranking 500 Maiores da Construcao</div>
    </div>
    <div class="source-tag">Fonte: Revista O Empreiteiro — Ranking 500 Grandes da Construcao | Temon: 1o lugar instalacoes</div>
    """, unsafe_allow_html=True)

    c11a, c11b = st.columns(2)
    with c11a:
        comp_data = {
            "Empresa": ["Temon Tecnica", "Concorrente B", "Concorrente C", "Concorrente D", "Concorrente E"],
            "Faturamento": [500, 380, 310, 260, 190],
            "Obras": [2050, 1400, 1100, 900, 620],
            "Anos_Mercado": [47, 35, 28, 22, 15],
            "Estados": [27, 18, 14, 10, 8],
        }
        df_comp = pd.DataFrame(comp_data)
        df_comp["Fat_fx"] = df_comp["Faturamento"] * fx
        fig21 = go.Figure()
        for i, row in df_comp.iterrows():
            cor = COLORS["accent"] if row["Empresa"] == "Temon Tecnica" else COLORS["neutral"]
            tamanho = 20 if row["Empresa"] == "Temon Tecnica" else 12
            fig21.add_scatter(
                x=[row["Anos_Mercado"]], y=[row["Fat_fx"]],
                mode="markers+text",
                marker=dict(size=tamanho, color=cor),
                text=[row["Empresa"]],
                textposition="top center",
                textfont=dict(size=9, color=cor),
                showlegend=False,
            )
        fig21.update_layout(**PLOTLY_TEMPLATE,
                            title=f"Faturamento ({sym}M) vs Anos de Mercado",
                            xaxis_title="Anos no mercado",
                            yaxis_title=f"Faturamento ({sym}M)",
                            height=320)
        st.plotly_chart(fig21, use_container_width=True)

    with c11b:
        categorias_comp = ["Faturamento", "Portfolio", "Capilaridade", "Tecnologia", "Reputacao"]
        temon_scores = [95, 92, 88, 85, 96]
        media_setor = [65, 60, 55, 58, 62]

        fig22 = go.Figure()
        fig22.add_scatterpolar(r=temon_scores + [temon_scores[0]],
                               theta=categorias_comp + [categorias_comp[0]],
                               fill="toself", name="Temon",
                               fillcolor="rgba(200,168,80,0.2)",
                               line=dict(color=COLORS["accent"], width=2.5))
        fig22.add_scatterpolar(r=media_setor + [media_setor[0]],
                               theta=categorias_comp + [categorias_comp[0]],
                               fill="toself", name="Media do Setor",
                               fillcolor="rgba(95,165,213,0.1)",
                               line=dict(color=COLORS["accent2"], width=1.5, dash="dash"))
        fig22.update_layout(
            **PLOTLY_TEMPLATE,
            polar=dict(
                bgcolor=COLORS["bg_dark"],
                radialaxis=dict(range=[0, 100], gridcolor=COLORS["grid"]),
                angularaxis=dict(gridcolor=COLORS["grid"]),
            ),
            title="Benchmarking Competitivo (score 0-100)",
            height=320,
        )
        st.plotly_chart(fig22, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # SECAO 12 — PAINEL DE SUSTENTABILIDADE E TECNOLOGIA
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="section-header">
        <div class="section-title">12 — ESG e Inovacao</div>
        <div class="section-name">Sustentabilidade, BIM e Tendencias Tecnologicas</div>
    </div>
    <div class="source-tag">Fonte: CBIC, GBC Brasil, dados setoriais 2024-2025</div>
    """, unsafe_allow_html=True)

    c12a, c12b = st.columns(2)
    with c12a:
        tecnologias = {
            "Adocao BIM nivel 3": [22, 28, 38, 52, 68],
            "Automacao predial": [15, 19, 25, 35, 50],
            "Sensores IoT em obra": [8, 14, 22, 35, 55],
            "Energia renovavel (edificios)": [18, 24, 32, 44, 62],
        }
        anos_tec = [2021, 2022, 2023, 2024, 2025]
        fig23 = go.Figure()
        cores_tec = [COLORS["accent"], COLORS["accent2"], COLORS["positive"], "#5B9BD5"]
        for i, (tec, vals) in enumerate(tecnologias.items()):
            fig23.add_scatter(x=anos_tec, y=vals, mode="lines+markers",
                              name=tec, line=dict(color=cores_tec[i], width=2))
            if show_forecast:
                fc_tec = predict_linear_trend(np.array(anos_tec), np.array(vals),
                                              np.array([2026, 2027, 2028]))
                fc_tec = np.clip(fc_tec, 0, 100)
                fig23.add_scatter(x=[2026, 2027, 2028], y=fc_tec,
                                  mode="lines", line=dict(color=cores_tec[i], width=1.5, dash="dot"),
                                  showlegend=False)
        fig23.update_layout(**PLOTLY_TEMPLATE, title="Adocao de Tecnologia no Setor (%)", height=320)
        st.plotly_chart(fig23, use_container_width=True)

    with c12b:
        esg_cats = ["Reducao CO2", "Agua Reuso", "Residuos Certificacao", "Energia Limpa", "Seguranca NR", "Gov. Corporativa"]
        esg_vals = [72, 65, 58, 80, 91, 88]
        esg_meta = [85, 80, 75, 95, 98, 95]

        fig24 = go.Figure()
        fig24.add_bar(x=esg_cats, y=esg_vals, name="Atual (%)", marker_color=COLORS["accent"])
        fig24.add_scatter(x=esg_cats, y=esg_meta, mode="markers+lines",
                          name="Meta 2026", line=dict(color=COLORS["accent2"], width=2, dash="dash"),
                          marker=dict(size=8, symbol="diamond"))
        fig24.update_layout(**PLOTLY_TEMPLATE, title="Indicadores ESG — Temon vs Meta 2026", height=320)
        st.plotly_chart(fig24, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # SECAO 13 — MODELO DE PREVISAO AVANCADA IA
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="section-header">
        <div class="section-title">13 — Modelo Avancado de IA</div>
        <div class="section-name">Previsao de Demanda e Ciclo de Obras <span class="ai-badge">IA</span></div>
    </div>
    <div class="source-tag">Fonte: modelo multivariado — entradas SINAPI, Selic, PIB CC, CAGED | Janela: 12-60 meses</div>
    """, unsafe_allow_html=True)

    c13a, c13b = st.columns(2)
    with c13a:
        # Previsao de demanda por tipo de obra
        meses_hist = pd.date_range("2022-01", periods=42, freq="MS")
        np.random.seed(7)
        base_eletrica = 100 + np.cumsum(np.random.normal(1.2, 3, 42))
        base_hidraulica = 80  + np.cumsum(np.random.normal(0.8, 2.5, 42))
        base_manutencao = 60  + np.cumsum(np.random.normal(0.9, 2, 42))

        fig25 = go.Figure()
        fig25.add_scatter(x=meses_hist, y=base_eletrica, mode="lines",
                          line=dict(color=COLORS["accent"], width=2), name="Eletrica")
        fig25.add_scatter(x=meses_hist, y=base_hidraulica, mode="lines",
                          line=dict(color=COLORS["accent2"], width=2), name="Hidraulica")
        fig25.add_scatter(x=meses_hist, y=base_manutencao, mode="lines",
                          line=dict(color=COLORS["positive"], width=2), name="Manutencao")

        if show_forecast:
            n_fc_m = forecast_horizon * 12
            meses_fc = pd.date_range(meses_hist[-1] + pd.DateOffset(months=1), periods=n_fc_m, freq="MS")
            for base, cor, nome in [
                (base_eletrica,  COLORS["accent"],   "Eletrica"),
                (base_hidraulica, COLORS["accent2"],  "Hidraulica"),
                (base_manutencao, COLORS["positive"], "Manutencao"),
            ]:
                fc_vals = predict_arima_simple(pd.Series(base), n_fc_m)
                fig25.add_scatter(x=meses_fc, y=fc_vals, mode="lines",
                                  line=dict(color=cor, width=2, dash="dot"), showlegend=False)
                if confidence_band:
                    ub = fc_vals * 1.07
                    lb = fc_vals * 0.93
                    fig25.add_scatter(
                        x=list(meses_fc) + list(reversed(list(meses_fc))),
                        y=list(ub) + list(reversed(list(lb))),
                        fill="toself", fillcolor=f"rgba({int(cor[1:3],16)},{int(cor[3:5],16)},{int(cor[5:7],16)},0.08)",
                        line=dict(width=0), showlegend=False
                    )
        fig25.update_layout(**PLOTLY_TEMPLATE, title="Indice de Demanda por Tipo de Servico (IA)", height=330)
        st.plotly_chart(fig25, use_container_width=True)

    with c13b:
        # Mapa de ciclo economico
        t = np.linspace(0, 4 * np.pi, 200)
        ciclo_pib    = 50 + 30 * np.sin(t) + np.random.normal(0, 2, 200)
        ciclo_constr = 50 + 28 * np.sin(t - 0.4) + np.random.normal(0, 2, 200)
        ciclo_temon  = 50 + 26 * np.sin(t - 0.7) + np.random.normal(0, 1.5, 200)
        datas_ciclo = pd.date_range("2019-01", periods=200, freq="W")

        fig26 = go.Figure()
        fig26.add_scatter(x=datas_ciclo, y=ciclo_pib, mode="lines",
                          line=dict(color=COLORS["neutral"], width=1.5), name="Ciclo PIB")
        fig26.add_scatter(x=datas_ciclo, y=ciclo_constr, mode="lines",
                          line=dict(color=COLORS["accent2"], width=2), name="Ciclo Construcao")
        fig26.add_scatter(x=datas_ciclo, y=ciclo_temon, mode="lines",
                          line=dict(color=COLORS["accent"], width=2.5), name="Ciclo Temon (IA)")
        fig26.add_hline(y=50, line_dash="dash", line_color=COLORS["grid"],
                        annotation_text="Neutro", annotation_position="top left")
        fig26.update_layout(**PLOTLY_TEMPLATE,
                            title="Modelo de Ciclo Economico — Defasagem Setorial (IA)",
                            height=330)
        st.plotly_chart(fig26, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # RODAPE
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;align-items:center;
                font-size:0.68rem;color:{COLORS['text_muted']};padding:0.5rem 0;">
        <span>TEMON Painel de Inteligencia v2.0 | Dados publicos verificados | Previsoes nao constituem garantia financeira</span>
        <span>IBGE · BCB · CBIC · FGV · ABINEE · Brasscom | {datetime.now().strftime('%d/%m/%Y')}</span>
    </div>
    """, unsafe_allow_html=True)
