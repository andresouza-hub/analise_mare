"""
GAC Tidal Insight - Aplicativo Streamlit para Análise de Aquíferos Costeiros
Versão 2.0 - Interface Melhorada com UX otimizado

Desenvolvido por: André Souza
Especialista em Gerenciamento de Áreas Contaminadas (GAC)
"""

from __future__ import annotations
import re
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional, Any, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import altair as alt
from datetime import datetime

from src.core.orquestrador_b import executar_analise_b

# =============================================================================
# Configuração da página
# =============================================================================
st.set_page_config(
    page_title="GAC Tidal Insight",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Utilitários
# =============================================================================

def salvar_upload_tmp(uploaded_file, pasta: Path) -> Optional[str]:
    """Salva arquivo uploadado em diretório temporário."""
    if uploaded_file is None:
        return None
    nome = Path(uploaded_file.name).name
    nome = re.sub(r'[\\/:*?"<>\n\r\t]', "_", nome)
    caminho = pasta / nome
    with open(caminho, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(caminho)


def base_nome(uploaded_file) -> str:
    """Extrai nome base de arquivo uploadado."""
    return Path(uploaded_file.name).stem.strip()


def _cardinal_16(angle_deg: float) -> str:
    """Converte ângulo para direção cardinal."""
    dirs = [
        "E", "ENE", "NE", "NNE", "N", "NNW", "NW", "WNW",
        "W", "WSW", "SW", "SSW", "S", "SSE", "SE", "ESE"
    ]
    a = angle_deg % 360
    return dirs[int((a + 11.25) // 22.5) % 16]


def _pick_col(df: pd.DataFrame, candidatos: Tuple[str, ...]) -> str:
    """Seleciona primeira coluna disponível."""
    for c in candidatos:
        if c in df.columns:
            return c
    raise KeyError(f"Nenhuma coluna encontrada entre {candidatos}")


def _ensure_dt(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Garante coluna como datetime."""
    df = df.copy()
    df[col] = pd.to_datetime(df[col])
    return df


# =============================================================================
# Constantes de estilo
# =============================================================================
COLORS = {
    'primary': '#1E88E5',
    'secondary': '#43A047',
    'accent': '#FB8C00',
    'error': '#E53935',
    'warning': '#FFB300',
    'sizigia': '#E53935',
    'quadratura': '#1E88E5',
    'intermediario': '#757575'
}

# =============================================================================
# CSS Customizado
# =============================================================================
st.markdown("""
<style>
    /* Cabeçalho principal */
    .main-header {
        background: linear-gradient(135deg, #1E88E5 0%, #43A047 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }

    /* Cards de métricas */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #1E88E5;
    }

    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    .status-success { background: #E8F5E9; color: #2E7D32; }
    .status-warning { background: #FFF3E0; color: #EF6C00; }
    .status-error { background: #FFEBEE; color: #C62828; }

    /* Info boxes */
    .info-box {
        background: #E3F2FD;
        border-left: 4px solid #1E88E5;
        padding: 15px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
    }

    /* Tooltip style */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #333;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 8px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 11px;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px 8px 0 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR - Configurações e Upload
# =============================================================================
with st.sidebar:
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; margin: 0; font-size: 24px;">🌊 GAC Tidal Insight</h1>
        <p style="color: rgba(255,255,255,0.9); margin: 5px 0 0 0; font-size: 12px;">
            Análise de Aquíferos Costeiros
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Logo do autor
    st.markdown("""
    <div style="text-align: center; padding: 10px; background: white; border-radius: 10px; margin-bottom: 20px;">
        <h3 style="margin: 0; color: #1E88E5;">👨‍🔬 André Souza</h3>
        <p style="margin: 5px 0 0 0; font-size: 11px; color: #666;">Especialista GAC</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Seção de Upload com expander
    with st.expander("📁 1. Upload de Poços", expanded=True):
        st.caption("Arquivos CSV com dados de nível d'água")
        pocos_files = st.file_uploader(
            "Selecione os CSVs",
            type=["csv"],
            accept_multiple_files=True,
            key="pocos_uploader",
            help="Formato esperado: colunas 'hora' (datetime) e 'Nivel' (m)"
        )
        if pocos_files:
            st.success(f"✅ {len(pocos_files)} arquivo(s) carregado(s)")

    # Data/hora de início
    with st.expander("⏰ 2. Data/Hora de Início", expanded=True):
        inicios_por_poco: Dict[str, pd.Timestamp] = {}
        if pocos_files:
            for f in pocos_files:
                poco = base_nome(f)
                inicio_txt = st.text_input(
                    f"Início – {poco}",
                    value="01/01/2026 00:00",
                    key=f"inicio_{poco}",
                    help="Data no formato dd/mm/aaaa hh:mm"
                )
                try:
                    inicios_por_poco[poco] = pd.to_datetime(inicio_txt, dayfirst=True)
                except Exception:
                    st.error(f"Data inválida para {poco}")

    # Upload de maré
    with st.expander("🌊 3. Dados de Maré (Opcional)", expanded=False):
        st.caption("Arquivo CSV com eventos de maré")
        mare_file = st.file_uploader(
            "Maré (CSV)",
            type=["csv"],
            key="mare_uploader",
            help="Formato: colunas 'Mês', 'Dia', 'Hora', 'Altura maré'"
        )
        if mare_file:
            st.success("✅ Maré carregada")

    # Upload de gradiente
    with st.expander("🧭 4. Gradiente 2D (Opcional)", expanded=False):
        st.caption("Arquivos para cálculo do gradiente hidráulico")
        cotas_file = st.file_uploader(
            "Planilha de Cotas (XLSX)",
            type=["xlsx"],
            key="cotas_uploader",
            help="Colunas: 'Poco', 'Cota_TOC_m'"
        )
        kmz_file = st.file_uploader(
            "Arquivo KMZ (poços)",
            type=["kmz"],
            key="kmz_uploader",
            help="Arquivo com coordenadas XY dos poços"
        )
        if cotas_file and kmz_file:
            st.success("✅ Arquivos de gradiente carregados")

    st.divider()

    # Configurações avançadas
    with st.expander("⚙️ Configurações Avançadas", expanded=False):
        st.caption("Parâmetros de análise")
        col_config1, col_config2 = st.columns(2)
        with col_config1:
            janela_h = st.number_input("Janela (h)", value=71, min_value=24, max_value=168)
        with col_config2:
            passo_h = st.number_input("Passo (h)", value=24, min_value=1, max_value=72)
        mostrar_ia = st.checkbox("Ativar interpretação IA", value=True)
        mostrar_pig = st.checkbox("Análise PIG Quinzenal", value=True)

    st.divider()

    # Informações
    st.markdown("""
    <div class="info-box" style="font-size: 11px;">
        <strong>📋 Metodologia:</strong><br>
        • Método Serfes (1991) - nível médio<br>
        • FFT - decomposição espectral<br>
        • Lag maré-poço<br>
        • Gradiente hidráulico 2D<br>
        • Modulação quinzenal
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# Utilitários para gráficos
# =============================================================================

# Tamanhos das figuras
FIG_TS = (8.0, 3.0)
FIG_VIOLIN = (6.0, 4.0)
FIG_MAP = (5.0, 5.0)
FIG_GRAD_SER = (8.0, 2.5)
FIG_ENV = (8.0, 3.0)

def plot_poco_nivel_serfes_mare(
    df, col_hora, col_nivel, serfes,
    df_mare, col_mare_dt, col_mare, titulo
):
    """Gráfico de série temporal com nível, Serfes e maré."""
    fig, ax1 = plt.subplots(figsize=FIG_TS)
    fig.set_dpi(120)

    # Nível do poço
    ax1.plot(df[col_hora], df[col_nivel], linewidth=1.0, label="Nível do poço", color='#1E88E5')
    ax1.axhline(serfes, linestyle="--", linewidth=1.2, label="Serfes", color='#FB8C00')
    ax1.fill_between(df[col_hora], df[col_nivel], serfes, alpha=0.2, color='#1E88E5')
    ax1.set_ylabel("Nível (m)", fontsize=10, color='#1E88E5')
    ax1.tick_params(axis="y", labelcolor='#1E88E5')
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    h, l = ax1.get_legend_handles_labels()

    # Maré no eixo secundário
    if df_mare is not None and not df_mare.empty and col_mare_dt and col_mare:
        ax2 = ax1.twinx()
        ax2.plot(
            df_mare[col_mare_dt],
            df_mare[col_mare],
            marker="o",
            markersize=3,
            linewidth=1.0,
            color="#43A047",
            label="Maré"
        )
        ax2.set_ylabel("Altura da maré (m)", fontsize=10, color='#43A047')
        ax2.tick_params(axis="y", labelcolor='#43A047')
        h2, l2 = ax2.get_legend_handles_labels()
        h += h2
        l += l2

    ax1.legend(h, l, loc="upper right", fontsize=9)
    ax1.set_title(titulo, fontsize=11, fontweight='bold')
    ax1.tick_params(axis="x", labelsize=8, rotation=30)

    fig.tight_layout()
    return fig


def plot_violin_box(df, col_nivel, titulo):
    """Gráfico violin + boxplot."""
    y = df[col_nivel].dropna().values
    fig, ax = plt.subplots(figsize=FIG_VIOLIN)
    fig.set_dpi(120)

    parts = ax.violinplot(y, showextrema=False, showmeans=True)
    for pc in parts['bodies']:
        pc.set_facecolor('#1E88E5')
        pc.set_alpha(0.6)
    parts['cmeans'].set_color('#FB8C00')

    ax.boxplot(y, widths=0.25, patch_artist=True,
               boxprops=dict(facecolor='white', color='#1E88E5'),
               medianprops=dict(color='#E53935', linewidth=2),
               whiskerprops=dict(color='#1E88E5'),
               capprops=dict(color='#1E88E5'))

    ax.set_title(titulo, fontsize=11, fontweight='bold')
    ax.set_ylabel("Nível (m)", fontsize=10)
    ax.set_xticks([])
    ax.grid(True, axis="y", alpha=0.3, linestyle='-', linewidth=0.5)

    # Adicionar estatísticas
    stats_text = f"μ={np.mean(y):.3f}\nσ={np.std(y):.3f}"
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.tight_layout()
    return fig


def plot_gradiente_mapa(df_xy: pd.DataFrame, angulo: float, titulo: str):
    """Mapa de gradiente com seta direcional."""
    fig, ax = plt.subplots(figsize=FIG_MAP)
    fig.set_dpi(120)

    ax.scatter(df_xy["X_m"], df_xy["Y_m"], s=100, c='#1E88E5', edgecolors='white', linewidth=2, zorder=5)

    for _, r in df_xy.iterrows():
        ax.annotate(
            r["Poco"],
            (r["X_m"], r["Y_m"]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold',
            color='#333'
        )

    # Seta de gradiente
    x0, y0 = df_xy["X_m"].mean(), df_xy["Y_m"].mean()
    amp_x = float(df_xy["X_m"].max() - df_xy["X_m"].min())
    amp_y = float(df_xy["Y_m"].max() - df_xy["Y_m"].min())
    L = max(amp_x, amp_y) * 0.20
    if L == 0:
        L = 1.0

    dx = np.cos(np.deg2rad(angulo)) * L
    dy = np.sin(np.deg2rad(angulo)) * L

    ax.annotate('', xy=(x0 + dx, y0 + dy), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color='#E53935', lw=3))

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(titulo, fontsize=11, fontweight='bold')
    ax.set_xlabel("X (m)", fontsize=10)
    ax.set_ylabel("Y (m)", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_facecolor('#f8f9fa')

    # Adicionar bússola
    circle = plt.Circle((0.95, 0.95), 0.08, transform=ax.transAxes,
                        facecolor='white', edgecolor='#333', zorder=10)
    ax.add_patch(circle)
    ax.text(0.95, 0.95, 'N', transform=ax.transAxes, ha='center', va='center', fontsize=8, fontweight='bold')

    fig.tight_layout()
    return fig


def plot_gradiente_series(df_v: pd.DataFrame):
    """Séries temporais de ângulo e módulo do gradiente."""
    fig1, ax1 = plt.subplots(figsize=FIG_GRAD_SER)
    fig1.set_dpi(120)
    ax1.plot(df_v["hora"], df_v["angulo_fluxo_deg"], linewidth=1.2, color='#1E88E5')
    ax1.fill_between(df_v["hora"], df_v["angulo_fluxo_deg"], alpha=0.3, color='#1E88E5')
    ax1.set_title("Ângulo do fluxo (°)", fontsize=10, fontweight='bold')
    ax1.tick_params(labelsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylabel("Ângulo (°)", fontsize=9)
    ax1.set_ylim(0, 360)
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(figsize=FIG_GRAD_SER)
    fig2.set_dpi(120)
    ax2.plot(df_v["hora"], df_v["modulo_gradiente"], linewidth=1.2, color='#43A047')
    ax2.fill_between(df_v["hora"], df_v["modulo_gradiente"], alpha=0.3, color='#43A047')
    ax2.set_title("Módulo do gradiente (m/m)", fontsize=10, fontweight='bold')
    ax2.tick_params(labelsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylabel("Gradiente (m/m)", fontsize=9)
    fig2.tight_layout()

    return fig1, fig2


def plot_envelope_quinzenal(df_env: pd.DataFrame, poco: str | None = None):
    """Gráfico de envelope quinzenal."""
    if df_env is None or df_env.empty:
        return None

    if poco:
        d = df_env[df_env["Poco"] == poco].copy()
        titulo = f"Envelope – Poço {poco}"
    else:
        d = df_env.groupby(["dia", "Regime_lunar"], as_index=False)[["amp_poco", "amp_mare"]].mean()
        d["Poco"] = "(média)"
        titulo = "Envelope – Média dos poços"

    if d.empty:
        return None

    cores = {"SIZIGIA": COLORS['sizigia'], "QUADRATURA": COLORS['quadratura'], "INTERMEDIARIO": COLORS['intermediario']}

    fig, ax1 = plt.subplots(figsize=FIG_ENV)
    fig.set_dpi(120)

    for reg, g in d.groupby("Regime_lunar"):
        ax1.plot(g["dia"], g["amp_poco"], marker="o", linestyle="-", linewidth=1.2,
                 markersize=4, color=cores.get(reg, "gray"), label=f"Poço – {reg.title()}")

    ax1.set_ylabel("Amp. diária do poço (m)", fontsize=9, color='#1E88E5')
    ax1.tick_params(axis="y", labelcolor='#1E88E5')
    ax1.tick_params(axis="x", labelsize=8, rotation=30)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(d["dia"], d["amp_mare"], color="#43A047", linewidth=1.5, label="Maré")
    ax2.set_ylabel("Amp. diária da maré (m)", fontsize=9, color='#43A047')
    ax2.tick_params(axis="y", labelcolor='#43A047')

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)

    ax1.set_title(titulo, fontsize=11, fontweight='bold')
    fig.tight_layout()
    return fig


# =============================================================================
# Gráficos Altair Interativos
# =============================================================================

def create_altair_time_series(df, col_time, col_value, title, color='#1E88E5'):
    """Cria gráfico de série temporal com Altair."""
    chart = alt.Chart(df).mark_line(color=color).encode(
        x=alt.X(f'{col_time}:T', title='Tempo', axis=alt.Axis(format='%d/%m %H:%M')),
        y=alt.Y(f'{col_value}:Q', title='Valor', scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip(f'{col_time}:T', title='Tempo', format='%d/%m %H:%M'),
                 alt.Tooltip(f'{col_value}:Q', title='Valor', format='.4f')]
    ).properties(
        title=title,
        height=250
    ).interactive()
    return chart


def create_altair_bar_chart(df, x_col, y_col, title, color='#1E88E5'):
    """Cria gráfico de barras com Altair."""
    chart = alt.Chart(df).mark_bar(color=color).encode(
        x=alt.X(f'{x_col}:N', title=x_col, axis=alt.Axis(labelAngle=-45)),
        y=alt.Y(f'{y_col}:Q', title=y_col),
        tooltip=[alt.Tooltip(f'{x_col}:N'), alt.Tooltip(f'{y_col}:Q', format='.4f')]
    ).properties(
        title=title,
        height=200
    )
    return chart


# =============================================================================
# Funções de Interpretação
# =============================================================================

def interpretar_resultados_avancado(resultado: dict, df_pig: Optional[pd.DataFrame] = None) -> str:
    """Gera interpretação técnica consolidada cruzando R, lag, FFT, IMQ e gradiente."""
    meta = resultado.get("meta", {})
    df_con = resultado.get("consolidado")
    grad = resultado.get("gradiente2d")

    linhas: List[str] = []
    linhas.append("## 🧠 Interpretação Técnica Automática")
    linhas.append("")
    linhas.append("### 1. QAQC do Processamento")
    linhas.append(f"| Indicador | Valor |")
    linhas.append(f"|------------|-------|")
    linhas.append(f"| Poços processados | **{meta.get('processados', 0)}** |")
    linhas.append(f"| Poços descartados | **{meta.get('pulados', 0)}** |")
    linhas.append(f"| Dados de maré | **{'Sim' if meta.get('tem_mare') else 'Não'}** |")
    linhas.append(f"| Insumos gradiente 2D | **{'Sim' if meta.get('tem_insumos_gradiente') else 'Não'}** |")

    if isinstance(df_con, pd.DataFrame) and not df_con.empty:
        col_poco = "Poço" if "Poço" in df_con.columns else ("Poco" if "Poco" in df_con.columns else "Poco")
        linhas.append("")
        linhas.append("### 2. Diagnóstico por Poço")

        for poco, g in df_con.groupby(col_poco):
            amp_p2p = g.get("Amplitude pico-a-pico (m)")
            rms = g.get("Amplitude RMS (m)")
            rss12 = g.get("Amp semidiurna (RSS, ~12h) (m)")
            rss24 = g.get("Amp diurna (RSS, ~24h) (m)")
            lag = g.get("Lag médio total (h)")

            amp_med = float(amp_p2p.median()) if amp_p2p is not None else np.nan
            rms_med = float(rms.median()) if rms is not None else np.nan
            lag_med = float(lag.median()) if lag is not None else np.nan

            # Dominância espectral
            dom_txt = "indefinida"
            if rss12 is not None and rss24 is not None:
                r12 = float(rss12.median())
                r24 = float(rss24.median())
                if pd.notna(r12) and pd.notna(r24) and r24 > 0:
                    ratio = r12 / r24
                    if ratio > 1.2:
                        dom_txt = "semidiurna (~12 h)"
                    elif ratio < 0.8:
                        dom_txt = "diurna (~24 h)"
                    else:
                        dom_txt = "mista (12–24 h)"

            # Classificação de lag
            if pd.isna(lag_med):
                lag_txt = "sem maré"
            elif lag_med <= 2:
                lag_txt = "resposta rápida (≤2 h)"
            elif lag_med <= 6:
                lag_txt = "resposta moderada (2–6 h)"
            elif lag_med <= 12:
                lag_txt = "resposta lenta (6–12 h)"
            else:
                lag_txt = "resposta muito lenta (>12 h)"

            # Índice R (se disponível em df_pig)
            r_txt = "—"
            if df_pig is not None and not df_pig.empty:
                rp = df_pig[(df_pig["Poco"] == poco) & df_pig["R_indice"].notna()]["R_indice"].mean()
                if pd.notna(rp):
                    if rp > 0.6:
                        r_txt = f"R={rp:.2f} (influência forte)"
                    elif rp >= 0.3:
                        r_txt = f"R={rp:.2f} (influência moderada)"
                    elif rp >= 0.05:
                        r_txt = f"R={rp:.2f} (resposta amortecida)"
                    else:
                        r_txt = f"R≈0 (desacoplado)"

            linhas.append(f"**Poço {poco}**")
            linhas.append(f"- Amplitude p2p: **{amp_med:.3f} m** | RMS: **{rms_med:.3f} m**")
            linhas.append(f"- Dominância: **{dom_txt}** | Lag: **{lag_txt}**")
            linhas.append(f"- Índice de resposta à maré: {r_txt}")

    # Gradiente 2D
    if isinstance(grad, dict) and grad.get("status") == "OK":
        df_jg = grad.get("janelas")
        if isinstance(df_jg, pd.DataFrame) and not df_jg.empty:
            ang = float(df_jg.iloc[0]["Ângulo dominante (°)"])
            mod = float(df_jg.iloc[0]["Módulo médio gradiente (m/m)"])
            linhas.append("")
            linhas.append("### 3. Gradiente Hidráulico 2D")
            linhas.append(f"- Direção: **{ang:.1f}° ({_cardinal_16(ang)})** | Módulo: **{mod:.4f} m/m**")
            inv_cols = [c for c in df_jg.columns if str(c).startswith("Inversão (")]
            if inv_cols:
                col_inv = inv_cols[0]
                inv_pct = float(100.0 * (df_jg[col_inv].astype(str).str.upper() == "SIM").mean())
                linhas.append(f"- Inversões nas janelas: **{inv_pct:.1f}%**")

    # Modulação quinzenal
    if df_pig is not None and not df_pig.empty:
        linhas.append("")
        linhas.append("### 4. Modulação Quinzenal (sizígia × quadratura)")
        imq_vals = []
        for poco, g in df_pig.groupby("Poco"):
            a_s = g.loc[g["Regime"] == "SIZIGIA", "Amp_p2p_m"].mean()
            a_q = g.loc[g["Regime"] == "QUADRATURA", "Amp_p2p_m"].mean()
            if pd.notna(a_s) and pd.notna(a_q) and a_q > 0:
                imq_vals.append(a_s / a_q)
        if imq_vals:
            linhas.append(f"- IMQ médio (Amp_sizígia/Amp_quadratura): **{np.nanmean(imq_vals):.2f}**")

        dom_list = []
        for poco, g in df_pig.groupby("Poco"):
            r12 = g["RSS_12h_m"].mean()
            r24 = g["RSS_24h_m"].mean()
            if pd.notna(r12) and pd.notna(r24):
                if r12 > r24:
                    dom_list.append("12h")
                elif r24 > r12:
                    dom_list.append("24h")
                else:
                    dom_list.append("mista")
        if dom_list:
            frac_12 = 100.0 * dom_list.count("12h") / len(dom_list)
            linhas.append(f"- Dominância espectral: **{frac_12:.0f}%** dos poços com banda ~12 h dominante")

    # Conclusão técnica consolidada
    linhas.append("")
    linhas.append("### 5. Conclusão Técnica Consolidada")
    linhas.append(
        "A resposta hidráulica mostra coerência com o forçamento periódico (maré) quando presente, "
        "com variação de lag e dominância espectral indicativas da assinatura local do aquífero. "
        "Valores elevados de R e IMQ sugerem forte conectividade hidráulica com o sistema marinho; "
        "inversões frequentes do gradiente indicam dinâmica de fluxo bidirecional sob influência mareal."
    )

    return "\n".join(linhas)


# =============================================================================
# Funções auxiliares para análise lunar (sizígia × quadratura)
# =============================================================================
SINODICO_DIAS = 29.53058867
LUA_REF = pd.Timestamp("2000-01-06 18:14:00", tz=None)


def calcular_idade_lunar(idx) -> pd.Series:
    """Calcula idade lunar (dias desde a última lua nova) usando mês sinódico."""
    if isinstance(idx, pd.DatetimeIndex):
        s = pd.Series(idx, index=idx)
    else:
        s = pd.to_datetime(pd.Series(idx))
    delta = (s - LUA_REF).dt.total_seconds() / 86400.0
    idade = delta.mod(SINODICO_DIAS)
    return idade.astype(float)


def classificar_regime_lunar(idade_dias: float) -> str:
    """Classifica como SIZIGIA, QUADRATURA ou INTERMEDIARIO conforme idade lunar."""
    if pd.isna(idade_dias):
        return "INTERMEDIARIO"
    if idade_dias <= 3.0 or idade_dias >= 26.5 or (14.0 <= idade_dias <= 17.0):
        return "SIZIGIA"
    if (5.0 <= idade_dias <= 9.0) or (20.0 <= idade_dias <= 24.0):
        return "QUADRATURA"
    return "INTERMEDIARIO"


def _amp_diaria(df: pd.DataFrame, col_tempo: str, col_val: str) -> pd.DataFrame:
    """Amplitude diária (max - min) por dia."""
    df = df.copy()
    df[col_tempo] = pd.to_datetime(df[col_tempo])
    df["dia"] = df[col_tempo].dt.floor("D")
    g = df.groupby("dia")[col_val].agg(["min", "max"]).reset_index()
    g["amp"] = g["max"] - g["min"]
    return g[["dia", "amp"]]


def _duracao_total(resultado: dict) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp], float]:
    """Retorna (início, fim, duração em dias) da série de poços."""
    series = resultado.get("series_horarias", {})
    if not series:
        return None, None, 0.0
    inicios = []
    fins = []
    for df in series.values():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        col_h = _pick_col(df, ("hora", "datetime", "datahora"))
        df = _ensure_dt(df, col_h)
        inicios.append(df[col_h].min())
        fins.append(df[col_h].max())
    if not inicios:
        return None, None, 0.0
    ini = min(inicios)
    fim = max(fins)
    dur_d = float((fim - ini).total_seconds() / 86400.0)
    return ini, fim, dur_d


def analise_pig_quinzenal(resultado: dict) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Análise de modulação quinzenal (sizígia × quadratura).

    Requisitos: presença de maré e série com ≥ 20 dias.
    Calcula amplitudes diárias por poço, classifica regime lunar (sizígia/quadratura),
    e agrega métricas (R, lag, RSS espectrais) por poço × regime.
    """
    df_mare = resultado.get("mare_eventos")
    if not isinstance(df_mare, pd.DataFrame) or df_mare.empty:
        return pd.DataFrame(), pd.DataFrame(), {"status": "SKIP", "motivo": "Sem maré"}

    ini, fim, dur_d = _duracao_total(resultado)
    if dur_d < 20:
        return pd.DataFrame(), pd.DataFrame(), {
            "status": "SKIP",
            "motivo": f"Período de {dur_d:.1f} dias < 20 dias mínimos"
        }

    # Amplitude diária da maré
    col_md = _pick_col(df_mare, ("datetime", "hora", "datahora"))
    col_m = _pick_col(df_mare, ("maré", "mare", "altura"))
    df_mare_d = _amp_diaria(_ensure_dt(df_mare, col_md), col_md, col_m)

    # Envelope por poço (amplitude diária do nível)
    series = resultado.get("series_horarias", {})
    env_rows: List[pd.DataFrame] = []
    for poco, df_h in series.items():
        if not isinstance(df_h, pd.DataFrame) or df_h.empty:
            continue
        col_h = _pick_col(df_h, ("hora", "datetime", "datahora"))
        col_n = _pick_col(df_h, ("Nivel", "Nível", "nivel", "level"))
        df_env = _amp_diaria(_ensure_dt(df_h, col_h), col_h, col_n)
        if df_env.empty:
            continue
        df_env["Poco"] = poco
        env_rows.append(df_env)
    if not env_rows:
        return pd.DataFrame(), pd.DataFrame(), {"status": "SKIP", "motivo": "Sem amplitude diária de poço"}
    df_env_all = pd.concat(env_rows, ignore_index=True)

    df_env_all = df_env_all.merge(df_mare_d.rename(columns={"amp": "amp_mare"}), on="dia", how="left")
    df_env_all.rename(columns={"amp": "amp_poco"}, inplace=True)

    # Idade lunar e regime por dia
    idade = calcular_idade_lunar(df_env_all["dia"])
    df_env_all["idade_lua_dias"] = idade.values
    df_env_all["Regime_lunar"] = df_env_all["idade_lua_dias"].apply(classificar_regime_lunar)

    # Tabela PIG por poço e regime
    df_con = resultado.get("consolidado")
    pig_rows = []
    if isinstance(df_con, pd.DataFrame) and not df_con.empty:
        dfc = df_con.copy()
        col_poco = "Poço" if "Poço" in dfc.columns else ("Poco" if "Poco" in dfc.columns else None)
        if not col_poco:
            col_poco = "Poco"
            dfc["Poco"] = dfc.get("Poço", "?")

        if "Início" in dfc.columns and "Fim" in dfc.columns:
            dfc["Início"] = pd.to_datetime(dfc["Início"])
            dfc["Fim"] = pd.to_datetime(dfc["Fim"])
            dfc["centro"] = dfc["Início"] + (dfc["Fim"] - dfc["Início"]) / 2
            dfc["idade_lua_dias"] = calcular_idade_lunar(dfc["centro"]).values
            dfc["Regime_lunar"] = dfc["idade_lua_dias"].apply(classificar_regime_lunar)
        else:
            dfc["centro"] = pd.NaT
            dfc["idade_lua_dias"] = np.nan
            dfc["Regime_lunar"] = "INTERMEDIARIO"

        col_amp_p2p = "Amplitude pico-a-pico (m)"
        col_rms = "Amplitude RMS (m)"
        col_rss12 = "Amp semidiurna (RSS, ~12h) (m)"
        col_rss24 = "Amp diurna (RSS, ~24h) (m)"
        col_lag = "Lag médio total (h)"

        df_mare_dt = _ensure_dt(df_mare, col_md)[[col_md, col_m]].rename(columns={col_md: "t", col_m: "mare"})

        def _amp_mare_na_janela(t0: pd.Timestamp, t1: pd.Timestamp) -> float:
            seg = df_mare_dt[(df_mare_dt["t"] >= t0) & (df_mare_dt["t"] <= t1)]["mare"].dropna()
            if seg.empty:
                return np.nan
            return float(seg.max() - seg.min())

        for (poco, regime), g in dfc.groupby([col_poco, "Regime_lunar"], dropna=False):
            if len(g) == 0:
                continue
            amp_p2p = float(g[col_amp_p2p].mean()) if col_amp_p2p in g.columns else np.nan
            rms = float(g[col_rms].mean()) if col_rms in g.columns else np.nan
            rss12 = float(g[col_rss12].mean()) if col_rss12 in g.columns else np.nan
            rss24 = float(g[col_rss24].mean()) if col_rss24 in g.columns else np.nan
            lag_med = float(g[col_lag].mean()) if col_lag in g.columns else np.nan

            r_vals = []
            if "Início" in g.columns and "Fim" in g.columns and col_amp_p2p in g.columns:
                for _, row in g.iterrows():
                    a_poco = row.get(col_amp_p2p, np.nan)
                    if pd.isna(a_poco):
                        continue
                    a_mare = _amp_mare_na_janela(row["Início"], row["Fim"])
                    if pd.notna(a_mare) and a_mare > 0:
                        r_vals.append(float(a_poco) / float(a_mare))
            r_med = float(np.nanmean(r_vals)) if len(r_vals) else np.nan

            inv_pct = np.nan
            grad = resultado.get("gradiente2d")
            if isinstance(grad, dict) and grad.get("status") == "OK":
                df_jg = grad.get("janelas")
                if isinstance(df_jg, pd.DataFrame) and not df_jg.empty and {"Início", "Fim"}.issubset(df_jg.columns):
                    jg = df_jg.copy()
                    jg["Início"] = pd.to_datetime(jg["Início"])
                    jg["Fim"] = pd.to_datetime(jg["Fim"])
                    jg["centro"] = jg["Início"] + (jg["Fim"] - jg["Início"]) / 2
                    jg["idade_lua_dias"] = calcular_idade_lunar(jg["centro"]).values
                    jg["Regime_lunar"] = jg["idade_lua_dias"].apply(classificar_regime_lunar)
                    jj = jg[jg["Regime_lunar"] == regime]
                    if not jj.empty:
                        inv_cols = [c for c in jj.columns if str(c).startswith("Inversão (")]
                        if inv_cols:
                            col_inv = inv_cols[0]
                            inv_pct = float(100.0 * (jj[col_inv].astype(str).str.upper() == "SIM").mean())

            pig_rows.append({
                "Poco": poco,
                "Regime": regime,
                "Amp_p2p_m": amp_p2p,
                "RMS_m": rms,
                "RSS_12h_m": rss12,
                "RSS_24h_m": rss24,
                "R_indice": r_med,
                "Lag_médio_h": lag_med,
                "Inversão_%": inv_pct,
            })

    df_pig = pd.DataFrame(pig_rows)

    resumo = {"status": "OK"}
    if not df_pig.empty:
        imq_vals = []
        for poco, g in df_pig.groupby("Poco"):
            a_s = g.loc[g["Regime"] == "SIZIGIA", "Amp_p2p_m"].mean()
            a_q = g.loc[g["Regime"] == "QUADRATURA", "Amp_p2p_m"].mean()
            if pd.notna(a_s) and pd.notna(a_q) and a_q > 0:
                imq_vals.append(a_s / a_q)
        resumo["IMQ_médio"] = float(np.nanmean(imq_vals)) if imq_vals else np.nan

        dom_semidiurna = []
        for poco, g in df_pig.groupby("Poco"):
            r12 = g["RSS_12h_m"].mean()
            r24 = g["RSS_24h_m"].mean()
            if pd.notna(r12) and pd.notna(r24):
                dom_semidiurna.append(float(r12 > r24))
        resumo["%_poços_semidiurna_dom"] = float(100.0 * np.mean(dom_semidiurna)) if dom_semidiurna else np.nan

    df_envelope = df_env_all.sort_values(["Poco", "dia"]).reset_index(drop=True)
    return df_pig, df_envelope, resumo


# =============================================================================
# CABEÇALHO PRINCIPAL
# =============================================================================

st.markdown("""
<div class="main-header">
    <h1 style="color: white; margin: 0;">🌊 GAC Tidal Insight</h1>
    <p style="color: rgba(255,255,255,0.9); margin: 5px 0 0 0;">
        Plataforma para Análise de Dinâmica Hidráulica de Aquíferos Costeiros
    </p>
</div>
""", unsafe_allow_html=True)

# Descrição do projeto
with st.expander("📖 Sobre o Projeto", expanded=False):
    st.markdown("""
    **GAC Tidal Insight** é uma plataforma analítica para avaliação integrada da dinâmica hidráulica de aquíferos rasos costeiros,
    incorporando:

    - 📊 **Método de Serfes (1991)** - Estimativa de nível médio
    - 📈 **Análise Espectral (FFT)** - Decomposição harmônica
    - 🌊 **Defasagem Maré-Poço (Lag)** - Análise de resposta hidráulica
    - 🧭 **Gradiente Hidráulico 2D** - Vetores de fluxo
    - 🌙 **Modulação Quinzenal** - Sizígia × Quadratura

    O sistema permite quantificar conectividade hidráulica, resposta do aquífero às variações de maré,
    estabilidade direcional e inversões do fluxo subterrâneo.
    """)

# Download de dados de exemplo para teste
with st.expander("📦 Baixar dados de exemplo para testar", expanded=False):
    st.markdown("""
    Cenário sintético de **4 poços em 30 dias** com modulação quinzenal alinhada ao
    calendário lunar real. Use estes arquivos para conhecer o app sem precisar de dados próprios.

    **Início recomendado para todos os poços:** `27/05/2025 00:00`
    """)

    exemplos_dir = Path(__file__).parent / "data" / "exemplos"

    def _download_se_existir(nome_arquivo: str, label: str, mime: str = "text/csv") -> None:
        caminho = exemplos_dir / nome_arquivo
        if caminho.exists():
            with open(caminho, "rb") as f:
                st.download_button(
                    label=label,
                    data=f.read(),
                    file_name=nome_arquivo,
                    mime=mime,
                    key=f"dl_{nome_arquivo}",
                    use_container_width=True,
                )

    col_ex1, col_ex2 = st.columns(2)

    with col_ex1:
        st.markdown("**Poços (formato Levellogger)**")
        for i in range(1, 5):
            _download_se_existir(f"PM-EX-0{i}.csv", f"⬇️ PM-EX-0{i}.csv")

    with col_ex2:
        st.markdown("**Maré e gradiente 2D**")
        _download_se_existir("mare_tabua.csv", "⬇️ Tábua de maré")
        _download_se_existir(
            "cotas.xlsx", "⬇️ Cotas (XLSX)",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        _download_se_existir(
            "pocos.kmz", "⬇️ Coordenadas (KMZ)",
            mime="application/vnd.google-earth.kmz",
        )

    st.caption(
        "Os dados são fictícios mas reproduzem dinâmica realista de aquífero costeiro. "
        "Detalhes técnicos do cenário em `data/exemplos/README.md` no repositório."
    )

st.divider()

# =============================================================================
# EXECUÇÃO DA ANÁLISE
# =============================================================================

def rodar_analise() -> Dict[str, Any]:
    """Executa análise com os arquivos carregados."""
    with tempfile.TemporaryDirectory(prefix="agente_gac_") as tmp:
        p = Path(tmp)
        caminhos_pocos = [salvar_upload_tmp(f, p) for f in pocos_files]
        caminho_mare = salvar_upload_tmp(mare_file, p) if mare_file else None
        caminho_cotas = salvar_upload_tmp(cotas_file, p) if cotas_file else None
        caminho_kmz = salvar_upload_tmp(kmz_file, p) if kmz_file else None

        return executar_analise_b(
            caminhos_pocos=caminhos_pocos,
            inicios_por_poco=inicios_por_poco,
            caminho_mare=caminho_mare,
            caminho_cotas=caminho_cotas,
            caminho_kmz=caminho_kmz,
        )

col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
with col_btn1:
    run_button = st.button("🚀 Executar Análise", type="primary", use_container_width=True)

with col_btn2:
    st.caption("ou")

with col_btn3:
    clear_button = st.button("🗑️ Limpar", use_container_width=True)

if run_button:
    if not pocos_files:
        st.error("⚠️ Envie pelo menos um arquivo de poço.")
        st.stop()

    with st.spinner("⏳ Processando dados..."):
        try:
            st.session_state["resultado"] = rodar_analise()
            st.success("✅ Análise concluída com sucesso!")
        except Exception as e:
            st.error("❌ Erro durante a execução")
            st.exception(e)
            st.stop()

if clear_button:
    if "resultado" in st.session_state:
        del st.session_state["resultado"]
    st.rerun()

resultado = st.session_state.get("resultado")

# =============================================================================
# RESULTADOS
# =============================================================================

if isinstance(resultado, dict):
    df_pig, df_envelope, pig_resumo = analise_pig_quinzenal(resultado)
    meta = resultado.get("meta", {})

    # Dashboard de Métricas
    st.markdown("### 📊 Dashboard de Métricas")

    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)

    with metrics_col1:
        st.metric(
            "Poços Processados",
            meta.get('processados', 0),
            meta.get('pulados', 0),
            delta_color="inverse"
        )

    with metrics_col2:
        status_mare = "✅ Com maré" if meta.get('tem_mare') else "❌ Sem maré"
        st.metric("Dados de Maré", status_mare)

    with metrics_col3:
        status_grad = "✅ Calculado" if meta.get('gradiente2d_calculado') else "—"
        st.metric("Gradiente 2D", status_grad)

    with metrics_col4:
        status_pig = pig_resumo.get('status', 'N/A')
        st.metric("Análise PIG", status_pig)

    st.divider()

    # Tabs de resultados
    tab_resumo, tab_pocos, tab_grad, tab_ia, tab_dl = st.tabs(
        ["📌 Resumo", "📄 Poços", "🧭 Gradiente 2D", "🧠 Interpretação", "⬇️ Download"]
    )

    # Tab Resumo
    with tab_resumo:
        st.subheader("QAQC – Qualidade do Processamento")

        df_qaqc = pd.DataFrame([
            {"Indicador": "Poços processados", "Status": meta.get("processados", 0), "Tipo": "success"},
            {"Indicador": "Poços pulados", "Status": meta.get("pulados", 0), "Tipo": "warning" if meta.get('pulados', 0) > 0 else "success"},
            {"Indicador": "Maré fornecida", "Status": "Sim" if meta.get("tem_mare") else "Não", "Tipo": "success" if meta.get("tem_mare") else "warning"},
            {"Indicador": "Insumos gradiente 2D", "Status": "Sim" if meta.get("tem_insumos_gradiente") else "Não", "Tipo": "success" if meta.get("tem_insumos_gradiente") else "warning"},
            {"Indicador": "PIG Quinzenal", "Status": "Ativo" if pig_resumo.get("status") == "OK" else "Inativo", "Tipo": "success" if pig_resumo.get("status") == "OK" else "warning"},
        ])

        st.table(df_qaqc.style.apply(
            lambda x: ['background-color: #E8F5E9' if v == 'success' else ('background-color: #FFF3E0' if v == 'warning' else '') for v in x],
            axis=1, subset=['Tipo']
        ).hide(['Tipo'], axis=1))

        st.divider()

        df_con = resultado.get("consolidado")
        if isinstance(df_con, pd.DataFrame) and not df_con.empty:
            col_poco = "Poço" if "Poço" in df_con.columns else ("Poco" if "Poco" in df_con.columns else "Poco")

            st.subheader("Tabelas Consolidadas")

            # Filtros
            filter_col1, filter_col2 = st.columns(2)
            with filter_col1:
                pocos_disponiveis = df_con[col_poco].unique().tolist()
                poco_selecionado = st.multiselect(
                    "Filtrar por poço",
                    pocos_disponiveis,
                    default=pocos_disponiveis[:1] if pocos_disponiveis else [],
                    help="Selecione um ou mais poços"
                )

            df_filtered = df_con[df_con[col_poco].isin(poco_selecionado)] if poco_selecionado else df_con

            # Abas para diferentes métricas
            table_tabs = st.tabs(["Hidráulica", "Espectral", "Maré/Lag", "Completo"])

            with table_tabs[0]:
                cols_hidraulica = [col_poco, "Janela", "Nível médio (Serfes) (m)", "Nível mínimo (m)", "Nível máximo (m)", "Amplitude pico-a-pico (m)", "Amplitude RMS (m)"]
                cols_hidraulica = [c for c in cols_hidraulica if c in df_filtered.columns]
                st.dataframe(df_filtered[cols_hidraulica], use_container_width=True)

            with table_tabs[1]:
                cols_espectral = [col_poco, "Janela", "Amp semidiurna (RSS, ~12h) (m)", "Amp diurna (RSS, ~24h) (m)"]
                cols_espectral = [c for c in cols_espectral if c in df_filtered.columns]
                st.dataframe(df_filtered[cols_espectral], use_container_width=True)

            with table_tabs[2]:
                cols_lag = [col_poco, "Janela", "Lag médio total (h)", "Fase da lua", "Idade da lua (dias)"]
                cols_lag = [c for c in cols_lag if c in df_filtered.columns]
                st.dataframe(df_filtered[cols_lag], use_container_width=True)

            with table_tabs[3]:
                st.dataframe(df_filtered, use_container_width=True)

    # Tab Poços
    with tab_pocos:
        porpoco = resultado.get("porpoco", {})
        series = resultado.get("series_horarias", {})
        df_mare = resultado.get("mare_eventos")

        if not porpoco:
            st.info("Nenhum poço processado.")
        else:
            poco_sel = st.selectbox("Selecione o poço", list(porpoco.keys()), key="poco_selector")
            df_janelas = porpoco[poco_sel]

            st.dataframe(df_janelas, use_container_width=True)

            df_h = series[poco_sel].copy()
            col_h = _pick_col(df_h, ("hora", "datetime", "datahora"))
            col_n = _pick_col(df_h, ("Nivel", "Nível", "nivel", "level"))
            df_h = _ensure_dt(df_h, col_h)

            dm = None
            col_md = None
            col_m = None
            if isinstance(df_mare, pd.DataFrame) and not df_mare.empty:
                dm = df_mare.copy()
                col_md = _pick_col(dm, ("datetime", "hora", "datahora"))
                col_m = _pick_col(dm, ("maré", "mare", "altura"))
                dm = _ensure_dt(dm, col_md)

            for i, row in df_janelas.reset_index(drop=True).iterrows():
                ini = row["Início"]
                fim = row["Fim"]
                serfes = row["Nível médio (Serfes) (m)"]
                df_win = df_h[(df_h[col_h] >= ini) & (df_h[col_h] <= fim)]
                df_mare_win = None
                if dm is not None:
                    df_mare_win = dm[(dm[col_md] >= ini) & (dm[col_md] <= fim)]

                with st.expander(f"📈 Janela {i+1} – {ini:%d/%m/%Y %H:%M} a {fim:%d/%m/%Y %H:%M}", expanded=True):
                    st.markdown(f"**Parâmetros:** Serfes = {serfes:.3f} m | Amplitude = {row.get('Amplitude pico-a-pico (m)', 'N/A')} m")

                    col_chart1, col_chart2 = st.columns(2)

                    with col_chart1:
                        fig_ts = plot_poco_nivel_serfes_mare(
                            df_win, col_h, col_n, serfes, df_mare_win, col_md, col_m,
                            f"Poço {poco_sel} – Nível x Serfes x Maré"
                        )
                        st.pyplot(fig_ts, use_container_width=True)

                    with col_chart2:
                        fig_v = plot_violin_box(
                            df_win,
                            col_n,
                            f"Distribuição – Poço {poco_sel} (janela {i+1})"
                        )
                        st.pyplot(fig_v, use_container_width=True)

                    with st.expander("📖 Metodologia"):
                        st.markdown("""
                        **Método de Serfes (1991):** O nível médio é estimado com janela efetiva de 71h
                        e passo de 24h, atenuando componentes diurna (~24h) e semidiurna (~12h),
                        preservando a tendência hidráulica.

                        **Distribuição Violin+Box:** A densidade de ocorrência, mediana, quartis e extremos
                        são representados. Distribuições estreitas indicam regime mais estável.
                        """)

    # Tab Gradiente 2D
    with tab_grad:
        grad = resultado.get("gradiente2d")
        if not isinstance(grad, dict) or grad.get("status") != "OK":
            st.info("📋 Gradiente hidráulico 2D não calculado. Forneça arquivos de cotas e KMZ para calcular.")
        else:
            st.success("✅ Gradiente hidráulico 2D calculado com sucesso")

            df_jg = grad.get("janelas")
            df_v = grad.get("vetores_horarios")
            df_xy = grad.get("pocos_xy")
            ang = None

            if isinstance(df_jg, pd.DataFrame) and not df_jg.empty:
                st.subheader("📊 Resumo por Janelas")
                st.dataframe(df_jg, use_container_width=True)
                ang = float(df_jg.iloc[0]["Ângulo dominante (°)"])
                mod = float(df_jg.iloc[0]["Módulo médio gradiente (m/m)"])

                st.markdown(f"""
                <div class="metric-card">
                    <h4>Direção Dominante</h4>
                    <h2 style="color: #1E88E5; margin: 0;">{ang:.1f}° ({_cardinal_16(ang)})</h2>
                </div>
                <div class="metric-card" style="margin-top: 10px;">
                    <h4>Módulo Médio</h4>
                    <h2 style="color: #43A047; margin: 0;">{mod:.4f} m/m</h2>
                </div>
                """, unsafe_allow_html=True)

            if isinstance(df_xy, pd.DataFrame) and not df_xy.empty and ang is not None:
                st.markdown("### 🗺️ Mapa de Gradiente")
                fig_map = plot_gradiente_mapa(
                    df_xy,
                    ang,
                    f"Gradiente Hidráulico Dominante – {ang:.1f}°"
                )
                st.pyplot(fig_map, use_container_width=True)

            with st.expander("📖 Metodologia – Gradiente Hidráulico 2D"):
                st.markdown("""
                O gradiente hidráulico 2D é calculado pelo ajuste de plano h(x,y) = a + bx + cy
                por mínimos quadrados. O vetor de fluxo é o gradiente negativo da superfície potenciométrica.

                **Parâmetros:**
                - Janela: 71h
                - Passo: 24h
                - Limiar angular: 160°
                - Mínimo de vetores: 60

                A direção dominante indica o sentido preferencial de escoamento; oscilações podem
                refletir influências de maré ou bombeamentos.
                """)

            if isinstance(df_v, pd.DataFrame) and not df_v.empty:
                st.markdown("### 📈 Séries Temporais do Gradiente")
                fig_a, fig_m = plot_gradiente_series(df_v)
                st.pyplot(fig_a, use_container_width=True)
                st.pyplot(fig_m, use_container_width=True)

    # Tab Interpretação
    with tab_ia:
        if mostrar_ia:
            try:
                st.markdown(interpretar_resultados_avancado(resultado, df_pig if not df_pig.empty else None))
            except Exception as e:
                st.error("Erro na interpretação avançada.")
                st.exception(e)

            if pig_resumo.get("status") == "OK" and mostrar_pig:
                st.divider()
                st.subheader("🌊 Análise PIG Quinzenal")

                st.markdown("""
                <div class="info-box">
                O <strong>Índice de Modulação Quinzenal (IMQ)</strong> representa a razão entre a amplitude
                durante sizígias (lua nova/cheia) e quadraturas (quartos). Valores > 1 indicam
                maior influência das fases lunares sobre o aquífero.
                </div>
                """, unsafe_allow_html=True)

                if not df_envelope.empty:
                    pocos_env = ["(média)"] + sorted(df_envelope["Poco"].unique().tolist())
                    poco_env = st.selectbox("Poço para envelope", pocos_env)
                    poco_sel_env = None if poco_env == "(média)" else poco_env

                    fig_env = plot_envelope_quinzenal(df_envelope, poco_sel_env)
                    if fig_env is not None:
                        st.pyplot(fig_env, use_container_width=True)

    # Tab Download
    with tab_dl:
        st.subheader("📥 Exportar Resultados")

        tabelas: Dict[str, pd.DataFrame] = {}
        df_con = resultado.get("consolidado")
        if isinstance(df_con, pd.DataFrame) and not df_con.empty:
            tabelas["Consolidado"] = df_con

        porpoco = resultado.get("porpoco", {})
        for poco, df in porpoco.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                tabelas[f"Poco_{poco}"] = df

        grad = resultado.get("gradiente2d")
        if isinstance(grad, dict) and grad.get("status") == "OK":
            df_v = grad.get("vetores_horarios")
            df_jg = grad.get("janelas")
            if isinstance(df_v, pd.DataFrame) and not df_v.empty:
                tabelas["Gradiente_vetores"] = df_v
            if isinstance(df_jg, pd.DataFrame) and not df_jg.empty:
                tabelas["Gradiente_janelas"] = df_jg

        if pig_resumo.get("status") == "OK":
            if isinstance(df_pig, pd.DataFrame) and not df_pig.empty:
                tabelas["PIG_tabela"] = df_pig
            if isinstance(df_envelope, pd.DataFrame) and not df_envelope.empty:
                tabelas["PIG_envelope"] = df_envelope

        if tabelas:
            st.markdown(f"**{len(tabelas)} tabela(s) disponível(s) para download:**")

            select_tabela = st.selectbox("Selecione a tabela", list(tabelas.keys()))
            df_download = tabelas[select_tabela]
            st.dataframe(df_download.head(10), use_container_width=True)
            st.caption(f"Mostrando 10 de {len(df_download)} linhas")

            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                bio = BytesIO()
                with pd.ExcelWriter(bio, engine="openpyxl") as writer:
                    df_download.to_excel(writer, index=False, sheet_name=select_tabela[:31])
                st.download_button(
                    f"⬇️ Baixar {select_tabela} (Excel)",
                    data=bio.getvalue(),
                    file_name=f"gac_{select_tabela}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

            with col_dl2:
                csv = df_download.to_csv(index=False).encode('utf-8')
                st.download_button(
                    f"📄 Baixar {select_tabela} (CSV)",
                    data=csv,
                    file_name=f"gac_{select_tabela}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        else:
            st.info("Nenhuma tabela disponível para download.")

else:
    # Estado inicial - esperando upload
    st.markdown("""
    <div style="text-align: center; padding: 60px 20px; background: #f8f9fa; border-radius: 15px; margin: 20px 0;">
        <h2 style="color: #666;">📤 Aguardando Upload de Dados</h2>
        <p style="color: #999; font-size: 16px;">
            Use o menu lateral para carregar os arquivos de poços e iniciar a análise.
        </p>
        <div style="margin-top: 30px;">
            <span style="display: inline-block; padding: 10px 20px; background: white; border-radius: 20px; margin: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                🌊 Dados de Maré
            </span>
            <span style="display: inline-block; padding: 10px 20px; background: white; border-radius: 20px; margin: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                📊 Poços
            </span>
            <span style="display: inline-block; padding: 10px 20px; background: white; border-radius: 20px; margin: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                🧭 Gradiente 2D
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
---
<div style="text-align: center; color: #999; font-size: 12px; padding: 20px;">
    <p>🌊 GAC Tidal Insight v2.0 | Desenvolvido por André Souza | Especialistas em GAC</p>
    <p>Baseado no Método de Serfes (1991) para estimativa de nível médio hidráulico</p>
</div>
""", unsafe_allow_html=True)