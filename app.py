
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

from src.core.orquestrador_b import executar_analise_b

# =============================================================================
# Configuração da página
# =============================================================================
st.set_page_config(page_title="Agente GAC", layout="wide")
st.title("🛠️ Agente GAC – Serfes + Amplitudes + FFT + Maré/Lag + Gradiente 2D + PIG (sizígia×quadratura)")

# 🎛️ Tamanhos dos gráficos (mantidos)
FIG_TS = (6.0, 2.0)        # Série temporal
FIG_VIOLIN = (4.6, 3.4)    # Violino/box
FIG_MAP = (3.2, 3.2)       # Mapa
FIG_GRAD_SER = (6.0, 1.7)  # Séries gradiente
FIG_ENV = (6.4, 2.2)       # Envelope quinzenal (novo)

# =============================================================================
# Utilidades
# =============================================================================

def salvar_upload_tmp(uploaded_file, pasta: Path) -> Optional[str]:
    if uploaded_file is None:
        return None
    nome = Path(uploaded_file.name).name
    nome = re.sub(r'[\\/:*?"<>\n\r\t]', "_", nome)
    caminho = pasta / nome
    with open(caminho, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(caminho)


def base_nome(uploaded_file) -> str:
    return Path(uploaded_file.name).stem.strip()


def _cardinal_16(angle_deg: float) -> str:
    dirs = [
        "E","ENE","NE","NNE","N","NNW","NW","WNW",
        "W","WSW","SW","SSW","S","SSE","SE","ESE"
    ]
    a = angle_deg % 360
    return dirs[int((a + 11.25) // 22.5) % 16]


def _pick_col(df: pd.DataFrame, candidatos: Tuple[str, ...]) -> str:
    for c in candidatos:
        if c in df.columns:
            return c
    raise KeyError(f"Nenhuma coluna encontrada entre {candidatos}")


def _ensure_dt(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.to_datetime(df[col])
    return df

# =============================================================================
# Gráficos básicos já existentes
# =============================================================================

def plot_poco_nivel_serfes_mare(
    df, col_hora, col_nivel, serfes,
    df_mare, col_mare_dt, col_mare, titulo
):
    fig, ax1 = plt.subplots(figsize=FIG_TS)
    fig.set_dpi(100)
    ax1.plot(df[col_hora], df[col_nivel], linewidth=0.9, label="Nível do poço")
    ax1.axhline(serfes, linestyle="--", linewidth=0.9, label="Serfes")
    ax1.set_ylabel("Nível (m)", fontsize=7)
    ax1.tick_params(axis="both", labelsize=6)
    ax1.grid(True, alpha=0.25)

    h, l = ax1.get_legend_handles_labels()
    if df_mare is not None and not df_mare.empty and col_mare_dt and col_mare:
        ax2 = ax1.twinx()
        ax2.plot(
            df_mare[col_mare_dt],
            df_mare[col_mare],
            marker="o",
            markersize=2.5,
            linewidth=0.8,
            color="green",
            label="Maré"
        )
        ax2.set_ylabel("Altura da maré (m)", fontsize=7)
        ax2.tick_params(axis="y", labelsize=6)
        h2, l2 = ax2.get_legend_handles_labels()
        h += h2
        l += l2
    ax1.legend(h, l, loc="upper right", fontsize=6)
    ax1.set_title(titulo, fontsize=8)
    fig.tight_layout()
    return fig


def plot_violin_box(df, col_nivel, titulo):
    y = df[col_nivel].dropna().values
    fig, ax = plt.subplots(figsize=FIG_VIOLIN)
    fig.set_dpi(100)
    ax.violinplot(y, showextrema=False)
    ax.boxplot(y, widths=0.25)
    ax.set_title(titulo, fontsize=11)
    ax.set_ylabel("Nível (m)", fontsize=10)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_xticks([])
    ax.set_xlabel("")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_gradiente_mapa(df_xy: pd.DataFrame, angulo: float, titulo: str):
    fig, ax = plt.subplots(figsize=FIG_MAP)
    fig.set_dpi(100)
    ax.scatter(df_xy["X_m"], df_xy["Y_m"], s=20)
    for _, r in df_xy.iterrows():
        ax.text(r["X_m"], r["Y_m"], r["Poco"], fontsize=6)
    x0, y0 = df_xy["X_m"].mean(), df_xy["Y_m"].mean()
    amp_x = float(df_xy["X_m"].max() - df_xy["X_m"].min())
    amp_y = float(df_xy["Y_m"].max() - df_xy["Y_m"].min())
    L = max(amp_x, amp_y) * 0.20
    if L == 0:
        L = 1.0
    dx = np.cos(np.deg2rad(angulo)) * L
    dy = np.sin(np.deg2rad(angulo)) * L
    ax.arrow(x0, y0, dx, dy, head_width=L * 0.08)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(titulo, fontsize=8)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_gradiente_series(df_v: pd.DataFrame):
    fig1, ax1 = plt.subplots(figsize=FIG_GRAD_SER)
    fig1.set_dpi(100)
    ax1.plot(df_v["hora"], df_v["angulo_fluxo_deg"], linewidth=0.9)
    ax1.set_title("Ângulo do fluxo (°)", fontsize=8)
    ax1.tick_params(labelsize=6)
    ax1.grid(True, alpha=0.25)
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(figsize=FIG_GRAD_SER)
    fig2.set_dpi(100)
    ax2.plot(df_v["hora"], df_v["modulo_gradiente"], linewidth=0.9)
    ax2.set_title("Módulo do gradiente (m/m)", fontsize=8)
    ax2.tick_params(labelsize=6)
    ax2.grid(True, alpha=0.25)
    fig2.tight_layout()
    return fig1, fig2

# =============================================================================
# IA – Interpretação técnica automática (básica original – mantida como fallback)
# =============================================================================

def interpretar_resultados(resultado: dict) -> str:
    """Fallback simples mantido por compatibilidade."""
    linhas = []
    meta = resultado.get("meta", {})
    df_con = resultado.get("consolidado")
    grad = resultado.get("gradiente2d")
    linhas.append("## 🧠 Interpretação técnica automática")
    linhas.append("")
    linhas.append("### 1) Escopo e qualidade do processamento")
    linhas.append(
        f"- Poços processados: **{meta.get('processados', 0)}**\n"
        f"- Poços descartados: **{meta.get('pulados', 0)}**"
    )
    if meta.get("tem_mare"):
        linhas.append("- Dados de maré utilizados.")
    else:
        linhas.append("- Sem dados de maré.")

    if isinstance(df_con, pd.DataFrame) and not df_con.empty:
        linhas.append("")
        linhas.append("### 2) Assinatura hidráulica")
        if "Amplitude pico-a-pico (m)" in df_con.columns:
            amp_med = float(df_con["Amplitude pico-a-pico (m)"].median())
            linhas.append(f"- Amplitude mediana: **{amp_med:.3f} m**.")

    if isinstance(grad, dict) and grad.get("status") == "OK":
        df_jg = grad.get("janelas")
        if isinstance(df_jg, pd.DataFrame) and not df_jg.empty:
            ang = float(df_jg.iloc[0]["Ângulo dominante (°)"])
            mod = float(df_jg.iloc[0]["Módulo médio gradiente (m/m)"])
            linhas.append("")
            linhas.append("### 3) Gradiente hidráulico")
            linhas.append(f"- Direção dominante: **{ang:.1f}° ({_cardinal_16(ang)})**")
            linhas.append(f"- Módulo médio: **{mod:.4f} m/m**")
            linhas.append("")
            linhas.append("### 4) Observação técnica")
            linhas.append(
                "Recomenda-se avaliar defasagem maré–poço e estabilidade angular."
            )
    return "\n".join(linhas)

# =============================================================================
# 🔬 NOVO: PIG (sizígia × quadratura) + interpretação avançada (sem IA externa)
# =============================================================================

SINODICO_DIAS = 29.53058867  # período sinódico médio
LUA_REF = pd.Timestamp("2000-01-06 18:14:00", tz=None)


def calcular_idade_lunar(idx: pd.DatetimeIndex | pd.Series) -> pd.Series:
    """
    Retorna idade da Lua (dias) para cada timestamp (0–29,53).
    Aceita Series, DatetimeIndex ou escalar; com/sem timezone.
    """
    # Converte para datetime, preservando NaT onde houver
    dt = pd.to_datetime(idx, errors="coerce")

    # Normaliza para DatetimeIndex "naive" (sem tz)
    if isinstance(dt, pd.Series):
        # Se for Series com tz, remove tz; se for naive, segue
        try:
            # pandas com dtype datetime64[ns, tz]
            if pd.api.types.is_datetime64tz_dtype(dt):
                dt = dt.dt.tz_convert(None)
        except Exception:
            # fallback caso a conversão falhe
            try:
                dt = dt.dt.tz_localize(None)
            except Exception:
                pass
        dt_idx = pd.DatetimeIndex(dt)
    elif isinstance(dt, pd.DatetimeIndex):
        # Remove tz se existir
        dt_idx = dt.tz_convert(None) if dt.tz is not None else dt
    else:
        # Escalar -> DatetimeIndex de 1 elemento
        ts = pd.Timestamp(dt)
        if getattr(ts, "tzinfo", None) is not None:
            try:
                ts = ts.tz_convert(None)
            except Exception:
                ts = ts.tz_localize(None)
        dt_idx = pd.DatetimeIndex([ts])

    # Diferença em dias (float)
    delta = dt_idx - LUA_REF
    dias = delta.total_seconds() / 86400.0

    # Idade (módulo do período sinódico)
    idade = np.mod(dias, SINODICO_DIAS)

    # Retorna Series indexada pelo próprio tempo
    return pd.Series(idade, index=dt_idx)


def classificar_regime_lunar(idade_dias: float) -> str:
    """Classifica regime: SIZIGIA / QUADRATURA / INTERMEDIARIO."""
    # Faixas com tolerância ±2 dias
    if (0 <= idade_dias <= 2) or (SINODICO_DIAS - 2 <= idade_dias <= SINODICO_DIAS) or (12.8 <= idade_dias <= 16.8):
        return "SIZIGIA"
    if (5.4 <= idade_dias <= 9.4) or (20.0 <= idade_dias <= 24.0):
        return "QUADRATURA"
    return "INTERMEDIARIO"


def _amp_diaria(df: pd.DataFrame, col_tempo: str, col_val: str) -> pd.DataFrame:
    """Amplitude diária (max-min) agrupada por dia."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["dia", "amp"])
    d = df.copy()
    d[col_tempo] = pd.to_datetime(d[col_tempo])
    d["dia"] = d[col_tempo].dt.floor("D")
    g = d.groupby("dia")[col_val]
    out = (g.max() - g.min()).rename("amp").reset_index()
    return out


def _duracao_total(resultado: dict) -> Tuple[pd.Timestamp | None, pd.Timestamp | None, float]:
    """Retorna (inicio, fim, duracao_dias) a partir das séries horárias dos poços."""
    series = resultado.get("series_horarias", {})
    tmins, tmaxs = [], []
    for df in series.values():
        if isinstance(df, pd.DataFrame) and not df.empty:
            col_h = _pick_col(df, ("hora", "datetime", "datahora"))
            ts = pd.to_datetime(df[col_h])
            tmins.append(ts.min())
            tmaxs.append(ts.max())
    if not tmins:
        return None, None, 0.0
    ini = min(tmins)
    fim = max(tmaxs)
    dur = (fim - ini).total_seconds() / 86400.0
    return ini, fim, float(dur)


def analise_pig_quinzenal(resultado: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Gera (df_pig, df_envelope, resumo) ou estruturas vazias quando não aplicável.
    - Requisitos: duração ≥ 20 dias e presença de maré.
    """
    df_mare = resultado.get("mare_eventos")
    if not isinstance(df_mare, pd.DataFrame) or df_mare.empty:
        return pd.DataFrame(), pd.DataFrame(), {"status": "SKIP", "motivo": "Sem maré"}

    ini, fim, dur_d = _duracao_total(resultado)
    if dur_d < 20:
        return pd.DataFrame(), pd.DataFrame(), {"status": "SKIP", "motivo": "Período < 20 dias"}

    # Amplitude diária da maré
    col_md = _pick_col(df_mare, ("datetime", "hora", "datahora"))
    col_m = _pick_col(df_mare, ("maré", "mare", "altura"))
    df_mare_d = _amp_diaria(_ensure_dt(df_mare, col_md), col_md, col_m)

    # Envelope por poço (amp diária do nível)
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

    # Junta maré diária
    df_env_all = df_env_all.merge(df_mare_d.rename(columns={"amp": "amp_mare"}), on="dia", how="left")
    df_env_all.rename(columns={"amp": "amp_poco"}, inplace=True)

    # Idade lunar e regime por dia
    idade = calcular_idade_lunar(df_env_all["dia"])  # série indexada por dia
    df_env_all["idade_lua_dias"] = idade.values
    df_env_all["Regime_lunar"] = df_env_all["idade_lua_dias"].apply(classificar_regime_lunar)

    # Tabela PIG por poço e regime (médias)
    # Para métricas espectrais/lag, usamos janelas do consolidado e classificamos por idade lunar no centro da janela
    df_con = resultado.get("consolidado")
    pig_rows = []
    if isinstance(df_con, pd.DataFrame) and not df_con.empty:
        dfc = df_con.copy()
        # Colunas esperadas
        col_poco = "Poço" if "Poço" in dfc.columns else ("Poco" if "Poco" in dfc.columns else None)
        if not col_poco:
            col_poco = "Poco"
            dfc["Poco"] = dfc.get("Poço", "?")
        # Centro da janela para classificação lunar
        if "Início" in dfc.columns and "Fim" in dfc.columns:
            dfc["Início"] = pd.to_datetime(dfc["Início"])
            dfc["Fim"] = pd.to_datetime(dfc["Fim"])
            dfc["centro"] = dfc["Início"] + (dfc["Fim"] - dfc["Início"]) / 2
            dfc["idade_lua_dias"] = calcular_idade_lunar(dfc["centro"]).values
            dfc["Regime_lunar"] = dfc["idade_lua_dias"].apply(classificar_regime_lunar)
        else:
            # fallback: usar dia inteiro
            dfc["centro"] = pd.NaT
            dfc["idade_lua_dias"] = np.nan
            dfc["Regime_lunar"] = "INTERMEDIARIO"

        # Nomes das colunas de interesse
        col_amp_p2p = "Amplitude pico-a-pico (m)"
        col_rms = "Amplitude RMS (m)"
        col_rss12 = "Amp semidiurna (RSS, ~12h) (m)"
        col_rss24 = "Amp diurna (RSS, ~24h) (m)"
        col_lag = "Lag médio total (h)"

        # Para R (amp_poço/amp_maré) calculamos maré na janela
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
            # R médio por janela (amp_poço / amp_maré)
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

            # % de inversão por regime (se gradiente existir)
            inv_pct = np.nan
            grad = resultado.get("gradiente2d")
            if isinstance(grad, dict) and grad.get("status") == "OK":
                df_jg = grad.get("janelas")
                if isinstance(df_jg, pd.DataFrame) and not df_jg.empty and {"Início", "Fim"}.issubset(df_jg.columns):
                    # Classificar janelas do gradiente por regime lunar (centro)
                    jg = df_jg.copy()
                    jg["Início"] = pd.to_datetime(jg["Início"])
                    jg["Fim"] = pd.to_datetime(jg["Fim"])
                    jg["centro"] = jg["Início"] + (jg["Fim"] - jg["Início"]) / 2
                    jg["idade_lua_dias"] = calcular_idade_lunar(jg["centro"]).values
                    jg["Regime_lunar"] = jg["idade_lua_dias"].apply(classificar_regime_lunar)
                    jj = jg[jg["Regime_lunar"] == regime]
                    if not jj.empty:
                        # considerar evento marcado na coluna que contém 'Inversão (' ...
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

    # Resumo agregado
    resumo = {"status": "OK"}
    if not df_pig.empty:
        # IMQ por poço (amp sizígia / quadratura)
        imq_vals = []
        for poco, g in df_pig.groupby("Poco"):
            a_s = g.loc[g["Regime"] == "SIZIGIA", "Amp_p2p_m"].mean()
            a_q = g.loc[g["Regime"] == "QUADRATURA", "Amp_p2p_m"].mean()
            if pd.notna(a_s) and pd.notna(a_q) and a_q > 0:
                imq_vals.append(a_s / a_q)
        resumo["IMQ_médio"] = float(np.nanmean(imq_vals)) if imq_vals else np.nan

        # Dominância espectral (12h vs 24h) por poço – proporção de poços com RSS_12h>RSS_24h
        dom_semidiurna = []
        for poco, g in df_pig.groupby("Poco"):
            r12 = g["RSS_12h_m"].mean()
            r24 = g["RSS_24h_m"].mean()
            if pd.notna(r12) and pd.notna(r24):
                dom_semidiurna.append(float(r12 > r24))
        resumo["%_poços_semidiurna_dom"] = float(100.0 * np.mean(dom_semidiurna)) if dom_semidiurna else np.nan

    # df_envelope: por dia e poço, com maré e regime
    df_envelope = df_env_all.sort_values(["Poco", "dia"]).reset_index(drop=True)

    return df_pig, df_envelope, resumo


def plot_envelope_quinzenal(df_env: pd.DataFrame, poco: str | None = None):
    """Gráfico do envelope (amplitude diária do poço + maré) para TODO o período.
       Se 'poco' for fornecido, filtra; senão plota todos (média por dia).
    """
    if df_env is None or df_env.empty:
        return None

    if poco:
        d = df_env[df_env["Poco"] == poco].copy()
        titulo = f"Envelope (amplitude diária) – Poço {poco}"
    else:
        # média das amplitudes diárias por dia (sobre poços)
        d = df_env.groupby(["dia", "Regime_lunar"], as_index=False)[["amp_poco", "amp_mare"]].mean()
        d["Poco"] = "(média)"
        titulo = "Envelope (amplitude diária) – Média dos poços"

    if d.empty:
        return None

    # Cores por regime
    cores = {"SIZIGIA": "crimson", "QUADRATURA": "royalblue", "INTERMEDIARIO": "gray"}

    fig, ax1 = plt.subplots(figsize=FIG_ENV)
    fig.set_dpi(100)

    # Barras/linhas do poço
    for reg, g in d.groupby("Regime_lunar"):
        ax1.plot(g["dia"], g["amp_poco"], marker="o", linestyle="-", linewidth=0.9,
                 markersize=3, color=cores.get(reg, "gray"), label=f"Poço – {reg.title()}")

    ax1.set_ylabel("Amp. diária do poço (m)", fontsize=7)
    ax1.tick_params(axis="both", labelsize=6)
    ax1.grid(True, alpha=0.25)

    # Maré na direita
    ax2 = ax1.twinx()
    ax2.plot(d["dia"], d["amp_mare"], color="green", linewidth=1.0, label="Maré (amp diária)")
    ax2.set_ylabel("Amp. diária da maré (m)", fontsize=7)
    ax2.tick_params(axis="y", labelsize=6)

    # Legendas combinadas
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=6)

    ax1.set_title(titulo, fontsize=8)
    fig.tight_layout()
    return fig


def interpretar_resultados_avancado(resultado: dict, df_pig: Optional[pd.DataFrame] = None) -> str:
    """Gera texto técnico consolidado por poço e por regime, cruzando R, lag, FFT, IMQ e gradiente."""
    meta = resultado.get("meta", {})
    df_con = resultado.get("consolidado")
    grad = resultado.get("gradiente2d")

    linhas: List[str] = []
    linhas.append("## 🧠 Interpretação técnica automática (avançada)")
    linhas.append("")
    linhas.append("### 1) QAQC do processamento")
    linhas.append(
        f"- Poços processados: **{meta.get('processados', 0)}**  \n"
        f"- Poços descartados: **{meta.get('pulados', 0)}**  \n"
        f"- Maré: **{'sim' if meta.get('tem_mare') else 'não'}**  \n"
        f"- Insumos p/ gradiente 2D: **{'sim' if meta.get('tem_insumos_gradiente') else 'não'}**"
    )

    # ---------------- Por poço (varrendo consolidado) -----------------
    if isinstance(df_con, pd.DataFrame) and not df_con.empty:
        col_poco = "Poço" if "Poço" in df_con.columns else ("Poco" if "Poco" in df_con.columns else None)
        if not col_poco:
            col_poco = "Poco"
        linhas.append("")
        linhas.append("### 2) Diagnóstico por poço")
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
                lag_txt = "sem maré / não aplicável"
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

            linhas.append(f"**Poço {poco}**  ")
            linhas.append(
                f"- Amplitude pico-a-pico mediana: **{amp_med:.3f} m**; RMS mediana: **{rms_med:.3f} m**.  \n"
                f"- Dominância espectral: **{dom_txt}**.  \n"
                f"- Lag maré–poço: **{lag_txt}**.  \n"
                f"- Índice de resposta à maré: {r_txt}."
            )

    # ---------------- Gradiente regional -----------------
    if isinstance(grad, dict) and grad.get("status") == "OK":
        df_jg = grad.get("janelas")
        if isinstance(df_jg, pd.DataFrame) and not df_jg.empty:
            ang = float(df_jg.iloc[0]["Ângulo dominante (°)"])
            mod = float(df_jg.iloc[0]["Módulo médio gradiente (m/m)"])
            linhas.append("")
            linhas.append("\n### 3) Gradiente 2D (regional)")
            linhas.append(
                f"- Direção dominante: **{ang:.1f}° ({_cardinal_16(ang)})**; módulo médio: **{mod:.4f} m/m**."
            )
            # Inversão (contagem)
            inv_cols = [c for c in df_jg.columns if str(c).startswith("Inversão (")]
            if inv_cols:
                col_inv = inv_cols[0]
                inv_pct = float(100.0 * (df_jg[col_inv].astype(str).str.upper() == "SIM").mean())
                linhas.append(f"- Ocorrência de inversões nas janelas: **{inv_pct:.1f}%**.")

    # ---------------- PIG / Modulação quinzenal -----------------
    if df_pig is not None and not df_pig.empty:
        linhas.append("")
        linhas.append("\n### 4) Modulação quinzenal (sizígia × quadratura)")
        # IMQ
        imq_vals = []
        for poco, g in df_pig.groupby("Poco"):
            a_s = g.loc[g["Regime"] == "SIZIGIA", "Amp_p2p_m"].mean()
            a_q = g.loc[g["Regime"] == "QUADRATURA", "Amp_p2p_m"].mean()
            if pd.notna(a_s) and pd.notna(a_q) and a_q > 0:
                imq_vals.append(a_s / a_q)
        if imq_vals:
            linhas.append(f"- IMQ médio (Amp_sízigia/Amp_quadratura): **{np.nanmean(imq_vals):.2f}**.")
        # Quem domina: 12h ou 24h (por poço)
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
            linhas.append(f"- Dominância espectral por poço: **{frac_12:.0f}%** com banda ~12 h dominante.")

    linhas.append("")
    linhas.append("\n### 5) Conclusão técnica consolidada")
    linhas.append(
        "A resposta hidráulica mostra coerência com o forçamento periódico (maré) quando presente,"
        " com variação de lag e dominância espectral indicativas da assinatura local do aquífero."
    )

    return "\n".join(linhas)

# =============================================================================
# Upload dos dados
# =============================================================================
st.markdown("### 1) Upload dos arquivos dos poços (nível d’água)")
pocos_files = st.file_uploader(
    "Arquivos CSV dos poços",
    type=["csv"],
    accept_multiple_files=True
)

st.markdown("### 2) Data/hora de início por poço")
inicios_por_poco: Dict[str, pd.Timestamp] = {}
if pocos_files:
    for f in pocos_files:
        poco = base_nome(f)
        inicio_txt = st.text_input(
            f"Início do teste – {poco} (dd/mm/aaaa hh:mm)",
            value="01/01/2026 00:00",
            key=f"inicio_{poco}"
        )
        try:
            inicios_por_poco[poco] = pd.to_datetime(inicio_txt, dayfirst=True)
        except Exception:
            st.error(f"Data inválida para o poço {poco}")

st.markdown("### 3) Maré (opcional)")
mare_file = st.file_uploader("Arquivo de maré (CSV)", type=["csv"])

st.markdown("### 4) Gradiente hidráulico 2D (opcional)")
col1, col2 = st.columns(2)
with col1:
    cotas_file = st.file_uploader("Planilha de cotas (XLSX)", type=["xlsx"])
with col2:
    kmz_file = st.file_uploader("Arquivo KMZ (poços)", type=["kmz"])

st.divider()

# =============================================================================
# Execução
# =============================================================================

def rodar_analise() -> Dict[str, Any]:
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

if st.button("🚀 Rodar análise", type="primary"):
    if not pocos_files:
        st.error("Envie pelo menos um arquivo de poço.")
        st.stop()
    with st.spinner("Processando dados..."):
        try:
            st.session_state["resultado"] = rodar_analise()
        except Exception as e:
            st.error("Erro durante a execução")
            st.exception(e)
            st.stop()

resultado = st.session_state.get("resultado")

# =============================================================================
# Resultados – Abas
# =============================================================================
if isinstance(resultado, dict):
    # --- Executa PIG/Envelope antes de abrir as abas ---
    df_pig, df_envelope, pig_resumo = analise_pig_quinzenal(resultado)

    tab_resumo, tab_pocos, tab_grad, tab_ia, tab_dl = st.tabs(
        ["📌 Resumo", "📄 Poços", "🧭 Gradiente 2D", "🧠 Interpretação", "⬇️ Download"]
    )

    # ---------------------------------------------------------------------
    # Aba Resumo
    # ---------------------------------------------------------------------
    with tab_resumo:
        st.subheader("QAQC – Qualidade do Processamento")
        meta = resultado.get("meta", {})
        df_qaqc = pd.DataFrame([
            {"Indicador": "Poços processados", "Status": meta.get("processados", 0)},
            {"Indicador": "Poços pulados", "Status": meta.get("pulados", 0)},
            {"Indicador": "Maré fornecida", "Status": "Sim" if meta.get("tem_mare") else "Não"},
            {"Indicador": "Insumos para gradiente 2D", "Status": "Sim" if meta.get("tem_insumos_gradiente") else "Não"},
            {"Indicador": "PIG (sizígia×quadratura)", "Status": "Ativo" if pig_resumo.get("status") == "OK" else "Inativo"},
        ])
        st.table(df_qaqc)

        st.divider()
        df_con = resultado.get("consolidado")
        if isinstance(df_con, pd.DataFrame) and not df_con.empty:
            col_poco = "Poço" if "Poço" in df_con.columns else ("Poco" if "Poco" in df_con.columns else None)
            col_jan = "Janela" if "Janela" in df_con.columns else None
            chaves = [c for c in [col_poco, col_jan] if c]

            def _show_tbl(titulo: str, palavras: List[str]):
                cols = [c for c in df_con.columns if any(p.lower() in c.lower() for p in palavras)]
                cols = chaves + [c for c in cols if c not in chaves]
                cols = [c for c in cols if c in df_con.columns]
                if len(cols) > len(chaves):
                    st.markdown(f"**{titulo}**")
                    st.table(df_con[cols])

            _show_tbl(
                "Hidráulica (Serfes, mínimos, máximos, amplitudes)",
                ["serfes", "mín", "min", "máx", "max", "pico", "semiampl", "\nN-Nmedio\n", "rms"],
            )
            _show_tbl(
                "Espectral (FFT ~12h / ~24h)",
                ["semidiurna", "~12", "diurna", "~24", "fft", "rss"],
            )
            _show_tbl(
                "Maré e defasagem (lag)",
                ["mare", "maré", "lag", "fase", "lua", "idade"],
            )
            with st.expander("Ver consolidado completo"):
                st.dataframe(df_con, use_container_width=True)
        else:
            st.info("Tabela consolidada vazia.")

        # Resumo PIG no painel
        if pig_resumo.get("status") == "OK":
            c1, c2 = st.columns(2)
            with c1:
                st.metric("IMQ médio (Amp_sízigia/Amp_quadratura)", f"{pig_resumo.get('IMQ_médio', np.nan):.2f}")
            with c2:
                st.metric("Poços com dominância ~12 h (%)", f"{pig_resumo.get('%_poços_semidiurna_dom', np.nan):.0f}%")
            st.markdown("**Top 5 poços por IMQ**")
            if not df_pig.empty:
                # calcula IMQ por poço
                imq_by_poco = []
                for poco, g in df_pig.groupby("Poco"):
                    a_s = g.loc[g["Regime"] == "SIZIGIA", "Amp_p2p_m"].mean()
                    a_q = g.loc[g["Regime"] == "QUADRATURA", "Amp_p2p_m"].mean()
                    if pd.notna(a_s) and pd.notna(a_q) and a_q > 0:
                        imq_by_poco.append({"Poco": poco, "IMQ": a_s / a_q})
                if imq_by_poco:
                    df_imq = pd.DataFrame(imq_by_poco).sort_values("IMQ", ascending=False).head(5)
                    st.table(df_imq)

    # ---------------------------------------------------------------------
    # Aba Poços (mantida)
    # ---------------------------------------------------------------------
    with tab_pocos:
        porpoco = resultado.get("porpoco", {})
        series = resultado.get("series_horarias", {})
        df_mare = resultado.get("mare_eventos")
        if not porpoco:
            st.info("Nenhum poço processado.")
        else:
            poco_sel = st.selectbox("Selecione o poço", list(porpoco.keys()))
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

                st.markdown(f"#### Janela {i+1} – {ini:%d/%m/%Y %H:%M} a {fim:%d/%m/%Y %H:%M}")
                fig_ts = plot_poco_nivel_serfes_mare(
                    df_win, col_h, col_n, serfes, df_mare_win, col_md, col_m,
                    f"Poço {poco_sel} – Nível x Serfes x Maré"
                )
                st.pyplot(fig_ts, use_container_width=False)

                with st.expander("📘 Metodologia – Série temporal e nível médio (Serfes)"):
                    st.markdown(
                        """
                        O nível médio foi estimado pelo **método de Serfes (1991)**, com janela efetiva de 71 h 
                        e passo de 24 h, para atenuar componentes diurna (~24 h) e semidiurna (~12 h), preservando a tendência hidráulica.
                        """
                    )

                fig_v = plot_violin_box(
                    df_win,
                    col_n,
                    f"Poço {poco_sel} – Distribuição dos níveis (janela {i+1})"
                )
                st.pyplot(fig_v, use_container_width=False)

                with st.expander("📘 Metodologia – Distribuição estatística"):
                    st.markdown(
                        """
                        O gráfico violin + boxplot representa a densidade de ocorrência, mediana, quartis 
                        e extremos; distribuições estreitas indicam regime mais estável.
                        """
                    )

    # ---------------------------------------------------------------------
    # Aba Gradiente 2D (mantida)
    # ---------------------------------------------------------------------
    with tab_grad:
        grad = resultado.get("gradiente2d")
        if not isinstance(grad, dict) or grad.get("status") != "OK":
            st.info("Gradiente hidráulico 2D não calculado.")
        else:
            st.success("Gradiente hidráulico 2D calculado com sucesso")
            df_jg = grad.get("janelas")
            df_v = grad.get("vetores_horarios")
            df_xy = grad.get("pocos_xy")
            ang = None
            if isinstance(df_jg, pd.DataFrame) and not df_jg.empty:
                st.subheader("Resumo por janelas")
                st.dataframe(df_jg, use_container_width=True)
                ang = float(df_jg.iloc[0]["Ângulo dominante (°)"])
                mod = float(df_jg.iloc[0]["Módulo médio gradiente (m/m)"])
                st.markdown(
                    f"**Direção dominante:** {ang:.1f}° ({_cardinal_16(ang)})  \n"
                    f"**Módulo médio:** {mod:.4f} m/m"
                )
            if isinstance(df_xy, pd.DataFrame) and not df_xy.empty and ang is not None:
                fig_map = plot_gradiente_mapa(
                    df_xy,
                    ang,
                    f"Gradiente hidráulico dominante – {ang:.1f}°"
                )
                st.pyplot(fig_map, use_container_width=False)
            with st.expander("📘 Metodologia – Gradiente hidráulico 2D"):
                st.markdown(
                    """
                    Ajuste de plano h(x,y)=a+bx+cy por mínimos quadrados; o vetor de fluxo é o gradiente negativo. 
                    A direção dominante indica o sentido preferencial de escoamento; oscilações podem refletir maré/bombeamentos.
                    """
                )
            if isinstance(df_v, pd.DataFrame) and not df_v.empty:
                fig_a, fig_m = plot_gradiente_series(df_v)
                st.pyplot(fig_a, use_container_width=False)
                st.pyplot(fig_m, use_container_width=False)
                with st.expander("📘 Interpretação – Variação temporal do gradiente"):
                    st.markdown(
                        """
                        A variação do ângulo pode indicar mudança do sentido de fluxo; a do módulo, intensificação 
                        ou enfraquecimento do gradiente. Inversões com módulo baixo sugerem instabilidade por gradiente fraco; 
                        com módulo alto, reversão hidrodinâmica relevante.
                        """
                    )

    # ---------------------------------------------------------------------
    # Aba Interpretação (nova avançada + PIG/envelope)
    # ---------------------------------------------------------------------
    with tab_ia:
        try:
            st.markdown(interpretar_resultados_avancado(resultado, df_pig if not df_pig.empty else None))
        except Exception as e:
            st.error("Erro na interpretação avançada. O restante do app permanece válido.")
            st.exception(e)

        # Se PIG ativo, mostrar envelope e tabela
        if pig_resumo.get("status") == "OK" and not df_envelope.empty:
            st.divider()
            st.subheader("🌊 Modulação Quinzenal – Envelope (todo o período)")
            # Seletor de poço para o envelope
            pocos_env = ["(média)"] + sorted(df_envelope["Poco"].unique().tolist())
            poco_env = st.selectbox("Poço para envelope", pocos_env)
            poco_sel_env = None if poco_env == "(média)" else poco_env
            fig_env = plot_envelope_quinzenal(df_envelope, poco_sel_env)
            if fig_env is not None:
                st.pyplot(fig_env, use_container_width=False)

            with st.expander("Tabela PIG completa (por poço e regime)"):
                st.dataframe(df_pig, use_container_width=True)

    # ---------------------------------------------------------------------
    # Aba Download (incluir PIG se existir)
    # ---------------------------------------------------------------------
    with tab_dl:
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
        # Novas tabelas PIG
        if pig_resumo.get("status") == "OK":
            if isinstance(df_pig, pd.DataFrame) and not df_pig.empty:
                tabelas["PIG_tabela"] = df_pig
            if isinstance(df_envelope, pd.DataFrame) and not df_envelope.empty:
                tabelas["PIG_envelope"] = df_envelope

        if tabelas:
            bio = BytesIO()
            with pd.ExcelWriter(bio, engine="openpyxl") as writer:
                for nome, df in tabelas.items():
                    df.to_excel(writer, index=False, sheet_name=nome[:31])
            st.download_button(
                "⬇️ Baixar resultados (Excel)",
                data=bio.getvalue(),
                file_name="agente_gac_resultados.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        else:
            st.info("Nenhuma tabela disponível para download.")
