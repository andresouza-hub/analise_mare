# src/core/mare_lag.py
"""
Maré + Lag (inflexão hidráulica) — funções científicas sem dependência de UI.
Inclui:
  - classificar_eventos_mare_alta_baixa(df_mare)
  - lag_evento_resposta_inflexao(...)
  - lag_evento_resposta_inflexao_por_tipo(...)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import medfilt


# -----------------------------
# Classificação Alta / Baixa
# -----------------------------
def classificar_eventos_mare_alta_baixa(df_mare: pd.DataFrame) -> pd.DataFrame:
    """
    Classifica eventos de maré como 'Alta' (máximo local) ou 'Baixa' (mínimo local).
    Entrada esperada: df_mare com colunas ['datetime', 'maré'].

    Retorna df ordenado com coluna extra 'Tipo' ('Alta' / 'Baixa' / 'Indef').
    """
    df = df_mare.sort_values('datetime').reset_index(drop=True).copy()
    h = df['maré'].to_numpy(dtype=float)
    tipo = np.full(len(df), 'Indef', dtype=object)

    for i in range(len(df)):
        prev = h[i - 1] if i - 1 >= 0 else np.nan
        nxt  = h[i + 1] if i + 1 < len(df) else np.nan

        if not np.isnan(prev) and not np.isnan(nxt):
            if h[i] >= prev and h[i] >= nxt:
                tipo[i] = 'Alta'
            elif h[i] <= prev and h[i] <= nxt:
                tipo[i] = 'Baixa'
            else:
                # decisão por proximidade relativa
                tipo[i] = 'Alta' if h[i] >= (prev + nxt) / 2 else 'Baixa'
        else:
            # bordas (primeiro/último) ou NaN vizinhos
            if np.isnan(prev) and not np.isnan(nxt):
                tipo[i] = 'Alta' if h[i] > nxt else 'Baixa'
            elif np.isnan(nxt) and not np.isnan(prev):
                tipo[i] = 'Alta' if h[i] > prev else 'Baixa'
            else:
                tipo[i] = 'Indef'

    df['Tipo'] = tipo
    return df


# -----------------------------
# Detecção de inflexão
# -----------------------------
def _primeira_inflexao_pos_evento(
    df_poco_h: pd.DataFrame,
    t_evento: pd.Timestamp,
    t_fim_busca: pd.Timestamp,
    kernel_mediana: int = 3,
) -> pd.Timestamp | None:
    """
    Retorna o timestamp da primeira inflexão (mudança de sinal da derivada)
    da série 'Nivel' entre t_evento e t_fim_busca. Usa filtro de mediana
    para robustez.
    """
    seg = df_poco_h[
        (df_poco_h['hora'] >= t_evento) & (df_poco_h['hora'] <= t_fim_busca)
    ].copy()
    if len(seg) < 4:
        return None

    y = seg['Nivel'].to_numpy(dtype=float)

    # Suaviza com mediana (ímpar e >=3)
    if kernel_mediana and kernel_mediana % 2 == 1 and kernel_mediana >= 3:
        y_s = medfilt(y, kernel_size=kernel_mediana)
    else:
        y_s = y

    dy = np.diff(y_s)
    s = np.sign(dy)

    # Propaga último sinal não-zero para reduzir zigue-zague por empates
    s2 = s.copy()
    last = 0
    for i in range(len(s2)):
        if s2[i] == 0:
            s2[i] = last
        else:
            last = s2[i]

    for i in range(1, len(s2)):
        if s2[i - 1] == 0 or s2[i] == 0:
            continue
        if s2[i] != s2[i - 1]:
            return seg['hora'].iloc[i]

    return None


# -----------------------------
# Lag (todos os eventos)
# -----------------------------
def lag_evento_resposta_inflexao(
    df_poco_h: pd.DataFrame,
    df_mare_eventos: pd.DataFrame,
    ini_janela: pd.Timestamp,
    fim_janela: pd.Timestamp,
    max_lag_h: int = 12,
    kernel_mediana: int = 3,
) -> tuple[float, float, int]:
    """
    Calcula lags (h) entre cada evento de maré na janela e a primeira inflexão
    da série do poço. Retorna (média, mediana, N).
    """
    if df_mare_eventos is None or df_mare_eventos.empty:
        return np.nan, np.nan, 0

    eventos = df_mare_eventos[
        (df_mare_eventos['datetime'] >= ini_janela) &
        (df_mare_eventos['datetime'] <= fim_janela)
    ].copy()

    if eventos.empty:
        return np.nan, np.nan, 0

    lags: list[float] = []
    for _, ev in eventos.iterrows():
        t0 = ev['datetime']
        t1 = min(fim_janela, t0 + pd.Timedelta(hours=max_lag_h))
        t_inf = _primeira_inflexao_pos_evento(df_poco_h, t0, t1, kernel_mediana)
        if t_inf is not None:
            lag_h = (t_inf - t0).total_seconds() / 3600.0
            if lag_h >= 0:
                lags.append(lag_h)

    if not lags:
        return np.nan, np.nan, 0

    return float(np.mean(lags)), float(np.median(lags)), int(len(lags))


# -----------------------------
# Lag por tipo (Alta / Baixa)
# -----------------------------
def lag_evento_resposta_inflexao_por_tipo(
    df_poco_h: pd.DataFrame,
    df_mare_tipo: pd.DataFrame,
    ini_janela: pd.Timestamp,
    fim_janela: pd.Timestamp,
    tipo: str = 'Alta',
    max_lag_h: int = 12,
    kernel_mediana: int = 3,
) -> tuple[float, float, int]:
    """
    Idêntico ao lag geral, mas restringe a eventos do 'tipo' ('Alta' ou 'Baixa').
    """
    if df_mare_tipo is None or df_mare_tipo.empty:
        return np.nan, np.nan, 0

    eventos = df_mare_tipo[
        (df_mare_tipo['Tipo'] == tipo) &
        (df_mare_tipo['datetime'] >= ini_janela) &
        (df_mare_tipo['datetime'] <= fim_janela)
    ].copy()

    if eventos.empty:
        return np.nan, np.nan, 0

    lags: list[float] = []
    for _, ev in eventos.iterrows():
        t0 = ev['datetime']
        t1 = min(fim_janela, t0 + pd.Timedelta(hours=max_lag_h))
        t_inf = _primeira_inflexao_pos_evento(df_poco_h, t0, t1, kernel_mediana)
        if t_inf is not None:
            lag_h = (t_inf - t0).total_seconds() / 3600.0
            if lag_h >= 0:
                lags.append(lag_h)

    if not lags:
        return np.nan, np.nan, 0

    return float(np.mean(lags)), float(np.median(lags)), int(len(lags))