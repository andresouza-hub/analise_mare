# src/core/analise_poco.py
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

from src.core.serfes import metodo_serfes
from src.core.amplitudes import amplitudes_basicas, amplitudes_fft

# Tolerância a variações do nome do módulo: mare_lag.py vs marelag.py
try:
    from src.core.mare_lag import (
        lag_evento_resposta_inflexao,
        lag_evento_resposta_inflexao_por_tipo,
    )
except ImportError:
    from src.core.marelag import (
        lag_evento_resposta_inflexao,
        lag_evento_resposta_inflexao_por_tipo,
    )

from src.core.lua import fase_da_lua


def analisar_poco(
    df_poco_h: pd.DataFrame,
    nome_poco: str,
    *,
    # keyword-only para evitar erros silenciosos
    inicio_teste: Optional[pd.Timestamp] = None,
    df_mare: Optional[pd.DataFrame] = None,
    df_mare_tipo: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Analisa UM poço a partir da série horária agregada (janela de 71 h, passo 24 h).
    Retorna série, resultados por janela e resumo.

    df_poco_h: colunas obrigatórias 'hora' (datetime) e 'Nivel' (m, numérico); len >= 71.
    nome_poco: identificador do poço.
    inicio_teste: metadado (não entra no cálculo).
    df_mare: eventos de maré (opcional) p/ lags.
    df_mare_tipo: eventos de maré com 'Tipo' (Alta/Baixa), opcional.
    """

    # --------- Validações ----------
    if df_poco_h is None or len(df_poco_h) < 71:
        raise ValueError(f"Poço {nome_poco} não possui dados suficientes (mín. 71 horas).")

    falt = {"hora", "Nivel"}.difference(df_poco_h.columns)
    if falt:
        raise ValueError(
            f"Poço {nome_poco}: faltam colunas obrigatórias {sorted(falt)} em df_poco_h."
        )

    # --------- Tipos/limpeza ----------
    if not is_datetime64_any_dtype(df_poco_h["hora"]):
        try:
            df_poco_h = df_poco_h.copy()
            df_poco_h["hora"] = pd.to_datetime(df_poco_h["hora"])
        except Exception as e:
            raise ValueError(
                f"Poço {nome_poco}: 'hora' não é datetime e não pôde ser convertida."
            ) from e

    if not is_numeric_dtype(df_poco_h["Nivel"]):
        try:
            df_poco_h = df_poco_h.copy()
            df_poco_h["Nivel"] = pd.to_numeric(df_poco_h["Nivel"], errors="coerce")
        except Exception as e:
            raise ValueError(
                f"Poço {nome_poco}: 'Nivel' não é numérico e não pôde ser convertido."
            ) from e

    if df_poco_h["Nivel"].isna().any():
        df_poco_h = df_poco_h.dropna(subset=["Nivel"]).reset_index(drop=True)

    if len(df_poco_h) < 71:
        raise ValueError(
            f"Poço {nome_poco}: após limpar 'Nivel', restaram menos de 71 horas."
        )

    df_poco_h = df_poco_h.sort_values("hora").reset_index(drop=True)

    # --------- Janelas 71h / passo 24h ----------
    resultados: List[Dict[str, Any]] = []
    i = 0
    janela = 1

    while i + 70 < len(df_poco_h):
        bloco = df_poco_h.iloc[i : i + 71].copy()
        niv = bloco["Nivel"].to_numpy(dtype=float)

        ini_j = bloco["hora"].iloc[0]
        fim_j = bloco["hora"].iloc[-1]
        centro_j = ini_j + (fim_j - ini_j) / 2

        fase_lua_txt, idade_lua = fase_da_lua(centro_j)

        # Calcula Serfes uma única vez e usa como referência para tudo
        n_serfes = float(metodo_serfes(niv))

        linha: Dict[str, Any] = {
            "Poço": nome_poco,
            "Janela": janela,
            "Início": ini_j,
            "Fim": fim_j,
            "Nível médio (Serfes) (m)": n_serfes,
            "Fase da lua": fase_lua_txt,
            "Idade da lua (dias)": idade_lua,
        }

        # --------- Amplitudes (tempo) com defesa ---------
        ab = amplitudes_basicas(niv, nref=n_serfes)
        if not isinstance(ab, dict):
            raise TypeError(f"amplitudes_basicas deve retornar dict, veio {type(ab)}")
        linha.update(ab)

        # --------- Amplitudes (frequência) com defesa ---------
        af = amplitudes_fft(niv, nref=n_serfes, dt_horas=1.0)
        if not isinstance(af, dict):
            raise TypeError(f"amplitudes_fft deve retornar dict, veio {type(af)}")
        linha.update(af)

        # --------- Lags com maré (se disponível) ----------
        if df_mare is not None:
            lag_mean, lag_med, n_lags = lag_evento_resposta_inflexao(
                df_poco_h=df_poco_h,
                df_mare_eventos=df_mare,
                ini_janela=ini_j,
                fim_janela=fim_j,
            )
            linha["Lag médio total (h)"] = float(lag_mean) if pd.notna(lag_mean) else np.nan
            linha["Lag mediano total (h)"] = float(lag_med) if pd.notna(lag_med) else np.nan
            linha["N eventos lag"] = int(n_lags)

            if df_mare_tipo is not None:
                lagA_mean, _, _ = lag_evento_resposta_inflexao_por_tipo(
                    df_poco_h, df_mare_tipo, ini_j, fim_j, tipo="Alta"
                )
                lagB_mean, _, _ = lag_evento_resposta_inflexao_por_tipo(
                    df_poco_h, df_mare_tipo, ini_j, fim_j, tipo="Baixa"
                )
            else:
                lagA_mean = lagB_mean = np.nan

            linha["Lag médio maré alta (h)"] = (
                float(lagA_mean) if pd.notna(lagA_mean) else np.nan
            )
            linha["Lag médio maré baixa (h)"] = (
                float(lagB_mean) if pd.notna(lagB_mean) else np.nan
            )
        else:
            linha["Lag médio total (h)"] = np.nan
            linha["Lag mediano total (h)"] = np.nan
            linha["Lag médio maré alta (h)"] = np.nan
            linha["Lag médio maré baixa (h)"] = np.nan
            linha["N eventos lag"] = 0

        resultados.append(linha)

        i += 24
        janela += 1

    # --------- Consolidação ----------
    df_result = pd.DataFrame(resultados)
    if df_result.empty:
        resumo = {
            "n_janelas": 0,
            "lag_medio_total": np.nan,
            "lag_medio_alta": np.nan,
            "lag_medio_baixa": np.nan,
        }
    else:
        resumo = {
            "n_janelas": int(len(df_result)),
            "lag_medio_total": float(df_result["Lag médio total (h)"].mean()),
            "lag_medio_alta": float(df_result["Lag médio maré alta (h)"].mean()),
            "lag_medio_baixa": float(df_result["Lag médio maré baixa (h)"].mean()),
        }

    return {
        "poco": nome_poco,
        "inicio_teste": inicio_teste,
        "serie_horaria": df_poco_h,
        "janelas": df_result,
        "resumo": resumo,
    }
