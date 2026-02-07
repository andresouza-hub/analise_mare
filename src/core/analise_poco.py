import numpy as np
import pandas as pd
from datetime import timedelta

from src.core.serfes import metodo_serfes
from src.core.amplitudes import amplitudes_basicas, amplitudes_fft
from src.core.mare_lag import (
    lag_evento_resposta_inflexao,
    lag_evento_resposta_inflexao_por_tipo
)
from src.core.lua import fase_da_lua


def analisar_poco(
    df_poco_h,
    nome_poco,
    inicio_teste,
    df_mare=None,
    df_mare_tipo=None
):
    """
    Analisa UM poço a partir da série horária agregada.
    Retorna objeto científico completo.
    """

    if df_poco_h is None or len(df_poco_h) < 71:
        raise ValueError(f"Poço {nome_poco} não possui dados suficientes (mín. 71 horas).")

    resultados = []
    i = 0
    janela = 1

    while i + 70 < len(df_poco_h):
        bloco = df_poco_h.iloc[i:i+71].copy()
        niv = bloco["Nivel"].to_numpy(dtype=float)

        ini_j = bloco["hora"].iloc[0]
        fim_j = bloco["hora"].iloc[-1]
        centro_j = ini_j + (fim_j - ini_j) / 2

        fase_lua_txt, idade_lua = fase_da_lua(centro_j)

        linha = {
            "Poço": nome_poco,
            "Janela": janela,
            "Início": ini_j,
            "Fim": fim_j,
            "Nível médio (Serfes) (m)": metodo_serfes(niv),
            "Fase da lua": fase_lua_txt,
            "Idade da lua (dias)": idade_lua
        }

        # amplitudes
        linha.update(amplitudes_basicas(niv))
        linha.update(amplitudes_fft(niv))

        # lags
        if df_mare is not None:
            lag_mean, lag_med, n_lags = lag_evento_resposta_inflexao(
                df_poco_h=df_poco_h,
                df_mare_eventos=df_mare,
                ini_janela=ini_j,
                fim_janela=fim_j
            )

            linha["Lag médio total (h)"] = lag_mean
            linha["Lag mediano total (h)"] = lag_med
            linha["N eventos lag"] = n_lags

            if df_mare_tipo is not None:
                lagA_mean, _, _ = lag_evento_resposta_inflexao_por_tipo(
                    df_poco_h, df_mare_tipo, ini_j, fim_j, tipo="Alta"
                )
                lagB_mean, _, _ = lag_evento_resposta_inflexao_por_tipo(
                    df_poco_h, df_mare_tipo, ini_j, fim_j, tipo="Baixa"
                )
            else:
                lagA_mean = lagB_mean = np.nan

            linha["Lag médio maré alta (h)"] = lagA_mean
            linha["Lag médio maré baixa (h)"] = lagB_mean
        else:
            linha["Lag médio total (h)"] = np.nan
            linha["Lag médio maré alta (h)"] = np.nan
            linha["Lag médio maré baixa (h)"] = np.nan
            linha["N eventos lag"] = 0

        resultados.append(linha)
        i += 24
        janela += 1

    df_result = pd.DataFrame(resultados)

    resumo = {
        "n_janelas": len(df_result),
        "lag_medio_total": df_result["Lag médio total (h)"].mean(),
        "lag_medio_alta": df_result["Lag médio maré alta (h)"].mean(),
        "lag_medio_baixa": df_result["Lag médio maré baixa (h)"].mean()
    }

    return {
        "poco": nome_poco,
        "inicio_teste": inicio_teste,
        "serie_horaria": df_poco_h,
        "janelas": df_result,
        "resumo": resumo
    }
