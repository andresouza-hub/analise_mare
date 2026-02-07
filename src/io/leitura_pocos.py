# =========================================================
# LEITURA DE POÇOS (LEVELLOGGER)
# =========================================================
# - Lê múltiplos CSVs de poços
# - Padroniza dados
# - Agrega nível por hora
# - NÃO faz cálculo científico
# =========================================================

from __future__ import annotations

import os
import pandas as pd


def _normalizar_nome_poco(caminho: str) -> str:
    """
    Extrai o nome do poço a partir do nome do arquivo.
    Ex: 'PM-285-776.csv' -> 'PM-285-776'
    """
    return os.path.splitext(os.path.basename(caminho))[0]


def _ler_csv_levelogger(caminho_csv: str) -> pd.DataFrame:
    """
    Lê um CSV de Levelogger no formato padrão.
    """
    df = pd.read_csv(
        caminho_csv,
        sep=';',
        skiprows=11,
        encoding='latin1'
    )

    # Renomeia colunas por posição (padrão Levelogger)
    df = df.rename(columns={
        df.columns[0]: 'Data',
        df.columns[1]: 'Hora',
        df.columns[3]: 'Nivel'
    })

    # Cria datetime
    df['datetime'] = pd.to_datetime(
        df['Data'].astype(str) + ' ' + df['Hora'].astype(str),
        dayfirst=True,
        errors='coerce'
    )

    # Converte nível para float
    df['Nivel'] = (
        df['Nivel']
        .astype(str)
        .str.replace(',', '.', regex=False)
        .astype(float)
    )

    df = df.dropna(subset=['datetime', 'Nivel']).copy()

    return df[['datetime', 'Nivel']]


def _agregar_por_hora(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega o nível por hora (média).
    """
    df['hora'] = df['datetime'].dt.floor('h')
    df_h = (
        df
        .groupby('hora', as_index=False)['Nivel']
        .mean()
        .sort_values('hora')
        .reset_index(drop=True)
    )
    return df_h


def ler_pocos_e_agregar_hora(caminhos_csv: list[str]) -> dict[str, pd.DataFrame]:
    """
    Lê múltiplos arquivos de poços e retorna um dicionário:

    {
        'PM-285-776': df_h,
        'PM-285-781': df_h,
        ...
    }
    """
    resultados: dict[str, pd.DataFrame] = {}

    for caminho in caminhos_csv:
        nome_poco = _normalizar_nome_poco(caminho)
        df_raw = _ler_csv_levelogger(caminho)
        df_h = _agregar_por_hora(df_raw)

        if df_h.empty:
            raise ValueError(f'Poço {nome_poco} não possui dados válidos.')

        resultados[nome_poco] = df_h

    return resultados
