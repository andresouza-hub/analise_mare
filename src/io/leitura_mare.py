# =========================================================
# LEITURA DE MARÉ (EVENTOS)
# =========================================================
# - Lê CSV de maré por eventos (alta/baixa)
# - Cria coluna datetime
# - NÃO classifica eventos
# - NÃO calcula lag
# =========================================================

from __future__ import annotations

import pandas as pd


def ler_mare_eventos(caminho_csv: str) -> pd.DataFrame:
    """
    Lê planilha de maré por eventos.

    Espera colunas:
      - Mês
      - Dia (num)
      - Hora
      - Altura maré
      - Ano  (obrigatória)

    Retorna DataFrame com:
      - datetime
      - mare
    """

    df = pd.read_csv(caminho_csv, sep=';', encoding='latin1')

    # Corrige cabeçalhos comuns com BOM / acento quebrado
    df = df.rename(columns={
        'ï»¿MÃªs': 'Mês',
        'Altura marÃ©': 'Altura maré',
        'ï»¿Ano': 'Ano'
    })

    df.columns = [str(c).strip() for c in df.columns]

    colunas_obrigatorias = ['Mês', 'Dia (num)', 'Hora', 'Altura maré', 'Ano']
    faltantes = [c for c in colunas_obrigatorias if c not in df.columns]

    if faltantes:
        raise KeyError(
            f'Planilha de maré deve conter as colunas {colunas_obrigatorias}. '
            f'Faltando: {faltantes}'
        )

    mapa_meses = {
        'janeiro': 1,
        'fevereiro': 2,
        'março': 3,
        'marco': 3,
        'abril': 4,
        'maio': 5,
        'junho': 6,
        'julho': 7,
        'agosto': 8,
        'setembro': 9,
        'outubro': 10,
        'novembro': 11,
        'dezembro': 12
    }

    df['mes_num'] = (
        df['Mês']
        .astype(str)
        .str.strip()
        .str.lower()
        .map(mapa_meses)
    )

    horas = pd.to_datetime(
        df['Hora'].astype(str).str.strip(),
        format='%H:%M',
        errors='coerce'
    )

    anos = pd.to_numeric(df['Ano'], errors='coerce')

    if anos.isna().any():
        raise ValueError("Coluna 'Ano' contém valores inválidos.")

    df['datetime'] = pd.to_datetime(
        dict(
            year=anos.astype(int),
            month=df['mes_num'],
            day=pd.to_numeric(df['Dia (num)'], errors='coerce'),
            hour=horas.dt.hour,
            minute=horas.dt.minute
        ),
        errors='coerce'
    )

    df['mare'] = (
        df['Altura maré']
        .astype(str)
        .str.replace(',', '.', regex=False)
        .astype(float)
    )

    df = df.dropna(subset=['datetime', 'mare'])
    df = df.sort_values('datetime').reset_index(drop=True)

    return df[['datetime', 'mare']]
