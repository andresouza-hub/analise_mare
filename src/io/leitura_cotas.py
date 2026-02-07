# =========================================================
# LEITURA DE COTAS TOPOGRÁFICAS
# =========================================================
# - Lê planilha Excel de cotas
# - Padroniza nomes de poços
# - NÃO calcula gradiente
# =========================================================

from __future__ import annotations

import pandas as pd
import re


def _normalizar_nome_poco(nome: str) -> str:
    """
    Normaliza nome do poço para comparação entre arquivos.
    """
    nome = str(nome).strip().upper()
    nome = re.sub(r'\s+', ' ', nome)
    return nome


def ler_cotas(caminho_excel: str) -> pd.DataFrame:
    """
    Lê planilha de cotas topográficas.

    Espera colunas:
      - Poco
      - Cota_TOC_m

    Retorna DataFrame com:
      - Poco
      - Cota_TOC_m
    """

    df = pd.read_excel(caminho_excel)

    df.columns = [str(c).strip() for c in df.columns]

    colunas_obrigatorias = ['Poco', 'Cota_TOC_m']
    faltantes = [c for c in colunas_obrigatorias if c not in df.columns]

    if faltantes:
        raise KeyError(
            f"Planilha de cotas deve conter as colunas {colunas_obrigatorias}. "
            f"Faltando: {faltantes}"
        )

    df = df[['Poco', 'Cota_TOC_m']].copy()

    df['Poco'] = df['Poco'].apply(_normalizar_nome_poco)
    df['Cota_TOC_m'] = pd.to_numeric(df['Cota_TOC_m'], errors='coerce')

    df = df.dropna(subset=['Poco', 'Cota_TOC_m'])
    df = df.drop_duplicates(subset=['Poco']).reset_index(drop=True)

    if df.empty:
        raise ValueError("Planilha de cotas não contém dados válidos.")

    return df
