# src/io/leitura_pocos.py
from __future__ import annotations

import pandas as pd
import numpy as np


def _ler_csv_levellogger(caminho_csv: str) -> pd.DataFrame:
    """
    Lê o CSV do Level Logger com o padrão típico:
      - skiprows=11, sep=';', encoding='latin1'
    """
    df = pd.read_csv(caminho_csv, skiprows=11, sep=';', encoding='latin1')
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _achar_coluna(df: pd.DataFrame, nomes_possiveis: list[str]) -> str | None:
    cols = list(df.columns)
    low = {c.lower(): c for c in cols}
    for n in nomes_possiveis:
        if n.lower() in low:
            return low[n.lower()]
    return None


def ler_pocos_e_agregar_hora(caminho_csv: str, inicio: pd.Timestamp) -> pd.DataFrame:
    """
    Lê Level Logger e agrega por hora (média), mantendo a lógica do Colab:
      - Date e Time em colunas separadas (ou 1ª e 2ª colunas como fallback)
      - Nível: tenta por nome; se não encontrar, usa a 4ª coluna (índice 3) como fallback

    Retorna df_h com colunas: ['hora', 'Nivel']
    """
    df = _ler_csv_levellogger(caminho_csv)

    # Date e Time
    col_date = _achar_coluna(df, ["Date", "Data"])
    col_time = _achar_coluna(df, ["Time", "Hora"])
    if col_date is None or col_time is None:
        if len(df.columns) < 2:
            raise ValueError(f"CSV inválido: poucas colunas. Colunas: {list(df.columns)}")
        col_date = df.columns[0]
        col_time = df.columns[1]

    # Nível
    col_nivel = _achar_coluna(df, ["Nivel", "Nível", "Level", "Water Level"])
    if col_nivel is None:
        if len(df.columns) <= 3:
            raise ValueError(
                f"CSV inválido: não achei coluna de nível e não existe coluna 4. "
                f"Colunas: {list(df.columns)}"
            )
        col_nivel = df.columns[3]

    # Parse
    dt = pd.to_datetime(
        df[col_date].astype(str).str.strip() + " " + df[col_time].astype(str).str.strip(),
        dayfirst=True,
        errors="coerce",
    )
    nivel = pd.to_numeric(df[col_nivel].astype(str).str.replace(",", "."), errors="coerce")

    out = pd.DataFrame({"datetime": dt, "Nivel": nivel}).dropna(subset=["datetime", "Nivel"]).copy()
    if inicio is not None:
        inicio = pd.to_datetime(inicio)
        out = out[out["datetime"] >= inicio].copy()

    out["hora"] = out["datetime"].dt.floor("h")
    df_h = out.groupby("hora")["Nivel"].mean().reset_index()
    return df_h


# Aliases (compatibilidade com variações de import)
ler_poco_e_agregar_hora = ler_pocos_e_agregar_hora
ler_pocos_agregar_hora = ler_pocos_e_agregar_hora
ler_poco_agregar_hora = ler_pocos_e_agregar_hora
ler_poco_horario = ler_pocos_e_agregar_hora