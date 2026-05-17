# src/io/leitura_pocos.py
"""
Leitor de arquivos de poços (nível d'água).

Aceita dois formatos automaticamente detectados:

1. FORMATO LEVELLOGGER (Solinst e similares):
   - Cabeçalho com 11 linhas de metadados (Serial_number, Project ID, Location, etc.)
   - Linha 12 com cabeçalho: Date;Time;ms;LEVEL;TEMPERATURE
   - Separador: ; (ponto e vírgula)
   - Decimal: vírgula ou ponto
   - Encoding: latin1

2. FORMATO CSV SIMPLES (recomendado para testes):
   - Cabeçalho na primeira linha com colunas: datetime,nivel
     OU: data,hora,nivel  OU:  Date,Time,Level
   - Separador: , (vírgula) ou ; (ponto e vírgula) — auto-detectado
   - Decimal: ponto ou vírgula — auto-detectado
   - Encoding: utf-8 ou latin1
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional


def _detectar_formato(caminho_csv: str) -> str:
    """Inspeciona as primeiras linhas para detectar o formato.

    Retorna 'levellogger' se as primeiras linhas têm cara de metadado
    (Serial_number, Project ID, etc.) ou 'simples' caso contrário.
    """
    marcadores = ("serial_number", "project id", "level", "unit:", "offset")
    try:
        with open(caminho_csv, "r", encoding="latin1", errors="ignore") as f:
            head = [next(f, "").lower() for _ in range(11)]
    except Exception:
        return "simples"

    pontos = sum(1 for linha in head for m in marcadores if m in linha)
    return "levellogger" if pontos >= 2 else "simples"


def _ler_csv_levellogger(caminho_csv: str) -> pd.DataFrame:
    """Lê CSV do Level Logger pulando 11 linhas de cabeçalho."""
    df = pd.read_csv(caminho_csv, skiprows=11, sep=";", encoding="latin1")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _ler_csv_simples(caminho_csv: str) -> pd.DataFrame:
    """Lê CSV simples (datetime/data/hora + nível) com auto-detecção de separador e encoding."""
    encodings = ["utf-8", "latin1", "iso-8859-1", "cp1252"]
    separadores = [",", ";", "\t"]

    ultima_excecao = None
    for enc in encodings:
        for sep in separadores:
            try:
                df = pd.read_csv(caminho_csv, sep=sep, encoding=enc)
                if len(df.columns) >= 2 and len(df) > 0:
                    df.columns = [str(c).strip() for c in df.columns]
                    return df
            except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
                ultima_excecao = e
                continue
    raise ValueError(f"Não foi possível ler CSV simples: {ultima_excecao}")


def _achar_coluna(df: pd.DataFrame, nomes_possiveis: list[str]) -> Optional[str]:
    """Encontra coluna por correspondência case-insensitive."""
    low = {str(c).lower(): c for c in df.columns}
    for n in nomes_possiveis:
        if n.lower() in low:
            return low[n.lower()]
    return None


def ler_pocos_e_agregar_hora(caminho_csv: str, inicio: pd.Timestamp) -> pd.DataFrame:
    """Lê arquivo de poço (Levellogger ou CSV simples) e agrega por hora (média).

    Parameters
    ----------
    caminho_csv : str
        Caminho do arquivo CSV.
    inicio : pd.Timestamp
        Timestamp inicial — registros anteriores são descartados.

    Returns
    -------
    pd.DataFrame
        DataFrame com colunas ['hora', 'Nivel'] (agregado horário, média).
    """
    formato = _detectar_formato(caminho_csv)

    if formato == "levellogger":
        df = _ler_csv_levellogger(caminho_csv)
        col_date = _achar_coluna(df, ["Date", "Data"])
        col_time = _achar_coluna(df, ["Time", "Hora"])
        if col_date is None or col_time is None:
            if len(df.columns) < 2:
                raise ValueError(f"CSV Levellogger inválido. Colunas: {list(df.columns)}")
            col_date = df.columns[0]
            col_time = df.columns[1]

        col_nivel = _achar_coluna(df, ["Nivel", "Nível", "Level", "LEVEL", "Water Level"])
        if col_nivel is None:
            if len(df.columns) <= 3:
                raise ValueError(f"Coluna de nível não encontrada. Colunas: {list(df.columns)}")
            col_nivel = df.columns[3]

        dt = pd.to_datetime(
            df[col_date].astype(str).str.strip() + " " + df[col_time].astype(str).str.strip(),
            dayfirst=True,
            errors="coerce",
        )
    else:
        df = _ler_csv_simples(caminho_csv)

        col_dt = _achar_coluna(df, ["datetime", "datahora", "data_hora", "timestamp"])
        if col_dt is not None:
            dt = pd.to_datetime(df[col_dt].astype(str).str.strip(), dayfirst=True, errors="coerce")
        else:
            col_date = _achar_coluna(df, ["data", "date", "dia"])
            col_time = _achar_coluna(df, ["hora", "time", "horario"])
            if col_date and col_time:
                dt = pd.to_datetime(
                    df[col_date].astype(str).str.strip() + " " + df[col_time].astype(str).str.strip(),
                    dayfirst=True,
                    errors="coerce",
                )
            else:
                dt = pd.to_datetime(df.iloc[:, 0].astype(str).str.strip(), dayfirst=True, errors="coerce")

        col_nivel = _achar_coluna(df, ["nivel", "nível", "level", "agua", "water_level"])
        if col_nivel is None:
            for c in df.columns[::-1]:
                amostra = pd.to_numeric(
                    df[c].astype(str).str.replace(",", ".", regex=False), errors="coerce"
                )
                if amostra.notna().sum() > len(df) * 0.5:
                    col_nivel = c
                    break
        if col_nivel is None:
            raise ValueError(f"Coluna de nível não identificada. Colunas: {list(df.columns)}")

    nivel = pd.to_numeric(
        df[col_nivel].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )

    out = pd.DataFrame({"datetime": dt, "Nivel": nivel}).dropna(subset=["datetime", "Nivel"]).copy()

    if inicio is not None:
        inicio = pd.to_datetime(inicio)
        out = out[out["datetime"] >= inicio].copy()

    if out.empty:
        raise ValueError(
            f"Nenhum dado válido após {inicio}. "
            f"Verifique o formato de datas e o timestamp de início."
        )

    out["hora"] = out["datetime"].dt.floor("h")
    df_h = out.groupby("hora")["Nivel"].mean().reset_index()
    return df_h


# Aliases (compatibilidade)
ler_poco_e_agregar_hora = ler_pocos_e_agregar_hora
ler_pocos_agregar_hora = ler_pocos_e_agregar_hora
ler_poco_agregar_hora = ler_pocos_e_agregar_hora
ler_poco_horario = ler_pocos_e_agregar_hora
