# src/io/leitura_mare.py
"""
Leitor de tabela de maré.

Aceita dois formatos automaticamente detectados:

1. FORMATO TÁBUA (Marinha do Brasil e similares):
   - Colunas: Mês, Dia (num), Dia (semana), Hora, Altura maré, Ano
   - Cada linha = um evento (preamar/baixamar)
   - Separador: ;
   - Tolerante a acentos, BOM, encoding latin1/utf-8

2. FORMATO SIMPLES (recomendado para testes):
   - Colunas: datetime, mare  (ou: data, hora, altura)
   - Separador: , ou ; (auto-detectado)
   - Decimal: . ou , (auto-detectado)
"""
from __future__ import annotations

import pandas as pd
from typing import Optional


def _ler_csv_robusto(caminho_csv: str) -> pd.DataFrame:
    """Lê CSV tentando múltiplos encodings e separadores."""
    encodings = ["utf-8", "latin1", "iso-8859-1", "cp1252"]
    separadores = [";", ",", "\t"]

    ultima = None
    for enc in encodings:
        for sep in separadores:
            try:
                df = pd.read_csv(caminho_csv, sep=sep, encoding=enc)
                if len(df.columns) >= 2:
                    return df
            except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
                ultima = e
                continue
    raise ValueError(f"Não foi possível ler arquivo de maré: {ultima}")


def _achar_coluna(df: pd.DataFrame, nomes_possiveis: list[str]) -> Optional[str]:
    """Encontra coluna por correspondência case-insensitive (ignorando espaços)."""
    low = {str(c).lower().strip(): c for c in df.columns}
    for n in nomes_possiveis:
        if n.lower().strip() in low:
            return low[n.lower().strip()]
    return None


def _normalizar_cabecalho(df: pd.DataFrame) -> pd.DataFrame:
    """Corrige cabeçalhos com encoding quebrado (BOM, mojibake)."""
    df = df.rename(columns={
        "ï»¿MÃªs": "Mês",
        "ï»¿Mês": "Mês",
        "MÃªs": "Mês",
        "Altura marÃ©": "Altura maré",
        "Altura marÃ© ": "Altura maré",
        "ï»¿Ano": "Ano",
        "Ano ": "Ano",
    })
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _eh_formato_tabua(df: pd.DataFrame) -> bool:
    """Heurística: tem colunas Mês (ou Mes) E Altura maré (ou Altura mare)?"""
    cols = [c.lower().replace("ê", "e").replace("é", "e") for c in df.columns]
    tem_mes = any("mes" in c for c in cols)
    tem_altura = any("altura" in c for c in cols)
    return tem_mes and tem_altura


def ler_mare_csv_eventos(caminho_csv: str, ano_ref: Optional[int] = None) -> pd.DataFrame:
    """Lê tabela de maré (formato tábua ou CSV simples) e retorna eventos.

    Returns
    -------
    pd.DataFrame
        DataFrame com colunas ['datetime', 'maré']
    """
    df = _ler_csv_robusto(caminho_csv)
    df = _normalizar_cabecalho(df)

    if _eh_formato_tabua(df):
        return _processar_formato_tabua(df, ano_ref)
    else:
        return _processar_formato_simples(df)


def _processar_formato_tabua(df: pd.DataFrame, ano_ref: Optional[int]) -> pd.DataFrame:
    """Processa formato Marinha (Mês, Dia, Hora, Altura, Ano)."""
    # Tolerância: aceita com ou sem acento
    col_mes = _achar_coluna(df, ["Mês", "Mes", "mês", "mes"])
    col_dia = _achar_coluna(df, ["Dia (num)", "Dia", "dia"])
    col_hora = _achar_coluna(df, ["Hora", "hora"])
    col_alt = _achar_coluna(df, ["Altura maré", "Altura mare", "altura", "Altura"])

    faltando = [n for n, c in [("Mês", col_mes), ("Dia", col_dia),
                                ("Hora", col_hora), ("Altura maré", col_alt)] if c is None]
    if faltando:
        raise KeyError(f"Tabela de maré: colunas faltando {faltando}")

    mapa_meses = {
        "janeiro": 1, "fevereiro": 2, "março": 3, "marco": 3, "abril": 4,
        "maio": 5, "junho": 6, "julho": 7, "agosto": 8,
        "setembro": 9, "outubro": 10, "novembro": 11, "dezembro": 12,
    }
    df["mes_num"] = df[col_mes].astype(str).str.strip().str.lower().map(mapa_meses)
    horas = pd.to_datetime(df[col_hora].astype(str).str.strip(), format="%H:%M", errors="coerce")

    col_ano = _achar_coluna(df, ["Ano", "ano", "year"])
    if col_ano is not None:
        anos = pd.to_numeric(df[col_ano], errors="coerce")
        if anos.isna().all():
            if ano_ref is None:
                raise ValueError("Coluna 'Ano' vazia e ano_ref não fornecido.")
            anos = anos.fillna(int(ano_ref))
        elif ano_ref is not None:
            anos = anos.fillna(int(ano_ref))
    else:
        if ano_ref is None:
            raise ValueError("Tabela sem 'Ano'. Forneça ano_ref.")
        anos = pd.Series(int(ano_ref), index=df.index)

    df["datetime"] = pd.to_datetime(
        dict(
            year=anos.astype("Int64"),
            month=df["mes_num"],
            day=pd.to_numeric(df[col_dia], errors="coerce"),
            hour=horas.dt.hour,
            minute=horas.dt.minute,
        ),
        errors="coerce",
    )
    df["maré"] = pd.to_numeric(
        df[col_alt].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )
    df = df.dropna(subset=["datetime", "maré"]).sort_values("datetime").reset_index(drop=True)
    return df[["datetime", "maré"]]


def _processar_formato_simples(df: pd.DataFrame) -> pd.DataFrame:
    """Processa formato simples: datetime + mare (ou data + hora + altura)."""
    col_dt = _achar_coluna(df, ["datetime", "datahora", "timestamp", "data_hora"])
    if col_dt is not None:
        dt = pd.to_datetime(df[col_dt].astype(str).str.strip(), dayfirst=True, errors="coerce")
    else:
        col_date = _achar_coluna(df, ["data", "date", "dia"])
        col_time = _achar_coluna(df, ["hora", "time"])
        if col_date and col_time:
            dt = pd.to_datetime(
                df[col_date].astype(str).str.strip() + " " + df[col_time].astype(str).str.strip(),
                dayfirst=True, errors="coerce",
            )
        else:
            dt = pd.to_datetime(df.iloc[:, 0].astype(str).str.strip(), dayfirst=True, errors="coerce")

    col_alt = _achar_coluna(df, ["maré", "mare", "altura", "nivel", "level", "tide"])
    if col_alt is None:
        # Fallback: última coluna numérica
        for c in df.columns[::-1]:
            amostra = pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False), errors="coerce")
            if amostra.notna().sum() > len(df) * 0.5:
                col_alt = c
                break
    if col_alt is None:
        raise ValueError(f"Coluna de maré não identificada. Colunas: {list(df.columns)}")

    altura = pd.to_numeric(
        df[col_alt].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )

    out = pd.DataFrame({"datetime": dt, "maré": altura})
    out = out.dropna(subset=["datetime", "maré"]).sort_values("datetime").reset_index(drop=True)
    return out
