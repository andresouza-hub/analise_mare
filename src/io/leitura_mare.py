# src/io/leitura_mare.py
from __future__ import annotations

import pandas as pd


def ler_mare_csv_eventos(caminho_csv: str, ano_ref: int | None = None) -> pd.DataFrame:
    """
    Lê planilha de maré por eventos e retorna DataFrame com:
      - 'datetime'
      - 'maré'  (float, em metros)

    Espera colunas: 'Mês', 'Dia (num)', 'Hora' e 'Altura maré'.
    A coluna 'Ano' é OPCIONAL. Se não existir, informe ano_ref; caso contrário, erro.

    Observações:
    - Robusto a BOM/acentos quebrados e variações comuns de cabeçalho.
    - Suporta múltiplos encodings (UTF-8, Latin-1, ISO-8859-1).
    - 'Hora' no formato '%H:%M'.
    """
    # Tentar múltiplos encodings
    encodings_to_try = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
    df = None
    last_error = None

    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(caminho_csv, sep=';', encoding=encoding)
            break  # Sucesso!
        except (UnicodeDecodeError, pd.errors.ParserError) as e:
            last_error = e
            continue

    if df is None:
        raise ValueError(f"Não foi possível ler o arquivo de maré com nenhum encoding suportado. Último erro: {last_error}")
    # Correções comuns de encoding / BOM
    df = df.rename(columns={
        'ï»¿MÃªs': 'Mês',
        'ï»¿Mês': 'Mês',
        'MÃªs': 'Mês',
        'Altura marÃ©': 'Altura maré',
        'Altura marÃ© ': 'Altura maré',
        'ï»¿Ano': 'Ano',
        'Ano ': 'Ano',
    })
    df.columns = [str(c).strip() for c in df.columns]

    obrig = ['Mês', 'Dia (num)', 'Hora', 'Altura maré']
    falt = [c for c in obrig if c not in df.columns]
    if falt:
        raise KeyError(
            f"Planilha de maré deve conter as colunas {obrig}. Faltando: {falt}"
        )

    mapa_meses = {
        'janeiro': 1, 'fevereiro': 2, 'março': 3, 'marco': 3, 'abril': 4,
        'maio': 5, 'junho': 6, 'julho': 7, 'agosto': 8,
        'setembro': 9, 'outubro': 10, 'novembro': 11, 'dezembro': 12
    }

    df['mes_num'] = (
        df['Mês'].astype(str).str.strip().str.lower().map(mapa_meses)
    )
    horas = pd.to_datetime(
        df['Hora'].astype(str).str.strip(),
        format='%H:%M',
        errors='coerce'
    )

    # Ano: opcional (se não existir, exige ano_ref)
    if 'Ano' in df.columns:
        anos = pd.to_numeric(df['Ano'], errors='coerce')
        if anos.isna().all():
            if ano_ref is None:
                raise ValueError("Coluna 'Ano' inválida/vazia e ano_ref não foi fornecido.")
            anos = anos.fillna(int(ano_ref))
        else:
            if ano_ref is not None:
                anos = anos.fillna(int(ano_ref))
    else:
        if ano_ref is None:
            raise ValueError("Planilha sem 'Ano'. Forneça ano_ref.")
        anos = pd.Series(int(ano_ref), index=df.index)

    df['datetime'] = pd.to_datetime(
        dict(
            year=anos.astype('Int64'),
            month=df['mes_num'],
            day=pd.to_numeric(df['Dia (num)'], errors='coerce'),
            hour=horas.dt.hour,
            minute=horas.dt.minute,
        ),
        errors='coerce'
    )

    # Coluna 'maré' com acento, compatível com mare_lag.py
    df['maré'] = (
        df['Altura maré'].astype(str).str.replace(',', '.', regex=False)
    )
    df['maré'] = pd.to_numeric(df['maré'], errors='coerce')

    df = (
        df.dropna(subset=['datetime', 'maré'])
          .sort_values('datetime')
          .reset_index(drop=True)
    )
    return df[['datetime', 'maré']]