# =========================================================
# MARÉ + LAG (inflexão hidráulica)
# Módulo científico isolado (sem Streamlit)
# =========================================================

import pandas as pd
import numpy as np
from scipy.signal import medfilt


# =========================================================
# LEITURA DA PLANILHA DE MARÉ (EVENTOS)
# =========================================================
def ler_mare_csv_eventos(arq_mare, ano_ref=None):
    """
    Lê planilha de maré (eventos) com colunas:
    Mês | Dia (num) | Hora | Altura maré | Ano (opcional)

    Retorna DataFrame com:
    datetime | maré
    """
    df = pd.read_csv(arq_mare, sep=';', encoding='latin1')

    # Correções comuns de encoding
    df = df.rename(columns={
        'ï»¿MÃªs': 'Mês',
        'Altura marÃ©': 'Altura maré',
        'ï»¿Ano': 'Ano'
    })
    df.columns = [str(c).strip() for c in df.columns]

    obrig = ['Mês', 'Dia (num)', 'Hora', 'Altura maré']
    falt = [c for c in obrig if c not in df.columns]
    if falt:
        raise KeyError(
            f"Colunas faltantes na maré: {falt}. "
            f"Colunas detectadas: {list(df.columns)}"
        )

    meses = {
        'janeiro': 1, 'fevereiro': 2, 'março': 3, 'marco': 3,
        'abril': 4, 'maio': 5, 'junho': 6, 'julho': 7,
        'agosto': 8, 'setembro': 9, 'outubro': 10,
        'novembro': 11, 'dezembro': 12
    }

    df['mes_num'] = (
        df['Mês']
        .astype(str)
        .str.strip()
        .str.lower()
        .map(meses)
    )

    hora = pd.to_datetime(
        df['Hora'].astype(str).str.strip(),
        format='%H:%M',
        errors='coerce'
    )

    if 'Ano' in df.columns:
        ano = pd.to_numeric(df['Ano'], errors='coerce')
        if ano.isna().all():
            if ano_ref is None:
                raise ValueError("Ano ausente e ano_ref não fornecido.")
            ano = ano.fillna(int(ano_ref))
    else:
        if ano_ref is None:
            raise ValueError("Planilha sem coluna 'Ano' e ano_ref não fornecido.")
        ano = pd.Series(int(ano_ref), index=df.index)

    df['datetime'] = pd.to_datetime(
        dict(
            year=ano.astype('Int64'),
            month=df['mes_num'],
            day=pd.to_numeric(df['Dia (num)'], errors='coerce'),
            hour=hora.dt.hour,
            minute=hora.dt.minute
        ),
        errors='coerce'
    )

    df['maré'] = pd.to_numeric(
        df['Altura maré'].astype(str).str.replace(',', '.'),
        errors='coerce'
    )

    df = (
        df.dropna(subset=['datetime', 'maré'])
          .sort_values('datetime')
          .reset_index(drop=True)
    )

    return df[['datetime', 'maré']]


# =========================================================
# CLASSIFICAÇÃO DE EVENTOS: MARÉ ALTA / BAIXA
# =========================================================
def classificar_eventos_mare_alta_baixa(df_mare):
    df = df_mare.sort_values('datetime').reset_index(drop=True).copy()
    h = df['maré'].to_numpy(dtype=float)
    tipo = np.full(len(df), 'Indef', dtype=object)

    for i in range(len(df)):
        prev = h[i - 1] if i > 0 else np.nan
        nxt = h[i + 1] if i < len(df) - 1 else np.nan

        if not np.isnan(prev) and not np.isnan(nxt):
            if h[i] >= prev and h[i] >= nxt:
                tipo[i] = 'Alta'
            elif h[i] <= prev and h[i] <= nxt:
                tipo[i] = 'Baixa'
        elif np.isnan(prev) and not np.isnan(nxt):
            tipo[i] = 'Alta' if h[i] > nxt else 'Baixa'
        elif np.isnan(nxt) and not np.isnan(prev):
            tipo[i] = 'Alta' if h[i] > prev else 'Baixa'

    df['Tipo'] = tipo
    return df


# =========================================================
# DETECÇÃO DE INFLEXÃO NO POÇO
# =========================================================
def _primeira_inflexao_pos_evento(df_poco_h, t_evento, t_fim_busca, kernel_mediana=3):
    seg = df_poco_h[
        (df_poco_h['hora'] >= t_evento) &
        (df_poco_h['hora'] <= t_fim_busca)
    ].copy()

    if len(seg) < 4:
        return None

    y = seg['Nivel'].to_numpy(dtype=float)

    if kernel_mediana >= 3 and kernel_mediana % 2 == 1:
        y = medfilt(y, kernel_size=kernel_mediana)

    dy = np.diff(y)
    s = np.sign(dy)

    for i in range(1, len(s)):
        if s[i] != s[i - 1] and s[i] != 0 and s[i - 1] != 0:
            return seg['hora'].iloc[i]

    return None


# =========================================================
# LAG GERAL (TODOS OS EVENTOS)
# =========================================================
def lag_evento_resposta_inflexao(
    df_poco_h,
    df_mare_eventos,
    ini_janela,
    fim_janela,
    max_lag_h=12,
    kernel_mediana=3
):
    eventos = df_mare_eventos[
        (df_mare_eventos['datetime'] >= ini_janela) &
        (df_mare_eventos['datetime'] <= fim_janela)
    ]

    lags = []

    for _, ev in eventos.iterrows():
        t0 = ev['datetime']
        t1 = min(fim_janela, t0 + pd.Timedelta(hours=max_lag_h))

        t_inf = _primeira_inflexao_pos_evento(
            df_poco_h, t0, t1, kernel_mediana
        )

        if t_inf is not None:
            lag = (t_inf - t0).total_seconds() / 3600
            if lag >= 0:
                lags.append(lag)

    if not lags:
        return np.nan, np.nan, 0

    return float(np.mean(lags)), float(np.median(lags)), len(lags)


# =========================================================
# LAG POR TIPO (ALTA / BAIXA)
# =========================================================
def lag_evento_resposta_inflexao_por_tipo(
    df_poco_h,
    df_mare_tipo,
    ini_janela,
    fim_janela,
    tipo='Alta',
    max_lag_h=12,
    kernel_mediana=3
):
    eventos = df_mare_tipo[
        (df_mare_tipo['Tipo'] == tipo) &
        (df_mare_tipo['datetime'] >= ini_janela) &
        (df_mare_tipo['datetime'] <= fim_janela)
    ]

    lags = []

    for _, ev in eventos.iterrows():
        t0 = ev['datetime']
        t1 = min(fim_janela, t0 + pd.Timedelta(hours=max_lag_h))

        t_inf = _primeira_inflexao_pos_evento(
            df_poco_h, t0, t1, kernel_mediana
        )

        if t_inf is not None:
            lag = (t_inf - t0).total_seconds() / 3600
            if lag >= 0:
                lags.append(lag)

    if not lags:
        return np.nan, np.nan, 0

    return float(np.mean(lags)), float(np.median(lags)), len(lags)
