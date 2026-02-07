# =========================================================
# ORQUESTRADOR MULTI-POÇOS (GRADIENTE / INVERSÃO)
# - NÃO lê KMZ/Excel aqui (isso fica em I/O)
# - NÃO faz Streamlit
# - NÃO plota
# - Só orquestra vetores + inversão
# =========================================================

from __future__ import annotations

import math
import numpy as np
import pandas as pd


def montar_head_long(
    series_por_poco: dict[str, pd.DataFrame],
    cotas_por_poco: dict[str, float],
) -> pd.DataFrame:
    """
    Monta tabela longa de carga hidráulica:
      hora | Poco | Head

    series_por_poco[p] deve ter colunas ['hora','Nivel']
    cotas_por_poco[p] = Cota_TOC_m

    Head = Cota_TOC_m - Nivel
    """
    rows = []
    for poco, df_h in series_por_poco.items():
        if poco not in cotas_por_poco:
            continue
        df = df_h.copy()
        df['hora'] = pd.to_datetime(df['hora'], errors='coerce')
        df['Nivel'] = pd.to_numeric(df['Nivel'], errors='coerce')
        df = df.dropna(subset=['hora','Nivel']).copy()

        cota = float(cotas_por_poco[poco])
        df['Poco'] = str(poco)
        df['Head'] = cota - df['Nivel'].astype(float)
        rows.append(df[['hora','Poco','Head']])

    if not rows:
        return pd.DataFrame(columns=['hora','Poco','Head'])

    out = pd.concat(rows, ignore_index=True)
    out = out.dropna(subset=['hora','Poco','Head']).sort_values(['hora','Poco']).reset_index(drop=True)
    return out


def ajustar_plano_gradiente(X, Y, H):
    """
    Ajusta H = a*X + b*Y + c por mínimos quadrados.
    Retorna (a, b, c).
    """
    A = np.column_stack([X, Y, np.ones_like(X)])
    coef, _, _, _ = np.linalg.lstsq(A, H, rcond=None)
    a, b, c = coef
    return float(a), float(b), float(c)


def angulo_0_360(vx, vy):
    ang = math.degrees(math.atan2(vy, vx))
    if ang < 0:
        ang += 360.0
    return ang


def delta_angular(a1, a2):
    d = abs(a2 - a1) % 360.0
    return min(d, 360.0 - d)


def circular_mean_deg(angles_deg):
    ang = np.asarray(angles_deg, dtype=float)
    ang = ang[~np.isnan(ang)]
    if len(ang) == 0:
        return np.nan
    rad = np.deg2rad(ang)
    s = np.mean(np.sin(rad))
    c = np.mean(np.cos(rad))
    if np.isclose(s, 0) and np.isclose(c, 0):
        return np.nan
    m = np.rad2deg(np.arctan2(s, c))
    if m < 0:
        m += 360.0
    return float(m)


def calcular_vetores_horarios(
    df_head_long: pd.DataFrame,
    df_coords: pd.DataFrame,
    min_pocos: int = 3
) -> pd.DataFrame:
    """
    Calcula 1 vetor por hora (quando houver >= min_pocos).
    df_head_long: ['hora','Poco','Head']
    df_coords: ['Poco','X_m','Y_m']
    Retorna: hora, n_pocos, a_dhdx, b_dhdy, vx, vy, modulo_gradiente, angulo_fluxo_deg
    """
    if df_head_long.empty:
        return pd.DataFrame()

    dfm = df_head_long.merge(df_coords[['Poco','X_m','Y_m']], on='Poco', how='inner')
    rows = []

    for hora, g in dfm.groupby('hora'):
        g = g.dropna(subset=['Head','X_m','Y_m']).copy()
        if len(g) < min_pocos:
            continue

        X = g['X_m'].to_numpy(dtype=float)
        Y = g['Y_m'].to_numpy(dtype=float)
        H = g['Head'].to_numpy(dtype=float)

        a, b, c = ajustar_plano_gradiente(X, Y, H)

        # gradiente de carga = (a,b); fluxo no sentido oposto -> v = (-a,-b)
        vx = -a
        vy = -b
        mod = math.sqrt(a*a + b*b)
        ang = angulo_0_360(vx, vy)

        rows.append({
            'hora': hora,
            'n_pocos': int(len(g)),
            'a_dhdx': a,
            'b_dhdy': b,
            'vx': vx,
            'vy': vy,
            'modulo_gradiente': mod,
            'angulo_fluxo_deg': ang
        })

    dfv = pd.DataFrame(rows).sort_values('hora').reset_index(drop=True)
    return dfv


def detectar_inversao_por_angulo(
    df_vetores: pd.DataFrame,
    limiar_deg: float = 160.0,
    consec_h: int = 2
):
    """
    Marca inversão se Δθ entre horas consecutivas >= limiar_deg por >= consec_h consecutivas.
    Retorna:
      (tem_inversao, t_ini, t_fim, dur_h)
    """
    if df_vetores is None or df_vetores.empty or len(df_vetores) < 3:
        return False, None, None, 0

    horas = df_vetores['hora'].to_list()
    angs = df_vetores['angulo_fluxo_deg'].to_numpy(dtype=float)

    deltas = np.array([delta_angular(angs[i-1], angs[i]) for i in range(1, len(angs))], dtype=float)
    flag = deltas >= limiar_deg

    run_start = None
    run_len = 0
    best = None

    for i, ok in enumerate(flag):
        if ok:
            if run_start is None:
                run_start = i
                run_len = 1
            else:
                run_len += 1
        else:
            if run_start is not None and run_len >= consec_h:
                best = (run_start, run_len)
                break
            run_start = None
            run_len = 0

    if best is None and run_start is not None and run_len >= consec_h:
        best = (run_start, run_len)

    if best is None:
        return False, None, None, 0

    s, L = best
    t_ini = horas[s]
    t_fim = horas[s + L]
    dur = int((t_fim - t_ini).total_seconds() / 3600.0)
    return True, t_ini, t_fim, dur


def resumir_gradiente_por_janelas(
    df_vetores_horarios: pd.DataFrame,
    janela_h: int = 71,
    passo_h: int = 24,
    limiar_deg: float = 160.0,
    consec_h: int = 2,
    min_vetores_na_janela: int = 60
) -> pd.DataFrame:
    """
    Faz janelas sobre a série de vetores horários.
    Retorna uma linha por janela com ângulo dominante, módulo médio e evento de inversão (se houver).
    """
    if df_vetores_horarios is None or df_vetores_horarios.empty:
        return pd.DataFrame()

    dfv = df_vetores_horarios.copy()
    dfv['hora'] = pd.to_datetime(dfv['hora'], errors='coerce')
    dfv = dfv.dropna(subset=['hora']).sort_values('hora').reset_index(drop=True)

    inicio_global = dfv['hora'].min().floor('h')
    fim_global = dfv['hora'].max().floor('h')

    rows = []
    t0 = inicio_global
    janela_id = 1

    while t0 + pd.Timedelta(hours=janela_h-1) <= fim_global:
        t1 = t0 + pd.Timedelta(hours=janela_h-1)
        seg = dfv[(dfv['hora'] >= t0) & (dfv['hora'] <= t1)].copy()

        if len(seg) < min_vetores_na_janela:
            t0 = t0 + pd.Timedelta(hours=passo_h)
            janela_id += 1
            continue

        ang_dom = circular_mean_deg(seg['angulo_fluxo_deg'].to_numpy())
        mod_med = float(seg['modulo_gradiente'].mean())

        tem_inv, tinv, tfim, dur = detectar_inversao_por_angulo(
            seg, limiar_deg=limiar_deg, consec_h=consec_h
        )

        rows.append({
            'Janela_grad': janela_id,
            'Início': t0,
            'Fim': t1,
            'N vetores (horas válidas)': int(len(seg)),
            'Ângulo dominante (°)': ang_dom,
            'Módulo médio gradiente (m/m)': mod_med,
            f'Inversão (Δθ≥{limiar_deg:.0f}° por ≥{consec_h}h)': 'SIM' if tem_inv else 'NÃO',
            'Início inversão': tinv,
            'Fim inversão': tfim,
            'Duração inversão (h)': dur
        })

        t0 = t0 + pd.Timedelta(hours=passo_h)
        janela_id += 1

    return pd.DataFrame(rows)
