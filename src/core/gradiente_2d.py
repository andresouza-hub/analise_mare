from __future__ import annotations

import math
import numpy as np
import pandas as pd


def _angulo_0_360(vx: float, vy: float) -> float:
    ang = math.degrees(math.atan2(vy, vx))
    return ang + 360 if ang < 0 else ang


def _delta_angular(a1: float, a2: float) -> float:
    """
    Menor diferença angular entre dois ângulos (graus).
    """
    d = abs(a2 - a1) % 360
    return min(d, 360 - d)


def _ajustar_plano(X: np.ndarray, Y: np.ndarray, H: np.ndarray):
    """
    Ajusta H = a*X + b*Y + c
    """
    A = np.column_stack([X, Y, np.ones(len(X))])
    coef, _, _, _ = np.linalg.lstsq(A, H, rcond=None)
    return coef  # a, b, c


def calcular_gradiente_2d_horario(
    series_horarias_por_poco: dict[str, pd.DataFrame],
    df_cotas: pd.DataFrame,
    df_xy: pd.DataFrame,
    min_pocos: int = 3,
) -> pd.DataFrame:
    """
    Calcula o vetor de gradiente hidráulico hora a hora (2D).

    Retorna DataFrame com:
      hora | n_pocos | a | b | modulo | angulo
    """

    # Normalização
    df_cotas = df_cotas.copy()
    df_xy = df_xy.copy()

    df_cotas["Poco"] = df_cotas["Poco"].astype(str).str.strip()
    df_xy["Poco"] = df_xy["Poco"].astype(str).str.strip()

    cotas = dict(zip(df_cotas["Poco"], df_cotas["Cota_TOC_m"]))

    # Monta base longa: hora | Poco | Head | X | Y
    linhas = []

    for poco, df_h in series_horarias_por_poco.items():
        if poco not in cotas:
            continue
        if poco not in df_xy["Poco"].values:
            continue

        cota = cotas[poco]
        df_tmp = df_h.copy()
        df_tmp["Poco"] = poco
        df_tmp["Head"] = cota - df_tmp["Nivel"]

        df_tmp = df_tmp.merge(df_xy, on="Poco", how="inner")
        linhas.append(df_tmp[["hora", "Poco", "Head", "X_m", "Y_m"]])

    if not linhas:
        return pd.DataFrame()

    df_long = pd.concat(linhas, ignore_index=True)

    resultados = []

    for hora, g in df_long.groupby("hora"):
        if len(g) < min_pocos:
            continue

        X = g["X_m"].to_numpy(float)
        Y = g["Y_m"].to_numpy(float)
        H = g["Head"].to_numpy(float)

        a, b, _ = _ajustar_plano(X, Y, H)

        # fluxo = -gradiente
        vx, vy = -a, -b
        ang = _angulo_0_360(vx, vy)
        mod = math.sqrt(a * a + b * b)

        resultados.append({
            "hora": hora,
            "n_pocos": len(g),
            "dH_dx (a)": a,
            "dH_dy (b)": b,
            "modulo_gradiente": mod,
            "angulo_fluxo_deg": ang,
        })

    return pd.DataFrame(resultados).sort_values("hora").reset_index(drop=True)


def detectar_inversao_gradiente(
    df_gradiente: pd.DataFrame,
    limiar_deg: float = 160.0,
    horas_consecutivas: int = 2,
):
    """
    Detecta inversão de fluxo com base em rotação angular.

    Retorna:
      dict com status, horário inicial, final e duração
    """

    if df_gradiente.empty or len(df_gradiente) < horas_consecutivas + 1:
        return {
            "inversao": False,
            "motivo": "dados insuficientes"
        }

    angs = df_gradiente["angulo_fluxo_deg"].to_numpy()
    horas = df_gradiente["hora"].to_list()

    deltas = [
        _delta_angular(angs[i - 1], angs[i])
        for i in range(1, len(angs))
    ]

    cont = 0
    inicio = None

    for i, d in enumerate(deltas):
        if d >= limiar_deg:
            cont += 1
            if cont == 1:
                inicio = horas[i]
            if cont >= horas_consecutivas:
                fim = horas[i + 1]
                dur = (fim - inicio).total_seconds() / 3600
                return {
                    "inversao": True,
                    "inicio": inicio,
                    "fim": fim,
                    "duracao_h": dur,
                    "criterio": f"Δθ ≥ {limiar_deg}° por ≥ {horas_consecutivas}h",
                }
        else:
            cont = 0
            inicio = None

    return {
        "inversao": False,
        "motivo": "sem rotação angular suficiente"
    }
