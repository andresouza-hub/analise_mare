from __future__ import annotations

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _vetor_de_angulo(ang_deg: float):
    rad = math.radians(float(ang_deg))
    return math.cos(rad), math.sin(rad)


def salvar_mapa_vetor(
    df_xy: pd.DataFrame,
    angulo_fluxo_deg: float,
    out_png: str,
    titulo: str = "",
):
    """
    Plota poços (X_m,Y_m) e vetor de fluxo (seta) no centróide.
    """
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    d = df_xy.copy()
    d["Poco"] = d["Poco"].astype(str)

    Xc = float(d["X_m"].mean())
    Yc = float(d["Y_m"].mean())

    plt.figure(figsize=(7, 7))
    plt.scatter(d["X_m"], d["Y_m"])

    for _, r in d.iterrows():
        plt.text(float(r["X_m"]), float(r["Y_m"]), r["Poco"], fontsize=9)

    vx, vy = _vetor_de_angulo(angulo_fluxo_deg)

    base = max(d["X_m"].max() - d["X_m"].min(), d["Y_m"].max() - d["Y_m"].min())
    L = 0.25 * base if base and base > 0 else 10.0

    plt.arrow(Xc, Yc, vx * L, vy * L, head_width=0.04 * L, length_includes_head=True)

    plt.title(titulo or f"Vetor de fluxo | {angulo_fluxo_deg:.1f}°")
    plt.xlabel("X (m) [UTM]")
    plt.ylabel("Y (m) [UTM]")
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def angulo_dominante_por_hora(df_grad_h: pd.DataFrame) -> float:
    """
    Média circular dos ângulos.
    """
    ang = df_grad_h["angulo_fluxo_deg"].to_numpy(float)
    ang = ang[~np.isnan(ang)]
    if len(ang) == 0:
        return float("nan")
    rad = np.deg2rad(ang)
    s = np.mean(np.sin(rad))
    c = np.mean(np.cos(rad))
    if np.isclose(s, 0) and np.isclose(c, 0):
        return float("nan")
    m = np.rad2deg(np.arctan2(s, c))
    if m < 0:
        m += 360
    return float(m)
