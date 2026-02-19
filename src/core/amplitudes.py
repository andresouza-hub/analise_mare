# src/core/amplitudes.py
from __future__ import annotations

import numpy as np


def _to_1d_float(a) -> np.ndarray:
    """Converte para array 1D float e remove NaN/inf."""
    x = np.asarray(a, dtype=float).ravel()
    x = x[np.isfinite(x)]
    return x


def amplitudes_basicas(n, nref=None) -> dict:
    """
    Amplitudes no domínio do tempo (m).

    n: vetor 1D (nível d'água na janela).
    nref: nível de referência (ex.: Serfes). Se None, usa média aritmética.

    Retorna SEMPRE dict (instância).
    """
    n = _to_1d_float(n)

    if n.size == 0:
        return {
            "Nível mínimo (m)": np.nan,
            "Nível máximo (m)": np.nan,
            "Amplitude pico-a-pico (m)": np.nan,
            "Semiamplitude (m)": np.nan,
            "Amplitude média |N-Nmedio| (m)": np.nan,
            "Amplitude RMS (m)": np.nan,
        }

    nref = float(np.mean(n)) if nref is None else float(nref)

    nmin = float(np.min(n))
    nmax = float(np.max(n))
    p2p = nmax - nmin
    semi = p2p / 2.0

    dev = n - nref
    amp_abs = float(np.mean(np.abs(dev)))
    amp_rms = float(np.sqrt(np.mean(dev ** 2)))

    return {
        "Nível mínimo (m)": nmin,
        "Nível máximo (m)": nmax,
        "Amplitude pico-a-pico (m)": float(p2p),
        "Semiamplitude (m)": float(semi),
        "Amplitude média |N-Nmedio| (m)": amp_abs,
        "Amplitude RMS (m)": amp_rms,
    }


def amplitudes_fft(n, nref=None, dt_horas: float = 1.0) -> dict:
    """
    Amplitudes no domínio da frequência (FFT), em metros (m),
    em bandas ~12h (semidiurna) e ~24h (diurna).

    - nref: referência para remover componente média (ex.: Serfes).
    - dt_horas: passo amostral em horas (para série horária, dt=1.0).

    Observação: usa escala 2*|FFT|/N para amplitude (pico) do seno.

    Retorna SEMPRE dict (instância).
    """
    n = _to_1d_float(n)

    if n.size < 8:
        return {
            "Amp semidiurna (máx, ~12h) (m)": np.nan,
            "Amp semidiurna (RSS, ~12h) (m)": np.nan,
            "Amp diurna (máx, ~24h) (m)": np.nan,
            "Amp diurna (RSS, ~24h) (m)": np.nan,
        }

    nref = float(np.mean(n)) if nref is None else float(nref)
    y = n - nref

    N = len(y)
    Y = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(N, d=dt_horas)  # ciclos por hora

    amps = 2.0 * np.abs(Y) / N
    if amps.size > 0:
        amps[0] = 0.0  # remove DC

    # frequências-alvo
    f_semi = 1.0 / 12.0
    f_diur = 1.0 / 24.0

    # tolerância em ciclos/hora (ajuste simples e robusto)
    tol = 0.02

    def band_metrics(f0: float):
        idx = np.where(np.abs(freqs - f0) < tol)[0]
        if idx.size == 0:
            return np.nan, np.nan
        a = amps[idx]
        return float(np.max(a)), float(np.sqrt(np.sum(a ** 2)))

    semi_max, semi_rss = band_metrics(f_semi)
    diur_max, diur_rss = band_metrics(f_diur)

    return {
        "Amp semidiurna (máx, ~12h) (m)": semi_max,
        "Amp semidiurna (RSS, ~12h) (m)": semi_rss,
        "Amp diurna (máx, ~24h) (m)": diur_max,
        "Amp diurna (RSS, ~24h) (m)": diur_rss,
    }
