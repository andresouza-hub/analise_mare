import numpy as np
import pandas as pd


def metodo_serfes(n):
    """
    Método de Serfes para cálculo do nível médio (hora central),
    exatamente como validado no Colab.

    Parâmetro
    ---------
    n : array-like
        Série horária de níveis (71 valores).

    Retorno
    -------
    float
        Nível médio (Serfes).
    """
    n = pd.Series(n).astype(float).reset_index(drop=True)

    # 48 médias móveis de 24h
    m1 = [n[i:i+24].mean() for i in range(48)]

    # 25 médias das médias
    m2 = [np.mean(m1[i:i+24]) for i in range(25)]

    # Valor final (hora central)
    nivel_medio = float(np.mean(m2))

    return nivel_medio
