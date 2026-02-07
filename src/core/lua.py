import pandas as pd

def fase_da_lua(dt):
    """
    Retorna (fase_texto, idade_dias) para um datetime.
    Implementação simples e estável para uso no relatório por janela.
    """
    if dt is None or (isinstance(dt, float) and pd.isna(dt)) or pd.isna(dt):
        return (pd.NA, pd.NA)

    dt = pd.Timestamp(dt)

    # Referência clássica aproximada (lua nova perto de 2000-01-06 18:14 UTC)
    ref = pd.Timestamp("2000-01-06 18:14:00")
    lunacao = 29.53058867  # dias

    dias = (dt - ref).total_seconds() / 86400.0
    idade = dias % lunacao
    frac = idade / lunacao

    if frac < 1/16 or frac >= 15/16:
        fase = "Lua Nova"
    elif frac < 3/16:
        fase = "Crescente"
    elif frac < 5/16:
        fase = "Quarto Crescente"
    elif frac < 7/16:
        fase = "Gibosa Crescente"
    elif frac < 9/16:
        fase = "Lua Cheia"
    elif frac < 11/16:
        fase = "Gibosa Minguante"
    elif frac < 13/16:
        fase = "Quarto Minguante"
    else:
        fase = "Minguante"

    return fase, float(idade)
