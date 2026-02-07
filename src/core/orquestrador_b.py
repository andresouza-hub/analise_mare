from __future__ import annotations

import os
import math
import pandas as pd

# I/O
import src.io.leitura_pocos as leitura_pocos
from src.io.leitura_mare import ler_mare_eventos
from src.io.leitura_cotas import ler_cotas
from src.io.leitura_kmz import ler_kmz_pocos

# Core científico
from src.core.analise_poco import analisar_poco

# Gradiente 2D
from src.core.gradiente_2d import (
    calcular_gradiente_2d_horario,
    detectar_inversao_gradiente,
)


def _nome_poco_do_caminho(caminho: str) -> str:
    base = os.path.basename(caminho)
    return os.path.splitext(base)[0].strip()


def _resolver_funcao_leitura_poco():
    candidatos = [
        "ler_pocos_agregar_hora",      # seu nome atual
        "ler_poco_e_agregar_hora",
        "ler_poco_agregar_hora",
        "ler_poco_horario",
    ]
    for nome in candidatos:
        fn = getattr(leitura_pocos, nome, None)
        if callable(fn):
            return fn, nome

    disponiveis = [n for n in dir(leitura_pocos) if not n.startswith("_")]
    raise ImportError(
        "Não encontrei no src/io/leitura_pocos.py uma função de leitura/agregação horária.\n"
        f"Nomes tentados: {', '.join(candidatos)}\n"
        f"Disponíveis no módulo: {disponiveis}\n"
        "Ação: abra src/io/leitura_pocos.py e confirme o nome da função pública."
    )


def _pegar_serfes_medio(df_res: pd.DataFrame) -> float:
    """
    Extrai o nível Serfes médio do dataframe do poço.

    Regra:
      - tenta achar a linha 'MÉDIA' (se existir)
      - senão, usa a média numérica da coluna
    """
    col = "Nível médio (Serfes) (m)"
    if col not in df_res.columns:
        raise KeyError(f"Coluna '{col}' não existe no resultado do poço.")

    # tenta linha média
    if "Janela" in df_res.columns:
        mask = df_res["Janela"].astype(str).str.upper().eq("MÉDIA")
        if mask.any():
            v = pd.to_numeric(df_res.loc[mask, col], errors="coerce").iloc[0]
            if pd.notna(v):
                return float(v)

    v = pd.to_numeric(df_res[col], errors="coerce").mean()
    return float(v)


def _gradiente_2d_por_serfes(resultados_por_poco: dict[str, pd.DataFrame],
                            df_cotas: pd.DataFrame,
                            df_xy: pd.DataFrame):
    """
    Calcula 1 vetor 2D "médio" usando:
      Head_serfes = Cota_TOC_m - Serfes_medio

    Ajusta plano Head = aX + bY + c; fluxo = (-a, -b).
    Retorna dicionário com módulo e ângulo.
    """
    df_c = df_cotas.copy()
    df_xy2 = df_xy.copy()
    df_c["Poco"] = df_c["Poco"].astype(str).str.strip()
    df_xy2["Poco"] = df_xy2["Poco"].astype(str).str.strip()

    cotas = dict(zip(df_c["Poco"], df_c["Cota_TOC_m"]))

    rows = []
    for poco, df_res in resultados_por_poco.items():
        if poco not in cotas:
            continue
        if poco not in df_xy2["Poco"].values:
            continue
        serfes_med = _pegar_serfes_medio(df_res)
        head = float(cotas[poco]) - float(serfes_med)
        xy = df_xy2[df_xy2["Poco"] == poco].iloc[0]
        rows.append({"Poco": poco, "Head": head, "X_m": float(xy["X_m"]), "Y_m": float(xy["Y_m"])})

    if len(rows) < 3:
        return {"status": "INSUFICIENTE", "motivo": "Precisa >=3 poços com cota+xy+serfes."}

    dfh = pd.DataFrame(rows)
    X = dfh["X_m"].to_numpy(float)
    Y = dfh["Y_m"].to_numpy(float)
    H = dfh["Head"].to_numpy(float)

    A = pd.DataFrame({"X": X, "Y": Y, "1": 1.0}).to_numpy()
    coef, _, _, _ = pd.np.linalg.lstsq(A, H, rcond=None)  # compatível
    a, b, c = float(coef[0]), float(coef[1]), float(coef[2])

    vx, vy = -a, -b
    ang = math.degrees(math.atan2(vy, vx))
    if ang < 0:
        ang += 360.0
    mod = math.sqrt(a*a + b*b)

    return {
        "status": "OK",
        "n_pocos": int(len(dfh)),
        "dH_dx (a)": a,
        "dH_dy (b)": b,
        "modulo_gradiente": mod,
        "angulo_fluxo_deg": float(ang),
        "tabela_head_serfes": dfh
    }


def executar_analise_b(
    caminhos_pocos: list[str],
    inicio_por_poco: dict[str, pd.Timestamp],
    caminho_mare: str | None = None,
    caminho_cotas: str | None = None,
    caminho_kmz: str | None = None,
):
    if not caminhos_pocos:
        raise ValueError("caminhos_pocos está vazio. Selecione 1 ou mais arquivos de poço.")

    ler_poco_hora, nome_func_leitura = _resolver_funcao_leitura_poco()

    # MARÉ (opcional)
    df_mare = None
    if caminho_mare:
        df_mare = ler_mare_eventos(caminho_mare)

    # PROCESSAMENTO POÇO A POÇO
    resultados_por_poco: dict[str, pd.DataFrame] = {}
    lista_consolidado: list[pd.DataFrame] = []
    series_horarias_por_poco: dict[str, pd.DataFrame] = {}

    inicio_por_poco_norm = {str(k).strip(): v for k, v in inicio_por_poco.items()}

    for caminho in caminhos_pocos:
        poco = _nome_poco_do_caminho(caminho)

        if poco not in inicio_por_poco_norm:
            chaves = sorted(list(inicio_por_poco_norm.keys()))
            raise ValueError(
                f"Início não definido para o poço '{poco}'.\n"
                "A chave do dicionário inicio_por_poco deve ser exatamente o nome do arquivo sem extensão.\n"
                f"Chaves recebidas: {chaves}"
            )

        inicio = inicio_por_poco_norm[poco]

        df_h = ler_poco_hora(caminho, inicio)

        df_res = analisar_poco(
            df_h=df_h,
            df_mare=df_mare,
            nome_poco=poco
        )

        resultados_por_poco[poco] = df_res
        lista_consolidado.append(df_res)
        series_horarias_por_poco[poco] = df_h

    df_consolidado = pd.concat(lista_consolidado, ignore_index=True)

    # INSUMOS GRADIENTE (opcional)
    insumos_gradiente = None
    gradiente_2d = None

    if caminho_cotas or caminho_kmz:
        if not (caminho_cotas and caminho_kmz):
            insumos_gradiente = {
                "status": "INCOMPLETO",
                "motivo": "Para gradiente 2D é necessário fornecer caminho_cotas E caminho_kmz.",
                "caminho_cotas": bool(caminho_cotas),
                "caminho_kmz": bool(caminho_kmz),
            }
        else:
            df_cotas = ler_cotas(caminho_cotas)      # Poco, Cota_TOC_m
            df_xy = ler_kmz_pocos(caminho_kmz)       # Poco, X_m, Y_m

            insumos_gradiente = {
                "status": "OK",
                "cotas": df_cotas,
                "coords_xy": df_xy,
                "pocos_com_serie": sorted(list(series_horarias_por_poco.keys())),
            }

            # ---- GRADIENTE 2D: HORÁRIO + INVERSÃO
            df_grad_h = calcular_gradiente_2d_horario(
                series_horarias_por_poco=series_horarias_por_poco,
                df_cotas=df_cotas,
                df_xy=df_xy,
                min_pocos=3,
            )
            inv = detectar_inversao_gradiente(df_grad_h, limiar_deg=160.0, horas_consecutivas=2)

            # ---- GRADIENTE MÉDIO: SERFES (1 vetor)
            serfes_grad = _gradiente_2d_por_serfes(resultados_por_poco, df_cotas, df_xy)

            gradiente_2d = {
                "horario": df_grad_h,
                "inversao": inv,
                "serfes": serfes_grad,
            }

    return {
        "meta": {
            "leitura_pocos_usou": nome_func_leitura,
            "tem_mare": df_mare is not None,
            "tem_insumos_gradiente": insumos_gradiente is not None,
        },
        "por_poco": resultados_por_poco,
        "consolidado": df_consolidado,
        "insumos_gradiente": insumos_gradiente,
        "gradiente_2d": gradiente_2d,
    }
