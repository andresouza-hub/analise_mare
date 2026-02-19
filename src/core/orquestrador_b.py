# src/core/orquestrador_b.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable

import pandas as pd


# =========================
# Import helpers (robusto)
# =========================
def _import_first(candidates: list[tuple[str, str]]) -> Callable:
    last_err = None
    for mod, fn in candidates:
        try:
            m = __import__(mod, fromlist=[fn])
            return getattr(m, fn)
        except Exception as e:
            last_err = e
    raise ImportError(f"Não consegui importar nenhuma opção: {candidates}. Último erro: {last_err}")


# ---- Leitura poços (tenta variações de módulo e nome da função)
ler_poco_e_agregar_hora = _import_first([
    ("src.io.leitura_pocos", "ler_poco_e_agregar_hora"),
    ("src.io.leitura_pocos", "ler_pocos_e_agregar_hora"),
    ("src.core.leitura_pocos", "ler_poco_e_agregar_hora"),
    ("src.core.leitura_pocos", "ler_pocos_e_agregar_hora"),
])

# ---- Leitura maré
ler_mare_csv_eventos = _import_first([
    ("src.io.leitura_mare", "ler_mare_csv_eventos"),
    ("src.core.leitura_mare", "ler_mare_csv_eventos"),
])

# ---- Classificação alta/baixa
classificar_eventos_mare_alta_baixa = _import_first([
    ("src.core.mare_lag", "classificar_eventos_mare_alta_baixa"),
    ("src.core.marelag", "classificar_eventos_mare_alta_baixa"),
])

# ---- Núcleo análise por poço
analisar_poco = _import_first([
    ("src.core.analise_poco", "analisar_poco"),
    ("src.core.analisepoco", "analisar_poco"),
])

# ---- Cotas e KMZ (I/O)
ler_cotas = _import_first([
    ("src.io.leitura_cotas", "ler_cotas"),
    ("src.core.leitura_cotas", "ler_cotas"),
])

ler_kmz_pocos = _import_first([
    ("src.io.leitura_kmz", "ler_kmz_pocos"),
    ("src.core.leitura_kmz", "ler_kmz_pocos"),
])

# ---- Gradiente 2D (cálculo)
montar_head_long = _import_first([
    ("src.core.analise_gradiente", "montar_head_long"),
    ("src.core.gradiente", "montar_head_long"),
])
calcular_vetores_horarios = _import_first([
    ("src.core.analise_gradiente", "calcular_vetores_horarios"),
    ("src.core.gradiente", "calcular_vetores_horarios"),
])
resumir_gradiente_por_janelas = _import_first([
    ("src.core.analise_gradiente", "resumir_gradiente_por_janelas"),
    ("src.core.gradiente", "resumir_gradiente_por_janelas"),
])


def _norm_poco(s: str) -> str:
    s = str(s).strip().upper()
    s = re.sub(r"\s+", " ", s)
    return s


def _nome_poco_do_caminho(caminho: str | Path) -> str:
    p = Path(caminho)
    return _norm_poco(p.stem)


def executar_analise_b(
    *,
    caminhos_pocos: List[str],
    inicios_por_poco: Dict[str, pd.Timestamp],
    caminho_mare: Optional[str] = None,
    caminho_cotas: Optional[str] = None,
    caminho_kmz: Optional[str] = None,
) -> Dict[str, Any]:

    if not caminhos_pocos:
        raise ValueError("Nenhum caminho de poço informado em 'caminhos_pocos'.")

    # ---- Maré (se houver)
    df_mare = None
    df_mare_tipo = None
    if caminho_mare:
        df_mare = ler_mare_csv_eventos(caminho_mare)
        try:
            df_mare_tipo = classificar_eventos_mare_alta_baixa(df_mare)
        except Exception:
            df_mare_tipo = None

    # ---- Normaliza inícios
    inicios_norm = {_norm_poco(k): pd.to_datetime(v) for k, v in inicios_por_poco.items()}

    porpoco: Dict[str, pd.DataFrame] = {}
    janelas_list: List[pd.DataFrame] = []
    series_horarias: Dict[str, pd.DataFrame] = {}

    processados = 0
    pulados = 0
    pulados_por_motivo: Dict[str, str] = {}

    for caminho in caminhos_pocos:
        poco_id = _nome_poco_do_caminho(caminho)

        if poco_id not in inicios_norm:
            pulados += 1
            pulados_por_motivo[poco_id] = "Início do teste não informado."
            continue

        inicio = inicios_norm[poco_id]

        # ---- Série horária do poço
        try:
            df_h = ler_poco_e_agregar_hora(caminho, inicio)
        except Exception as e:
            pulados += 1
            pulados_por_motivo[poco_id] = f"Falha na leitura/agregação: {e}"
            continue

        if len(df_h) < 71:
            pulados += 1
            pulados_por_motivo[poco_id] = "Menos de 71 horas após o início."
            continue

        series_horarias[poco_id] = df_h

        # ---- Análise por poço
        out = analisar_poco(
            df_poco_h=df_h,
            nome_poco=poco_id,
            inicio_teste=inicio,
            df_mare=df_mare,
            df_mare_tipo=df_mare_tipo,
        )

        df_janelas = out["janelas"]
        porpoco[poco_id] = df_janelas
        janelas_list.append(df_janelas)
        processados += 1

    df_consolidado = pd.concat(janelas_list, ignore_index=True) if janelas_list else pd.DataFrame()

    # ---- Status de insumos de gradiente
    if caminho_cotas or caminho_kmz:
        if not (caminho_cotas and caminho_kmz):
            insumos_gradiente = {
                "status": "INCOMPLETO",
                "motivo": "Para gradiente 2D forneça 'caminho_cotas' e 'caminho_kmz'.",
                "caminho_cotas_recebido": bool(caminho_cotas),
                "caminho_kmz_recebido": bool(caminho_kmz),
            }
        else:
            insumos_gradiente = {
                "status": "OK",
                "caminho_cotas": str(caminho_cotas),
                "caminho_kmz": str(caminho_kmz),
            }
    else:
        insumos_gradiente = None

    # ---- Gradiente 2D
    gradiente2d = None
    if insumos_gradiente and insumos_gradiente.get("status") == "OK":
        try:
            df_cotas = ler_cotas(caminho_cotas)  # Poco, Cota_TOC_m
            df_xy = ler_kmz_pocos(caminho_kmz)   # Poco, X_m, Y_m

            df_cotas = df_cotas.copy()
            df_xy = df_xy.copy()
            df_cotas["Poco"] = df_cotas["Poco"].apply(_norm_poco)
            df_xy["Poco"] = df_xy["Poco"].apply(_norm_poco)

            cotas_por_poco = dict(zip(df_cotas["Poco"], df_cotas["Cota_TOC_m"]))

            pocos_series = set(series_horarias.keys())
            pocos_cota = set(df_cotas["Poco"].tolist())
            pocos_xy = set(df_xy["Poco"].tolist())

            inter = sorted(list(pocos_series & pocos_cota & pocos_xy))
            faltam_cota = sorted(list(pocos_series - pocos_cota))
            faltam_xy = sorted(list(pocos_series - pocos_xy))

            df_head_long = montar_head_long(series_horarias, cotas_por_poco)
            df_vetores = calcular_vetores_horarios(df_head_long, df_xy, min_pocos=3)

            df_janelas_grad = resumir_gradiente_por_janelas(
                df_vetores,
                janela_h=71,
                passo_h=24,
                limiar_deg=160.0,
                consec_h=2,
                min_vetores_na_janela=60,
            )

            gradiente2d = {
                "status": "OK",
                "meta": {
                    "pocos_com_series": len(pocos_series),
                    "pocos_com_cota": len(pocos_cota),
                    "pocos_com_xy": len(pocos_xy),
                    "pocos_intersecao": len(inter),
                    "pocos_intersecao_lista": inter,
                    "pocos_sem_cota": faltam_cota,
                    "pocos_sem_xy": faltam_xy,
                    "n_heads": int(len(df_head_long)),
                    "n_vetores_horarios": int(len(df_vetores)),
                    "n_janelas_grad": int(len(df_janelas_grad)) if isinstance(df_janelas_grad, pd.DataFrame) else 0,
                },
                "vetores_horarios": df_vetores,
                "janelas": df_janelas_grad,
                # para plotar o mapa no app (como no Colab)
                "pocos_xy": df_xy[df_xy["Poco"].isin(inter)].reset_index(drop=True),
            }

        except Exception as e:
            gradiente2d = {"status": "ERRO", "erro": str(e)}

    meta = {
        "processados": int(processados),
        "pulados": int(pulados),
        "motivos_pulados": pulados_por_motivo,
        "tem_mare": df_mare is not None,
        "tem_insumos_gradiente": insumos_gradiente is not None,
        "gradiente2d_calculado": isinstance(gradiente2d, dict) and gradiente2d.get("status") == "OK",
    }

    return {
        "meta": meta,
        "porpoco": porpoco,
        "consolidado": df_consolidado,
        "insumos_gradiente": insumos_gradiente,
        "gradiente2d": gradiente2d,
        # essenciais para os gráficos do Colab no Streamlit
        "series_horarias": series_horarias,
        "mare_eventos": df_mare,          # colunas: datetime, maré
        "mare_eventos_tipo": df_mare_tipo # opcional
    }
