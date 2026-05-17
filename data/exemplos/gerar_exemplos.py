#!/usr/bin/env python3
"""
Gerador de dados sintéticos realistas para o GAC Tidal Insight.

Cria um cenário fictício de monitoramento de aquífero costeiro em região de macromaré,
com 4 poços de comportamento hidráulico distinto:

- PM-EX-01 (poço próximo da linha de costa, fortemente acoplado à maré)
- PM-EX-02 (poço intermediário, moderadamente acoplado)
- PM-EX-03 (poço distante, fracamente acoplado)
- PM-EX-04 (poço em aquífero confinado, praticamente desacoplado)

A maré sintética é construída a partir das principais componentes harmônicas:
- M2 (semidiurna lunar, 12.42 h)     amplitude ~1.0 m
- S2 (semidiurna solar, 12.00 h)     amplitude ~0.3 m
- K1 (diurna luni-solar, 23.93 h)    amplitude ~0.2 m
- O1 (diurna lunar, 25.82 h)         amplitude ~0.15 m
A interação M2+S2 produz a modulação quinzenal (sizígia/quadratura) de período ~14.77 dias.

A resposta de cada poço inclui:
- Amortecimento exponencial da amplitude conforme distância da costa
- Atraso (lag) de fase proporcional à difusividade hidráulica do meio
- Ruído gaussiano para simular variabilidade natural
- Tendência leve (recarga/declínio sazonal)

Parâmetros podem ser ajustados via constantes ou argumentos de linha de comando.

Uso:
    python gerar_exemplos.py [--dias 30] [--saida ./]
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Configuração do cenário (cliente fictício "Terminal Aquaviário Beta")
# =============================================================================
# Data de início alinhada à lua nova de 27/05/2025 (idade lunar ≈ 0).
# Isso garante que o batimento das componentes M2/S2 coincida com o calendário lunar:
# sizígia da maré = sizígia astronômica → IMQ > 1 conforme esperado.
DATA_INICIO = datetime(2025, 5, 27, 0, 0, 0)
N_DIAS = 30                                    # duração total
DT_AMOSTRA_MIN = 30                            # passo de amostragem (min)

# Cenário geográfico fictício (zona costeira hipotética)
# Coordenadas em torno de Madre de Deus/BA mas deslocadas e renomeadas
POCOS = [
    # (nome, lon, lat, cota_TOC_m, dist_costa_m, perfil)
    ("PM-EX-01", -38.6201, -12.7398, 4.85, 30,   "forte"),
    ("PM-EX-02", -38.6195, -12.7392, 5.42, 90,   "moderada"),
    ("PM-EX-03", -38.6189, -12.7386, 6.18, 180,  "fraca"),
    ("PM-EX-04", -38.6183, -12.7380, 7.95, 320,  "confinado"),
]

# Parâmetros hidráulicos por perfil de resposta
PERFIS = {
    # nome:      (atenuação relativa,  lag em horas,  nível base m,  ruído σ)
    "forte":      (0.55, 1.5,  3.20, 0.012),
    "moderada":   (0.28, 4.5,  3.95, 0.010),
    "fraca":      (0.08, 9.0,  4.70, 0.008),
    "confinado":  (0.02, 16.0, 5.85, 0.005),
}


# =============================================================================
# Construção da maré sintética
# =============================================================================
def gerar_serie_mare_horaria(dt_horario: pd.DatetimeIndex) -> np.ndarray:
    """Maré sintética horária = nível médio + componentes harmônicas + modulação.

    Período em horas desde o início. Componentes principais:
    M2 (12.42h), S2 (12.00h), K1 (23.93h), O1 (25.82h).
    A interferência M2-S2 gera o ciclo quinzenal natural (≈14.77 dias).
    """
    t_h = (dt_horario - dt_horario[0]).total_seconds().values / 3600.0

    nivel_medio = 1.45  # nível médio do mar fictício (m)
    # Fases zeradas para M2 e S2 no instante inicial (lua nova) — assim o batimento
    # quinzenal (período ~14.77 dias) alinha-se ao calendário lunar.
    M2 = 1.00 * np.cos(2 * np.pi * t_h / 12.4206)
    S2 = 0.45 * np.cos(2 * np.pi * t_h / 12.0000)
    K1 = 0.20 * np.cos(2 * np.pi * t_h / 23.9345 + 1.00)
    O1 = 0.15 * np.cos(2 * np.pi * t_h / 25.8193 + 0.60)

    return nivel_medio + M2 + S2 + K1 + O1


def extrair_eventos_preamar_baixamar(
    dt_horario: pd.DatetimeIndex,
    mare_horaria: np.ndarray,
) -> pd.DataFrame:
    """A partir da série horária, identifica preamares e baixamares (extremos locais).

    Retorna DataFrame no formato de tábua da Marinha do Brasil:
    Mês, Dia (num), Dia (semana), Hora, Altura maré, Ano.
    """
    # Detecção simples: ponto i é máximo (mín) se maior (menor) que vizinhos
    eventos_idx = []
    for i in range(1, len(mare_horaria) - 1):
        if mare_horaria[i] > mare_horaria[i - 1] and mare_horaria[i] > mare_horaria[i + 1]:
            eventos_idx.append(i)
        elif mare_horaria[i] < mare_horaria[i - 1] and mare_horaria[i] < mare_horaria[i + 1]:
            eventos_idx.append(i)

    dias_semana = {0: "segunda", 1: "terça", 2: "quarta", 3: "quinta",
                   4: "sexta", 5: "sábado", 6: "domingo"}
    meses = {1: "Janeiro", 2: "Fevereiro", 3: "Março", 4: "Abril",
             5: "Maio", 6: "Junho", 7: "Julho", 8: "Agosto",
             9: "Setembro", 10: "Outubro", 11: "Novembro", 12: "Dezembro"}

    linhas = []
    for i in eventos_idx:
        ts = dt_horario[i]
        linhas.append({
            "Mês": meses[ts.month],
            "Dia (num)": ts.day,
            "Dia (semana)": dias_semana[ts.weekday()],
            "Hora": ts.strftime("%H:%M"),
            "Altura maré": f"{mare_horaria[i]:.2f}".replace(".", ","),
            "Ano": ts.year,
        })
    return pd.DataFrame(linhas)


# =============================================================================
# Resposta do poço à maré (modelo simplificado de aquífero costeiro)
# =============================================================================
def gerar_serie_poco(
    dt_amostra: pd.DatetimeIndex,
    mare_horaria_completa: np.ndarray,
    dt_horario: pd.DatetimeIndex,
    nivel_base: float,
    atenuacao: float,
    lag_h: float,
    ruido_sigma: float,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Gera série de nível do poço como resposta atenuada e defasada da maré.

    Modelo: h_poço(t) = h_base + α · (h_maré(t - Δt) - h_médio) + tendência + ruído
    """
    rng = np.random.default_rng(seed)

    # Interpola a maré horária para o passo de amostragem do poço, com lag
    t_h_amostra = (dt_amostra - dt_horario[0]).total_seconds().values / 3600.0
    t_h_amostra_com_lag = t_h_amostra - lag_h

    t_h_mare = (dt_horario - dt_horario[0]).total_seconds().values / 3600.0
    mare_interp = np.interp(t_h_amostra_com_lag, t_h_mare, mare_horaria_completa)

    # Componente da maré atenuada (desviada do nível médio)
    mare_medio = float(np.mean(mare_horaria_completa))
    componente_mare = atenuacao * (mare_interp - mare_medio)

    # Tendência sazonal suave (recarga/declínio leve no período)
    tendencia = -0.0002 * t_h_amostra  # mm por hora

    # Ruído gaussiano
    ruido = rng.normal(0, ruido_sigma, size=len(dt_amostra))

    nivel = nivel_base + componente_mare + tendencia + ruido
    # Temperatura: ciclo diurno + ruído
    temp = 25.5 + 1.8 * np.sin(2 * np.pi * t_h_amostra / 24.0) + rng.normal(0, 0.3, len(dt_amostra))

    return nivel, temp


# =============================================================================
# Escrita dos arquivos
# =============================================================================
def escrever_poco_levellogger(
    caminho: Path,
    nome_poco: str,
    location: str,
    dt: pd.DatetimeIndex,
    nivel: np.ndarray,
    temp: np.ndarray,
) -> None:
    """Escreve CSV no formato Solinst Levellogger (compatível com app)."""
    df = pd.DataFrame({
        "Date": [t.strftime("%d/%m/%Y") for t in dt],
        "Time": [t.strftime("%H:%M:%S") for t in dt],
        "ms": [0] * len(dt),
        "LEVEL": [f"{v:.4f}".replace(".", ",") for v in nivel],
        "TEMPERATURE": [f"{v:.2f}".replace(".", ",") for v in temp],
    })

    with open(caminho, "w", encoding="latin1", newline="") as f:
        f.write("Serial_number:;;;;\n")
        f.write("2100000;;;;\n")
        f.write("Project ID:;;;;\n")
        f.write(f"{nome_poco};;;;\n")
        f.write("Location:;;;;\n")
        f.write(f"{location};;;;\n")
        f.write("LEVEL;;;;\n")
        f.write("UNIT: m;;;;\n")
        f.write("Offset: 0,000000 m;;;;\n")
        f.write("TEMPERATURE;;;;\n")
        f.write("UNIT: C;;;;\n")
        df.to_csv(f, sep=";", index=False, encoding="latin1")


def escrever_poco_simples(caminho: Path, dt: pd.DatetimeIndex, nivel: np.ndarray) -> None:
    """Escreve CSV simples (datetime, nivel) — formato amigável para testes."""
    df = pd.DataFrame({
        "datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
        "nivel": np.round(nivel, 4),
    })
    df.to_csv(caminho, index=False, encoding="utf-8")


def escrever_kmz(caminho: Path, pocos: list) -> None:
    """Escreve KMZ com Placemarks de cada poço (lon, lat)."""
    import zipfile

    kml_template = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
<name>Pocos GAC Exemplo</name>
{placemarks}
</Document>
</kml>
"""
    placemark_template = """<Placemark>
  <name>{nome}</name>
  <Point><coordinates>{lon},{lat},0</coordinates></Point>
</Placemark>"""

    placemarks = "\n".join(
        placemark_template.format(nome=p[0], lon=p[1], lat=p[2]) for p in pocos
    )
    kml = kml_template.format(placemarks=placemarks)

    with zipfile.ZipFile(caminho, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("doc.kml", kml)


def escrever_cotas(caminho: Path, pocos: list) -> None:
    """Escreve XLSX com Poco e Cota_TOC_m."""
    df = pd.DataFrame({
        "Poco": [p[0] for p in pocos],
        "Cota_TOC_m": [p[3] for p in pocos],
    })
    df.to_excel(caminho, index=False)


def escrever_mare_tabua(caminho: Path, df_eventos: pd.DataFrame) -> None:
    """Escreve maré em formato tábua (compatível com app)."""
    df_eventos.to_csv(caminho, sep=";", index=False, encoding="utf-8")


def escrever_mare_simples(caminho: Path, dt: pd.DatetimeIndex, mare: np.ndarray) -> None:
    """Escreve maré em formato simples (datetime, mare)."""
    df = pd.DataFrame({
        "datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
        "mare": np.round(mare, 3),
    })
    df.to_csv(caminho, index=False, encoding="utf-8")


# =============================================================================
# Pipeline principal
# =============================================================================
def gerar(saida: Path, dias: int = N_DIAS) -> None:
    saida.mkdir(parents=True, exist_ok=True)

    # --- Eixos temporais
    fim = DATA_INICIO + timedelta(days=dias)

    # Maré horária (resolução fina para extração de extremos)
    dt_h = pd.date_range(DATA_INICIO, fim, freq="h", inclusive="left")
    mare_h = gerar_serie_mare_horaria(dt_h)

    # Eventos de preamar/baixamar
    df_mare_eventos = extrair_eventos_preamar_baixamar(dt_h, mare_h)

    # --- Poços (cada um com seu perfil)
    print(f"Gerando {dias} dias de dados sintéticos em: {saida}")
    print(f"Período: {DATA_INICIO:%d/%m/%Y} a {fim:%d/%m/%Y}")
    print()

    dt_amostra = pd.date_range(DATA_INICIO, fim, freq=f"{DT_AMOSTRA_MIN}min", inclusive="left")

    for i, p in enumerate(POCOS):
        nome, lon, lat, cota, dist, perfil = p
        atenuacao, lag, nivel_base, ruido = PERFIS[perfil]
        nivel, temp = gerar_serie_poco(
            dt_amostra, mare_h, dt_h,
            nivel_base=nivel_base,
            atenuacao=atenuacao,
            lag_h=lag,
            ruido_sigma=ruido,
            seed=42 + i,
        )

        # Levellogger
        out_ll = saida / f"{nome}.csv"
        escrever_poco_levellogger(out_ll, nome, f"AREA-COSTEIRA-{i+1:02d}", dt_amostra, nivel, temp)
        print(f"  ✓ {out_ll.name}  ({len(dt_amostra)} registros, perfil '{perfil}')")

        # CSV simples (mesmo poço, formato alternativo)
        out_simples = saida / f"{nome}_simples.csv"
        escrever_poco_simples(out_simples, dt_amostra, nivel)

    # --- Maré (dois formatos)
    out_mare_tabua = saida / "mare_tabua.csv"
    escrever_mare_tabua(out_mare_tabua, df_mare_eventos)
    print(f"\n  ✓ {out_mare_tabua.name}  ({len(df_mare_eventos)} eventos preamar/baixamar)")

    out_mare_simples = saida / "mare_simples.csv"
    escrever_mare_simples(out_mare_simples, dt_h, mare_h)
    print(f"  ✓ {out_mare_simples.name}  ({len(dt_h)} pontos horários)")

    # --- KMZ e cotas
    out_kmz = saida / "pocos.kmz"
    escrever_kmz(out_kmz, POCOS)
    print(f"\n  ✓ {out_kmz.name}  ({len(POCOS)} poços)")

    out_cotas = saida / "cotas.xlsx"
    escrever_cotas(out_cotas, POCOS)
    print(f"  ✓ {out_cotas.name}")

    # --- Resumo de início por poço (para o usuário copiar)
    print("\n📋 Para usar no app, configure os timestamps de início:")
    print(f"   {DATA_INICIO:%d/%m/%Y %H:%M}  para todos os 4 poços")

    print("\n✅ Cenário sintético gerado com sucesso.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gerador de dados sintéticos para GAC Tidal Insight")
    parser.add_argument("--dias", type=int, default=N_DIAS, help=f"Duração em dias (padrão: {N_DIAS})")
    parser.add_argument("--saida", type=Path, default=Path(__file__).parent,
                        help="Diretório de saída (padrão: ao lado deste script)")
    args = parser.parse_args()
    gerar(args.saida, dias=args.dias)


if __name__ == "__main__":
    main()
