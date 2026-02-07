import sys
from pathlib import Path

# adiciona a raiz do projeto ao PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from datetime import datetime
from src.core.orquestrador_b import executar_analise_b

# ============================
# CAMINHOS DOS DADOS
# ============================

arquivos_pocos = [
    r"C:\Users\andresouza\Agente_GAC\dados\pocos\PM-330.csv",
    r"C:\Users\andresouza\Agente_GAC\dados\pocos\PM-331.csv",
]

arquivo_mare = r"C:\Users\andresouza\Agente_GAC\dados\mare\mare_eventos.csv"

# ============================
# INÍCIO DO TESTE POR POÇO
# ============================

inicio_por_poco = {
    "PM-330": datetime(2023, 9, 16, 16, 0),
    "PM-331": datetime(2023, 9, 16, 16, 0),
}

# ============================
# EXECUÇÃO
# ============================

resultado = executar_analise_b(
    caminhos_pocos=arquivos_pocos,
    inicio_por_poco=inicio_por_poco,
    caminho_mare=arquivo_mare,
    caminho_cotas=None,
    caminho_kmz=None
)

# ============================
# INSPEÇÃO RÁPIDA
# ============================

print("\nPOÇOS PROCESSADOS:")
print(resultado["por_poco"].keys())

print("\nCOLUNAS DO CONSOLIDADO:")
print(resultado["consolidado"].columns)

print("\nAMOSTRA DO CONSOLIDADO:")
print(resultado["consolidado"].head())
