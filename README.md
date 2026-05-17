# GAC Tidal Insight

**Análise de Aquíferos Costeiros com Influência de Maré**

![Streamlit](https://img.shields.io/badge/Streamlit-1.36.0-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python&logoColor=white)

## Descrição

O **GAC Tidal Insight** é uma plataforma analítica para avaliação integrada da
dinâmica hidráulica de aquíferos rasos costeiros, voltada a profissionais de
Gerenciamento de Áreas Contaminadas (GAC). O sistema permite quantificar
conectividade hidráulica, resposta do aquífero às variações de maré,
estabilidade direcional e inversões do fluxo subterrâneo, apoiando interpretações
hidrogeológicas e a tomada de decisão.

## Funcionalidades

- **Método de Serfes (1991)** — estimativa de nível médio com janela móvel de 71 h
- **Análise espectral (FFT)** — decomposição em bandas semidiurna (~12 h) e diurna (~24 h)
- **Defasagem maré–poço (lag)** — quantificação da resposta hidráulica
- **Gradiente hidráulico 2D** — ajuste de plano por mínimos quadrados, direção e módulo
- **Modulação quinzenal** — análise sizígia × quadratura com Índice IMQ
- **Interpretação técnica automática** — relatório consolidado por poço

## Formatos de entrada aceitos

A aplicação aceita dois formatos para cada tipo de arquivo, com **detecção automática**.

### Poços (nível d'água)

**Formato A — Levellogger (Solinst):**
Arquivo do datalogger com 11 linhas de metadados (Serial_number, Project ID, etc.)
seguidas do cabeçalho `Date;Time;ms;LEVEL;TEMPERATURE`. Separador `;`, decimal vírgula,
encoding latin1.

**Formato B — CSV simples (recomendado para testes):**
Cabeçalho `datetime,nivel` na primeira linha, separador `,` (ou `;`), encoding UTF-8.
Aceita também `data,hora,nivel` ou variações case-insensitive.

### Maré

**Formato A — Tábua da Marinha do Brasil:**
Cabeçalho `Mês;Dia (num);Dia (semana);Hora;Altura maré;Ano`, separador `;`.
Cada linha = um evento (preamar ou baixamar).

**Formato B — CSV simples:**
Cabeçalho `datetime,mare`, série contínua (horária ou outra resolução).

### Gradiente 2D (opcional)

- **Cotas:** XLSX com colunas `Poco;Cota_TOC_m`
- **Localização:** KMZ com Placemarks/Points nomeados conforme os poços

## Instalação local

```bash
git clone https://github.com/SEU-USUARIO/gac-tidal-insight.git
cd gac-tidal-insight
python -m venv .venv
source .venv/bin/activate          # Linux/Mac
# .venv\Scripts\activate           # Windows
pip install -r requirements.txt
streamlit run app.py
```

## Dados de exemplo

A pasta `data/exemplos/` contém um cenário sintético completo (4 poços, 30 dias,
maré com modulação quinzenal alinhada ao calendário lunar real, KMZ e cotas).
Consulte `data/exemplos/README.md` para a descrição completa do cenário e dos
resultados esperados.

Para regenerar os exemplos (ou criar variações):

```bash
cd data/exemplos
python gerar_exemplos.py --dias 30
```

## Estrutura do projeto

```
gac-tidal-insight/
├── app.py                          # Interface Streamlit (entrypoint)
├── requirements.txt                # Dependências pinadas
├── README.md
├── DEPLOY_GUIDE.md                 # Guia de deploy no Streamlit Cloud
├── src/
│   ├── core/
│   │   ├── orquestrador_b.py       # Pipeline de execução
│   │   ├── analise_poco.py         # Análise por poço (Serfes + FFT + lag)
│   │   ├── mare_lag.py             # Cálculo de defasagem
│   │   ├── serfes.py               # Método de Serfes (1991)
│   │   ├── gradiente_2d.py         # Gradiente hidráulico bidimensional
│   │   ├── analise_gradiente.py    # Janelas e vetores horários
│   │   ├── amplitudes.py           # Amplitudes p2p e RMS
│   │   └── lua.py                  # Fases lunares
│   ├── io/
│   │   ├── leitura_pocos.py        # Levellogger + CSV simples
│   │   ├── leitura_mare.py         # Tábua Marinha + CSV simples
│   │   ├── leitura_cotas.py        # Excel de cotas
│   │   └── leitura_kmz.py          # KMZ → coordenadas UTM
│   └── viz/
│       └── mapas_gradiente.py      # Mapa do gradiente 2D
└── data/
    └── exemplos/
        ├── gerar_exemplos.py       # Gerador paramétrico
        ├── PM-EX-0[1-4].csv        # 4 poços formato Levellogger
        ├── PM-EX-0[1-4]_simples.csv# Mesmos poços, formato simples
        ├── mare_tabua.csv          # Maré formato tábua
        ├── mare_simples.csv        # Maré formato simples
        ├── pocos.kmz               # Coordenadas
        ├── cotas.xlsx              # Cotas TOC
        └── README.md               # Descrição do cenário
```

## Deploy no Streamlit Cloud

Consulte `DEPLOY_GUIDE.md` para o passo-a-passo completo.

## Autor

**André Souza** — Especialista em Gerenciamento de Áreas Contaminadas (GAC)
