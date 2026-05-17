# GAC Tidal Insight

**Análise de Aquíferos Costeiros com Influência de Maré**

![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

## Descrição

O **GAC Tidal Insight** é uma ferramenta de análise hidrológica para aquíferos costeiros, desenvolvida especificamente para especialistas em Gerenciamento de Áreas Contaminadas (GAC). A aplicação permite:

- **Análise de poços de monitoramento** com dados de nível piezométrico
- **Correlação com dados maregráficos** em tempo real
- **Cálculo do atraso de maré** (tidal lag) entre o mar e os poços
- **Análise espectral FFT** para identificação de componentes de maré
- **Cálculo de gradiente hidráulico 2D** com visualização de mapas
- **Estimativa de direção de fluxo** subterráneo

## Metodologia

A aplicação implementa metodologias estabelecidas na literatura técnica:

1. **Método Serfes (1991)** - Estimativa de nível médio d'água
2. **Análise FFT** - Decomposição espectral (semi-diurnal ~12.42h, diurna ~24h)
3. **Análise de fase/amplitude** - Determinação do tidal lag
4. **Modulação fortnightly** - Identificação de sizígia vs quadratura

## Instalação

### Requisitos

- Python 3.10+
- pip ou conda

### Passos

```bash
# Clonar ou baixar o projeto
git clone https://github.com/SEU-USUARIO/gac-tidal-insight.git
cd gac-tidal-insight

# Criar ambiente virtual (recomendado)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou .venv\Scripts\activate  # Windows

# Instalar dependências
pip install -r requirements.txt

# Executar
streamlit run app.py
```

## Uso

### 1. Upload de Dados

Carregue os arquivos no formato esperado:

| Tipo | Formato | Colunas |
|------|---------|---------|
| **Poços** | CSV | Date, Time, ms, LEVEL, TEMPERATURE |
| **Maré** | CSV | Mês, Dia (num), Hora, Altura maré, Ano |
| **Cotas** | XLSX | Poco, Cota_TOC_m |

### 2. Configurações

No painel lateral:
- Período de referência para maré
- Intervalo FFT desejado
- Parâmetros de filtragem

### 3. Resultados

A aplicação gera:
- **Dashboard** com métricas consolidadas
- **Análise por poço** com gráficos de nível
- **Mapa de gradiente 2D** com vetores de fluxo
- **Relatório em Excel** para download

## Estrutura do Projeto

```
gac-tidal-insight/
├── app.py                    # Interface Streamlit
├── requirements.txt          # Dependências
├── src/
│   ├── core/
│   │   ├── orquestrador_b.py  # Orquestrador de análise
│   │   ├── analise_poco.py    # Análise individual
│   │   ├── mare_lag.py       # Cálculo de tidal lag
│   │   ├── serfes.py         # Método Serfes
│   │   └── gradiente_2d.py   # Gradiente 2D
│   ├── io/
│   │   ├── leitura_pocos.py   # Leitura CSV poços
│   │   ├── leitura_mare.py   # Leitura CSV maré
│   │   └── leitura_cotas.py  # Leitura XLSX cotas
│   ├── viz/
│   │   └── mapas_gradiente.py # Visualizações
│   └── export/
│       └── exportador_excel.py # Exportação
└── data/
    └── exemplos/            # Arquivos de exemplo
```

## Dados de Exemplo

A pasta `data/exemplos/` contém arquivos de exemplo para teste:

- `poco_exemplo.csv` - 20 dias de dados (30 min interval)
- `mare_exemplo.csv` - 5 dias de eventos maregráficos
- `cotas_exemplo.xlsx` - 4 poços com coordenadas

## Deploy

O projeto pode ser implantado no **Streamlit Cloud** gratuitamente:

1. Crie um repositório no GitHub
2. Faça push do código
3. Acesse https://share.streamlit.io
4. Selecione o repositório e faça deploy

Consulte `DEPLOY_GUIDE.md` para instruções detalhadas.

## Autor

**André Souza**
Especialista em Gerenciamento de Áreas Contaminadas (GAC)

## Licença

MIT License - Uso livre para fins educacionais e comerciais.

## Citação

Se você usar esta ferramenta em pesquisa, por favor cite:

```
Souza, A. (2025). GAC Tidal Insight - Análise de Aquíferos Costeiros.
https://github.com/SEU-USUARIO/gac-tidal-insight
```