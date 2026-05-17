# Guia de Deploy — GAC Tidal Insight

Este documento descreve o deploy no **Streamlit Community Cloud**, o ambiente
gratuito recomendado para uso da plataforma.

## Pré-requisitos

- Conta no GitHub
- Conta no [Streamlit Community Cloud](https://share.streamlit.io)
- Repositório com o código (público ou privado)

## Estrutura mínima do repositório

```
seu-repo/
├── app.py                   # Entrypoint da aplicação
├── requirements.txt         # Dependências pinadas
├── src/                     # Pacote de análise
└── data/
    └── exemplos/            # Dados sintéticos (opcional, mas recomendado)
```

## requirements.txt

As versões devem ficar **pinadas** para evitar regressões silenciosas em deploys futuros.
O arquivo atual já vem pinado:

```
streamlit==1.36.0
altair==4.2.2
pandas>=2.0.0,<3.0.0
numpy>=1.24.0,<2.0.0
matplotlib>=3.7.0,<4.0.0
scipy>=1.10.0,<2.0.0
seaborn>=0.12.0,<1.0.0
openpyxl>=3.1.0,<4.0.0
pyproj>=3.6.0,<4.0.0
```

**Importante:** `altair==4.2.2` está pinado em versão exata porque o app foi
construído sobre a API 4.x. O altair 5.x tem mudanças incompatíveis.

## Passo a passo

### 1. Subir o código para o GitHub

```bash
cd gac-tidal-insight
git init
git add .
git commit -m "Versão funcional do GAC Tidal Insight"
git branch -M main
git remote add origin https://github.com/SEU-USUARIO/gac-tidal-insight.git
git push -u origin main
```

### 2. Conectar ao Streamlit Cloud

1. Acesse [share.streamlit.io](https://share.streamlit.io)
2. Clique em **New app**
3. Selecione seu repositório
4. **Branch:** `main`
5. **Main file path:** `app.py`
6. (Opcional) Defina uma URL customizada
7. Clique em **Deploy**

O Streamlit Cloud vai:
- Clonar o repo
- Instalar dependências do `requirements.txt`
- Iniciar o app

A primeira execução leva alguns minutos. Deploys subsequentes (após push) são mais rápidos.

### 3. Verificações após deploy

- [ ] App carrega sem erros na URL pública
- [ ] Expander "📦 Baixar dados de exemplo" mostra os botões de download
- [ ] Upload de poços funciona (testar com `PM-EX-01.csv` baixado do próprio app)
- [ ] Análise executa após clicar em "🚀 Executar Análise"
- [ ] Aba "Interpretação" mostra texto técnico não-vazio
- [ ] Aba "Gradiente 2D" exibe ângulo e mapa quando KMZ + cotas estão presentes

## Atualizações

Para atualizar o app deployado:

```bash
# Faça as mudanças localmente
git add .
git commit -m "Descrição da mudança"
git push origin main
```

O Streamlit Cloud detecta o push e faz redeploy automaticamente.

## Configurações avançadas (opcional)

### `.streamlit/config.toml`

Para customizar tema ou comportamento, crie `.streamlit/config.toml`:

```toml
[theme]
primaryColor="#1E88E5"
backgroundColor="#FFFFFF"
secondaryBackgroundColor="#F0F2F6"
textColor="#262730"

[server]
maxUploadSize=200
```

### Limites do plano gratuito

- 1 GB RAM
- Hibernação após ~7 dias sem acesso (acorda no primeiro acesso, ~30 s)
- Arquivos enviados: limitado pelo `maxUploadSize` (padrão 200 MB)

Para uso intenso, considere o Streamlit for Teams ou auto-hospedagem.

## Solução de problemas comuns

### App falha ao iniciar

Cheque os logs no painel do Streamlit Cloud (botão "Manage app" → "Logs"):
- `ModuleNotFoundError` → dependência faltando no `requirements.txt`
- `ImportError` no nível de `src/...` → estrutura de pastas quebrada (verifique `__init__.py`)
- `SyntaxError` → erro no código que passou pelo git

### Análise não executa

- Confira que os arquivos de poço estão no formato Levellogger OU no CSV simples (datetime,nivel)
- Verifique se o timestamp de início é compatível com as datas dos arquivos
- Para análise quinzenal: precisa de pelo menos 20 dias de dados

### Análise quinzenal não aparece

A análise PIG quinzenal requer:
- Maré fornecida
- Série de poços com duração ≥ 20 dias
- Checkbox "Análise PIG Quinzenal" marcada no painel lateral

Se algum destes falta, a aba Interpretação mostra apenas as seções aplicáveis.
