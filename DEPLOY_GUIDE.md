# Deploy no Streamlit Cloud - GAC Tidal Insight

## Pré-requisitos

1. Conta no GitHub (https://github.com)
2. Conta no Streamlit Cloud (https://streamlit.io/cloud)

## Passo a Passo

### 1. Criar Repositório no GitHub

1. Acesse https://github.com/new
2. Nome do repositório: `gac-tidal-insight`
3. Descrição: "Análise de aquíferos costeiros com influência de maré"
4. Visibilidade: Público (necessário para Streamlit Cloud gratuito)
5. Não inicialize com README

### 2. Preparar o Projeto

O projeto já está pronto! Mas precisamos criar um arquivo `.gitkeep` na pasta `data` para que ela seja incluída no repositório:

```bash
cd /workspace/analise_mare_project/Agente_GAC
touch data/.gitkeep
touch data/exemplos/.gitkeep
touch data/tmp/.gitkeep
```

### 3. Estrutura do Repositório

```
gac-tidal-insight/
├── app.py                 # Aplicação principal
├── requirements.txt      # Dependências Python
├── .gitignore           # Arquivos ignorados
├── src/
│   ├── core/            # Módulos de análise
│   ├── io/              # Leitura de dados
│   ├── viz/             # Visualizações
│   ├── analysis/        # Processamento
│   └── export/          # Exportação
├── data/
│   └── exemplos/        # Arquivos de exemplo (para download)
└── README.md
```

### 4. Fazer Upload para GitHub

#### Opção A: Via Git (linha de comando)

```bash
cd /workspace/analise_mare_project/Agente_GAC

# Inicializar repositório (se ainda não existir)
git init
git add .
git commit -m "Initial commit - GAC Tidal Insight v2.0"

# Adicionar origin (substitua pelo URL do seu repositório)
git remote add origin https://github.com/SEU-USUARIO/gac-tidal-insight.git
git branch -M main
git push -u origin main
```

#### Opção B: Via GitHub Desktop

1. Clone o repositório criado
2. Copie todos os arquivos do projeto para a pasta do repositório
3. Commit e Push

#### Opção C: Via GitHub Web

1. Acesse seu repositório vazio
2. Clique em "uploading an existing file"
3. Arraste todos os arquivos do projeto
4. Commit

### 5. Deploy no Streamlit Cloud

1. Acesse https://share.streamlit.io
2. Clique em "New App"
3. Configure:
   - **Repository**: `SEU-USUARIO/gac-tidal-insight`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Clique em "Deploy!"

### 6. URL da Aplicação

Após o deploy, seu app estará disponível em:
```
https://SEU-USUARIO-gac-tidal-insight.streamlit.app
```

---

## Solução de Problemas

### Erro: "ModuleNotFoundError"

Verifique se o `requirements.txt` está no raiz do projeto com todas as dependências:

```
streamlit==1.29.0
altair>=4.2.2
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
scipy>=1.10.0
seaborn>=0.12.0
openpyxl>=3.1.0
```

### Erro: "File not found"

O Streamlit Cloud executa de um diretório virtual. Use caminhos relativos:
- ✅ `src/core/orquestrador_b.py`
- ❌ `/workspace/analise_mare_project/Agente_GAC/src/core/orquestrador_b.py`

### Erro: "Secrets not configured"

Para variáveis de ambiente, use o Streamlit Secrets:
1. No Streamlit Cloud, vá em Settings → Secrets
2. Adicione suas variáveis em formato `KEY = "value"`

---

## Atualizações Posteriores

Após fazer alterações locally:

```bash
cd /workspace/analise_mare_project/Agente_GAC
git add .
git commit -m "Descrição da atualização"
git push
```

O Streamlit Cloud detecta alterações automaticamente e refaz o deploy.