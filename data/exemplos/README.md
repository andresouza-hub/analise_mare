# Dados de Exemplo — GAC Tidal Insight

Este diretório contém **dados sintéticos** para teste e demonstração do aplicativo.
Os dados são fictícios mas estatisticamente realistas, baseados na dinâmica de
aquíferos costeiros sob regime de macromaré (amplitude > 2 m).

## Cenário simulado

Campanha hipotética de **30 dias** (27/05/2025 a 26/06/2025) em "Terminal Aquaviário Beta",
zona costeira fictícia em latitude ~12,7° S. A data de início coincide com uma **lua nova real**,
de modo que o batimento M2-S2 (modulação quinzenal) das marés alinha-se ao calendário lunar.

São 4 poços de monitoramento dispostos em transeto perpendicular à linha de costa,
com perfis hidráulicos distintos para exercitar todas as funcionalidades do app:

| Poço      | Distância da costa | Perfil hidráulico | Acoplamento esperado |
|-----------|--------------------|--------------------|----------------------|
| PM-EX-01  | 30 m               | forte              | R alto, lag curto    |
| PM-EX-02  | 90 m               | moderada           | R intermediário      |
| PM-EX-03  | 180 m              | fraca              | R baixo, lag longo   |
| PM-EX-04  | 320 m              | confinado          | R≈0 (desacoplado)    |

## Resultados esperados

Ao processar este cenário no app:

- **Amplitude pico-a-pico**: ~1,8 m em sizígia (PM-EX-01) até ~0,07 m (PM-EX-04)
- **IMQ (Amp_sizígia/Amp_quadratura)**: 1,5 – 1,9 em todos os poços
- **Lag**: ~1,5 h (PM-EX-01) até ~16 h (PM-EX-04)
- **Dominância espectral**: semidiurna (~12 h) em todos os poços
- **Gradiente 2D**: direção dominante consistente, módulo ~0,003 m/m

## Arquivos disponíveis

### Poços (nível d'água)
- `PM-EX-01.csv` a `PM-EX-04.csv` — **formato Levellogger (Solinst)**, separador `;`,
  decimal vírgula, encoding latin1, 11 linhas de cabeçalho de metadados
- `PM-EX-01_simples.csv` a `PM-EX-04_simples.csv` — **formato CSV simples**
  (`datetime,nivel`), separador `,`, decimal ponto, UTF-8

Ambos os formatos são aceitos pelo app (detecção automática). O simples é mais fácil
de produzir em planilhas e editores de texto.

### Maré
- `mare_tabua.csv` — **formato tábua da Marinha**, colunas `Mês;Dia (num);Dia (semana);Hora;Altura maré;Ano`,
  só preamares e baixamares (115 eventos em 30 dias)
- `mare_simples.csv` — **formato CSV simples** (`datetime,mare`), série horária completa (720 pontos)

### Gradiente 2D
- `pocos.kmz` — coordenadas dos 4 poços (Placemark/Point) em WGS84
- `cotas.xlsx` — cotas topográficas do TOC, colunas `Poco;Cota_TOC_m`

## Como usar no app

1. Abra o aplicativo no Streamlit Cloud (ou local)
2. No menu lateral, faça upload dos 4 arquivos de poço (`PM-EX-0[1-4].csv`)
3. Para cada poço, configure o início como **27/05/2025 00:00**
4. Faça upload de `mare_tabua.csv` na seção de maré
5. Opcional (gradiente 2D): upload de `cotas.xlsx` e `pocos.kmz`
6. Clique em **🚀 Executar Análise**

## Regenerar os dados

Para criar uma nova versão dos exemplos (ex.: com duração diferente):

```bash
cd data/exemplos
python gerar_exemplos.py --dias 45
```

Parâmetros aceitos:
- `--dias N`     duração da campanha em dias (padrão: 30)
- `--saida PATH` diretório de saída (padrão: ao lado do script)

Para customizar perfis dos poços, edite as constantes `POCOS` e `PERFIS` no início
de `gerar_exemplos.py`.

## Modelo físico

A maré sintética é a soma das componentes harmônicas principais:

| Componente | Período (h) | Amplitude (m) | Origem               |
|------------|-------------|---------------|----------------------|
| M2         | 12,4206     | 1,00          | Semidiurna lunar     |
| S2         | 12,0000     | 0,45          | Semidiurna solar     |
| K1         | 23,9345     | 0,20          | Diurna luni-solar    |
| O1         | 25,8193     | 0,15          | Diurna lunar         |

A interferência entre M2 e S2 produz o ciclo quinzenal de 14,77 dias
(sizígia × quadratura), reproduzindo o padrão real observado em portos costeiros.

A resposta de cada poço modela um aquífero raso costeiro como:

```
h_poço(t) = h_base + α · (h_maré(t − Δt) − h̄_maré) + tendência + ε
```

onde α é o coeficiente de atenuação (função inversa da distância da costa),
Δt é o lag hidráulico (função direta da distância) e ε é ruído gaussiano.
