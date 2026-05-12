# Análise Retrona – Juiz TJSP · Concluso para Decisão → Publicação

> **Expert-level jurídico + data science** — previsão probabilística de publicação de decisões a partir do histórico real de movimentações processuais do TJSP.

---

## 📋 Objetivo

Responder às seguintes perguntas com análise estatística rigorosa:

1. **Quantos dias úteis jurídicos** este juiz costuma levar entre cada *Concluso para Decisão* e a *Publicação da Decisão*?
2. **Estimativa probabilística** para a publicação do último concluso de **03/02/2026**, incluindo:
   - Coluna `P(pub. NESTE dia)` — probabilidade de publicação em cada dia útil específico.
   - Coluna `P(pub. ATÉ este dia)` — probabilidade acumulada, do 1º ao 75º dia útil jurídico TJSP.

---

## 🗂️ Arquivos

| Arquivo | Descrição |
|---|---|
| `movimentacoes.csv` | Histórico de movimentações processuais (conclusões e publicações) |
| `analise_tjsp.py` | Script de análise estatística completo |
| `requirements.txt` | Dependências Python necessárias |

---

## 🚀 Como executar

```bash
pip install -r requirements.txt
python analise_tjsp.py
```

---

## 🔬 Metodologia

### 1. Calendário de Dias Úteis Jurídicos TJSP

O script implementa o calendário completo de dias úteis jurídicos TJSP, excluindo:

- **Fins de semana** (sábado e domingo)
- **Feriados nacionais fixos**: Confraternização Universal (1/jan), Tiradentes (21/abr), Dia do Trabalhador (1/mai), Independência (7/set), N. Sra. Aparecida (12/out), Finados (2/nov), Proclamação da República (15/nov), Consciência Negra (20/nov), Natal (25/dez)
- **Feriados nacionais móveis**: Sexta-Feira Santa, Páscoa, Corpus Christi
- **Feriados estaduais SP**: Aniversário de SP (25/jan), Revolução Constitucionalista (9/jul)
- **Recesso judiciário TJSP** (Resolução TJSP): Janeiro (2–31), Julho (1–31), Dezembro (20–31)

### 2. Classificação Contextual por Tipo de Decisão

A análise não usa média simples. Cada movimentação é classificada por tipo:

| Tipo | Prazo típico (d.u.) | Característica |
|---|---|---|
| **Tutela de Urgência/Antecipada** | 8–12 d.u. | Urgência legal; juiz prioriza |
| **Embargos de Declaração** | 7–8 d.u. | Prazo curto por natureza processual |
| **Decisão Interlocutória** | 3–14 d.u. | Variável conforme complexidade |
| **Saneamento** | 10–14 d.u. | Análise detalhada necessária |
| **Sentença** | 11–32 d.u. | Maior complexidade; maior variância |

### 3. Ajuste Estatístico por MLE (Maximum Likelihood Estimation)

São testados dois modelos de distribuição de probabilidade:

- **Distribuição Gamma** — adequada para dados de tempo positivos com assimetria à direita
- **Distribuição Log-Normal** — adequada quando o log dos dados segue distribuição normal

O modelo com maior p-value no teste de Kolmogorov-Smirnov é selecionado automaticamente.

### 4. Modelo de Mistura

Para maximizar a acurácia, é utilizado um **modelo de mistura ponderada**:

```
P(k) = 0.65 × dist_tipo(k) + 0.35 × dist_geral(k)
```

- `dist_tipo`: distribuição ajustada especificamente para o tipo de decisão do concluso-alvo
- `dist_geral`: distribuição ajustada a todos os dados históricos

---

## 📊 Resultados

### Estatísticas Históricas por Tipo

| Tipo | n | Mín (d.u.) | Mediana (d.u.) | Média (d.u.) | Máx (d.u.) |
|---|---|---|---|---|---|
| Decisão Interlocutória | 8 | 3 | 11 | 10,5 | 14 |
| Embargos de Declaração | 2 | 7 | 7,5 | 7,5 | 8 |
| Saneamento | 3 | 10 | 12 | 12,0 | 14 |
| Sentença | 11 | 11 | 18 | 20,3 | 32 |
| Tutela de Urgência/Antecipada | 5 | 8 | 9 | 9,6 | 12 |

### Previsão para 03/02/2026 (Decisão Interlocutória)

| Percentil | Dia Útil | Data Prevista | Interpretação |
|---|---|---|---|
| **P25** | 8 d.u. | **13/02/2026** | Cenário otimista |
| **P50** | 11 d.u. | **18/02/2026** | Cenário mais provável |
| **P75** | 14 d.u. | **23/02/2026** | Cenário conservador |
| **P90** | 19 d.u. | **02/03/2026** | Cenário pessimista |

> ⚖️ **Recomendação processual**: Caso não haja publicação até o 19º dia útil (02/03/2026), considerar peticionamento para agilização nos termos do art. 235 do CPC/2015.

---

## 📅 Tabela Probabilística Completa (1–75 d.u.)

Execute `python analise_tjsp.py` para gerar a tabela completa com:
- Data calendário de cada dia útil
- `P(pub. NESTE dia)` — probabilidade pontual
- `P(pub. ATÉ este dia)` — probabilidade acumulada

A tabela destaca os **30 primeiros dias úteis** (◀) e marca os percentis P50, P75 e P90.

---

## 🏛️ Contexto Jurídico TJSP

### Por que o prazo pós-recesso de janeiro é maior?

O concluso de **03/02/2026** ocorre imediatamente após o recesso de janeiro, período em que:
1. O juiz retoma os trabalhos com **backlog acumulado** de processos que ficaram conclusos durante o recesso.
2. Há concentração de petições protocoladas no início do ano.
3. A prioridade de julgamento segue critérios de urgência e antiguidade (art. 12 CPC/2015).

### Referência Legal

- **Art. 12 CPC/2015**: Julgamento em ordem cronológica
- **Art. 226 CPC/2015**: Prazo de 10 dias para decisões interlocutórias (prazo impróprio — não gera nulidade)
- **Art. 235 CPC/2015**: Reclamação por excesso de prazo
- **Resolução TJSP nº 549/2011** e atualizações: regulamenta o recesso judiciário

---

## ⚙️ Dependências

```
pandas>=2.0
numpy>=1.24
scipy>=1.10
python-dateutil>=2.8
```

