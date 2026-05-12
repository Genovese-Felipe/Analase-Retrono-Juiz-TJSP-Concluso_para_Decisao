"""
==========================================================================
ANÁLISE RETRONA DO JUIZ – TJSP
Concluso para Decisão → Publicação da Decisão
==========================================================================
Objetivo:
  1. Calcular os dias úteis jurídicos que este juiz tipicamente leva entre
     cada 'Concluso para Decisão' e a 'Publicação da Decisão'.
  2. Estimar, com análise probabilística expert-level, a provável data de
     publicação para o último Concluso de 03/02/2026.
  3. Gerar tabela com P(publicação no dia k) e P(publicação até o dia k)
     para k = 1 … 75 dias úteis jurídicos TJSP.

Metodologia:
  - Calendário TJSP: exclui fins-de-semana, feriados nacionais/estaduais e
    recesso judiciário (jan. e jul.).
  - Análise contextual por TIPO de movimentação (Sentença, Decisão
    Interlocutória, etc.), pois prazos diferem significativamente.
  - Ajuste estatístico: distribuição Gamma/Log-Normal via MLE (Maximum
    Likelihood Estimation) — NÃO apenas média simples.
  - Probabilidades geradas por função de densidade de probabilidade (PDF)
    e função de distribuição acumulada (CDF).
==========================================================================
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import date, timedelta
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────
# 1. CALENDÁRIO TJSP – DIAS ÚTEIS JURÍDICOS
# ──────────────────────────────────────────────────────────────────────────

def get_feriados_nacionais(ano: int) -> list[date]:
    """Retorna feriados nacionais fixos + móveis para o ano."""
    from dateutil.easter import easter
    pascoa = easter(ano)

    feriados = [
        date(ano, 1, 1),   # Confraternização Universal
        date(ano, 4, 21),  # Tiradentes
        date(ano, 5, 1),   # Dia do Trabalhador
        date(ano, 9, 7),   # Independência
        date(ano, 10, 12), # N. Sra. Aparecida
        date(ano, 11, 2),  # Finados
        date(ano, 11, 15), # Proclamação da República
        date(ano, 11, 20), # Consciência Negra (lei 14.759/2023)
        date(ano, 12, 25), # Natal
        # Páscoa e Sexta-feira Santa
        pascoa,
        pascoa - timedelta(days=2),
        # Corpus Christi (60 dias após Páscoa)
        pascoa + timedelta(days=60),
    ]
    return feriados


def get_feriados_sp(ano: int) -> list[date]:
    """Feriados estaduais e municipais de São Paulo."""
    return [
        date(ano, 1, 25),  # Aniversário de São Paulo
        date(ano, 7, 9),   # Revolução Constitucionalista
    ]


def get_recesso_tjsp(ano: int) -> list[date]:
    """
    Recesso judiciário TJSP (Resolução TJSP).
    Janeiro: 2 a 31 (dias não úteis)
    Julho:   1 a 31 (dias não úteis)
    Dezembro: 20 a 31 (dias não úteis)
    """
    recesso = []
    # Janeiro (2–31)
    d = date(ano, 1, 2)
    while d <= date(ano, 1, 31):
        recesso.append(d)
        d += timedelta(days=1)
    # Julho (1–31)
    d = date(ano, 7, 1)
    while d <= date(ano, 7, 31):
        recesso.append(d)
        d += timedelta(days=1)
    # Dezembro (20–31)
    d = date(ano, 12, 20)
    while d <= date(ano, 12, 31):
        recesso.append(d)
        d += timedelta(days=1)
    return recesso


def build_nao_uteis(ano_inicio: int, ano_fim: int) -> set[date]:
    """Constrói conjunto de todos os dias NÃO úteis jurídicos TJSP."""
    nao_uteis: set[date] = set()
    for ano in range(ano_inicio, ano_fim + 1):
        nao_uteis.update(get_feriados_nacionais(ano))
        nao_uteis.update(get_feriados_sp(ano))
        nao_uteis.update(get_recesso_tjsp(ano))
    return nao_uteis


NAO_UTEIS = build_nao_uteis(2023, 2027)


def is_dia_util_tjsp(d: date) -> bool:
    """Retorna True se o dia é útil jurídico TJSP."""
    return d.weekday() < 5 and d not in NAO_UTEIS


def dias_uteis_entre(d_inicio: date, d_fim: date) -> int:
    """
    Conta dias úteis jurídicos TJSP entre d_inicio (exclusive) e d_fim
    (inclusive), conforme ocorre no cômputo processual.
    """
    if d_fim <= d_inicio:
        return 0
    count = 0
    d = d_inicio + timedelta(days=1)
    while d <= d_fim:
        if is_dia_util_tjsp(d):
            count += 1
        d += timedelta(days=1)
    return count


def nth_dia_util_apos(d_inicio: date, n: int) -> date:
    """Retorna a data do n-ésimo dia útil TJSP após d_inicio."""
    count = 0
    d = d_inicio + timedelta(days=1)
    while True:
        if is_dia_util_tjsp(d):
            count += 1
            if count == n:
                return d
        d += timedelta(days=1)


# ──────────────────────────────────────────────────────────────────────────
# 2. CARGA E PRÉ-PROCESSAMENTO DOS DADOS
# ──────────────────────────────────────────────────────────────────────────

df_raw = pd.read_csv("movimentacoes.csv", parse_dates=["data"])
df_raw["data"] = df_raw["data"].dt.date

# Classifica o tipo de conclusão
def classifica_tipo(obs: str) -> str:
    obs = obs.lower()
    if "sentença" in obs or "extinção" in obs:
        return "Sentença"
    elif "embargos de declaração" in obs:
        return "Embargos de Declaração"
    elif "saneamento" in obs:
        return "Saneamento"
    elif "tutela" in obs:
        return "Tutela de Urgência/Antecipada"
    else:
        return "Decisão Interlocutória"

df_raw["tipo"] = df_raw["observacao"].apply(classifica_tipo)

# Separa conclusões e publicações, pareando-as em ordem cronológica
conclusoes = df_raw[df_raw["tipo_movimentacao"] == "Concluso para Decisão"].copy()
publicacoes = df_raw[df_raw["tipo_movimentacao"] == "Publicação da Decisão"].copy()

pares = []
for i, (_, conc) in enumerate(conclusoes.iterrows()):
    # Procura a publicação imediatamente seguinte a esta conclusão
    pubs_seguintes = publicacoes[publicacoes["data"] > conc["data"]].sort_values("data")
    if len(pubs_seguintes) > 0:
        pub = pubs_seguintes.iloc[0]
        du = dias_uteis_entre(conc["data"], pub["data"])
        pares.append({
            "conclusao":    conc["data"],
            "publicacao":   pub["data"],
            "tipo":         conc["tipo"],
            "observacao":   conc["observacao"],
            "dias_uteis":   du,
        })

df = pd.DataFrame(pares)
# Remove o último par (03/02/2026 – ainda sem publicação)
df_hist = df[df["publicacao"].notna() & (df["conclusao"] < date(2026, 1, 1))].copy()

# ──────────────────────────────────────────────────────────────────────────
# 3. ESTATÍSTICAS DESCRITIVAS POR TIPO
# ──────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("ANÁLISE RETRONA – JUIZ TJSP")
print("Concluso para Decisão → Publicação (Dias Úteis Jurídicos TJSP)")
print("=" * 70)

print("\n📊 ESTATÍSTICAS DESCRITIVAS POR TIPO DE MOVIMENTAÇÃO")
print("-" * 70)

stats_por_tipo = (
    df_hist.groupby("tipo")["dias_uteis"]
    .agg(["count", "min", "median", "mean", "max", "std"])
    .rename(columns={
        "count":  "n",
        "min":    "Mín",
        "median": "Mediana",
        "mean":   "Média",
        "max":    "Máx",
        "std":    "DesvPad",
    })
    .round(1)
)
print(stats_por_tipo.to_string())

print("\n📊 ESTATÍSTICAS GERAIS (todos os tipos)")
print("-" * 70)
du = df_hist["dias_uteis"].values
print(f"  N observações  : {len(du)}")
print(f"  Mínimo         : {du.min()} d.u.")
print(f"  Percentil 25   : {np.percentile(du, 25):.1f} d.u.")
print(f"  Mediana (P50)  : {np.median(du):.1f} d.u.")
print(f"  Média          : {du.mean():.1f} d.u.")
print(f"  Percentil 75   : {np.percentile(du, 75):.1f} d.u.")
print(f"  Percentil 90   : {np.percentile(du, 90):.1f} d.u.")
print(f"  Máximo         : {du.max()} d.u.")
print(f"  Desvio padrão  : {du.std():.1f} d.u.")

# ──────────────────────────────────────────────────────────────────────────
# 4. AJUSTE ESTATÍSTICO – MLE (Gamma e Log-Normal)
# ──────────────────────────────────────────────────────────────────────────

print("\n🔬 AJUSTE ESTATÍSTICO (MLE) – TODOS OS TIPOS")
print("-" * 70)

# Gamma
gamma_params = stats.gamma.fit(du, floc=0)
gamma_a, gamma_loc, gamma_scale = gamma_params
# Log-Normal
lognorm_params = stats.lognorm.fit(du, floc=0)
lognorm_s, lognorm_loc, lognorm_scale = lognorm_params

# Kolmogorov-Smirnov
ks_gamma  = stats.kstest(du, "gamma",   args=gamma_params)
ks_lognorm = stats.kstest(du, "lognorm", args=lognorm_params)

print(f"  Gamma    : shape={gamma_a:.3f}, scale={gamma_scale:.3f}")
print(f"             KS p-value = {ks_gamma.pvalue:.4f}")
print(f"  Log-Normal: s={lognorm_s:.3f}, scale={lognorm_scale:.3f}")
print(f"             KS p-value = {ks_lognorm.pvalue:.4f}")

# Escolhe o melhor ajuste
if ks_lognorm.pvalue >= ks_gamma.pvalue:
    best_dist_name = "Log-Normal"
    best_dist = stats.lognorm(*lognorm_params)
else:
    best_dist_name = "Gamma"
    best_dist = stats.gamma(*gamma_params)

print(f"\n  ✔ Melhor ajuste: {best_dist_name}")

# ──────────────────────────────────────────────────────────────────────────
# 5. ANÁLISE CONTEXTUAL – CONCLUSO DE 03/02/2026
# ──────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("🎯 PREVISÃO PARA O CONCLUSO DE 03/02/2026")
print("=" * 70)

# O último concluso é do tipo "Decisão Interlocutória" (sem tipo específico)
tipo_alvo = "Decisão Interlocutória"

print(f"\n  Tipo do concluso : {tipo_alvo} (ausência de especificação adicional)")
print(f"  Data do concluso : 03/02/2026")
print(f"  Período           : pós-recesso jan/2026 — início do ano processual")

# Dados do tipo específico
df_tipo = df_hist[df_hist["tipo"] == tipo_alvo]["dias_uteis"].values

if len(df_tipo) >= 5:
    tipo_params_gamma   = stats.gamma.fit(df_tipo, floc=0)
    tipo_params_lognorm = stats.lognorm.fit(df_tipo, floc=0)
    ks_t_g = stats.kstest(df_tipo, "gamma",   args=tipo_params_gamma)
    ks_t_l = stats.kstest(df_tipo, "lognorm", args=tipo_params_lognorm)
    if ks_t_l.pvalue >= ks_t_g.pvalue:
        dist_tipo = stats.lognorm(*tipo_params_lognorm)
        dist_tipo_nome = "Log-Normal"
    else:
        dist_tipo = stats.gamma(*tipo_params_gamma)
        dist_tipo_nome = "Gamma"
    print(f"  Distribuição tipo: {dist_tipo_nome} (MLE sobre {len(df_tipo)} obs.)")
    use_dist = dist_tipo
else:
    print(f"  (poucos dados específicos — usando distribuição geral)")
    use_dist = best_dist

# Estatísticas do tipo
if len(df_tipo) > 0:
    print(f"\n  Histórico '{tipo_alvo}':")
    print(f"    n={len(df_tipo)} | Mín={df_tipo.min()} | "
          f"Mediana={np.median(df_tipo):.0f} | "
          f"Média={df_tipo.mean():.1f} | Máx={df_tipo.max()} d.u.")

# Data-base
concluso_alvo = date(2026, 2, 3)

# ──────────────────────────────────────────────────────────────────────────
# 6. TABELA DE PROBABILIDADES (d.u. 1 … 75)
# ──────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("📅 TABELA PROBABILÍSTICA DE PUBLICAÇÃO")
print("   Concluso: 03/02/2026 | Distribuição: " + best_dist_name)
print("=" * 70)
print(f"\n{'DU':>4}  {'Data Calendário':<18}  {'Dia Semana':<12}  "
      f"{'P(pub. NESTE dia)':>17}  {'P(pub. ATÉ este dia)':>20}")
print("-" * 80)

# Ajuste: mistura ponderada entre dist. geral e dist. tipo
W_TIPO  = 0.65   # peso para distribuição específica do tipo
W_GERAL = 0.35   # peso para distribuição geral

registros = []
for k in range(1, 76):
    cal_date = nth_dia_util_apos(concluso_alvo, k)
    dia_semana = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"][cal_date.weekday()]

    # PDF e CDF da mistura
    if len(df_tipo) >= 5:
        pdf_k = W_TIPO  * use_dist.pdf(k) + W_GERAL * best_dist.pdf(k)
        cdf_k = W_TIPO  * use_dist.cdf(k) + W_GERAL * best_dist.cdf(k)
    else:
        pdf_k = best_dist.pdf(k)
        cdf_k = best_dist.cdf(k)

    registros.append({
        "k":           k,
        "data":        cal_date,
        "dia_semana":  dia_semana,
        "pdf":         pdf_k,
        "cdf":         cdf_k,
    })

# Normaliza a PDF para que a soma (k=1..75) = CDF(75)
pdf_vals  = np.array([r["pdf"] for r in registros])
cdf_final = registros[-1]["cdf"]
if pdf_vals.sum() > 0:
    pdf_norm = pdf_vals / pdf_vals.sum() * cdf_final
else:
    pdf_norm = pdf_vals

# Calcula CDF acumulada a partir da PDF normalizada
cdf_acum = np.cumsum(pdf_norm)

DIAS_UTEIS_MES = 22   # referência mensal

for i, r in enumerate(registros):
    k         = r["k"]
    cal_date  = r["data"]
    dia_sem   = r["dia_semana"]
    p_este    = pdf_norm[i]
    p_acum    = cdf_acum[i]

    # Destaque nos 30 primeiros dias úteis
    marca = "◀" if k <= 30 else ""
    if   i > 0 and p_acum >= 0.90 and pdf_norm[i - 1] < p_acum - p_este * 0.1:
        marca += " P90"
    elif p_acum >= 0.75:
        marca += " P75"
    elif p_acum >= 0.50:
        marca += " P50"

    print(f"  {k:>3}  {str(cal_date):<18}  {dia_sem:<10}  "
          f"  {p_este:>8.4%}         {p_acum:>8.4%}   {marca}")

# ──────────────────────────────────────────────────────────────────────────
# 7. RESUMO EXECUTIVO
# ──────────────────────────────────────────────────────────────────────────

# Encontra percentis a partir da CDF acumulada
def find_percentile_day(target_cdf: float) -> tuple[int, date]:
    for i, cum in enumerate(cdf_acum):
        if cum >= target_cdf:
            return registros[i]["k"], registros[i]["data"]
    return registros[-1]["k"], registros[-1]["data"]

p25_k, p25_d  = find_percentile_day(0.25)
p50_k, p50_d  = find_percentile_day(0.50)
p75_k, p75_d  = find_percentile_day(0.75)
p90_k, p90_d  = find_percentile_day(0.90)

print("\n" + "=" * 70)
print("🏛️  RESUMO EXECUTIVO – PREVISÃO DE PUBLICAÇÃO")
print("=" * 70)
print(f"""
  📌 Concluso para Decisão  : 03/02/2026
  📌 Tipo de decisão         : {tipo_alvo}
  📌 Modelo estatístico      : Mistura ({best_dist_name} geral + {dist_tipo_nome if len(df_tipo)>=5 else best_dist_name} tipo)
  📌 Base histórica          : {len(df_hist)} pares conclusão → publicação

  ┌──────────────────────────────────────────────────────────────────┐
  │   PERCENTIL │ DIA ÚTIL │ DATA PREVISTA  │ INTERPRETAÇÃO          │
  ├──────────────────────────────────────────────────────────────────┤
  │     P25     │  {p25_k:>4} d.u. │  {str(p25_d):<14}  │ Cenário otimista       │
  │     P50     │  {p50_k:>4} d.u. │  {str(p50_d):<14}  │ Cenário mais provável  │
  │     P75     │  {p75_k:>4} d.u. │  {str(p75_d):<14}  │ Cenário conservador    │
  │     P90     │  {p90_k:>4} d.u. │  {str(p90_d):<14}  │ Cenário pessimista     │
  └──────────────────────────────────────────────────────────────────┘

  🔑 ANÁLISE CONTEXTUAL:
  • O concluso ocorreu em 03/02/2026, logo após o recesso de janeiro.
    O juiz normalmente retoma os trabalhos com backlog acumulado,
    o que tende a AUMENTAR ligeiramente o prazo médio de resposta.
  • Decisões interlocutórias sem urgência tendem a ser prolatadas
    entre o {p25_k}º e o {p75_k}º dia útil (intervalo P25–P75).
  • A ESTIMATIVA CENTRAL (P50) é {str(p50_d)} ({p50_k} d.u.).
  • Há 75% de chance de publicação até {str(p75_d)} ({p75_k} d.u.).
  • Se não houver publicação até o {p90_k}º d.u. ({str(p90_d)}),
    sugere-se peticionar para agilização (art. 235 CPC/2015).
""")

print("=" * 70)
print("Análise concluída. Dados em: movimentacoes.csv")
print("=" * 70)
