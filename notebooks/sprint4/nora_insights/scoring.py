"""From segment probabilities to meeting indicators, and from indicators to actions.

Scores are **percentiles against the TOTVS base**: a churn-risk score of 85
means the meeting carries more risk language than 85% of the 1,126 meetings in
the dataset. A commercial team works a queue, and a relative score is what
orders a queue; an absolute probability from a model trained on weak labels
would claim a calibration it does not have.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

# How much a lexicon hit is trusted on its own when combined with the model (noisy-OR).
RULE_WEIGHT = 0.6

LEVELS = [(80, "Alto"), (50, "Medio"), (0, "Baixo")]

ERP_PRODUCTS = {"Protheus", "RM", "Datasul", "Logix", "Winthor", "Consinco"}

CROSS_SELL = [
    # (product, trigger pattern over the normalized meeting text, minimum hits, pitch)
    ("Fluig", r"\b(workflow|fluxo de aprovac\w*|aprovac\w*|assinatura eletronica|gestao de documentos|ged)\b", 3,
     "Apresentar o Fluig (BPM/ECM) para automatizar os fluxos de aprovacao e documentos citados"),
    ("TOTVS Techfin", r"\b(credito|cobranca|recebiveis|antecipac\w*|inadimpl\w*|capital de giro)\b", 3,
     "Oferecer solucoes TOTVS Techfin (credito, cobranca, antecipacao de recebiveis)"),
    ("RD Station", r"\b(leads?|marketing|funil|automacao de marketing|captac\w* de clientes)\b", 3,
     "Apresentar o RD Station para geracao e gestao de leads"),
    ("TOTVS Cloud", r"\b(servidor\w*|infraestrutura|backup|data ?center|on premise|hospedagem)\b", 4,
     "Avaliar migracao para TOTVS Cloud (infraestrutura, backup e disponibilidade foram citados)"),
]


def level(score: float) -> str:
    return next(name for cut, name in LEVELS if score >= cut)


def combine(p_model: np.ndarray, rule_hit: np.ndarray) -> np.ndarray:
    """Noisy-OR of two evidence sources: the classifier and the precise-but-narrow lexicon."""
    return 1 - (1 - p_model) * (1 - RULE_WEIGHT * rule_hit.astype(float))


def meeting_raw(seg_scores: np.ndarray) -> float:
    """Mixes the peak (a meeting can turn on one bad moment) with the overall level."""
    if len(seg_scores) == 0:
        return 0.0
    k = max(3, math.ceil(0.1 * len(seg_scores)))
    top = np.sort(seg_scores)[::-1][:k]
    return float(0.5 * top.mean() + 0.5 * seg_scores.mean())


def percentile(value: float, reference: np.ndarray) -> float:
    return float(100.0 * np.searchsorted(reference, value, side="right") / len(reference))


@dataclass
class Recommendation:
    acao: str
    motivo: str
    prioridade: str  # alta | media | baixa


@dataclass
class Indicators:
    risk: float
    opportunity: float
    sentiment_index: float
    sentiment_start: float
    sentiment_end: float
    products: dict[str, int]
    competitors: dict[str, int]
    clean_text: str
    flagged_risk: int
    flagged_opportunity: int
    recommendations: list[Recommendation] = field(default_factory=list)

    @property
    def sentiment_label(self) -> str:
        if self.sentiment_index >= 10:
            return "positivo"
        if self.sentiment_index <= -10:
            return "negativo"
        return "neutro"

    @property
    def sentiment_trend(self) -> str:
        delta = self.sentiment_end - self.sentiment_start
        if delta >= 15:
            return "melhorou"
        if delta <= -15:
            return "piorou"
        return "estavel"

    @property
    def account_status(self) -> str:
        risk, opp = level(self.risk), level(self.opportunity)
        if risk == "Alto" and opp == "Alto":
            return "Negociacao sensivel"
        if risk == "Alto":
            return "Conta em risco"
        if opp == "Alto":
            return "Oportunidade quente"
        if risk == "Medio" or self.sentiment_label == "negativo":
            return "Atencao"
        return "Saudavel / acompanhamento"

    @property
    def priority(self) -> float:
        return max(self.risk, self.opportunity)


def recommend(ind: Indicators) -> list[Recommendation]:
    recs: list[Recommendation] = []
    risk, opp = level(ind.risk), level(ind.opportunity)
    competitors = ", ".join(ind.competitors)

    if risk == "Alto" and opp == "Alto":
        recs.append(Recommendation(
            "Resolver as pendencias levantadas antes de avancar a proposta; envolver o gestor da conta",
            f"Risco {ind.risk:.0f}/100 e oportunidade {ind.opportunity:.0f}/100 na mesma reuniao: "
            "avancar comercialmente sem tratar a insatisfacao tende a travar o fechamento", "alta"))
    elif risk == "Alto":
        recs.append(Recommendation(
            "Acionar Customer Success em ate 48h com um plano de acao para as dores citadas",
            f"Risco de churn {ind.risk:.0f}/100 — mais sinal de risco que {math.floor(ind.risk)}% das "
            f"reunioes da base TOTVS; {ind.flagged_risk} trecho(s) sinalizado(s)", "alta"))
    elif opp == "Alto":
        recs.append(Recommendation(
            "Enviar proposta comercial e agendar a proxima etapa em ate 5 dias uteis",
            f"Oportunidade {ind.opportunity:.0f}/100 — mais sinal de compra que "
            f"{math.floor(ind.opportunity)}% das reunioes; {ind.flagged_opportunity} trecho(s) com sinal "
            "de compra", "alta"))

    if ind.competitors:
        if risk != "Baixo":
            recs.append(Recommendation(f"Preparar comparativo (battlecard) frente a {competitors}",
                                       "Concorrente citado em reuniao com sinal de risco", "alta"))
        elif opp != "Baixo":
            recs.append(Recommendation(f"Reforcar os diferenciais TOTVS frente a {competitors} na proposta",
                                       "Concorrente citado em reuniao com oportunidade aberta", "media"))

    if ind.sentiment_trend == "piorou":
        recs.append(Recommendation(
            "Fazer follow-up de satisfacao em ate uma semana",
            f"O sentimento caiu ao longo da reuniao ({ind.sentiment_start:+.0f} no inicio, "
            f"{ind.sentiment_end:+.0f} no fim)", "media"))
    elif risk == "Medio" and ind.sentiment_label == "negativo":
        recs.append(Recommendation("Agendar conversa de acompanhamento com o cliente",
                                   "Risco moderado com sentimento geral negativo", "media"))

    if opp != "Baixo" or risk == "Baixo":
        for product, pattern, minimum, pitch in CROSS_SELL:
            if product in ind.products:
                continue
            n = len(re.findall(pattern, ind.clean_text))
            if n >= minimum and (product != "Fluig" or ERP_PRODUCTS & set(ind.products)):
                recs.append(Recommendation(pitch, f"{n} mencoes ao tema na conversa, sem citar {product}", "media"))

    if not ind.products:
        recs.append(Recommendation("Registrar no CRM o produto de interesse",
                                   "Nenhum produto TOTVS foi identificado na transcricao", "baixa"))
    if not any(r.prioridade == "alta" for r in recs):
        recs.append(Recommendation("Manter a cadencia de relacionamento e registrar o resumo no CRM",
                                   "Nenhum indicador em nivel alto", "baixa"))
    order = {"alta": 0, "media": 1, "baixa": 2}
    return sorted(recs, key=lambda r: order[r.prioridade])
