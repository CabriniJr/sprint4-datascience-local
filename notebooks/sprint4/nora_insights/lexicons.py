"""Weak supervision: the lexicons that produce the silver labels, and entity dictionaries.

The TOTVS dataset carries no per-segment label. The labels for churn risk,
commercial opportunity and sentiment are therefore produced by *labeling
functions* — hand-written patterns over the normalized text (lowercase, no
accents). They are precise but narrow: they fire only on the literal phrasing
they list. The classifiers are trained with those very phrases **masked out**
of the input (``mask``), so what a model learns is the *context* around risk or
opportunity language, which is what lets it score a segment that uses none of
the listed words.

Every pattern runs over ``text.clean_text`` output, so accents are already gone.
"""

from __future__ import annotations

import re
from collections import Counter

RISK_PATTERNS = [
    r"cancel(ar|amento|ou|ado)\s+(o\s+|a\s+|do\s+|da\s+)?(contrato|sistema|licenca|servico|assinatura|projeto|parceria)",
    r"\bcancelamento\b",
    r"\brescis\w*",
    r"\bmulta\s+(contratual|rescisoria)",
    r"\binsatisf\w*",
    r"\breclama\w*",
    r"\bdecepcion\w*",
    r"\bfrustra\w*",
    r"\bnao\s+(esta\s+|ta\s+)?funciona\w*",
    r"\bparou\s+de\s+funcionar",
    r"\blentid\w*",
    r"\btravand\w*",
    r"\bfora\s+do\s+ar\b",
    r"\bsem\s+resposta\b",
    r"\bninguem\s+(responde|retorna|resolve)",
    r"\bchamados?\s+abertos?\b",
    r"\bsla\s+(estourad|vencid)\w*",
    r"\btrocar\s+de\s+(sistema|fornecedor|erp|plataforma)",
    r"\bsair\s+d[ao]\s+totvs\b",
    r"\bconsiderando\s+(seriamente\s+)?sair\b",
    r"\bnao\s+(vamos|vai|quer\w*)\s+renovar",
    r"\bavaliando\s+outr[ao]s?\s+(fornecedor|sistema|solucao|opcao)\w*",
    r"\bmuito\s+caro\b",
    r"\bcaro\s+demais\b",
    r"\bpessim[oa]\b",
    r"\babsurd[oa]\b",
    r"\bdescaso\b",
    r"\bprejuizo\w*",
    r"\binsustentavel\b",
    r"\bnao\s+resolve\w*",
]

# ERP vocabulary collides with sales vocabulary: in a TOTVS meeting "cotacao", "comprar",
# "pedido" and "contratacao" usually describe the *customer's* purchasing or hiring process being
# demonstrated, not a purchase from TOTVS. Those words only count when the object is the solution.
_SOLUTION = r"(a\s+|o\s+)?(totvs|solucao|sistema|modulo|ferramenta|plataforma|licenca|projeto)"

OPPORTUNITY_PATTERNS = [
    r"\bpropostas?\s+(comercia\w*|formal|de\s+valor)\b",
    r"\b(mandar|enviar|apresentar|preparar)\s+(a\s+|uma\s+)?proposta\b",
    r"\borcamento\s+(aprovado|para\s+o\s+projeto)\b",
    r"\binvestimento\b",
    rf"\bcontrat(ar|acao\s+d[ao])\s+{_SOLUTION}",
    rf"\b(aquisicao|adquirir|comprar)\s+(d[ao]\s+)?{_SOLUTION}",
    # Software licences, not the HR kind (licenca-maternidade, licenca medica) that RM demos are full of.
    r"\blicencas?\b(?!\s+(maternidade|paternidade|medica|premio|saude|remunerada|nojo|gala|gestante))",
    r"\bnovos?\s+modulos?\b",
    r"\bexpan(dir|sao)\b",
    r"\bampliar\b",
    r"\bupgrade\b",
    r"\bfechar\s+(o\s+)?(negocio|contrato|pedido)",
    r"\bassinar\s+(o\s+)?contrato",
    r"\bproximos?\s+passos?\b",
    r"\bprova\s+de\s+conceito\b",
    r"\bpoc\b",
    r"\bcondic\w*\s+comercia\w*",
    r"\bdesconto\w*",
    r"\bprecifica\w*",
    r"\bbudget\b",
    r"\bverba\b",
    r"\bquanto\s+custa\b",
    r"\bvalor\s+(mensal|do\s+investimento|da\s+proposta)",
    r"\btem\s+interesse\b|\binteresse\s+em\b",
    r"\bfaz\s+sentido\s+para\s+(voces|nos|a\s+gente)\b",
    r"\bagendar\s+(uma\s+)?(demo|demonstracao|apresentacao|visita)",
]

SENTIMENT_POSITIVE = [
    r"\botim[oa]s?\b", r"\bexcelente\w*", r"\bmaravilh\w*", r"\bperfeit[oa]\b", r"\bgostei\b",
    r"\badorei\b", r"\bsensacional\b", r"\bfantastic\w*", r"\bparabens\b", r"\bsatisfeit[oa]s?\b",
    r"\bfacilit\w*", r"\bresolveu\b", r"\bfunciona\s+(muito\s+)?bem\b", r"\bmuito\s+bom\b",
    r"\bincrivel\b", r"\bshow\s+de\s+bola\b", r"\bsuper\s+(bem|bom|legal)\b", r"\bcontent[ea]s?\b",
]
# Courtesy ("agradeco", "obrigado") is left out on purpose: every meeting ends with it, so it
# measures where a segment sits in the call, not how the customer feels.

SENTIMENT_NEGATIVE = [
    r"\bruim\b", r"\bpessim[oa]\b", r"\bhorrivel\b", r"\bdificil\b", r"\bdificuldades?\b",
    r"\bproblemas?\b", r"\berros?\b", r"\bfalhas?\b", r"\blent[oa]\b", r"\blentid\w*",
    r"\btravand\w*", r"\binsatisf\w*", r"\breclama\w*", r"\bdecepcion\w*", r"\bfrustra\w*",
    r"\bpreocup\w*", r"\bdemor\w*", r"\batras\w*", r"\bnao\s+gostei\b", r"\bnao\s+funciona\w*",
    r"\bchatead\w*", r"\binfelizmente\b", r"\bdescaso\b", r"\babsurd[oa]\b",
]
# "sem problema", "nenhum erro", "nao tem problema" are reassurances, not complaints.
_NEGATED_NEGATIVE = re.compile(r"\b(sem|nenhum|nenhuma|nao\s+tem|nao\s+teve|zero)\s+(problemas?|erros?|falhas?|dificuldades?)\b")

PRODUCTS = {
    "Protheus": r"\b(protheus|proteus|protheu)\b",
    "RM": r"\b(totvs\s+rm|totsrm|rm)\b",
    "Fluig": r"\bfluig\w*",
    "Datasul": r"\b(datasul|data\s+sul)\b",
    "Logix": r"\blogix\b",
    "Winthor": r"\b(winthor|wintor)\b",
    "Consinco": r"\bconsinco\b",
    "RD Station": r"\brd\s+station\b",
    "TOTVS Techfin": r"\btechfin\b",
    "TOTVS Cloud": r"\b(totvs\s+cloud|totscloud)\b",
    "Carol (IA)": r"\btotvs\s+carol\b",
}

COMPETITORS = {
    "SAP": r"\b(sap|sap\s+business\s+one|s\s*4\s*hana)\b",
    "Oracle": r"\boracle\b",
    "Senior": r"\bsenior\s+sistemas\b",
    "Sankhya": r"\bsankhya\b",
    "Omie": r"\bomie\b",
    "MV": r"\bmv\s+(soul|sistemas|hospitalar)\b",
    "Tasy": r"\btasy\b",
    "Salesforce": r"\bsalesforce\b",
    "Microsoft Dynamics": r"\bdynamics\b",
    "Benner": r"\bbenner\b",
    "Linx": r"\blinx\b",
}

_COMPILED: dict[str, list[re.Pattern[str]]] = {
    "risk": [re.compile(p) for p in RISK_PATTERNS],
    "opportunity": [re.compile(p) for p in OPPORTUNITY_PATTERNS],
    "positive": [re.compile(p) for p in SENTIMENT_POSITIVE],
    "negative": [re.compile(p) for p in SENTIMENT_NEGATIVE],
}
_PRODUCT_RE = {name: re.compile(p) for name, p in PRODUCTS.items()}
_COMPETITOR_RE = {name: re.compile(p) for name, p in COMPETITORS.items()}


def hits(clean: str, family: str) -> int:
    """Number of lexicon matches of ``family`` in a normalized segment."""
    if family == "negative":
        clean = _NEGATED_NEGATIVE.sub(" ", clean)
    return sum(len(p.findall(clean)) for p in _COMPILED[family])


def weak_labels(clean: str) -> dict[str, int | str]:
    """Silver labels for one segment: risk 0/1, opportunity 0/1, sentiment pos/neu/neg."""
    pos, neg = hits(clean, "positive"), hits(clean, "negative")
    if neg > pos:
        sentiment = "negativo"
    elif pos > neg:
        sentiment = "positivo"
    else:
        sentiment = "neutro"
    return {
        "risk": int(hits(clean, "risk") > 0),
        "opportunity": int(hits(clean, "opportunity") > 0),
        "sentiment": sentiment,
    }


def mask(clean: str, families: tuple[str, ...]) -> str:
    """Removes the labeling phrases so a model cannot learn the label from its own definition."""
    for family in families:
        for p in _COMPILED[family]:
            clean = p.sub(" ", clean)
        if family == "negative":
            clean = _NEGATED_NEGATIVE.sub(" ", clean)
    return " ".join(clean.split())


def find_products(clean: str) -> Counter[str]:
    return Counter({name: n for name, rx in _PRODUCT_RE.items() if (n := len(rx.findall(clean)))})


def find_competitors(clean: str) -> Counter[str]:
    return Counter({name: n for name, rx in _COMPETITOR_RE.items() if (n := len(rx.findall(clean)))})
