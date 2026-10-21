"""Inference: Transcricao -> Processamento -> Modelo -> Indicadores -> Recomendacao.

``NoraInsights.load()`` reads the bundle written by ``train.py`` and
``analyze(text)`` returns a JSON-serializable report for one transcript.
"""

from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .lexicons import find_competitors, find_products, hits
from .scoring import Indicators, combine, level, meeting_raw, percentile, recommend
from .text import clean_text, parse_turns, segment

PRODUCT_WORDS = ("totvs", "protheus", "fluig", "datasul", "logix", "winthor", "consinco", "techfin")

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
BUNDLE = ARTIFACTS / "nora_insights.joblib"


class NoraInsights:
    def __init__(self, bundle: dict[str, Any]) -> None:
        self.models = bundle["models"]
        self.reference = bundle["reference"]          # sorted meeting raw scores per indicator
        self.terms = bundle["terms"]                  # meeting-level TF-IDF for "principais termos"
        self.meta = bundle.get("meta", {})

    @classmethod
    def load(cls, path: Path = BUNDLE) -> "NoraInsights":
        if not path.exists():
            raise SystemExit(f"Modelos nao encontrados em {path}. Rode: python train.py")
        return cls(joblib.load(path))

    def segment_scores(self, cleans: list[str]) -> dict[str, np.ndarray]:
        """Per-segment scores: model probability combined with the lexicon (noisy-OR)."""
        risk_rule = np.array([hits(c, "risk") > 0 for c in cleans])
        opp_rule = np.array([hits(c, "opportunity") > 0 for c in cleans])
        lex = np.array([np.sign(hits(c, "positive") - hits(c, "negative")) for c in cleans], dtype=float)
        sent = self.models["sentiment"].predict_proba(cleans)
        classes = list(self.models["sentiment"].classes_)
        polarity = sent[:, classes.index("positivo")] - sent[:, classes.index("negativo")]
        return {
            "risk": combine(self.models["risk"].predict_proba(cleans)[:, 1], risk_rule),
            "opportunity": combine(self.models["opportunity"].predict_proba(cleans)[:, 1], opp_rule),
            "polarity": np.clip(0.7 * polarity + 0.3 * lex, -1, 1),
        }

    def analyze(self, raw: str, evidence: int = 3) -> dict[str, Any]:
        turns = parse_turns(raw)
        segs = [s for s in segment(turns) if len(s.clean.split()) >= 8]
        if not segs:
            raise ValueError("Transcricao vazia ou curta demais para analise")
        cleans = [s.clean for s in segs]
        sc = self.segment_scores(cleans)
        full_clean = " ".join(cleans)

        # The trend ignores the opening and closing segments (greetings and goodbyes) when it can.
        body = sc["polarity"][1:-1] if len(segs) >= 5 else sc["polarity"]
        third = max(1, len(body) // 3)
        ind = Indicators(
            risk=percentile(meeting_raw(sc["risk"]), self.reference["risk"]),
            opportunity=percentile(meeting_raw(sc["opportunity"]), self.reference["opportunity"]),
            sentiment_index=float(100 * sc["polarity"].mean()),
            sentiment_start=float(100 * body[:third].mean()),
            sentiment_end=float(100 * body[-third:].mean()),
            products=dict(find_products(full_clean).most_common()),
            competitors=dict(find_competitors(full_clean).most_common()),
            clean_text=full_clean,
            flagged_risk=int((sc["risk"] >= self.meta.get("thresholds", {}).get("risk", 0.5)).sum()),
            flagged_opportunity=int(
                (sc["opportunity"] >= self.meta.get("thresholds", {}).get("opportunity", 0.5)).sum()),
        )
        ind.recommendations = recommend(ind)

        def top_segments(values: np.ndarray, reverse: bool = True) -> list[dict[str, Any]]:
            order = np.argsort(values)
            order = order[::-1] if reverse else order
            return [{"trecho": segs[i].index, "score": round(float(values[i]), 3),
                     "texto": _shorten(segs[i].text)} for i in order[:evidence]]

        return {
            "resumo": {"turnos": len(turns), "locutores": len({t.speaker for t in turns}),
                       "segmentos": len(segs), "palavras": sum(s.words for s in segs)},
            "classificacao_conta": ind.account_status,
            "indicadores": {
                "risco_churn": {"score": round(ind.risk, 1), "nivel": level(ind.risk),
                                "trechos_sinalizados": ind.flagged_risk},
                "oportunidade_comercial": {"score": round(ind.opportunity, 1), "nivel": level(ind.opportunity),
                                           "trechos_sinalizados": ind.flagged_opportunity},
                "sentimento": {"indice": round(ind.sentiment_index, 1), "rotulo": ind.sentiment_label,
                               "inicio": round(ind.sentiment_start, 1), "fim": round(ind.sentiment_end, 1),
                               "tendencia": ind.sentiment_trend},
                "prioridade": round(ind.priority, 1),
            },
            "produtos_identificados": ind.products,
            "concorrentes_citados": ind.competitors,
            "principais_termos": self.top_terms(full_clean, exclude=_proper_nouns(raw)),
            "recomendacoes": [asdict(r) for r in ind.recommendations],
            "evidencias": {
                "risco": top_segments(sc["risk"]),
                "oportunidade": top_segments(sc["opportunity"]),
                "mais_negativo": top_segments(sc["polarity"], reverse=False),
            },
            "trajetoria_sentimento": [round(float(v), 3) for v in sc["polarity"]],
        }

    def top_terms(self, clean: str, n: int = 12, exclude: frozenset[str] = frozenset()) -> list[dict[str, Any]]:
        vec = self.terms
        row = vec.transform([clean]).toarray()[0]
        names = vec.get_feature_names_out()
        idx = [i for i in np.argsort(row)[::-1] if row[i] > 0 and not set(names[i].split()) & exclude][:n]
        return [{"termo": names[i], "peso": round(float(row[i]), 3)} for i in idx]


_PROPER_NOUN_RE = re.compile(r"(?<=[a-zà-ú,] )([A-ZÀ-Ú][a-zà-ú]{2,})")


def _proper_nouns(raw: str) -> frozenset[str]:
    """Words written capitalized mid-sentence — names of people and companies, not topics."""
    return frozenset(clean_text(w) for w in _PROPER_NOUN_RE.findall(raw)) - set(PRODUCT_WORDS)


def _shorten(text: str, limit: int = 420) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " ..."
