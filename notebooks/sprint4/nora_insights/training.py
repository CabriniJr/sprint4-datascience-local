"""Training orchestration shared by ``train.py`` and the notebook."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold

from .lexicons import hits
from .models import (TASKS, Evaluation, candidates, choose_threshold, cross_validate, evaluate,
                     model_input, scores)
from .scoring import combine, meeting_raw
from .text import STOPWORDS


def compare_models(segments: pd.DataFrame, task: str, train_idx: np.ndarray, test_idx: np.ndarray,
                   n_splits: int = 3) -> dict[str, Any]:
    """Cross-validates every candidate on the training meetings and scores it on the held-out ones."""
    x = model_input(segments, task)
    y = segments[task].to_numpy()
    groups = segments["meeting_id"]
    x_tr, y_tr, g_tr = x.iloc[train_idx], y[train_idx], groups.iloc[train_idx]
    x_te, y_te = x.iloc[test_idx], y[test_idx]
    binary = TASKS[task]["positive"] is not None

    cv_rows, test_evals, fitted, thresholds = [], {}, {}, {}
    for name, pipe in candidates(task).items():
        folds, oof = cross_validate(pipe, x_tr, y_tr, g_tr, task, n_splits=n_splits)
        thr = choose_threshold(y_tr, oof, task) if binary else float("nan")
        # Fold metrics re-read at the model's own threshold, so models are compared on equal terms.
        if binary:
            fold_ids = list(GroupKFold(n_splits=n_splits).split(x_tr, y_tr, g_tr))
            folds = [evaluate(y_tr[v], oof[v], task, name, thr) for _, v in fold_ids]
        frame = pd.DataFrame([f.row() for f in folds]).drop(columns=["modelo", "threshold"])
        row = {"modelo": name}
        for metric in frame.columns:
            row[f"{metric}_media"] = frame[metric].mean()
            row[f"{metric}_dp"] = frame[metric].std()
        cv_rows.append(row)
        model = clone(pipe).fit(x_tr, y_tr)
        test_evals[name] = evaluate(y_te, scores(model, x_te, task), task, name, thr if binary else 0.5)
        fitted[name], thresholds[name] = model, thr

    cv = pd.DataFrame(cv_rows).set_index("modelo")
    key = "pr_auc_media" if binary else "f1_media"
    selected = cv[key].idxmax()
    return {"task": task, "cv": cv, "test": test_evals, "fitted": fitted, "thresholds": thresholds,
            "selected": selected, "selection_metric": key}


def unmasked_ceiling(segments: pd.DataFrame, task: str, train_idx: np.ndarray, test_idx: np.ndarray,
                     model: str = "Regressao Logistica") -> Evaluation:
    """Same model trained *without* masking: shows how circular weak labels inflate metrics."""
    x = model_input(segments, task, masked=False)
    y = segments[task].to_numpy()
    pipe = clone(candidates(task)[model]).fit(x.iloc[train_idx], y[train_idx])
    return evaluate(y[test_idx], scores(pipe, x.iloc[test_idx], task), task, f"{model} (sem mascara)")


def out_of_fold_meetings(segments: pd.DataFrame, meetings: pd.DataFrame, selected: dict[str, str],
                         n_splits: int = 5) -> pd.DataFrame:
    """Meeting-level raw scores where every meeting was scored by a model that never saw it."""
    seg_scores = pd.DataFrame(index=segments.index)
    groups = segments["meeting_id"]
    for task in ("risk", "opportunity", "sentiment"):
        x = model_input(segments, task)
        y = segments[task].to_numpy()
        _, oof = cross_validate(candidates(task)[selected[task]], x, y, groups, task, n_splits=n_splits)
        if task == "sentiment":
            lex = np.sign(segments["clean"].map(lambda c: hits(c, "positive") - hits(c, "negative")))
            seg_scores["polarity"] = np.clip(0.7 * (oof[:, 2] - oof[:, 0]) + 0.3 * lex, -1, 1)
        else:
            rule = segments["clean"].map(lambda c: hits(c, task) > 0).to_numpy()
            seg_scores[task] = combine(oof, rule)
    seg_scores["meeting_id"] = groups
    agg = seg_scores.groupby("meeting_id").agg(
        risk_raw=("risk", lambda s: meeting_raw(s.to_numpy())),
        opportunity_raw=("opportunity", lambda s: meeting_raw(s.to_numpy())),
        sentiment_index=("polarity", lambda s: 100 * s.mean()),
        segmentos=("risk", "size"),
    )
    cols = ["ID_MEETING", "NOTA_NPS", "nps_categoria", "NOME_SEGMENTO", "TP_RECURSO"]
    return agg.join(meetings[cols].set_index("ID_MEETING"), how="left")


def terms_vectorizer(segments: pd.DataFrame) -> TfidfVectorizer:
    """Meeting-level TF-IDF used for the "principais termos" of a new transcript."""
    docs = segments.groupby("meeting_id")["clean"].apply(" ".join)
    vec = TfidfVectorizer(stop_words=STOPWORDS, ngram_range=(1, 2), min_df=5, max_df=0.3,
                          sublinear_tf=True, max_features=40000, token_pattern=r"(?u)\b[a-z]{3,}\b")
    return vec.fit(docs)


def build_bundle(segments: pd.DataFrame, meetings: pd.DataFrame, selected: dict[str, str],
                 thresholds: dict[str, float], oof: pd.DataFrame, metrics: dict[str, Any]) -> dict[str, Any]:
    """Refits the selected models on every meeting and packs what inference needs."""
    models = {}
    for task, name in selected.items():
        x = model_input(segments, task)
        models[task] = clone(candidates(task)[name]).fit(x, segments[task].to_numpy())
    return {
        "models": models,
        "reference": {"risk": np.sort(oof["risk_raw"].to_numpy()),
                      "opportunity": np.sort(oof["opportunity_raw"].to_numpy())},
        "terms": terms_vectorizer(segments),
        "meta": {"selected": selected, "thresholds": thresholds, "metrics": metrics,
                 "meetings": int(len(meetings)), "segments": int(len(segments)),
                 "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
    }
