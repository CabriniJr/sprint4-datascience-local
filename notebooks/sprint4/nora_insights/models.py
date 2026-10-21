"""Segment classifiers: candidate models, evaluation protocol and threshold choice.

Three tasks, each trained on the weak labels of ``lexicons`` with the labeling
phrases masked out of the input:

* ``risk``         — risco de churn / insatisfacao (binary, rare class)
* ``opportunity``  — oportunidade comercial (binary)
* ``sentiment``    — positivo / neutro / negativo (3 classes)

Every split is grouped by meeting: no segment of a test meeting is ever seen in
training, so the numbers describe how the model behaves on a *new* transcript.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

from .lexicons import mask
from .text import STOPWORDS

RANDOM_STATE = 42

TASKS = {
    "risk": {"name": "Risco de churn", "mask": ("risk",), "positive": 1, "labels": [0, 1],
             "label_names": ["neutro", "risco"]},
    "opportunity": {"name": "Oportunidade comercial", "mask": ("opportunity",), "positive": 1,
                    "labels": [0, 1], "label_names": ["nao oportunidade", "oportunidade"]},
    "sentiment": {"name": "Sentimento", "mask": ("positive", "negative"), "positive": None,
                  "labels": ["negativo", "neutro", "positivo"],
                  "label_names": ["negativo", "neutro", "positivo"]},
}


def vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(stop_words=STOPWORDS, ngram_range=(1, 2), min_df=5, max_df=0.5,
                           sublinear_tf=True, max_features=30000, token_pattern=r"(?u)\b[a-z]{2,}\b")


def candidates(task: str) -> dict[str, Pipeline]:
    multiclass = TASKS[task]["positive"] is None
    return {
        "Naive Bayes": Pipeline([("tfidf", vectorizer()), ("clf", ComplementNB(alpha=0.5))]),
        "Regressao Logistica": Pipeline([
            ("tfidf", vectorizer()),
            ("clf", LogisticRegression(C=2.0, class_weight="balanced", max_iter=3000,
                                       solver="lbfgs" if multiclass else "liblinear")),
        ]),
        "Random Forest": Pipeline([
            ("tfidf", vectorizer()),
            ("kbest", SelectKBest(chi2, k=5000)),
            ("clf", RandomForestClassifier(n_estimators=200, min_samples_leaf=2, max_features="sqrt",
                                           class_weight="balanced_subsample", n_jobs=-1,
                                           random_state=RANDOM_STATE)),
        ]),
    }


def model_input(segments: pd.DataFrame, task: str, masked: bool = True) -> pd.Series:
    if not masked:
        return segments["clean"]
    return segments["clean"].map(lambda c: mask(c, TASKS[task]["mask"]))


def group_split(segments: pd.DataFrame, test_size: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    return next(gss.split(segments, groups=segments["meeting_id"]))


@dataclass
class Evaluation:
    model: str
    task: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    threshold: float
    confusion: np.ndarray

    def row(self) -> dict:
        return {"modelo": self.model, "accuracy": self.accuracy, "precision": self.precision,
                "recall": self.recall, "f1": self.f1, "roc_auc": self.roc_auc,
                "pr_auc": self.pr_auc, "threshold": self.threshold}


def scores(pipe: Pipeline, x: pd.Series, task: str) -> np.ndarray:
    """Probability of the positive class (binary) or the full probability matrix (sentiment)."""
    proba = pipe.predict_proba(x)
    if TASKS[task]["positive"] is None:
        order = [list(pipe.classes_).index(c) for c in TASKS[task]["labels"]]
        return proba[:, order]
    return proba[:, list(pipe.classes_).index(1)]


def evaluate(y: np.ndarray, s: np.ndarray, task: str, model: str, threshold: float = 0.5) -> Evaluation:
    spec = TASKS[task]
    if spec["positive"] is None:
        labels = np.array(spec["labels"])
        pred = labels[s.argmax(axis=1)]
        y_bin = np.stack([(y == c).astype(int) for c in labels], axis=1)
        return Evaluation(model, task, accuracy_score(y, pred),
                          precision_score(y, pred, average="macro", zero_division=0),
                          recall_score(y, pred, average="macro", zero_division=0),
                          f1_score(y, pred, average="macro", zero_division=0),
                          roc_auc_score(y_bin, s, average="macro"),
                          average_precision_score(y_bin, s, average="macro"),
                          float("nan"), confusion_matrix(y, pred, labels=labels))
    pred = (s >= threshold).astype(int)
    return Evaluation(model, task, accuracy_score(y, pred), precision_score(y, pred, zero_division=0),
                      recall_score(y, pred, zero_division=0), f1_score(y, pred, zero_division=0),
                      roc_auc_score(y, s), average_precision_score(y, s), threshold,
                      confusion_matrix(y, pred, labels=[0, 1]))


def choose_threshold(y: np.ndarray, s: np.ndarray, task: str) -> float:
    """Risk favours recall (a missed churn costs more than a false alarm); opportunity maximizes F1.

    For risk, the threshold is the highest one that still recovers at least 70% of the
    risky segments, which is the best precision available at that recall.
    """
    precision, recall, thresholds = precision_recall_curve(y, s)
    precision, recall = precision[:-1], recall[:-1]
    if task == "risk":
        ok = np.where(recall >= 0.70)[0]
        return float(thresholds[ok[-1]]) if len(ok) else 0.5
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-9, None)
    return float(thresholds[int(np.nanargmax(f1))])


def cross_validate(pipe: Pipeline, x: pd.Series, y: np.ndarray, groups: pd.Series, task: str,
                   n_splits: int = 3) -> tuple[list[Evaluation], np.ndarray]:
    """Grouped K-fold; returns per-fold evaluations and the out-of-fold scores."""
    oof = None
    folds: list[Evaluation] = []
    for train_idx, val_idx in GroupKFold(n_splits=n_splits).split(x, y, groups):
        fitted = clone(pipe).fit(x.iloc[train_idx], y[train_idx])
        s = scores(fitted, x.iloc[val_idx], task)
        if oof is None:
            oof = np.zeros((len(x),) + s.shape[1:])
        oof[val_idx] = s
        folds.append(evaluate(y[val_idx], s, task, "fold"))
    return folds, oof


def top_features(pipe: Pipeline, task: str, n: int = 20) -> dict[str, list[tuple[str, float]]]:
    """Most influential terms of a fitted logistic regression, per class."""
    clf = pipe.named_steps["clf"]
    names = pipe.named_steps["tfidf"].get_feature_names_out()
    if "kbest" in pipe.named_steps:
        names = names[pipe.named_steps["kbest"].get_support()]
    if not hasattr(clf, "coef_"):
        imp = clf.feature_importances_
        idx = np.argsort(imp)[::-1][:n]
        return {"importancia": [(names[i], float(imp[i])) for i in idx]}
    if clf.coef_.shape[0] == 1:
        coef = clf.coef_[0]
        return {"positiva": [(names[i], float(coef[i])) for i in np.argsort(coef)[::-1][:n]],
                "negativa": [(names[i], float(coef[i])) for i in np.argsort(coef)[:n]]}
    return {str(c): [(names[i], float(clf.coef_[k][i])) for i in np.argsort(clf.coef_[k])[::-1][:n]]
            for k, c in enumerate(clf.classes_)}
