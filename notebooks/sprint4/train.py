"""Trains the three segment classifiers on the TOTVS base and writes ``artifacts/``.

    python train.py                 # full run
    TOTVS_JSON=/caminho.json python train.py

Writes ``artifacts/nora_insights.joblib`` (models + reference distributions),
``artifacts/metrics.json`` and ``artifacts/meetings_oof.csv``. Nothing in
``artifacts/`` is versioned: it is derived from the TOTVS data, which is not.
"""

from __future__ import annotations

import json
import time

import joblib
import pandas as pd

from nora_insights.data import build_segments, load_meetings
from nora_insights.models import TASKS, group_split
from nora_insights.pipeline import ARTIFACTS, BUNDLE
from nora_insights.training import build_bundle, compare_models, out_of_fold_meetings

SEGMENTS_CACHE = ARTIFACTS / "segments.pkl"


def load_segments() -> tuple[pd.DataFrame, pd.DataFrame]:
    meetings = load_meetings()
    if SEGMENTS_CACHE.exists():
        return meetings, pd.read_pickle(SEGMENTS_CACHE)
    segments = build_segments(meetings)
    ARTIFACTS.mkdir(exist_ok=True)
    segments.to_pickle(SEGMENTS_CACHE)
    return meetings, segments


def main() -> None:
    started = time.time()
    meetings, segments = load_segments()
    print(f"{len(meetings)} reunioes, {len(segments)} segmentos ({time.time() - started:.0f}s)")
    train_idx, test_idx = group_split(segments)

    selected, thresholds, metrics = {}, {}, {}
    for task in TASKS:
        result = compare_models(segments, task, train_idx, test_idx)
        selected[task] = result["selected"]
        thresholds[task] = result["thresholds"][result["selected"]]
        metrics[task] = {"cv": result["cv"].reset_index().to_dict(orient="records"),
                         "test": {k: v.row() for k, v in result["test"].items()},
                         "selected": result["selected"]}
        print(f"\n== {TASKS[task]['name']} — selecionado: {result['selected']} ({time.time() - started:.0f}s)")
        print(pd.DataFrame([v.row() for v in result["test"].values()]).round(3).to_string(index=False))

    oof = out_of_fold_meetings(segments, meetings, selected)
    ARTIFACTS.mkdir(exist_ok=True)
    oof.to_csv(ARTIFACTS / "meetings_oof.csv")
    joblib.dump(build_bundle(segments, meetings, selected, thresholds, oof, metrics), BUNDLE, compress=3)
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    print(f"\nArtefatos em {ARTIFACTS} ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
