"""Loading the TOTVS dataset and building the segment-level training table."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from .lexicons import weak_labels
from .text import ROOT, parse_turns, segment

DEFAULT_DATASET = ROOT / "data" / "private" / "totvs" / "ANON_transcricao.json"
SYNTHETIC_DIR = ROOT / "data" / "synthetic" / "meetings"


def dataset_path() -> Path:
    return Path(os.environ.get("TOTVS_JSON", DEFAULT_DATASET))


def load_meetings(path: Path | None = None) -> pd.DataFrame:
    """One row per meeting. The export repeats a meeting once per CRM row; we keep one."""
    path = path or dataset_path()
    if not path.exists():
        raise SystemExit(
            f"Base nao encontrada em {path}.\n"
            "Coloque ANON_transcricao.json em data/private/totvs/ ou aponte com TOTVS_JSON=/caminho.json"
        )
    df = pd.read_json(path, lines=True, dtype={"ID_MEETING": str, "CODT": str})
    df["NOTA_NPS"] = pd.to_numeric(df["NOTA_NPS"], errors="coerce")
    # Several rows of the same meeting differ only in CRM columns; prefer the one that has an NPS.
    df = (df.sort_values("NOTA_NPS", na_position="last")
            .drop_duplicates(subset=["ID_MEETING", "ANON_TRANSCRICAO"])
            .drop_duplicates(subset=["ID_MEETING"])
            .reset_index(drop=True))
    df["nps_categoria"] = pd.cut(df["NOTA_NPS"], bins=[-1, 6, 8, 10],
                                 labels=["Detrator", "Neutro", "Promotor"])
    return df


def build_segments(meetings: pd.DataFrame, target_words: int = 120) -> pd.DataFrame:
    """Segments every meeting and attaches the weak labels to each segment."""
    rows: list[dict] = []
    for meeting_id, raw in zip(meetings["ID_MEETING"], meetings["ANON_TRANSCRICAO"]):
        for seg in segment(parse_turns(raw or ""), target_words=target_words):
            if len(seg.clean.split()) < 15:
                continue
            rows.append({"meeting_id": meeting_id, "seg_index": seg.index, "text": seg.text,
                         "clean": seg.clean, "words": seg.words, **weak_labels(seg.clean)})
    return pd.DataFrame(rows)


def load_synthetic() -> dict[str, str]:
    """The 17 synthetic meetings (``.txt`` version) — transcripts no model was trained on."""
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted(SYNTHETIC_DIR.glob("*.txt"))}
