"""Transcript ingestion: parsing, cleaning and segmentation.

Accepts the TOTVS export format (``[LOCUTOR N]: fala``), the synthetic
``Nome: fala`` plain-text format, WebVTT and SubRip, and turns any of them into
a list of speech turns. Turns are grouped into *segments* of roughly
``target_words`` words: the unit the classifiers score, small enough to carry
a single intent and large enough to carry context.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "packages" / "nlp-baseline" / "src"))
from nlp_baseline.normalize import normalize_text  # noqa: E402
from nlp_baseline.stopwords import get_ptbr_stopwords_normalized  # noqa: E402

_LOCUTOR_RE = re.compile(r"\[\s*LOCUTOR\s*(\d+)\s*\]\s*:?", re.IGNORECASE)
_VTT_VOICE_RE = re.compile(r"^<v\s+([^>]+)>(.*)$")
_NAMED_SPEAKER_RE = re.compile(r"^([A-ZÀ-Ú][\wÀ-ú.]*(?:\s[A-ZÀ-Ú][\wÀ-ú.]*){0,3})\s*:\s+(.+)$")
_TIMESTAMP_RE = re.compile(r"^\d{2}:\d{2}(:\d{2})?[.,]\d{3}\s*-->")
_HEADER_RE = re.compile(r"^(Reunião|Data|Horário|Participantes|WEBVTT|NOTE)\b", re.IGNORECASE)
_PLACEHOLDER_RE = re.compile(r"\[(PESSOA|EMPRESA|LOCAL|LOCUTOR\s*\d*)\]", re.IGNORECASE)
# The speech-to-text engine spells the vendor name several ways ("Totus", "Totos", "Tots").
_VENDOR_RE = re.compile(r"\b(totvs|totus|totos|tots|totes)\b")

# Negations and intensifiers carry sentiment, so they stay in the vocabulary.
_KEEP = {"nao", "nunca", "sem", "mal", "pouco", "muito", "mais", "menos", "bem", "sim"}
# Filler words of spoken Portuguese that dominate any transcript and discriminate nothing.
_FILLERS = {
    "ne", "ta", "tá", "entao", "ai", "assim", "gente", "coisa", "aqui", "la", "pra", "pro", "pras",
    "pros", "beleza", "ok", "okay", "tipo", "ah", "eh", "uhum", "hum", "opa", "olha", "cara", "tô",
    "to", "ce", "voce", "voces", "vc", "ja", "vai", "vou", "fazer", "tem", "ter", "estar", "esta",
    "isso", "aquilo", "nessa", "nesse", "dessa", "desse", "daqui", "tudo", "todo", "toda", "pessoal",
    "certo", "exatamente", "verdade", "realmente", "acho", "sabe", "entendeu", "falar", "falando",
    "dizer", "deixa", "ver", "vamos", "tambem", "porque", "agora", "hoje", "dia", "vez", "coisas",
    "pessoa", "empresa", "local", "locutor", "minutinho", "obrigado", "obrigada", "tarde", "bom",
    "boa", "dia", "noite", "oi", "ola", "fala", "alo", "consegue", "ouvir", "ouvindo", "tela",
}
STOPWORDS = sorted((set(get_ptbr_stopwords_normalized()) - _KEEP) | _FILLERS)


@dataclass(frozen=True)
class Turn:
    speaker: str
    text: str


@dataclass(frozen=True)
class Segment:
    index: int
    text: str            # raw speech, shown to the user as evidence
    clean: str           # normalized text, fed to the models
    speakers: int
    words: int


def parse_turns(raw: str) -> list[Turn]:
    """Splits a transcript in any supported format into speaker turns."""
    if _LOCUTOR_RE.search(raw):
        return _parse_locutor(raw)
    turns: list[Turn] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.isdigit() or _TIMESTAMP_RE.match(line) or _HEADER_RE.match(line):
            continue
        voice = _VTT_VOICE_RE.match(line)
        if voice:
            turns.append(Turn(voice.group(1).strip(), re.sub(r"</v>$", "", voice.group(2)).strip()))
            continue
        named = _NAMED_SPEAKER_RE.match(line)
        if named:
            turns.append(Turn(named.group(1), named.group(2)))
        elif turns:
            # Continuation of the previous speaker (wrapped subtitle line).
            last = turns[-1]
            turns[-1] = Turn(last.speaker, f"{last.text} {line}")
        else:
            turns.append(Turn("?", line))
    return turns


def _parse_locutor(raw: str) -> list[Turn]:
    turns: list[Turn] = []
    parts = _LOCUTOR_RE.split(raw)
    # parts = [prefix, id1, text1, id2, text2, ...]
    for speaker, text in zip(parts[1::2], parts[2::2]):
        text = " ".join(text.split())
        if text:
            turns.append(Turn(f"LOCUTOR {speaker}", text))
    return turns


def clean_text(text: str) -> str:
    """Normalization shared by training and inference (same function, same tokens)."""
    text = _PLACEHOLDER_RE.sub(" ", text)
    text = normalize_text(text, remove_digits=True)
    return _VENDOR_RE.sub("totvs", text)


def segment(turns: list[Turn], target_words: int = 120) -> list[Segment]:
    """Groups consecutive turns into segments of about ``target_words`` words."""
    segments: list[Segment] = []
    buf: list[Turn] = []
    words = 0
    for turn in _split_long_turns(turns, target_words):
        buf.append(turn)
        words += len(turn.text.split())
        if words >= target_words:
            segments.append(_make_segment(len(segments), buf, words))
            buf, words = [], 0
    # A short tail is merged into the previous segment instead of standing alone.
    if buf:
        if segments and words < target_words // 3:
            prev = segments.pop()
            merged_text = f"{prev.text} {' '.join(t.text for t in buf)}"
            segments.append(Segment(prev.index, merged_text, clean_text(merged_text),
                                    max(prev.speakers, len({t.speaker for t in buf})), prev.words + words))
        else:
            segments.append(_make_segment(len(segments), buf, words))
    return segments


def _split_long_turns(turns: list[Turn], target_words: int) -> list[Turn]:
    """Some exports lose the line breaks and arrive as one huge turn; cut those into windows."""
    out: list[Turn] = []
    for turn in turns:
        tokens = turn.text.split()
        if len(tokens) <= 2 * target_words:
            out.append(turn)
            continue
        for start in range(0, len(tokens), target_words):
            out.append(Turn(turn.speaker, " ".join(tokens[start:start + target_words])))
    return out


def _make_segment(index: int, turns: list[Turn], words: int) -> Segment:
    text = " ".join(t.text for t in turns)
    return Segment(index, text, clean_text(text), len({t.speaker for t in turns}), words)


def read_transcript_file(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")
