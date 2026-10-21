"""Helpers for the notebook: figures and safe display of transcript excerpts."""

from __future__ import annotations

import re

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

# Capitalized words the anonymization missed (residual names) are masked before display.
_ALLOWED = {"TOTVS", "Protheus", "RM", "Fluig", "Datasul", "Logix", "Winthor", "RD", "Station", "SAP",
            "Oracle", "Cloud", "Techfin", "Eu", "Ele", "Ela", "Então", "Mas", "Não", "Sim", "Tá", "Olha",
            "Isso", "A", "O", "E", "É", "Ok", "Aí", "Né", "Bom", "Boa", "Agora", "Hoje", "Quando", "Se"}
_CAPITALIZED = re.compile(r"(?<![.!?]\s)(?<!^)\b([A-ZÀ-Ú][a-zà-ú]{2,})\b")

PALETTE = {"primary": "#1f5fa8", "secondary": "#e07b39", "neutral": "#8c8c8c", "ok": "#2e8b57",
           "bad": "#c0392b"}


def scrub(text: str, limit: int = 260) -> str:
    """Shortens an excerpt and masks capitalized tokens that may be residual personal names."""
    text = " ".join(text.split())[:limit]
    text = _CAPITALIZED.sub(lambda m: m.group(1) if m.group(1) in _ALLOWED else "[NOME]", text)
    return text + (" ..." if len(text) == limit else "")


def style() -> None:
    plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})


def confusion_grid(evaluations: dict, label_names: list[str], title: str) -> plt.Figure:
    fig, axes = plt.subplots(1, len(evaluations), figsize=(4.2 * len(evaluations), 3.6))
    for ax, (name, ev) in zip(np.atleast_1d(axes), evaluations.items()):
        ConfusionMatrixDisplay(ev.confusion, display_labels=label_names).plot(
            ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(f"{name}\nF1={ev.f1:.2f}  recall={ev.recall:.2f}", fontsize=10)
        ax.grid(False)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig
