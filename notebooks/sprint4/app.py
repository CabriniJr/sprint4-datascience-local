"""Functional prototype: paste or upload a transcript, get the commercial readout.

    streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from nora_insights.data import SYNTHETIC_DIR
from nora_insights.pipeline import NoraInsights

EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"
PRIORITY_ICON ={"alta": "🔴", "media": "🟠", "baixa": "⚪"}

st.set_page_config(page_title="NORA Insights", page_icon="📈", layout="wide")


@st.cache_resource
def engine() -> NoraInsights:
    return NoraInsights.load()


def examples() -> dict[str, str]:
    files = (sorted(EXAMPLES_DIR.glob("*.txt")) + sorted(SYNTHETIC_DIR.glob("*.txt"))
             + sorted(SYNTHETIC_DIR.parent.parent.glob("samples/*.txt")))
    return {p.stem: p.read_text(encoding="utf-8") for p in files}


st.title("NORA Insights")
st.caption("Transcricao → Processamento → Modelo → Indicadores → Recomendacao  ·  "
           "Sprint 4 · Data Science & Statistical Computing · FIAP Challenge 2026 × TOTVS")

with st.sidebar:
    st.header("Transcricao")
    source = st.radio("Origem", ["Exemplo", "Upload", "Colar texto"])
    raw = ""
    if source == "Exemplo":
        ex = examples()
        names = list(ex)
        wanted = st.query_params.get("exemplo")
        name = st.selectbox("Reuniao de exemplo (ficticia)", names,
                            index=names.index(wanted) if wanted in names else 0)
        raw = ex[name]
    elif source == "Upload":
        up = st.file_uploader("Arquivo .txt, .vtt ou .srt", type=["txt", "vtt", "srt"])
        if up:
            raw = up.read().decode("utf-8", errors="replace")
    else:
        raw = st.text_area("Cole a transcricao ([LOCUTOR N]: fala, ou Nome: fala)", height=300)
    meta = engine().meta
    st.divider()
    st.caption(f"Modelos treinados em {meta.get('meetings', '?')} reunioes TOTVS "
               f"({meta.get('segments', '?')} trechos) em {meta.get('trained_at', '?')}.")
    for task, name in meta.get("selected", {}).items():
        st.caption(f"• {task}: {name}")

if not raw.strip():
    st.info("Escolha um exemplo, envie um arquivo ou cole uma transcricao.")
    st.stop()

try:
    report = engine().analyze(raw)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

ind = report["indicadores"]
st.subheader(f"Classificacao da conta: {report['classificacao_conta']}")
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Risco de churn · {ind['risco_churn']['nivel']}", f"{ind['risco_churn']['score']:.0f}/100")
c2.metric(f"Oportunidade comercial · {ind['oportunidade_comercial']['nivel']}",
          f"{ind['oportunidade_comercial']['score']:.0f}/100")
c3.metric(f"Sentimento · {ind['sentimento']['rotulo']} ({ind['sentimento']['tendencia']})",
          f"{ind['sentimento']['indice']:+.0f}")
c4.metric("Prioridade", f"{ind['prioridade']:.0f}/100")
st.caption("Scores de risco e oportunidade sao percentis contra a base TOTVS: 85 = mais sinal do que "
           "85% das reunioes analisadas.")

st.markdown("### Recomendacoes")
for r in report["recomendacoes"]:
    st.markdown(f"{PRIORITY_ICON[r['prioridade']]} **{r['acao']}**  \n<small>{r['motivo']}</small>",
                unsafe_allow_html=True)

left, right = st.columns(2)
with left:
    st.markdown("### Produtos TOTVS identificados")
    st.write(", ".join(f"**{k}** ({v})" for k, v in report["produtos_identificados"].items()) or "Nenhum")
    st.markdown("### Concorrentes citados")
    st.write(", ".join(f"**{k}** ({v})" for k, v in report["concorrentes_citados"].items()) or "Nenhum")
    st.markdown("### Principais termos")
    st.dataframe(pd.DataFrame(report["principais_termos"]), hide_index=True, use_container_width=True)
with right:
    st.markdown("### Sentimento ao longo da reuniao")
    st.line_chart(pd.DataFrame({"polaridade": report["trajetoria_sentimento"]}))
    s = report["resumo"]
    st.caption(f"{s['turnos']} falas · {s['locutores']} locutores · {s['segmentos']} trechos · "
               f"{s['palavras']} palavras")

st.markdown("### Evidencias (trechos que mais pesaram)")
tabs = st.tabs(["Risco", "Oportunidade", "Mais negativo"])
for tab, key in zip(tabs, ["risco", "oportunidade", "mais_negativo"]):
    with tab:
        for ev in report["evidencias"][key]:
            st.markdown(f"> {ev['texto']}  \n<small>trecho {ev['trecho']} · score {ev['score']}</small>",
                        unsafe_allow_html=True)

with st.expander("Saida completa (JSON)"):
    st.download_button("Baixar JSON", json.dumps(report, ensure_ascii=False, indent=2),
                       file_name="nora_insights.json", mime="application/json")
    st.json(report)
