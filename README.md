# NORA Insights — protótipo (Sprint 4 · Data Science)

FIAP Challenge 2026 × TOTVS · Data Science & Statistical Computing · Sprint 4.

Recebe uma transcrição de reunião e devolve: produto TOTVS identificado, sentimento, risco de
churn, oportunidade comercial, principais termos e recomendação de ação.

```
Transcrição → Processamento → Modelo → Indicadores → Recomendação
```

| Integrante | RM |
|---|---|
| Luigi Mendes Cabrini | 563552 |
| Bruno Koeke | 561309 |
| Rogério Cruz Arroyo | 563517 |
| Anthony Sforzin | 562096 |

## Rodar

```bash
pip install -r requirements.txt
streamlit run notebooks/sprint4/app.py
```

Ou pela linha de comando: `python notebooks/sprint4/analyze.py notebooks/sprint4/examples/novo-01-protheus-insatisfacao.txt`.

Os modelos treinados estão em `notebooks/sprint4/artifacts/`. A base de transcrições da TOTVS
**não** está neste repositório; o relatório, o notebook executado e o código de treino estão em
`notebooks/sprint4/` (veja o README de lá).
