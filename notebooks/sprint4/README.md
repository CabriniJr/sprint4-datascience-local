# NORA Insights — Sprint 4 de Data Science

Entregável da disciplina **Data Science & Statistical Computing** (FIAP Challenge 2026 × TOTVS,
Sprint 4): uma solução que recebe **uma transcrição nova** e devolve informação útil para a área
comercial.

```
Transcrição → Processamento → Modelo → Indicadores → Recomendação
   .txt/.vtt     parsing, limpeza,   3 classificadores   risco, oportunidade,   ações com motivo
   .srt/TOTVS    trechos de ~120     (TF-IDF + modelo    sentimento, produtos,  e evidência
                 palavras            selecionado)        termos, prioridade
```

| Arquivo | O que é |
|---|---|
| `sprint4_pipeline_insights.ipynb` | **O relatório**: pipeline, alvos, comparação de modelos, scoring, protótipo, validação, recomendações e limitações — executado, com saídas |
| `sprint4_pipeline_insights.py` | O mesmo notebook em formato *percent* (roda como script e abre no Jupyter/VS Code) |
| `nora_insights/` | O pacote: `text.py` (parsing/limpeza/trechos), `lexicons.py` (supervisão fraca, produtos, concorrentes), `models.py` (candidatos, protocolo, limiar), `training.py`, `scoring.py` (indicadores e recomendações), `pipeline.py` (inferência) |
| `train.py` | Treina e grava `artifacts/` sem passar pelo notebook |
| `analyze.py` | CLI: analisa um arquivo de transcrição |
| `app.py` | Protótipo web (Streamlit) |
| `figures/` | Gráficos gerados pelo notebook |

## Como rodar

```bash
pip install -r requirements.txt
# base TOTVS (não versionada): data/private/totvs/ANON_transcricao.json
python train.py                       # ~15 min; ou execute o notebook, que faz o mesmo
python analyze.py ../../data/synthetic/meetings/04-acme-churn-suporte-performance.txt
streamlit run app.py
```

`TOTVS_JSON=/outro/caminho.json` aponta para a base em outro lugar.

## Decisões que valem saber antes de ler o código

- **Rótulos por supervisão fraca, com mascaramento.** A base não tem rótulo por trecho. Padrões
  escritos à mão geram os rótulos, e **esses padrões são apagados do texto antes do treino**, para
  que o modelo aprenda o contexto, e não a própria regra. Sem a máscara o F1 sobe para 0,80–0,91 e
  não significa nada; o notebook mostra as duas versões.
- **Toda divisão é agrupada por reunião.** Nenhum trecho de uma reunião de teste aparece no treino.
- **Scores são percentis contra a base TOTVS**, não probabilidades: os modelos foram treinados com
  rótulos fracos e não são calibrados como "chance de cancelar".
- **Produto e concorrente são reconhecidos por dicionário**, não por classificador: o vocabulário é
  fechado, e o dicionário cobre as grafias do reconhecimento de voz ("proteus" → Protheus).
- **Nada derivado dos dados TOTVS é versionado.** `artifacts/` (modelos, cache de trechos, scores por
  reunião) está no `.gitignore`, como `data/private/`.
- O texto passa pela mesma `normalize_text` do pacote `packages/nlp-baseline`, usada no worker da
  NORA (ADR 0010).
