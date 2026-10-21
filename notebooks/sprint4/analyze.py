"""Analyzes one transcript from the command line.

    python analyze.py ../../data/synthetic/meetings/04-acme-churn-suporte-performance.txt
    python analyze.py reuniao.vtt --json > relatorio.json
    type reuniao.txt | python analyze.py -
"""

from __future__ import annotations

import argparse
import json
import sys

from nora_insights.pipeline import NoraInsights


def render(report: dict) -> str:
    ind = report["indicadores"]
    lines = [
        f"Classificacao da conta: {report['classificacao_conta']}  (prioridade {ind['prioridade']:.0f}/100)",
        f"  Risco de churn ........ {ind['risco_churn']['score']:5.1f}/100  {ind['risco_churn']['nivel']}",
        f"  Oportunidade comercial  {ind['oportunidade_comercial']['score']:5.1f}/100  "
        f"{ind['oportunidade_comercial']['nivel']}",
        f"  Sentimento ............ {ind['sentimento']['indice']:+5.1f}  {ind['sentimento']['rotulo']} "
        f"(tendencia: {ind['sentimento']['tendencia']})",
        "Produtos: " + (", ".join(f"{k} ({v})" for k, v in report["produtos_identificados"].items()) or "nenhum"),
        "Concorrentes: " + (", ".join(report["concorrentes_citados"]) or "nenhum"),
        "Termos: " + ", ".join(t["termo"] for t in report["principais_termos"]),
        "Recomendacoes:",
    ]
    lines += [f"  [{r['prioridade'].upper():5}] {r['acao']}\n          motivo: {r['motivo']}"
              for r in report["recomendacoes"]]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="NORA Insights — analise de uma transcricao")
    parser.add_argument("arquivo", help="transcricao .txt/.vtt/.srt, ou - para stdin")
    parser.add_argument("--json", action="store_true", help="imprime o relatorio completo em JSON")
    args = parser.parse_args()
    if args.arquivo == "-":
        raw = sys.stdin.read()
    else:
        with open(args.arquivo, encoding="utf-8", errors="replace") as fh:
            raw = fh.read()
    report = NoraInsights.load().analyze(raw)
    if args.json:
        sys.stdout.reconfigure(encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(render(report))


if __name__ == "__main__":
    main()
