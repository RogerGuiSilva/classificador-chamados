import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from app.config import Settings, get_settings
from app.engines.gemini_engine import GeminiEngine, GeminiEngineError
from app.engines.hybrid import needs_escalation
from app.engines.ml_engine import MLEngine
from app.schemas import Categoria, Prioridade, Resultado
from app.services.rules import apply_safety_rules

SEED = 42
CATEGORY_LABELS = [item.value for item in Categoria]
PRIORITY_LABELS = [item.value for item in Prioridade]


def evaluate_target(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict[str, Any]:
    _, _, f1_per_class, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "f1_por_classe": {label: float(value) for label, value in zip(labels, f1_per_class)},
        "matriz_confusao": {"labels": labels, "matriz": confusion_matrix(y_true, y_pred, labels=labels).tolist()},
    }


def percentile_95(values: list[float]) -> float:
    return float(np.percentile(values, 95)) if values else 0.0


def final_priority(result: Resultado, text: str) -> str:
    return apply_safety_rules(result.prioridade, text).prioridade.value


def collect_llm_results(
    engine: GeminiEngine, texts: list[str], pause_seconds: float
) -> list[Resultado | None]:
    results: list[Resultado | None] = []
    for text in texts:
        try:
            results.append(engine.classify(text))
        except GeminiEngineError:
            results.append(None)
        time.sleep(pause_seconds)
    return results


def summarize_engine(
    frame: pd.DataFrame,
    predictions: list[tuple[str, str]],
    latencies: list[float],
    costs: list[float],
    escalated: list[bool] | None = None,
) -> dict[str, Any]:
    summary = {
        "categoria": evaluate_target(frame["categoria"].tolist(), [item[0] for item in predictions], CATEGORY_LABELS),
        "prioridade": evaluate_target(frame["prioridade"].tolist(), [item[1] for item in predictions], PRIORITY_LABELS),
        "latencia_media_ms": float(np.mean(latencies)) if latencies else 0.0,
        "latencia_p95_ms": percentile_95(latencies),
        "custo_por_1000_usd": float(np.sum(costs) / len(frame) * 1000) if len(frame) else 0.0,
    }
    if escalated is not None:
        summary["pct_escalado_para_llm"] = float(np.mean(escalated) * 100)
    return summary


def disagreement_analysis(
    frame: pd.DataFrame, ml_results: list[Resultado], llm_results: list[Resultado | None]
) -> dict[str, Any]:
    cases = []
    for (_, row), ml, llm in zip(frame.iterrows(), ml_results, llm_results):
        if llm is None or (ml.categoria == llm.categoria and ml.prioridade == llm.prioridade):
            continue
        truth = (row["categoria"], row["prioridade"])
        ml_right = (ml.categoria.value, ml.prioridade.value) == truth
        llm_right = (llm.categoria.value, llm.prioridade.value) == truth
        cases.append(
            {
                "texto": row["texto"],
                "verdadeiro": {"categoria": truth[0], "prioridade": truth[1]},
                "ml": {"categoria": ml.categoria.value, "prioridade": ml.prioridade.value, "confianca": round(ml.confianca, 3)},
                "llm": {"categoria": llm.categoria.value, "prioridade": llm.prioridade.value, "confianca": round(llm.confianca, 3)},
                "quem_acertou": "ambos" if ml_right and llm_right else "ml" if ml_right else "llm" if llm_right else "nenhum",
            }
        )
    winners = pd.Series([case["quem_acertou"] for case in cases], dtype="object").value_counts().to_dict()
    return {"total": len(cases), "quem_acertou": winners, "casos": cases}


def run_benchmark(
    settings: Settings,
    sample_size: int | None,
    skip_gemini: bool,
    threshold: float,
    pause_seconds: float,
) -> dict[str, Any]:
    frame = pd.read_csv(settings.data_dir / "test_frozen.csv")
    if sample_size and sample_size < len(frame):
        frame = frame.sample(n=sample_size, random_state=SEED).reset_index(drop=True)
    texts = frame["texto"].tolist()

    ml_engine = MLEngine(settings.models_dir)
    ml_results = [ml_engine.classify(text) for text in texts]
    ml_latencies = [result.latencia_ms for result in ml_results]
    ml_predictions = [(result.categoria.value, final_priority(result, text)) for result, text in zip(ml_results, texts)]

    report: dict[str, Any] = {
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_amostras": len(frame),
        "limiar_confianca": threshold,
        "gemini_incluido": False,
        "modelo_gemini": None,
        "motores": {"ml": summarize_engine(frame, ml_predictions, ml_latencies, [0.0] * len(frame))},
    }
    if skip_gemini or not settings.gemini_configured:
        return report

    gemini = GeminiEngine.from_settings(settings)
    llm_results = collect_llm_results(gemini, texts, pause_seconds)
    failures = sum(result is None for result in llm_results)
    llm_final = [llm or ml for llm, ml in zip(llm_results, ml_results)]
    llm_predictions = [(result.categoria.value, final_priority(result, text)) for result, text in zip(llm_final, texts)]
    llm_latencies = [result.latencia_ms for result in llm_results if result is not None]
    llm_costs = [result.custo_estimado for result in llm_results if result is not None]

    escalated = [needs_escalation(ml, threshold) for ml in ml_results]
    hybrid_final = [llm if flag and llm is not None else ml for ml, llm, flag in zip(ml_results, llm_results, escalated)]
    hybrid_predictions = [(result.categoria.value, final_priority(result, text)) for result, text in zip(hybrid_final, texts)]
    hybrid_latencies = [
        ml.latencia_ms + (llm.latencia_ms if flag and llm is not None else 0.0)
        for ml, llm, flag in zip(ml_results, llm_results, escalated)
    ]
    hybrid_costs = [llm.custo_estimado if flag and llm is not None else 0.0 for llm, flag in zip(llm_results, escalated)]

    report["gemini_incluido"] = True
    report["modelo_gemini"] = settings.gemini_model
    report["falhas_gemini"] = failures
    report["motores"]["llm"] = summarize_engine(frame, llm_predictions, llm_latencies, llm_costs)
    report["motores"]["hybrid"] = summarize_engine(frame, hybrid_predictions, hybrid_latencies, hybrid_costs, escalated)
    report["discordancias"] = disagreement_analysis(frame, ml_results, llm_results)
    return report


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Benchmark de motores",
        "",
        f"Gerado em {report['gerado_em']} sobre {report['n_amostras']} chamados do conjunto de teste congelado.",
        "",
        "| Motor | Acc. categoria | F1 categoria | Acc. prioridade | F1 prioridade | Latência média (ms) | p95 (ms) | Custo/1000 (USD) | % escalado |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, data in report["motores"].items():
        escalated = f"{data['pct_escalado_para_llm']:.1f}" if "pct_escalado_para_llm" in data else "-"
        lines.append(
            f"| {name} | {data['categoria']['accuracy']:.3f} | {data['categoria']['f1_macro']:.3f} | "
            f"{data['prioridade']['accuracy']:.3f} | {data['prioridade']['f1_macro']:.3f} | "
            f"{data['latencia_media_ms']:.1f} | {data['latencia_p95_ms']:.1f} | {data['custo_por_1000_usd']:.4f} | {escalated} |"
        )
    if report["gemini_incluido"]:
        analysis = report["discordancias"]
        lines += [
            "",
            f"Modelo Gemini: `{report['modelo_gemini']}` · falhas do Gemini substituídas pelo ML: {report['falhas_gemini']}",
            "",
            f"## Discordâncias ML vs Gemini ({analysis['total']})",
            "",
            f"Quem acertou: {analysis['quem_acertou']}",
        ]
    else:
        lines += ["", "Gemini não incluído nesta execução."]
    return "\n".join(lines) + "\n"


def save_report(report: dict[str, Any], reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "benchmark.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (reports_dir / "benchmark.md").write_text(render_markdown(report), encoding="utf-8")


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--amostra", type=int, default=None)
    parser.add_argument("--sem-gemini", action="store_true")
    parser.add_argument("--limiar", type=float, default=settings.ml_confidence_threshold)
    parser.add_argument("--pausa", type=float, default=0.0)
    args = parser.parse_args()
    report = run_benchmark(settings, args.amostra, args.sem_gemini, args.limiar, args.pausa)
    save_report(report, settings.reports_dir)
    print(render_markdown(report))


if __name__ == "__main__":
    main()
