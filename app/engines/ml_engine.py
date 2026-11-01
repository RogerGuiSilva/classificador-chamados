import json
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from app.engines.base import Engine
from app.schemas import Categoria, Prioridade, Resultado, TermoExplicativo
from app.services.rules import suggested_action

WORD_FEATURE_PREFIX = "palavras__"


class ModelsNotLoadedError(RuntimeError):
    pass


def class_weights(classifier: Any, class_index: int) -> np.ndarray | None:
    if hasattr(classifier, "coef_"):
        return classifier.coef_[class_index]
    if hasattr(classifier, "calibrated_classifiers_"):
        return np.mean(
            [item.estimator.coef_[class_index] for item in classifier.calibrated_classifiers_], axis=0
        )
    if hasattr(classifier, "feature_log_prob_"):
        log_prob = classifier.feature_log_prob_
        others = np.delete(log_prob, class_index, axis=0).mean(axis=0)
        return log_prob[class_index] - others
    if hasattr(classifier, "feature_importances_"):
        return classifier.feature_importances_
    return None


def top_contributing_terms(
    pipeline: Pipeline, text: str, class_index: int, limit: int
) -> list[TermoExplicativo]:
    features = pipeline.named_steps["features"]
    weights = class_weights(pipeline.named_steps["classifier"], class_index)
    if weights is None:
        return []
    row = features.transform([text]).toarray()[0]
    names = features.get_feature_names_out()
    contributions = row * weights
    ranked = np.argsort(contributions)[::-1]
    terms = []
    for index in ranked:
        if contributions[index] <= 0 or len(terms) >= limit:
            break
        if names[index].startswith(WORD_FEATURE_PREFIX):
            terms.append(
                TermoExplicativo(
                    termo=names[index].removeprefix(WORD_FEATURE_PREFIX),
                    peso=round(float(contributions[index]), 4),
                )
            )
    return terms


class MLEngine(Engine):
    name = "ml"

    def __init__(self, models_dir: Path, top_terms: int = 5) -> None:
        self._models_dir = models_dir
        self._top_terms = top_terms
        self._category_model: Pipeline | None = None
        self._priority_model: Pipeline | None = None
        self.reload()

    def reload(self) -> None:
        self._category_model = self._load("categoria")
        self._priority_model = self._load("prioridade")

    def _load(self, target: str) -> Pipeline | None:
        path = self._models_dir / f"{target}.joblib"
        return joblib.load(path) if path.exists() else None

    def is_available(self) -> bool:
        return self._category_model is not None and self._priority_model is not None

    def loaded_models(self) -> dict[str, bool]:
        return {
            "categoria": self._category_model is not None,
            "prioridade": self._priority_model is not None,
        }

    def metrics(self) -> dict[str, Any] | None:
        path = self._models_dir / "metrics.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def classify(self, text: str) -> Resultado:
        if not self.is_available():
            raise ModelsNotLoadedError("Modelos de ML não encontrados. Execute o treinamento.")
        started = perf_counter()
        category, category_confidence, category_terms = self._predict(self._category_model, text)
        priority, priority_confidence, priority_terms = self._predict(self._priority_model, text)
        categoria = Categoria(category)
        prioridade = Prioridade(priority)
        return Resultado(
            motor=self.name,
            categoria=categoria,
            prioridade=prioridade,
            confianca=min(category_confidence, priority_confidence),
            confianca_categoria=category_confidence,
            confianca_prioridade=priority_confidence,
            acao_sugerida=suggested_action(categoria, prioridade),
            termos_ml=_merge_terms(category_terms, priority_terms, self._top_terms),
            latencia_ms=(perf_counter() - started) * 1000,
        )

    def _predict(self, pipeline: Pipeline, text: str) -> tuple[str, float, list[TermoExplicativo]]:
        probabilities = pipeline.predict_proba([text])[0]
        index = int(np.argmax(probabilities))
        label = str(pipeline.classes_[index])
        terms = top_contributing_terms(pipeline, text, index, self._top_terms)
        return label, float(min(max(probabilities[index], 0.0), 1.0)), terms


def _merge_terms(
    first: list[TermoExplicativo], second: list[TermoExplicativo], limit: int
) -> list[TermoExplicativo]:
    best: dict[str, float] = {}
    for term in first + second:
        best[term.termo] = max(best.get(term.termo, 0.0), term.peso)
    ordered = sorted(best.items(), key=lambda item: item[1], reverse=True)[:limit]
    return [TermoExplicativo(termo=name, peso=weight) for name, weight in ordered]
