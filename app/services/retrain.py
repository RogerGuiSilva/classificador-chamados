from typing import Any

import pandas as pd

from app import db
from app.config import Settings
from app.engines.ml_engine import MLEngine
from app.services.preprocess import normalize_text
from ml.train import TARGETS, load_splits, treinar

FEEDBACK_STATUS_LABEL_SOURCE = {"categoria": "categoria_corrigida", "prioridade": "prioridade_corrigida"}


class NoFeedbackError(RuntimeError):
    pass


def label_from_feedback(row: dict[str, Any], target: str) -> str | None:
    corrected = row[FEEDBACK_STATUS_LABEL_SOURCE[target]]
    if corrected:
        return corrected
    return row[target] if row["status"] == "confirmado" else None


def build_extras(feedback: list[dict[str, Any]], frozen_texts: set[str]) -> dict[str, pd.DataFrame]:
    extras: dict[str, pd.DataFrame] = {}
    for target in TARGETS:
        latest: dict[str, dict[str, str]] = {}
        for row in feedback:
            label = label_from_feedback(row, target)
            key = normalize_text(row["descricao"])
            if label and key not in frozen_texts:
                latest[key] = {"texto": row["descricao"], "rotulo": label}
        extras[target] = pd.DataFrame(list(latest.values()), columns=["texto", "rotulo"])
    return extras


def retreinar(
    settings: Settings,
    ml_engine: MLEngine,
    nomes: tuple[str, ...] | None = None,
    cv_folds: int = 5,
) -> dict[str, Any]:
    feedback = db.training_feedback(settings.database_path)
    train_df, test_df = load_splits(settings.data_dir)
    frozen_texts = {normalize_text(text) for text in test_df["texto"]}
    extras = build_extras(feedback, frozen_texts)
    if all(extra.empty for extra in extras.values()):
        raise NoFeedbackError("Nenhuma correção ou confirmação disponível para retreinar")
    before = ml_engine.metrics()
    after = treinar(train_df, test_df, settings.models_dir, extras=extras, nomes=nomes, cv_folds=cv_folds)
    ml_engine.reload()
    return {
        "correcoes_incorporadas": {target: len(extras[target]) for target in TARGETS},
        "antes": before,
        "depois": after,
    }
