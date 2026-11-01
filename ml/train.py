import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from app.config import get_settings
from app.services.preprocess import normalize_text

SEED = 42
TARGETS = ("categoria", "prioridade")


def build_features() -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "palavras",
                TfidfVectorizer(preprocessor=normalize_text, ngram_range=(1, 2), sublinear_tf=True),
            ),
            (
                "caracteres",
                TfidfVectorizer(
                    preprocessor=normalize_text, analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True
                ),
            ),
        ]
    )


def build_candidates(seed: int = SEED) -> dict[str, Pipeline]:
    classifiers = {
        "logistic_regression": LogisticRegression(C=10, max_iter=2000, class_weight="balanced"),
        "linear_svc_calibrado": CalibratedClassifierCV(LinearSVC(C=1.0, random_state=seed), cv=3),
        "multinomial_nb": MultinomialNB(alpha=0.1),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
    }
    return {
        name: Pipeline([("features", build_features()), ("classifier", classifier)])
        for name, classifier in classifiers.items()
    }


def evaluate_holdout(model: Pipeline, texts: list[str], labels: list[str]) -> dict[str, Any]:
    predictions = model.predict(texts)
    classes = [str(label) for label in model.classes_]
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=classes, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "por_classe": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(classes)
        },
        "matriz_confusao": {
            "labels": classes,
            "matriz": confusion_matrix(labels, predictions, labels=classes).tolist(),
        },
    }


def cross_validate_candidates(
    candidates: dict[str, Pipeline], texts: list[str], labels: list[str], cv_folds: int, seed: int
) -> dict[str, dict[str, float]]:
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    summary = {}
    for name, pipeline in candidates.items():
        scores = cross_validate(
            pipeline, texts, labels, cv=splitter, scoring=["accuracy", "f1_macro"], n_jobs=1
        )
        summary[name] = {
            "accuracy_media": float(np.mean(scores["test_accuracy"])),
            "f1_macro_media": float(np.mean(scores["test_f1_macro"])),
            "f1_macro_desvio": float(np.std(scores["test_f1_macro"])),
        }
    return summary


def train_target(
    texts: list[str],
    labels: list[str],
    test_texts: list[str],
    test_labels: list[str],
    nomes: tuple[str, ...] | None,
    cv_folds: int,
    seed: int,
) -> tuple[Pipeline, dict[str, Any]]:
    candidates = build_candidates(seed)
    if nomes is not None:
        candidates = {name: candidates[name] for name in nomes}
    cv_summary = cross_validate_candidates(candidates, texts, labels, cv_folds, seed)
    best_name = max(cv_summary, key=lambda name: cv_summary[name]["f1_macro_media"])
    best_model = candidates[best_name].fit(texts, labels)
    report = {
        "melhor_modelo": best_name,
        "n_treino": len(texts),
        "validacao_cruzada": cv_summary,
        "holdout": evaluate_holdout(best_model, test_texts, test_labels),
    }
    return best_model, report


def save_atomically(model: Pipeline, path: Path) -> None:
    temporary = path.with_suffix(".tmp")
    joblib.dump(model, temporary)
    os.replace(temporary, path)


def treinar(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    models_dir: Path,
    extras: dict[str, pd.DataFrame] | None = None,
    nomes: tuple[str, ...] | None = None,
    cv_folds: int = 5,
    seed: int = SEED,
) -> dict[str, Any]:
    models_dir.mkdir(parents=True, exist_ok=True)
    extras = extras or {}
    metrics: dict[str, Any] = {
        "treinado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_teste": len(test_df),
        "n_extras": {},
    }
    for target in TARGETS:
        texts = train_df["texto"].tolist()
        labels = train_df[target].tolist()
        extra = extras.get(target)
        if extra is not None and not extra.empty:
            texts += extra["texto"].tolist()
            labels += extra["rotulo"].tolist()
        metrics["n_extras"][target] = 0 if extra is None else len(extra)
        model, report = train_target(
            texts, labels, test_df["texto"].tolist(), test_df[target].tolist(), nomes, cv_folds, seed
        )
        save_atomically(model, models_dir / f"{target}.joblib")
        metrics[target] = report
    (models_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def load_splits(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(data_dir / "train.csv"), pd.read_csv(data_dir / "test_frozen.csv")


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=settings.data_dir)
    parser.add_argument("--models-dir", type=Path, default=settings.models_dir)
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()
    train_df, test_df = load_splits(args.data_dir)
    metrics = treinar(train_df, test_df, args.models_dir, cv_folds=args.cv_folds)
    for target in TARGETS:
        holdout = metrics[target]["holdout"]
        print(
            f"{target}: {metrics[target]['melhor_modelo']} "
            f"accuracy={holdout['accuracy']:.3f} f1_macro={holdout['f1_macro']:.3f}"
        )


if __name__ == "__main__":
    main()
