import json

import pytest

from app.engines.ml_engine import MLEngine, ModelsNotLoadedError
from app.schemas import Categoria, Prioridade


def test_result_shape_and_probability_range(ml_engine):
    result = ml_engine.classify("Motor da esteira com ruído anormal e vibração")
    assert isinstance(result.categoria, Categoria)
    assert isinstance(result.prioridade, Prioridade)
    for value in (result.confianca, result.confianca_categoria, result.confianca_prioridade):
        assert 0.0 <= value <= 1.0
    assert result.confianca == min(result.confianca_categoria, result.confianca_prioridade)
    assert result.latencia_ms >= 0


def test_explanatory_terms_come_from_the_text(ml_engine):
    result = ml_engine.classify("CLP da linha de envase perdeu comunicação")
    assert result.termos_ml
    text_tokens = {"clp", "linha", "envase", "perdeu", "comunicacao"}
    assert any(set(term.termo.split()) & text_tokens for term in result.termos_ml)
    assert all(term.peso > 0 for term in result.termos_ml)


def test_obvious_cases_are_classified_sensibly(ml_engine):
    assert ml_engine.classify("Painel elétrico com cheiro de queimado e faíscas").categoria == Categoria.ELETRICA
    assert ml_engine.classify("Preventiva: lubrificação programada do redutor").prioridade == Prioridade.BAIXA


def test_missing_models_raise_clear_error(tmp_path):
    engine = MLEngine(tmp_path)
    assert engine.is_available() is False
    with pytest.raises(ModelsNotLoadedError):
        engine.classify("qualquer texto")


def test_metrics_file_has_holdout_and_cross_validation(ml_engine):
    metrics = ml_engine.metrics()
    for target in ("categoria", "prioridade"):
        assert metrics[target]["validacao_cruzada"]
        assert {"accuracy", "f1_macro", "por_classe", "matriz_confusao"} <= set(metrics[target]["holdout"])
    json.dumps(metrics)


def test_frozen_test_set_does_not_overlap_training(dataset_dir):
    import pandas as pd

    from app.services.preprocess import normalize_text

    train = {normalize_text(text) for text in pd.read_csv(dataset_dir / "train.csv")["texto"]}
    test = {normalize_text(text) for text in pd.read_csv(dataset_dir / "test_frozen.csv")["texto"]}
    assert not train & test
