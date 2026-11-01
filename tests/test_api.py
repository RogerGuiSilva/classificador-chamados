import pytest
from fastapi.testclient import TestClient

from app import db
from app.main import create_app
from app.schemas import Categoria, Prioridade
from app.services.retrain import NoFeedbackError, retreinar
from tests.conftest import FakeLLMEngine, make_result


@pytest.fixture
def llm():
    return FakeLLMEngine()


@pytest.fixture
def client(settings, ml_engine, llm):
    with TestClient(create_app(settings, ml_engine, llm)) as test_client:
        yield test_client


def test_health_reports_models_and_gemini_status(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["modelos_carregados"] == {"categoria": True, "prioridade": True}
    assert body["gemini"]["disponivel"] is True


def test_classify_ml_mode_returns_full_contract(client):
    response = client.post("/classificar", json={"texto": "Motor da esteira com ruído anormal", "modo": "ml"})
    assert response.status_code == 200
    body = response.json()
    assert body["categoria"] in {item.value for item in Categoria}
    assert body["prioridade"] in {item.value for item in Prioridade}
    assert 0 <= body["confianca"] <= 1
    assert body["motor_usado"] == "ml" and body["id"] is not None
    assert {"regra_aplicada", "revisao_humana", "fallback"} <= set(body["flags"])
    assert body["latencia_ms"] >= 0 and body["acao_sugerida"]


def test_classify_llm_mode_uses_mock(client, llm):
    body = client.post("/classificar", json={"texto": "painel com risco de choque", "modo": "llm"}).json()
    assert body["motor_usado"] == "llm" and llm.calls == 1
    assert body["explicacao"]["justificativa_llm"]


def test_classify_without_persisting_does_not_store(client, settings):
    client.post("/classificar", json={"texto": "teste de comparação", "modo": "ml", "persistir": False})
    assert db.list_chamados(settings.database_path)[1] == 0


@pytest.mark.parametrize("payload", [{"texto": ""}, {"texto": "   "}, {"modo": "ml"}, {"texto": "ok", "modo": "invalido"}])
def test_invalid_input_returns_422(client, payload):
    assert client.post("/classificar", json=payload).status_code == 422


def test_text_over_limit_returns_422(client):
    assert client.post("/classificar", json={"texto": "a" * 201}).status_code == 422


def test_batch_classification_and_limit(client):
    ok = client.post("/classificar/lote", json={"textos": ["rolamento com ruído", "CLP sem comunicação"], "modo": "ml"})
    assert ok.status_code == 200 and ok.json()["total"] == 2
    too_many = client.post("/classificar/lote", json={"textos": ["a", "b", "c", "d"], "modo": "ml"})
    assert too_many.status_code == 422
    assert client.post("/classificar/lote", json={"textos": []}).status_code == 422


def test_degrades_to_ml_when_gemini_missing(settings, ml_engine):
    app = create_app(settings, ml_engine, FakeLLMEngine(available=False))
    with TestClient(app) as test_client:
        body = test_client.post("/classificar", json={"texto": "motor desarmando", "modo": "llm"}).json()
        assert body["flags"]["fallback"] is True and body["motor_usado"] == "ml"
        assert test_client.get("/health").json()["gemini"]["disponivel"] is False


def test_history_filters_and_pagination(client):
    for text in ["rolamento com ruído", "CLP perdeu comunicação", "painel com cheiro de queimado"]:
        client.post("/classificar", json={"texto": text, "modo": "ml"})
    listing = client.get("/chamados", params={"tamanho": 2}).json()
    assert listing["total"] == 3 and len(listing["itens"]) == 2
    assert client.get("/chamados", params={"categoria": "Inexistente"}).status_code == 422
    assert client.get("/chamados", params={"status": "pendente"}).json()["total"] == 3


def test_correction_flow_and_not_found(client):
    chamado_id = client.post("/classificar", json={"texto": "rolamento com ruído", "modo": "ml"}).json()["id"]
    patched = client.patch(f"/chamados/{chamado_id}/correcao", json={"categoria": "Automação", "prioridade": "Alta"})
    assert patched.status_code == 200
    assert patched.json()["categoria_corrigida"] == "Automação"
    assert patched.json()["status"] == "corrigido"
    assert client.patch("/chamados/9999/correcao", json={"categoria": "Elétrica"}).status_code == 404
    assert client.patch(f"/chamados/{chamado_id}/correcao", json={}).status_code == 422


def test_confirmation(client):
    chamado_id = client.post("/classificar", json={"texto": "rolamento com ruído", "modo": "ml"}).json()["id"]
    assert client.post(f"/chamados/{chamado_id}/confirmar").json()["status"] == "confirmado"
    assert client.post("/chamados/9999/confirmar").status_code == 404


def test_metrics_include_model_and_usage(client):
    client.post("/classificar", json={"texto": "painel com risco de choque", "modo": "llm"})
    body = client.get("/metricas").json()
    assert body["modelo"]["categoria"]["holdout"]["accuracy"] > 0
    assert body["uso"]["total_chamados"] == 1
    assert body["uso"]["gemini"]["chamadas"] == 1
    assert body["uso"]["gemini"]["custo_total_estimado"] > 0


def test_retrain_without_feedback_returns_conflict(client):
    assert client.post("/retreinar").status_code == 409


def test_benchmark_endpoint_404_until_report_exists(client):
    assert client.get("/benchmark").status_code == 404


def test_retrain_incorporates_corrections_and_reports_before_after(settings, ml_engine):
    db.init_db(settings.database_path)
    for text, categoria in [("sensor novo travando na linha 9", "Automação"), ("cabo esquentando no quadro 7", "Elétrica")]:
        chamado_id = db.insert_chamado(
            settings.database_path,
            {"descricao": text, "modo": "ml", "motor": "ml", "categoria": "Mecânica", "prioridade": "Média", "confianca": 0.5},
        )
        db.apply_correction(settings.database_path, chamado_id, categoria, "Média")
    outcome = retreinar(settings, ml_engine, nomes=("logistic_regression",), cv_folds=3)
    assert outcome["correcoes_incorporadas"] == {"categoria": 2, "prioridade": 2}
    assert outcome["antes"] is not None and outcome["depois"]["categoria"]["holdout"]["accuracy"] > 0
    assert ml_engine.is_available()


def test_retrain_never_uses_texts_from_frozen_test_set(settings, ml_engine):
    import pandas as pd

    frozen_text = pd.read_csv(settings.data_dir / "test_frozen.csv")["texto"].iloc[0]
    db.init_db(settings.database_path)
    chamado_id = db.insert_chamado(
        settings.database_path,
        {"descricao": frozen_text, "modo": "ml", "motor": "ml", "categoria": "Mecânica", "prioridade": "Média", "confianca": 0.5},
    )
    db.apply_correction(settings.database_path, chamado_id, "Automação", "Alta")
    with pytest.raises(NoFeedbackError):
        retreinar(settings, ml_engine, nomes=("logistic_regression",), cv_folds=3)
