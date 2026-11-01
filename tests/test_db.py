import pytest

from app import db


def make_record(**overrides):
    record = {
        "descricao": "motor com ruído",
        "modo": "ml",
        "motor": "ml",
        "categoria": "Mecânica",
        "prioridade": "Média",
        "confianca": 0.8,
        "ml_categoria": "Mecânica",
        "ml_prioridade": "Média",
        "ml_confianca": 0.8,
        "termos_ml": [{"termo": "ruido", "peso": 0.5}],
        "palavras_criticas": [],
        "latencia_ms": 4.0,
    }
    record.update(overrides)
    return record


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "test.db"
    db.init_db(path)
    return path


def test_insert_and_get_roundtrip(database):
    chamado_id = db.insert_chamado(database, make_record())
    stored = db.get_chamado(database, chamado_id)
    assert stored["descricao"] == "motor com ruído"
    assert stored["termos_ml"] == [{"termo": "ruido", "peso": 0.5}]
    assert stored["status"] == "pendente"


def test_get_missing_returns_none(database):
    assert db.get_chamado(database, 999) is None


def test_sql_injection_attempt_is_stored_as_plain_text(database):
    payload = "'); DROP TABLE chamados; --"
    db.insert_chamado(database, make_record(descricao=payload))
    items, total = db.list_chamados(database, categoria=payload)
    assert total == 0
    assert db.list_chamados(database)[1] == 1


def test_filters_and_pagination(database):
    for index in range(5):
        db.insert_chamado(database, make_record(categoria="Elétrica" if index % 2 else "Mecânica"))
    items, total = db.list_chamados(database, categoria="Elétrica")
    assert total == 2 and all(item["categoria"] == "Elétrica" for item in items)
    page, total = db.list_chamados(database, pagina=2, tamanho=2)
    assert total == 5 and len(page) == 2
    assert db.list_chamados(database, data_inicio="2999-01-01")[1] == 0


def test_correction_marks_status_and_filters_by_effective_value(database):
    chamado_id = db.insert_chamado(database, make_record())
    assert db.apply_correction(database, chamado_id, "Elétrica", None) is True
    stored = db.get_chamado(database, chamado_id)
    assert stored["categoria_corrigida"] == "Elétrica" and stored["status"] == "corrigido"
    assert db.list_chamados(database, categoria="Elétrica")[1] == 1
    assert db.list_chamados(database, categoria="Mecânica")[1] == 0


def test_correction_equal_to_prediction_counts_as_confirmation(database):
    chamado_id = db.insert_chamado(database, make_record())
    db.apply_correction(database, chamado_id, "Mecânica", None)
    assert db.get_chamado(database, chamado_id)["status"] == "confirmado"


def test_correction_of_missing_id_returns_false(database):
    assert db.apply_correction(database, 42, "Mecânica", None) is False
    assert db.confirm_chamado(database, 42) is False


def test_confirm_does_not_override_correction(database):
    chamado_id = db.insert_chamado(database, make_record())
    db.apply_correction(database, chamado_id, "Elétrica", None)
    db.confirm_chamado(database, chamado_id)
    assert db.get_chamado(database, chamado_id)["status"] == "corrigido"


def test_usage_stats_aggregates_llm_cost_and_correction_rate(database):
    first = db.insert_chamado(
        database,
        make_record(llm_categoria="Mecânica", llm_prioridade="Média", llm_confianca=0.9, llm_latencia_ms=800.0,
                    tokens_entrada=100, tokens_saida=20, custo_estimado=0.002, escalado_llm=1),
    )
    second = db.insert_chamado(database, make_record())
    db.apply_correction(database, first, "Elétrica", None)
    db.confirm_chamado(database, second)
    stats = db.usage_stats(database)
    assert stats["total_chamados"] == 2
    assert stats["taxa_correcao"] == 0.5
    assert stats["gemini"]["chamadas"] == 1
    assert stats["gemini"]["custo_total_estimado"] == pytest.approx(0.002)
    assert stats["escalados_para_llm"] == 1


def test_connections_are_closed_after_use(database):
    import sqlite3

    with db.connect(database) as connection:
        pass
    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")
