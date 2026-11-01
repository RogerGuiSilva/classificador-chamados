from app.services.preprocess import fold_text, normalize_text, strip_accents


def test_lowercase_and_accent_removal():
    assert strip_accents("Automação Elétrica") == "Automacao Eletrica"
    assert normalize_text("MOTOR ELÉTRICO") == "motor eletrico"


def test_error_codes_are_normalized_consistently():
    assert normalize_text("Erro E04 no inversor") == normalize_text("erro e-4 no inversor")
    assert "cod_e4" in normalize_text("Erro E04")


def test_current_loop_signal_is_normalized():
    assert "sinal_4_20ma" in normalize_text("Leitura 4-20mA oscilando")
    assert "sinal_4_20ma" in normalize_text("sinal 4 20 mA fora")


def test_stopwords_removed_but_negation_and_technical_terms_kept():
    result = normalize_text("O motor da esteira não liga com CLP e IHM")
    assert "nao" in result.split()
    assert {"motor", "esteira", "clp", "ihm"} <= set(result.split())
    assert "da" not in result.split()


def test_conjunction_before_number_is_not_an_error_code():
    assert "cod_" not in normalize_text("motor e 4 sensores")


def test_fold_text_keeps_stopwords():
    assert fold_text("  Vazamento DE Gás ") == "vazamento de gas"
