from app.engines.hybrid import HybridRouter
from app.schemas import Categoria, Modo, Prioridade
from tests.conftest import FakeLLMEngine, FakeMLEngine, make_result

THRESHOLD = 0.65


def router(ml_result, llm=None):
    llm = llm or FakeLLMEngine()
    return HybridRouter(FakeMLEngine(ml_result), llm, THRESHOLD), llm


def test_high_confidence_ml_does_not_call_llm():
    hybrid, llm = router(make_result(Categoria.MECANICA, Prioridade.MEDIA, 0.95, "ml"))
    response = hybrid.classify("rolamento com ruído", Modo.HYBRID)
    assert llm.calls == 0
    assert response.motor_usado == "ml" and response.flags.escalado_para_llm is False


def test_low_confidence_escalates_and_agreement_has_no_review_flag():
    hybrid, llm = router(
        make_result(Categoria.ELETRICA, Prioridade.ALTA, 0.4, "ml"),
    )
    response = hybrid.classify("painel desarmando", Modo.HYBRID)
    assert llm.calls == 1
    assert response.motor_usado == "ml+llm" and response.flags.escalado_para_llm is True
    assert response.flags.revisao_humana is False


def test_disagreement_flags_human_review_and_returns_both_opinions():
    hybrid, _ = router(make_result(Categoria.MECANICA, Prioridade.MEDIA, 0.4, "ml"))
    response = hybrid.classify("motor desarmando", Modo.HYBRID)
    assert response.flags.revisao_humana is True
    assert response.opiniao_ml.categoria == Categoria.MECANICA
    assert response.opiniao_llm.categoria == Categoria.ELETRICA
    assert response.categoria == Categoria.ELETRICA


def test_llm_failure_falls_back_to_ml_and_recommends_review_when_confidence_low():
    hybrid, _ = router(make_result(Categoria.MECANICA, Prioridade.MEDIA, 0.4, "ml"), FakeLLMEngine(error_kind="timeout"))
    response = hybrid.classify("motor desarmando", Modo.HYBRID)
    assert response.flags.fallback is True and response.flags.motivo_fallback == "timeout"
    assert response.flags.revisao_humana is True
    assert response.motor_usado == "ml"


def test_missing_key_in_llm_mode_degrades_to_ml():
    hybrid, llm = router(make_result(Categoria.MECANICA, Prioridade.BAIXA, 0.9, "ml"), FakeLLMEngine(available=False))
    response = hybrid.classify("lubrificação programada", Modo.LLM)
    assert llm.calls == 0
    assert response.flags.fallback is True and response.flags.motivo_fallback == "gemini_indisponivel"
    assert response.categoria == Categoria.MECANICA and response.flags.revisao_humana is False


def test_ml_mode_never_calls_llm():
    hybrid, llm = router(make_result(Categoria.MECANICA, Prioridade.BAIXA, 0.2, "ml"))
    hybrid.classify("texto", Modo.ML)
    assert llm.calls == 0


def test_llm_mode_uses_only_llm_when_available():
    hybrid, llm = router(make_result(Categoria.MECANICA, Prioridade.BAIXA, 0.9, "ml"))
    response = hybrid.classify("texto", Modo.LLM)
    assert llm.calls == 1 and response.motor_usado == "llm"
    assert response.opiniao_ml is None


def test_safety_rule_overrides_model_and_is_flagged():
    hybrid, _ = router(make_result(Categoria.ELETRICA, Prioridade.BAIXA, 0.95, "ml"))
    response = hybrid.classify("Preventiva do painel, saindo fumaça", Modo.ML)
    assert response.prioridade == Prioridade.ALTA
    assert response.flags.regra_aplicada is True
    assert "fumaça" in response.flags.palavras_criticas
