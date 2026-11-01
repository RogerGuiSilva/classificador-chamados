import pytest

from app.schemas import Categoria, Prioridade
from app.services.rules import apply_safety_rules, detect_critical_terms, suggested_action


@pytest.mark.parametrize(
    "text",
    ["Saindo FUMAÇA do painel", "risco de choque elétrico", "vazamento de gás na linha", "Emergência acionada", "princípio de incêndio"],
)
def test_critical_words_raise_priority_to_high(text):
    outcome = apply_safety_rules(Prioridade.BAIXA, text)
    assert outcome.prioridade == Prioridade.ALTA
    assert outcome.sobrescreveu is True
    assert outcome.palavras_criticas


def test_already_high_priority_is_not_flagged_as_override():
    outcome = apply_safety_rules(Prioridade.ALTA, "fumaça no motor")
    assert outcome.prioridade == Prioridade.ALTA
    assert outcome.sobrescreveu is False
    assert outcome.palavras_criticas == ["fumaça"]


def test_text_without_critical_words_is_untouched():
    outcome = apply_safety_rules(Prioridade.MEDIA, "ruído anormal no rolamento")
    assert outcome.prioridade == Prioridade.MEDIA
    assert outcome.palavras_criticas == []


def test_negated_smoke_is_ignored():
    assert detect_critical_terms("motor aquecendo, sem fumaça") == []


def test_every_combination_has_a_suggested_action():
    for categoria in Categoria:
        for prioridade in Prioridade:
            assert suggested_action(categoria, prioridade)
