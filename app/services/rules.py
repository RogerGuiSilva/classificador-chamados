import re
from dataclasses import dataclass, field

from app.schemas import Categoria, Prioridade
from app.services.preprocess import fold_text

CRITICAL_PATTERNS: dict[str, re.Pattern[str]] = {
    "fumaça": re.compile(r"(?<!sem )(?<!sem nenhuma )\bfumaca\w*"),
    "incêndio": re.compile(r"\bincendi\w*"),
    "fogo": re.compile(r"\bfogo\b|\bchamas?\b"),
    "choque": re.compile(r"(?<!sem )\bchoques?\b"),
    "vazamento de gás": re.compile(r"\bvazamento\s+de\s+gas\b|\bvazando\s+gas\b"),
    "emergência": re.compile(r"\bemergencia\b"),
    "explosão": re.compile(r"\bexplo\w+"),
    "faísca": re.compile(r"\bfaisca\w*"),
    "curto-circuito": re.compile(r"\bcurto[\s-]?circuito\b"),
}

SUGGESTED_ACTIONS: dict[tuple[Categoria, Prioridade], str] = {
    (Categoria.MECANICA, Prioridade.ALTA): "Parar o equipamento, isolar a área e acionar a equipe mecânica imediatamente.",
    (Categoria.MECANICA, Prioridade.MEDIA): "Programar inspeção mecânica no próximo turno e monitorar vibração e temperatura.",
    (Categoria.MECANICA, Prioridade.BAIXA): "Incluir na rotina de manutenção preventiva mecânica.",
    (Categoria.ELETRICA, Prioridade.ALTA): "Desenergizar o circuito, isolar a área e acionar a equipe elétrica imediatamente.",
    (Categoria.ELETRICA, Prioridade.MEDIA): "Agendar inspeção elétrica com medição de corrente, tensão e temperatura dos terminais.",
    (Categoria.ELETRICA, Prioridade.BAIXA): "Incluir na rotina de inspeção elétrica preventiva.",
    (Categoria.AUTOMACAO, Prioridade.ALTA): "Colocar o processo em estado seguro e acionar a equipe de automação imediatamente.",
    (Categoria.AUTOMACAO, Prioridade.MEDIA): "Verificar logs do CLP, sensores e comunicação no próximo turno.",
    (Categoria.AUTOMACAO, Prioridade.BAIXA): "Incluir na janela de manutenção programada de automação.",
}


@dataclass(frozen=True)
class SafetyOutcome:
    prioridade: Prioridade
    palavras_criticas: list[str] = field(default_factory=list)
    sobrescreveu: bool = False


def detect_critical_terms(text: str) -> list[str]:
    folded = fold_text(text)
    return [term for term, pattern in CRITICAL_PATTERNS.items() if pattern.search(folded)]


def apply_safety_rules(prioridade: Prioridade, text: str) -> SafetyOutcome:
    terms = detect_critical_terms(text)
    if terms and prioridade != Prioridade.ALTA:
        return SafetyOutcome(Prioridade.ALTA, terms, True)
    return SafetyOutcome(prioridade, terms, False)


def suggested_action(categoria: Categoria, prioridade: Prioridade) -> str:
    return SUGGESTED_ACTIONS[(categoria, prioridade)]
