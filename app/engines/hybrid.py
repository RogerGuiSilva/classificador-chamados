import logging
from time import perf_counter

from app.engines.base import Engine
from app.engines.gemini_engine import GeminiEngineError
from app.schemas import (
    Explicacao,
    Flags,
    Modo,
    Opiniao,
    Prioridade,
    RespostaClassificacao,
    Resultado,
)
from app.services.rules import apply_safety_rules, suggested_action

logger = logging.getLogger(__name__)


def needs_escalation(ml_result: Resultado, threshold: float) -> bool:
    return ml_result.confianca < threshold


def verdicts_disagree(first: Resultado, second: Resultado) -> bool:
    return first.categoria != second.categoria or first.prioridade != second.prioridade


def to_opinion(result: Resultado) -> Opiniao:
    return Opiniao(
        motor=result.motor,
        categoria=result.categoria,
        prioridade=result.prioridade,
        confianca=result.confianca,
        justificativa=result.justificativa,
        latencia_ms=result.latencia_ms,
    )


class HybridRouter:
    def __init__(self, ml_engine: Engine, llm_engine: Engine, confidence_threshold: float) -> None:
        self._ml_engine = ml_engine
        self._llm_engine = llm_engine
        self._threshold = confidence_threshold

    def classify(self, text: str, mode: Modo) -> RespostaClassificacao:
        started = perf_counter()
        ml_result: Resultado | None = None
        llm_result: Resultado | None = None
        fallback_reason: str | None = None
        escalated = False

        if mode in (Modo.ML, Modo.HYBRID):
            ml_result = self._ml_engine.classify(text)
        if mode == Modo.LLM or (mode == Modo.HYBRID and needs_escalation(ml_result, self._threshold)):
            escalated = mode == Modo.HYBRID
            llm_result, fallback_reason = self._try_llm(text)
        if llm_result is None and ml_result is None:
            ml_result = self._ml_engine.classify(text)

        return self._assemble(text, mode, ml_result, llm_result, escalated, fallback_reason, started)

    def _try_llm(self, text: str) -> tuple[Resultado | None, str | None]:
        if not self._llm_engine.is_available():
            return None, "gemini_indisponivel"
        try:
            return self._llm_engine.classify(text), None
        except GeminiEngineError as error:
            return None, error.kind
        except Exception:
            logger.exception("gemini_erro_inesperado")
            return None, "erro_inesperado"

    def _assemble(
        self,
        text: str,
        mode: Modo,
        ml_result: Resultado | None,
        llm_result: Resultado | None,
        escalated: bool,
        fallback_reason: str | None,
        started: float,
    ) -> RespostaClassificacao:
        decisive = llm_result or ml_result
        disagreement = bool(ml_result and llm_result and verdicts_disagree(ml_result, llm_result))
        fell_back = fallback_reason is not None
        low_confidence_fallback = fell_back and ml_result is not None and needs_escalation(ml_result, self._threshold)
        safety = apply_safety_rules(decisive.prioridade, text)
        action = (
            suggested_action(decisive.categoria, Prioridade.ALTA)
            if safety.sobrescreveu
            else decisive.acao_sugerida or suggested_action(decisive.categoria, safety.prioridade)
        )
        return RespostaClassificacao(
            categoria=decisive.categoria,
            prioridade=safety.prioridade,
            confianca=decisive.confianca,
            motor_usado="ml+llm" if ml_result and llm_result else decisive.motor,
            modo_solicitado=mode,
            explicacao=Explicacao(
                termos_ml=ml_result.termos_ml if ml_result else [],
                justificativa_llm=llm_result.justificativa if llm_result else None,
                causa_provavel=llm_result.causa_provavel if llm_result else None,
                equipamento_detectado=llm_result.equipamento_detectado if llm_result else None,
            ),
            acao_sugerida=action,
            flags=Flags(
                regra_aplicada=safety.sobrescreveu,
                revisao_humana=disagreement or low_confidence_fallback,
                fallback=fell_back,
                escalado_para_llm=escalated,
                resposta_em_cache=bool(llm_result and llm_result.em_cache),
                palavras_criticas=safety.palavras_criticas,
                motivo_fallback=fallback_reason,
            ),
            opiniao_ml=to_opinion(ml_result) if ml_result and llm_result else None,
            opiniao_llm=to_opinion(llm_result) if llm_result and ml_result else None,
            latencia_ms=(perf_counter() - started) * 1000,
            llm_latencia_ms=llm_result.latencia_ms if llm_result else None,
            tokens_entrada=llm_result.tokens_entrada if llm_result else 0,
            tokens_saida=llm_result.tokens_saida if llm_result else 0,
            custo_estimado=llm_result.custo_estimado if llm_result else 0.0,
        )
