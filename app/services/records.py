from typing import Any

from app.schemas import RespostaClassificacao


def to_registro(resposta: RespostaClassificacao, texto: str) -> dict[str, Any]:
    ml = resposta.opiniao_ml
    llm = resposta.opiniao_llm
    only_llm = resposta.motor_usado == "llm"
    only_ml = resposta.motor_usado == "ml"
    return {
        "descricao": texto,
        "modo": resposta.modo_solicitado.value,
        "motor": resposta.motor_usado,
        "categoria": resposta.categoria.value,
        "prioridade": resposta.prioridade.value,
        "confianca": resposta.confianca,
        "ml_categoria": ml.categoria.value if ml else (resposta.categoria.value if only_ml else None),
        "ml_prioridade": ml.prioridade.value if ml else (resposta.prioridade.value if only_ml else None),
        "ml_confianca": ml.confianca if ml else (resposta.confianca if only_ml else None),
        "llm_categoria": llm.categoria.value if llm else (resposta.categoria.value if only_llm else None),
        "llm_prioridade": llm.prioridade.value if llm else (resposta.prioridade.value if only_llm else None),
        "llm_confianca": llm.confianca if llm else (resposta.confianca if only_llm else None),
        "justificativa": resposta.explicacao.justificativa_llm,
        "causa_provavel": resposta.explicacao.causa_provavel,
        "equipamento_detectado": resposta.explicacao.equipamento_detectado,
        "acao_sugerida": resposta.acao_sugerida,
        "termos_ml": [termo.model_dump() for termo in resposta.explicacao.termos_ml],
        "regra_aplicada": int(resposta.flags.regra_aplicada),
        "palavras_criticas": resposta.flags.palavras_criticas,
        "revisao_humana": int(resposta.flags.revisao_humana),
        "fallback": int(resposta.flags.fallback),
        "escalado_llm": int(resposta.flags.escalado_para_llm),
        "em_cache": int(resposta.flags.resposta_em_cache),
        "latencia_ms": resposta.latencia_ms,
        "llm_latencia_ms": resposta.llm_latencia_ms,
        "tokens_entrada": resposta.tokens_entrada,
        "tokens_saida": resposta.tokens_saida,
        "custo_estimado": resposta.custo_estimado,
    }
