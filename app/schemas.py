from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class Categoria(str, Enum):
    MECANICA = "Mecânica"
    ELETRICA = "Elétrica"
    AUTOMACAO = "Automação"


class Prioridade(str, Enum):
    BAIXA = "Baixa"
    MEDIA = "Média"
    ALTA = "Alta"


class Modo(str, Enum):
    ML = "ml"
    LLM = "llm"
    HYBRID = "hybrid"


class StatusChamado(str, Enum):
    PENDENTE = "pendente"
    CONFIRMADO = "confirmado"
    CORRIGIDO = "corrigido"


class TermoExplicativo(BaseModel):
    termo: str
    peso: float


class Resultado(BaseModel):
    motor: str
    categoria: Categoria
    prioridade: Prioridade
    confianca: float = Field(ge=0, le=1)
    confianca_categoria: float = Field(ge=0, le=1)
    confianca_prioridade: float = Field(ge=0, le=1)
    justificativa: str | None = None
    causa_provavel: str | None = None
    acao_sugerida: str | None = None
    equipamento_detectado: str | None = None
    termos_ml: list[TermoExplicativo] = Field(default_factory=list)
    latencia_ms: float = 0.0
    tokens_entrada: int = 0
    tokens_saida: int = 0
    custo_estimado: float = 0.0
    em_cache: bool = False


class Opiniao(BaseModel):
    motor: str
    categoria: Categoria
    prioridade: Prioridade
    confianca: float
    justificativa: str | None = None
    latencia_ms: float = 0.0


class Explicacao(BaseModel):
    termos_ml: list[TermoExplicativo] = Field(default_factory=list)
    justificativa_llm: str | None = None
    causa_provavel: str | None = None
    equipamento_detectado: str | None = None


class Flags(BaseModel):
    regra_aplicada: bool = False
    revisao_humana: bool = False
    fallback: bool = False
    escalado_para_llm: bool = False
    resposta_em_cache: bool = False
    palavras_criticas: list[str] = Field(default_factory=list)
    motivo_fallback: str | None = None


class RespostaClassificacao(BaseModel):
    id: int | None = None
    categoria: Categoria
    prioridade: Prioridade
    confianca: float
    motor_usado: str
    modo_solicitado: Modo
    explicacao: Explicacao
    acao_sugerida: str
    flags: Flags
    opiniao_ml: Opiniao | None = None
    opiniao_llm: Opiniao | None = None
    latencia_ms: float
    llm_latencia_ms: float | None = None
    tokens_entrada: int = 0
    tokens_saida: int = 0
    custo_estimado: float = 0.0


def _strip_text(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError("O texto do chamado não pode ser vazio")
    return stripped


class ClassificarRequest(BaseModel):
    texto: str
    modo: Modo = Modo.HYBRID
    persistir: bool = True

    _normalize = field_validator("texto")(_strip_text)


class LoteRequest(BaseModel):
    textos: list[str] = Field(min_length=1)
    modo: Modo = Modo.HYBRID

    @field_validator("textos")
    @classmethod
    def strip_all(cls, values: list[str]) -> list[str]:
        return [_strip_text(value) for value in values]


class LoteResponse(BaseModel):
    total: int
    resultados: list[RespostaClassificacao]


class CorrecaoRequest(BaseModel):
    categoria: Categoria | None = None
    prioridade: Prioridade | None = None

    @model_validator(mode="after")
    def require_at_least_one(self) -> "CorrecaoRequest":
        if self.categoria is None and self.prioridade is None:
            raise ValueError("Informe categoria e/ou prioridade para corrigir")
        return self


class ChamadoOut(BaseModel):
    id: int
    data_hora: str
    descricao: str
    modo: str
    motor: str
    categoria: str
    prioridade: str
    confianca: float
    ml_categoria: str | None = None
    ml_prioridade: str | None = None
    ml_confianca: float | None = None
    llm_categoria: str | None = None
    llm_prioridade: str | None = None
    llm_confianca: float | None = None
    justificativa: str | None = None
    causa_provavel: str | None = None
    equipamento_detectado: str | None = None
    acao_sugerida: str | None = None
    termos_ml: list[TermoExplicativo] = Field(default_factory=list)
    regra_aplicada: bool = False
    palavras_criticas: list[str] = Field(default_factory=list)
    revisao_humana: bool = False
    fallback: bool = False
    escalado_llm: bool = False
    em_cache: bool = False
    latencia_ms: float = 0.0
    llm_latencia_ms: float | None = None
    tokens_entrada: int = 0
    tokens_saida: int = 0
    custo_estimado: float = 0.0
    categoria_corrigida: str | None = None
    prioridade_corrigida: str | None = None
    status: str


class ListaChamados(BaseModel):
    total: int
    pagina: int
    tamanho: int
    itens: list[ChamadoOut]


class GeminiHealth(BaseModel):
    configurado: bool
    disponivel: bool
    modelo: str | None = None
    ultimo_erro: str | None = None


class HealthResponse(BaseModel):
    status: str
    modelos_carregados: dict[str, bool]
    gemini: GeminiHealth


class RetreinarResponse(BaseModel):
    correcoes_incorporadas: dict[str, int]
    antes: dict[str, Any] | None
    depois: dict[str, Any]
