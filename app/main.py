import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import date
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import db
from app.config import Settings, get_settings
from app.engines.base import Engine
from app.engines.gemini_engine import GeminiEngine
from app.engines.hybrid import HybridRouter
from app.engines.ml_engine import MLEngine, ModelsNotLoadedError
from app.logging_config import configure_logging
from app.schemas import (
    Categoria,
    ChamadoOut,
    ClassificarRequest,
    CorrecaoRequest,
    GeminiHealth,
    HealthResponse,
    ListaChamados,
    LoteRequest,
    LoteResponse,
    Modo,
    Prioridade,
    RespostaClassificacao,
    RetreinarResponse,
    StatusChamado,
)
from app.services.records import to_registro
from app.services.retrain import NoFeedbackError, retreinar

logger = logging.getLogger("app.api")


def create_app(
    settings: Settings | None = None,
    ml_engine: Engine | None = None,
    llm_engine: Engine | None = None,
) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging(settings.log_level)
        db.init_db(settings.database_path)
        ml = ml_engine or MLEngine(settings.models_dir)
        llm = llm_engine or GeminiEngine.from_settings(settings)
        app.state.ml = ml
        app.state.llm = llm
        app.state.router = HybridRouter(ml, llm, settings.ml_confidence_threshold)
        logger.info(
            "api_iniciada",
            extra={"context": {"ml_disponivel": ml.is_available(), "gemini_disponivel": llm.is_available()}},
        )
        yield

    app = FastAPI(title="Classificador Inteligente de Chamados de Manutenção Industrial", version="1.0.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        started = time.perf_counter()
        request_id = uuid.uuid4().hex[:12]
        response = await call_next(request)
        logger.info(
            "requisicao",
            extra={
                "context": {
                    "request_id": request_id,
                    "metodo": request.method,
                    "caminho": request.url.path,
                    "status": response.status_code,
                    "duracao_ms": round((time.perf_counter() - started) * 1000, 2),
                }
            },
        )
        response.headers["X-Request-ID"] = request_id
        return response

    @app.exception_handler(ModelsNotLoadedError)
    async def models_not_loaded_handler(request: Request, error: ModelsNotLoadedError):
        return JSONResponse(status_code=503, content={"detail": str(error)})

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, error: RequestValidationError):
        details = [{"campo": ".".join(str(part) for part in item["loc"]), "erro": item["msg"]} for item in error.errors()]
        return JSONResponse(status_code=422, content={"detail": details})

    @app.exception_handler(Exception)
    async def unexpected_handler(request: Request, error: Exception):
        logger.exception("erro_nao_tratado", extra={"context": {"caminho": request.url.path}})
        return JSONResponse(status_code=500, content={"detail": "Erro interno do servidor"})

    def check_text_length(text: str) -> None:
        if len(text) > settings.max_text_length:
            raise HTTPException(
                status_code=422, detail=f"Texto excede o limite de {settings.max_text_length} caracteres"
            )

    def classify_and_store(request: Request, text: str, mode: Modo, persist: bool) -> RespostaClassificacao:
        response = request.app.state.router.classify(text, mode)
        if persist:
            response.id = db.insert_chamado(settings.database_path, to_registro(response, text))
        logger.info(
            "classificacao",
            extra={
                "context": {
                    "modo": mode.value,
                    "motor": response.motor_usado,
                    "categoria": response.categoria.value,
                    "prioridade": response.prioridade.value,
                    "confianca": round(response.confianca, 3),
                    "tamanho_texto": len(text),
                    "fallback": response.flags.fallback,
                    "latencia_ms": round(response.latencia_ms, 2),
                }
            },
        )
        return response

    @app.post("/classificar", response_model=RespostaClassificacao)
    def classificar(body: ClassificarRequest, request: Request):
        check_text_length(body.texto)
        return classify_and_store(request, body.texto, body.modo, body.persistir)

    @app.post("/classificar/lote", response_model=LoteResponse)
    def classificar_lote(body: LoteRequest, request: Request):
        if len(body.textos) > settings.max_batch_size:
            raise HTTPException(status_code=422, detail=f"Lote excede o limite de {settings.max_batch_size} textos")
        for text in body.textos:
            check_text_length(text)
        results = [classify_and_store(request, text, body.modo, True) for text in body.textos]
        return LoteResponse(total=len(results), resultados=results)

    @app.get("/chamados", response_model=ListaChamados)
    def listar_chamados(
        categoria: Categoria | None = None,
        prioridade: Prioridade | None = None,
        status: StatusChamado | None = None,
        data_inicio: date | None = None,
        data_fim: date | None = None,
        pagina: int = Query(1, ge=1),
        tamanho: int = Query(20, ge=1, le=200),
    ):
        items, total = db.list_chamados(
            settings.database_path,
            categoria=categoria.value if categoria else None,
            prioridade=prioridade.value if prioridade else None,
            status=status.value if status else None,
            data_inicio=data_inicio.isoformat() if data_inicio else None,
            data_fim=data_fim.isoformat() if data_fim else None,
            pagina=pagina,
            tamanho=tamanho,
        )
        return ListaChamados(total=total, pagina=pagina, tamanho=tamanho, itens=[ChamadoOut(**item) for item in items])

    @app.patch("/chamados/{chamado_id}/correcao", response_model=ChamadoOut)
    def corrigir_chamado(chamado_id: int, body: CorrecaoRequest):
        updated = db.apply_correction(
            settings.database_path,
            chamado_id,
            body.categoria.value if body.categoria else None,
            body.prioridade.value if body.prioridade else None,
        )
        if not updated:
            raise HTTPException(status_code=404, detail="Chamado não encontrado")
        return ChamadoOut(**db.get_chamado(settings.database_path, chamado_id))

    @app.post("/chamados/{chamado_id}/confirmar", response_model=ChamadoOut)
    def confirmar_chamado(chamado_id: int):
        if not db.confirm_chamado(settings.database_path, chamado_id):
            raise HTTPException(status_code=404, detail="Chamado não encontrado")
        return ChamadoOut(**db.get_chamado(settings.database_path, chamado_id))

    @app.get("/metricas")
    def metricas(request: Request) -> dict[str, Any]:
        ml = request.app.state.ml
        model_metrics = ml.metrics() if hasattr(ml, "metrics") else None
        return {"modelo": model_metrics, "uso": db.usage_stats(settings.database_path)}

    @app.post("/retreinar", response_model=RetreinarResponse)
    def retreinar_modelo(request: Request):
        try:
            outcome = retreinar(settings, request.app.state.ml)
        except NoFeedbackError as error:
            raise HTTPException(status_code=409, detail=str(error))
        return outcome

    @app.get("/benchmark")
    def benchmark() -> dict[str, Any]:
        path = settings.reports_dir / "benchmark.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail="Benchmark ainda não foi executado (python -m ml.benchmark)")
        return json.loads(path.read_text(encoding="utf-8"))

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request):
        ml = request.app.state.ml
        llm = request.app.state.llm
        loaded = ml.loaded_models() if hasattr(ml, "loaded_models") else {"categoria": ml.is_available(), "prioridade": ml.is_available()}
        return HealthResponse(
            status="ok" if ml.is_available() else "degradado",
            modelos_carregados=loaded,
            gemini=GeminiHealth(
                configurado=settings.gemini_configured,
                disponivel=llm.is_available(),
                modelo=settings.gemini_model or None,
                ultimo_erro=getattr(llm, "last_error", None),
            ),
        )

    return app


app = create_app()
