import shutil
from types import SimpleNamespace

import pandas as pd
import pytest

from app.config import Settings
from app.engines.base import Engine
from app.engines.gemini_engine import GeminiEngineError
from app.engines.ml_engine import MLEngine
from app.schemas import Categoria, Prioridade, Resultado
from ml import gerar_dataset
from ml.train import treinar

VALID_VERDICT_JSON = (
    '{"categoria": "Elétrica", "prioridade": "Alta", "confianca": 0.92, "justificativa": "Risco de choque.", '
    '"causa_provavel": "Isolamento danificado.", "acao_sugerida": "Desenergizar o circuito.", '
    '"equipamento_detectado": "painel elétrico"}'
)


class ScriptedClient:
    def __init__(self, outcomes: list) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []
        self.models = self

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeApiError(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(f"erro {code}")
        self.code = code


def make_response(text: str = VALID_VERDICT_JSON, tokens_in: int = 1000, tokens_out: int = 100):
    return SimpleNamespace(
        text=text,
        usage_metadata=SimpleNamespace(prompt_token_count=tokens_in, candidates_token_count=tokens_out),
    )


class FakeLLMEngine(Engine):
    name = "llm"

    def __init__(self, result: Resultado | None = None, error_kind: str | None = None, available: bool = True) -> None:
        self.result = result or make_result(Categoria.ELETRICA, Prioridade.ALTA, 0.9, "llm")
        self.error_kind = error_kind
        self.available = available
        self.calls = 0
        self.last_error = None

    def classify(self, text: str) -> Resultado:
        self.calls += 1
        if self.error_kind:
            raise GeminiEngineError(self.error_kind)
        return self.result

    def is_available(self) -> bool:
        return self.available


class FakeMLEngine(Engine):
    name = "ml"

    def __init__(self, result: Resultado) -> None:
        self.result = result

    def classify(self, text: str) -> Resultado:
        return self.result

    def is_available(self) -> bool:
        return True


def make_result(categoria: Categoria, prioridade: Prioridade, confianca: float, motor: str) -> Resultado:
    return Resultado(
        motor=motor,
        categoria=categoria,
        prioridade=prioridade,
        confianca=confianca,
        confianca_categoria=confianca,
        confianca_prioridade=confianca,
        justificativa="justificativa" if motor == "llm" else None,
        acao_sugerida="Ação de teste",
        latencia_ms=5.0,
        tokens_entrada=100 if motor == "llm" else 0,
        tokens_saida=20 if motor == "llm" else 0,
        custo_estimado=0.001 if motor == "llm" else 0.0,
    )


@pytest.fixture(scope="session")
def dataset_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("data")
    gerar_dataset.gerar_e_salvar(directory, forcar=True)
    return directory


@pytest.fixture(scope="session")
def trained_models_dir(dataset_dir, tmp_path_factory):
    directory = tmp_path_factory.mktemp("models")
    train_df = pd.read_csv(dataset_dir / "train.csv")
    test_df = pd.read_csv(dataset_dir / "test_frozen.csv")
    treinar(train_df, test_df, directory, nomes=("logistic_regression", "multinomial_nb"), cv_folds=3)
    return directory


@pytest.fixture
def settings(tmp_path, dataset_dir, trained_models_dir):
    models_copy = tmp_path / "models"
    shutil.copytree(trained_models_dir, models_copy)
    return Settings(
        _env_file=None,
        gemini_api_key="",
        gemini_model="",
        database_path=tmp_path / "chamados.db",
        data_dir=dataset_dir,
        models_dir=models_copy,
        reports_dir=tmp_path / "reports",
        max_text_length=200,
        max_batch_size=3,
        gemini_backoff_base_seconds=0.0,
    )


@pytest.fixture
def ml_engine(settings):
    return MLEngine(settings.models_dir)
