import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS chamados (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    data_hora TEXT NOT NULL,
    descricao TEXT NOT NULL,
    modo TEXT NOT NULL,
    motor TEXT NOT NULL,
    categoria TEXT NOT NULL,
    prioridade TEXT NOT NULL,
    confianca REAL NOT NULL,
    ml_categoria TEXT,
    ml_prioridade TEXT,
    ml_confianca REAL,
    llm_categoria TEXT,
    llm_prioridade TEXT,
    llm_confianca REAL,
    justificativa TEXT,
    causa_provavel TEXT,
    equipamento_detectado TEXT,
    acao_sugerida TEXT,
    termos_ml TEXT NOT NULL DEFAULT '[]',
    regra_aplicada INTEGER NOT NULL DEFAULT 0,
    palavras_criticas TEXT NOT NULL DEFAULT '[]',
    revisao_humana INTEGER NOT NULL DEFAULT 0,
    fallback INTEGER NOT NULL DEFAULT 0,
    escalado_llm INTEGER NOT NULL DEFAULT 0,
    em_cache INTEGER NOT NULL DEFAULT 0,
    latencia_ms REAL NOT NULL DEFAULT 0,
    llm_latencia_ms REAL,
    tokens_entrada INTEGER NOT NULL DEFAULT 0,
    tokens_saida INTEGER NOT NULL DEFAULT 0,
    custo_estimado REAL NOT NULL DEFAULT 0,
    categoria_corrigida TEXT,
    prioridade_corrigida TEXT,
    status TEXT NOT NULL DEFAULT 'pendente'
);
CREATE INDEX IF NOT EXISTS idx_chamados_data ON chamados (data_hora);
CREATE INDEX IF NOT EXISTS idx_chamados_status ON chamados (status);
"""

INSERT_COLUMNS = (
    "data_hora", "descricao", "modo", "motor", "categoria", "prioridade", "confianca",
    "ml_categoria", "ml_prioridade", "ml_confianca", "llm_categoria", "llm_prioridade", "llm_confianca",
    "justificativa", "causa_provavel", "equipamento_detectado", "acao_sugerida", "termos_ml",
    "regra_aplicada", "palavras_criticas", "revisao_humana", "fallback", "escalado_llm", "em_cache",
    "latencia_ms", "llm_latencia_ms", "tokens_entrada", "tokens_saida", "custo_estimado",
)
JSON_COLUMNS = ("termos_ml", "palavras_criticas")
EFFECTIVE_CATEGORY = "COALESCE(categoria_corrigida, categoria)"
EFFECTIVE_PRIORITY = "COALESCE(prioridade_corrigida, prioridade)"


@contextmanager
def connect(path: Path) -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with connect(path) as connection:
        connection.executescript(SCHEMA)


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    record = dict(row)
    for column in JSON_COLUMNS:
        record[column] = json.loads(record[column])
    return record


def insert_chamado(path: Path, registro: dict[str, Any]) -> int:
    values = {column: registro[column] for column in INSERT_COLUMNS if registro.get(column) is not None}
    values.setdefault("data_hora", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
    for column in JSON_COLUMNS:
        if column in values:
            values[column] = json.dumps(values[column], ensure_ascii=False)
    columns = list(values)
    placeholders = ", ".join("?" for _ in columns)
    with connect(path) as connection:
        cursor = connection.execute(
            f"INSERT INTO chamados ({', '.join(columns)}) VALUES ({placeholders})",
            [values[column] for column in columns],
        )
        return int(cursor.lastrowid)


def get_chamado(path: Path, chamado_id: int) -> dict[str, Any] | None:
    with connect(path) as connection:
        row = connection.execute("SELECT * FROM chamados WHERE id = ?", (chamado_id,)).fetchone()
    return row_to_dict(row) if row else None


def build_filters(
    categoria: str | None,
    prioridade: str | None,
    status: str | None,
    data_inicio: str | None,
    data_fim: str | None,
) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if categoria:
        clauses.append(f"{EFFECTIVE_CATEGORY} = ?")
        params.append(categoria)
    if prioridade:
        clauses.append(f"{EFFECTIVE_PRIORITY} = ?")
        params.append(prioridade)
    if status:
        clauses.append("status = ?")
        params.append(status)
    if data_inicio:
        clauses.append("date(data_hora) >= date(?)")
        params.append(data_inicio)
    if data_fim:
        clauses.append("date(data_hora) <= date(?)")
        params.append(data_fim)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    return where, params


def list_chamados(
    path: Path,
    categoria: str | None = None,
    prioridade: str | None = None,
    status: str | None = None,
    data_inicio: str | None = None,
    data_fim: str | None = None,
    pagina: int = 1,
    tamanho: int = 20,
) -> tuple[list[dict[str, Any]], int]:
    where, params = build_filters(categoria, prioridade, status, data_inicio, data_fim)
    with connect(path) as connection:
        total = connection.execute(f"SELECT COUNT(*) FROM chamados {where}", params).fetchone()[0]
        rows = connection.execute(
            f"SELECT * FROM chamados {where} ORDER BY id DESC LIMIT ? OFFSET ?",
            [*params, tamanho, (pagina - 1) * tamanho],
        ).fetchall()
    return [row_to_dict(row) for row in rows], int(total)


def apply_correction(path: Path, chamado_id: int, categoria: str | None, prioridade: str | None) -> bool:
    with connect(path) as connection:
        cursor = connection.execute(
            "UPDATE chamados SET categoria_corrigida = COALESCE(?, categoria_corrigida), "
            "prioridade_corrigida = COALESCE(?, prioridade_corrigida) WHERE id = ?",
            (categoria, prioridade, chamado_id),
        )
        if cursor.rowcount == 0:
            return False
        connection.execute(
            "UPDATE chamados SET status = CASE "
            "WHEN (categoria_corrigida IS NOT NULL AND categoria_corrigida != categoria) "
            "OR (prioridade_corrigida IS NOT NULL AND prioridade_corrigida != prioridade) "
            "THEN 'corrigido' ELSE 'confirmado' END WHERE id = ?",
            (chamado_id,),
        )
    return True


def confirm_chamado(path: Path, chamado_id: int) -> bool:
    with connect(path) as connection:
        exists = connection.execute("SELECT 1 FROM chamados WHERE id = ?", (chamado_id,)).fetchone()
        if not exists:
            return False
        connection.execute("UPDATE chamados SET status = 'confirmado' WHERE id = ? AND status = 'pendente'", (chamado_id,))
    return True


def training_feedback(path: Path) -> list[dict[str, Any]]:
    with connect(path) as connection:
        rows = connection.execute(
            "SELECT id, descricao, categoria, prioridade, categoria_corrigida, prioridade_corrigida, status "
            "FROM chamados WHERE status IN ('confirmado', 'corrigido') ORDER BY id"
        ).fetchall()
    return [dict(row) for row in rows]


def _count_by(connection: sqlite3.Connection, expression: str) -> dict[str, int]:
    rows = connection.execute(f"SELECT {expression} AS chave, COUNT(*) AS total FROM chamados GROUP BY chave").fetchall()
    return {row["chave"]: row["total"] for row in rows}


def usage_stats(path: Path) -> dict[str, Any]:
    with connect(path) as connection:
        total = connection.execute("SELECT COUNT(*) FROM chamados").fetchone()[0]
        by_status = _count_by(connection, "status")
        llm = connection.execute(
            "SELECT COUNT(*) AS chamadas, COALESCE(SUM(tokens_entrada), 0) AS tokens_entrada, "
            "COALESCE(SUM(tokens_saida), 0) AS tokens_saida, COALESCE(SUM(custo_estimado), 0) AS custo_total, "
            "AVG(llm_latencia_ms) AS latencia_media_ms FROM chamados WHERE llm_categoria IS NOT NULL AND em_cache = 0"
        ).fetchone()
        flags = connection.execute(
            "SELECT COALESCE(SUM(escalado_llm), 0) AS escalados, COALESCE(SUM(revisao_humana), 0) AS revisoes, "
            "COALESCE(SUM(fallback), 0) AS fallbacks, COALESCE(SUM(regra_aplicada), 0) AS regras, "
            "AVG(latencia_ms) AS latencia_media_ms FROM chamados"
        ).fetchone()
        daily = connection.execute(
            "SELECT date(data_hora) AS dia, COUNT(*) AS total FROM chamados GROUP BY dia ORDER BY dia"
        ).fetchall()
        by_category = _count_by(connection, EFFECTIVE_CATEGORY)
        by_priority = _count_by(connection, EFFECTIVE_PRIORITY)
        by_engine = _count_by(connection, "motor")
    corrected = by_status.get("corrigido", 0)
    reviewed = corrected + by_status.get("confirmado", 0)
    return {
        "total_chamados": total,
        "por_categoria": by_category,
        "por_prioridade": by_priority,
        "por_motor": by_engine,
        "por_status": by_status,
        "taxa_correcao": corrected / reviewed if reviewed else None,
        "taxa_correcao_sobre_total": corrected / total if total else None,
        "escalados_para_llm": flags["escalados"],
        "revisoes_humanas_recomendadas": flags["revisoes"],
        "fallbacks": flags["fallbacks"],
        "regras_aplicadas": flags["regras"],
        "latencia_media_ms": flags["latencia_media_ms"],
        "gemini": {
            "chamadas": llm["chamadas"],
            "tokens_entrada": llm["tokens_entrada"],
            "tokens_saida": llm["tokens_saida"],
            "custo_total_estimado": llm["custo_total"],
            "latencia_media_ms": llm["latencia_media_ms"],
        },
        "serie_diaria": [{"dia": row["dia"], "total": row["total"]} for row in daily],
    }
