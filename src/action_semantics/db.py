"""Optional helpers for the legacy PostgreSQL export path.

The current IndexedVideo workflow reads a supplied JSONL file and never opens a
database connection. Install the ``legacy-database`` extra only if the older
generic clips/steps/pairwise export is deliberately being revived.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

from .io_utils import write_jsonl


def get_database_url() -> str:
    value = os.environ.get("DATABASE_URL")
    if not value:
        raise RuntimeError("DATABASE_URL is not set in the environment.")
    return value


def make_engine(database_url: str | None = None) -> Engine:
    return create_engine(database_url or get_database_url(), pool_pre_ping=True)


def list_tables(engine: Engine, schema: str | None = None) -> list[str]:
    inspector = inspect(engine)
    return sorted(inspector.get_table_names(schema=schema))


def table_columns(engine: Engine, table_name: str, schema: str | None = None) -> list[str]:
    inspector = inspect(engine)
    return [column["name"] for column in inspector.get_columns(table_name, schema=schema)]


def stream_query(engine: Engine, sql: str, parameters: dict[str, Any] | None = None) -> Iterable[dict[str, Any]]:
    with engine.connect() as connection:
        result = connection.execution_options(stream_results=True).execute(text(sql), parameters or {})
        for row in result.mappings():
            yield dict(row)


def export_query_to_jsonl(
    engine: Engine,
    sql: str,
    output_path: Path,
    parameters: dict[str, Any] | None = None,
) -> int:
    return write_jsonl(output_path, stream_query(engine, sql, parameters))
