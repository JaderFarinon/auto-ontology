# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SQL Server connector for Auto Ontology."""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Optional
from urllib.parse import parse_qs, unquote, urlparse

import pandas as pd
import pytds
import sqlglot
from sqlglot import exp

from auto_ontology.catalog.constants import TableTypes
from auto_ontology.connectors.base import SQLDatabase, StatementTimeout

logger = logging.getLogger(__name__)

_CHAT_MAX_ROWS = 200


def _parse_connection_string(connection_string: str) -> dict[str, Any]:
    """Parse an MSSQL URI into pytds connection arguments."""
    parsed = urlparse(connection_string)
    if parsed.scheme.split("+", 1)[0].lower() not in {"mssql", "sqlserver"}:
        raise ValueError("Invalid SQL Server connection string")
    if not parsed.hostname:
        raise ValueError("SQL Server connection string is missing a host")
    database = unquote(parsed.path.lstrip("/"))
    if not database:
        raise ValueError("SQL Server connection string is missing a database")

    return {
        "server": parsed.hostname,
        "port": parsed.port or 1433,
        "user": unquote(parsed.username or ""),
        "password": unquote(parsed.password or ""),
        "database": database,
        "cafile": parse_qs(parsed.query).get("cafile", [None])[0],
    }


def _validate_read_query(sql: str) -> None:
    """Reject anything except one read-only query before opening SQL Server."""
    try:
        validation_sql = re.sub(r"%s", "NULL", sql)
        statements = [
            statement
            for statement in sqlglot.parse(validation_sql, read="tsql")
            if statement
        ]
    except Exception as exc:
        raise ValueError(f"SQL Server query could not be parsed safely: {exc}") from exc

    if len(statements) != 1 or not isinstance(statements[0], exp.Query):
        raise ValueError("Only a single read-only SQL query is allowed.")
    if statements[0].find(exp.Into) is not None:
        raise ValueError("SELECT INTO is not allowed on the read-only connection.")


class SQLServerDatabase(SQLDatabase):
    """Concrete SQLDatabase backed by the pure-Python python-tds driver."""

    supports_statement_timeout = True

    def __init__(
        self, connection_string: str, schemas: list[str] | None = None
    ) -> None:
        self._connection_string = connection_string
        self._connect_kwargs = _parse_connection_string(connection_string)
        self._database_name = str(self._connect_kwargs["database"])
        self._schemas = schemas or None
        self.ping()

    @property
    def dialect(self) -> str:
        """Return the SQLGlot dialect for Microsoft SQL Server."""
        return "tsql"

    @property
    def database_name(self) -> str:
        return self._database_name

    def _connect(self, timeout_s: float | None = None) -> Any:
        kwargs = dict(self._connect_kwargs)
        cafile = kwargs.pop("cafile", None)
        kwargs.update(
            readonly=True,
            autocommit=True,
            login_timeout=15,
            timeout=min(timeout_s, 30) if timeout_s is not None else 30,
            validate_host=True,
        )
        if cafile:
            kwargs["cafile"] = cafile
        return pytds.connect(**kwargs)

    def _schema_filter(self, alias: str) -> tuple[str, list[str]]:
        if not self._schemas:
            return "", []
        placeholders = ", ".join("%s" for _ in self._schemas)
        return f" AND {alias}.name IN ({placeholders})", list(self._schemas)

    def get_schemas(self) -> list[str]:
        frame = self.execute(
            """
            SELECT DISTINCT schemas.name AS schema_name
            FROM sys.objects AS objects
            JOIN sys.schemas AS schemas ON schemas.schema_id = objects.schema_id
            WHERE objects.type IN ('U', 'V')
            ORDER BY schemas.name
            """
        )
        return [str(name) for name in frame.get("schema_name", pd.Series(dtype=str))]

    def execute(
        self,
        sql: str,
        parameters: Optional[list] = None,
        *,
        timeout_s: float | None = None,
    ) -> pd.DataFrame:
        """Run one read-only statement; cap chat results at 200 rows."""
        _validate_read_query(sql)
        effective_timeout = min(timeout_s, 30) if timeout_s is not None else None
        connection = self._connect(effective_timeout)
        cursor = connection.cursor()
        started = time.monotonic()
        try:
            cursor.execute(sql, tuple(parameters) if parameters is not None else None)
            if cursor.description is None:
                return pd.DataFrame()
            columns = [description[0] for description in cursor.description]
            if timeout_s is not None:
                # Chat SQL is the only caller that supplies a timeout. Fetch one
                # extra row to report truncation without materializing large results.
                rows = cursor.fetchmany(_CHAT_MAX_ROWS + 1)
                truncated = len(rows) > _CHAT_MAX_ROWS
                frame = pd.DataFrame(rows[:_CHAT_MAX_ROWS], columns=columns)
                frame.attrs["truncated"] = truncated
                return frame
            return pd.DataFrame(cursor.fetchall(), columns=columns)
        except (TimeoutError, pytds.tds_base.TimeoutError) as exc:
            if timeout_s is None:
                raise
            raise StatementTimeout.since(effective_timeout, started) from exc
        finally:
            cursor.close()
            connection.close()

    def get_tables(self) -> pd.DataFrame:
        schema_filter, parameters = self._schema_filter("schemas")
        return self.execute(
            f"""
            SELECT
                schemas.name AS table_schema,
                objects.name AS table_name,
                CASE WHEN objects.type = 'V' THEN '{TableTypes.VIEW}'
                     ELSE '{TableTypes.BASE_TABLE}' END AS table_type
            FROM sys.objects AS objects
            JOIN sys.schemas AS schemas ON schemas.schema_id = objects.schema_id
            WHERE objects.type IN ('U', 'V') {schema_filter}
            ORDER BY schemas.name, objects.name
            """,
            parameters,
        )

    def get_columns(self) -> pd.DataFrame:
        schema_filter, parameters = self._schema_filter("schemas")
        return self.execute(
            f"""
            SELECT
                schemas.name AS table_schema,
                objects.name AS table_name,
                columns.name AS column_name,
                types.name AS data_type,
                columns.is_nullable AS is_nullable,
                columns.column_id AS ordinal_position
            FROM sys.columns AS columns
            JOIN sys.objects AS objects ON objects.object_id = columns.object_id
            JOIN sys.schemas AS schemas ON schemas.schema_id = objects.schema_id
            JOIN sys.types AS types ON types.user_type_id = columns.user_type_id
            WHERE objects.type IN ('U', 'V') {schema_filter}
            ORDER BY schemas.name, objects.name, columns.column_id
            """,
            parameters,
        )

    def get_queries(self, hours: int = 24) -> pd.DataFrame:
        # Query Store and DMVs need elevated permissions and are not required
        # for schema ingestion.
        return pd.DataFrame(columns=["end_time", "query_text"])

    def get_views(self) -> pd.DataFrame:
        schema_filter, parameters = self._schema_filter("schemas")
        return self.execute(
            f"""
            SELECT schemas.name AS table_schema,
                   objects.name AS table_name,
                   modules.definition AS view_definition
            FROM sys.views AS objects
            JOIN sys.schemas AS schemas ON schemas.schema_id = objects.schema_id
            JOIN sys.sql_modules AS modules ON modules.object_id = objects.object_id
            WHERE 1 = 1 {schema_filter}
            ORDER BY schemas.name, objects.name
            """,
            parameters,
        )

    def get_pks(self) -> pd.DataFrame:
        schema_filter, parameters = self._schema_filter("schemas")
        return self.execute(
            f"""
            SELECT schemas.name AS table_schema,
                   tables.name AS table_name,
                   columns.name AS column_name,
                   index_columns.key_ordinal AS ordinal_position
            FROM sys.indexes AS indexes
            JOIN sys.index_columns AS index_columns
              ON index_columns.object_id = indexes.object_id
             AND index_columns.index_id = indexes.index_id
            JOIN sys.tables AS tables ON tables.object_id = indexes.object_id
            JOIN sys.schemas AS schemas ON schemas.schema_id = tables.schema_id
            JOIN sys.columns AS columns
              ON columns.object_id = tables.object_id
             AND columns.column_id = index_columns.column_id
            WHERE indexes.is_primary_key = 1 {schema_filter}
            ORDER BY schemas.name, tables.name, index_columns.key_ordinal
            """,
            parameters,
        )

    def get_fks(self) -> pd.DataFrame:
        schema_filter, parameters = self._schema_filter("child_schema")
        return self.execute(
            f"""
            SELECT
                child_schema.name AS table_schema,
                child_table.name AS table_name,
                child_column.name AS column_name,
                parent_schema.name AS referenced_schema,
                parent_table.name AS referenced_table,
                parent_column.name AS referenced_column
            FROM sys.foreign_key_columns AS foreign_keys
            JOIN sys.tables AS child_table
              ON child_table.object_id = foreign_keys.parent_object_id
            JOIN sys.schemas AS child_schema
              ON child_schema.schema_id = child_table.schema_id
            JOIN sys.columns AS child_column
              ON child_column.object_id = foreign_keys.parent_object_id
             AND child_column.column_id = foreign_keys.parent_column_id
            JOIN sys.tables AS parent_table
              ON parent_table.object_id = foreign_keys.referenced_object_id
            JOIN sys.schemas AS parent_schema
              ON parent_schema.schema_id = parent_table.schema_id
            JOIN sys.columns AS parent_column
              ON parent_column.object_id = foreign_keys.referenced_object_id
             AND parent_column.column_id = foreign_keys.referenced_column_id
            WHERE 1 = 1 {schema_filter}
            ORDER BY child_schema.name, child_table.name, child_column.name
            """,
            parameters,
        )

    def ping(self) -> None:
        connection = self._connect()
        try:
            cursor = connection.cursor()
            try:
                cursor.execute("SELECT 1")
                cursor.fetchone()
            finally:
                cursor.close()
        finally:
            connection.close()

    def close(self) -> None:
        """No-op because each operation owns and closes its connection."""
