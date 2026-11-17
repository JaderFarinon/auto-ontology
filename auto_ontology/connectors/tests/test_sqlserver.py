# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pandas as pd
import pytest
from pytest import MonkeyPatch

from auto_ontology.connectors.connection_string_factory import build_connection_string
from auto_ontology.connectors.registry import CONNECTOR_REGISTRY, create_connector
from auto_ontology.connectors.sqlserver import (
    SQLServerDatabase,
    _parse_connection_string,
    _validate_read_query,
)


def test_parse_connection_string_decodes_credentials_and_database() -> None:
    parsed = _parse_connection_string(
        "mssql://user%40domain:secret%2Fvalue@clinic.example:1434/Clinic%20DB"
    )
    assert parsed == {
        "server": "clinic.example",
        "port": 1434,
        "user": "user@domain",
        "password": "secret/value",
        "database": "Clinic DB",
        "cafile": None,
    }


def test_build_connection_string() -> None:
    assert (
        build_connection_string(
            {
                "type": "mssql",
                "host": "clinic.example",
                "user": "user@domain",
                "password": "secret/value",
                "database": "Clinic DB",
            }
        )
        == "mssql://user%40domain:secret%2Fvalue@clinic.example:1433/Clinic%20DB"
    )


def test_registry_registers_sql_server_and_tsql_dialect() -> None:
    assert CONNECTOR_REGISTRY["mssql"] is SQLServerDatabase
    assert SQLServerDatabase.dialect.fget(None) == "tsql"


class Cursor:
    def __init__(self) -> None:
        self.description = [("value",)]
        self.statements: list[tuple[str, Any]] = []
        self.rows = [(1,), (2,)]
        self.closed = False

    def execute(self, statement: str, parameters: Any = None) -> None:
        self.statements.append((statement, parameters))

    def fetchone(self) -> tuple[int]:
        return (1,)

    def fetchall(self) -> list[tuple[int]]:
        return self.rows

    def fetchmany(self, size: int) -> list[tuple[int]]:
        return self.rows[:size]

    def close(self) -> None:
        self.closed = True


class Connection:
    def __init__(self, kwargs: dict[str, Any]) -> None:
        self.kwargs = kwargs
        self.cursor_instance = Cursor()
        self.closed = False

    def cursor(self) -> Cursor:
        return self.cursor_instance

    def close(self) -> None:
        self.closed = True


def test_connector_runs_read_only_and_caps_chat_rows(monkeypatch: MonkeyPatch) -> None:
    connections: list[Connection] = []

    def connect(**kwargs: Any) -> Connection:
        connection = Connection(kwargs)
        connections.append(connection)
        return connection

    monkeypatch.setattr("auto_ontology.connectors.sqlserver.pytds.connect", connect)
    database = create_connector("mssql://reader:secret@clinic.example:1433/Clinic")
    assert isinstance(database, SQLServerDatabase)
    assert database.database_name == "Clinic"
    assert connections[0].kwargs["readonly"] is True
    assert connections[0].kwargs["validate_host"] is True

    result = database.execute("SELECT %s AS value", [7], timeout_s=30)

    assert isinstance(result, pd.DataFrame)
    assert result.to_dict(orient="records") == [{"value": 1}, {"value": 2}]
    assert connections[1].cursor_instance.statements == [("SELECT %s AS value", (7,))]
    assert connections[1].closed
    assert result.attrs["truncated"] is False


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM dbo.Invoice",
        "UPDATE dbo.Invoice SET Amount = 0",
        "DROP TABLE dbo.Invoice",
        "SELECT * INTO dbo.Copy FROM dbo.Invoice",
        "SELECT 1; DELETE FROM dbo.Invoice",
    ],
)
def test_validate_read_query_rejects_writes(sql: str) -> None:
    with pytest.raises(ValueError):
        _validate_read_query(sql)


def test_validate_read_query_allows_select_with_cte() -> None:
    _validate_read_query(
        "WITH totals AS (SELECT 1 AS amount) SELECT amount FROM totals"
    )
