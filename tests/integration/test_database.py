import os

import pytest

from spark_rag.db.connection import check_database, get_connection
from spark_rag.db.models import initialize_schema

pytestmark = pytest.mark.integration

requires_database = pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION") != "1",
    reason="Define RUN_INTEGRATION=1 con PostgreSQL/pgvector disponible.",
)


@requires_database
def test_postgresql_y_pgvector_disponibles() -> None:
    status = check_database()
    assert status.ok, status.error
    assert status.server_version
    assert status.pgvector_version


@requires_database
def test_esquema_de_control_idempotente() -> None:
    with get_connection() as connection:
        initialize_schema(connection)
        initialize_schema(connection)
        row = connection.execute(
            "SELECT to_regclass('public.ingested_files') AS table_name"
        ).fetchone()

    assert row is not None
    assert row["table_name"] is not None
