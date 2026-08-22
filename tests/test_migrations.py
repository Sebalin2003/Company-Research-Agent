from __future__ import annotations

from pathlib import Path

import pytest
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, inspect, text

from backend.app.db.migrations import SchemaMigrationError, upgrade_database
from backend.app.db.models import Base


def sqlite_url(path: Path) -> str:
    return f"sqlite:///{path.as_posix()}"


def current_revision(engine) -> str | None:
    with engine.connect() as connection:
        return MigrationContext.configure(connection).get_current_revision()


def test_empty_database_is_upgraded_to_current_schema(tmp_path: Path) -> None:
    path = tmp_path / "fresh.db"
    url = sqlite_url(path)

    assert upgrade_database(url) is None

    engine = create_engine(url)
    assert set(Base.metadata.tables) <= set(inspect(engine).get_table_names())
    assert current_revision(engine) == "20260821_0001"
    engine.dispose()


def test_upgrade_does_not_disable_application_loggers(tmp_path: Path) -> None:
    import logging

    app_logger = logging.getLogger("backend.app.services.research_service")
    app_logger.disabled = False

    upgrade_database(sqlite_url(tmp_path / "logging.db"))

    assert app_logger.disabled is False


def test_matching_unversioned_database_is_backed_up_stamped_and_preserved(
    tmp_path: Path,
) -> None:
    path = tmp_path / "existing.db"
    url = sqlite_url(path)
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO companies "
                "(id, name, normalized_name, possible_aliases_json, created_at, updated_at) "
                "VALUES ('company-1', 'Acme', 'acme', '[]', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            )
        )
    engine.dispose()

    backup = upgrade_database(url)

    assert backup is not None and backup.exists()
    engine = create_engine(url)
    assert current_revision(engine) == "20260821_0001"
    with engine.connect() as connection:
        assert connection.scalar(text("SELECT name FROM companies WHERE id='company-1'")) == "Acme"
    engine.dispose()


def test_incompatible_unversioned_database_is_rejected_without_backup(tmp_path: Path) -> None:
    path = tmp_path / "incompatible.db"
    url = sqlite_url(path)
    engine = create_engine(url)
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE unexpected (id INTEGER PRIMARY KEY)"))
    engine.dispose()

    with pytest.raises(SchemaMigrationError, match="no coincide"):
        upgrade_database(url)

    assert not list(tmp_path.glob("*.bak-*"))
    engine = create_engine(url)
    assert "alembic_version" not in inspect(engine).get_table_names()
    engine.dispose()


def test_current_version_is_idempotent_and_does_not_create_backup(tmp_path: Path) -> None:
    path = tmp_path / "current.db"
    url = sqlite_url(path)
    upgrade_database(url)

    assert upgrade_database(url) is None
    assert not list(tmp_path.glob("*.bak-*"))
