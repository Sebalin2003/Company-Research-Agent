from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil

from alembic import command
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import make_url

from backend.app.core.config import PROJECT_ROOT
from backend.app.db.models import Base


ALEMBIC_INI = PROJECT_ROOT / "alembic.ini"


class SchemaMigrationError(RuntimeError):
    pass


def upgrade_database(database_url: str) -> Path | None:
    config = alembic_config(database_url)
    engine = create_engine(
        database_url,
        connect_args={"check_same_thread": False} if database_url.startswith("sqlite") else {},
    )
    backup: Path | None = None
    try:
        with engine.connect() as connection:
            tables = set(inspect(connection).get_table_names())
            current = MigrationContext.configure(connection).get_current_revision()

        if not tables:
            command.upgrade(config, "head")
            return None

        if current is None:
            validate_current_schema(engine)
            backup = backup_sqlite_database(database_url)
            command.stamp(config, "head")
            return backup

        head = ScriptDirectory.from_config(config).get_current_head()
        if current != head:
            backup = backup_sqlite_database(database_url)
        command.upgrade(config, "head")
        return backup
    finally:
        engine.dispose()


def validate_current_schema(engine) -> None:
    actual = inspect(engine)
    actual_tables = set(actual.get_table_names()) - {"alembic_version"}
    expected_tables = set(Base.metadata.tables)
    if actual_tables != expected_tables:
        missing = sorted(expected_tables - actual_tables)
        unexpected = sorted(actual_tables - expected_tables)
        raise SchemaMigrationError(
            "La base de datos local no coincide con el esquema actual. "
            f"Tablas faltantes: {missing or 'ninguna'}; "
            f"tablas inesperadas: {unexpected or 'ninguna'}."
        )

    for table_name, table in Base.metadata.tables.items():
        expected_columns = {column.name for column in table.columns}
        actual_columns = {column["name"] for column in actual.get_columns(table_name)}
        if actual_columns != expected_columns:
            raise SchemaMigrationError(
                "La base de datos local no coincide con el esquema actual. "
                f"Columnas incompatibles en {table_name}."
            )


def backup_sqlite_database(database_url: str) -> Path | None:
    url = make_url(database_url)
    if url.drivername != "sqlite" or not url.database or url.database == ":memory:":
        return None
    source = Path(url.database)
    if not source.is_absolute():
        source = (PROJECT_ROOT / source).resolve()
    if not source.exists():
        return None
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    destination = source.with_name(f"{source.name}.bak-{timestamp}")
    shutil.copy2(source, destination)
    return destination


def alembic_config(database_url: str) -> Config:
    config = Config(str(ALEMBIC_INI))
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    return config
