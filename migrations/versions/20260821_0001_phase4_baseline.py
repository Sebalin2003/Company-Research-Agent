"""Phase 4 baseline for the complete Radar Laboral schema."""

from alembic import op

from backend.app.db.models import Base


revision = "20260821_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    Base.metadata.create_all(bind=op.get_bind())


def downgrade() -> None:
    Base.metadata.drop_all(bind=op.get_bind())
