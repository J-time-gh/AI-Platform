"""persist RAG sources on chat messages

Revision ID: f27d93bc1a64
Revises: c1e9b78a40d2
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "f27d93bc1a64"
down_revision: str | Sequence[str] | None = "c1e9b78a40d2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Persist the search chunks and highlight terms returned for an answer."""
    op.add_column(
        "chat_messages",
        sa.Column(
            "sources",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'[]'::json"),
        ),
    )


def downgrade() -> None:
    """Remove persisted RAG source data."""
    op.drop_column("chat_messages", "sources")
