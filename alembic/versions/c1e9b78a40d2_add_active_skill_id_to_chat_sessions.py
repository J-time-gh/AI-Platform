"""add active_skill_id to chat sessions

Revision ID: c1e9b78a40d2
Revises: a80df3b9de4c
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op


revision: str = "c1e9b78a40d2"
down_revision: str | Sequence[str] | None = "a80df3b9de4c"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Persist the selected Skill on each chat session."""
    op.add_column(
        "chat_sessions",
        sa.Column(
            "active_skill_id",
            sa.String(length=64),
            nullable=True,
        ),
    )


def downgrade() -> None:
    """Remove the selected Skill field from chat sessions."""
    op.drop_column(
        "chat_sessions",
        "active_skill_id",
    )
