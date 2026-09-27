"""add document ingest status

Revision ID: a80df3b9de4c
Revises: 96cbce9a33f8
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "a80df3b9de4c"
down_revision: Union[str, Sequence[str], None] = "96cbce9a33f8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add asynchronous document-ingestion state fields."""
    op.add_column(
        "documents",
        sa.Column(
            "status",
            sa.String(length=16),
            server_default="pending",
            nullable=False,
        ),
    )

    op.add_column(
        "documents",
        sa.Column(
            "ingest_error",
            sa.String(length=1000),
            nullable=True,
        ),
    )

    # 当前已有的 Document 都来自旧同步入库流程，应视为已完成。
    op.execute(
        "UPDATE documents SET status = 'ready'",
    )

    op.create_check_constraint(
        "ck_documents_status",
        "documents",
        "status IN ('pending', 'processing', 'ready', 'failed')",
    )

    op.create_index(
        "ix_documents_user_id_status",
        "documents",
        ["user_id", "status"],
        unique=False,
    )


def downgrade() -> None:
    """Remove asynchronous document-ingestion state fields."""
    op.drop_index(
        "ix_documents_user_id_status",
        table_name="documents",
    )

    op.drop_constraint(
        "ck_documents_status",
        "documents",
        type_="check",
    )

    op.drop_column(
        "documents",
        "ingest_error",
    )

    op.drop_column(
        "documents",
        "status",
    )