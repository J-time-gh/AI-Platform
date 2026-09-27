"""add persistent document-ingestion flow progress

Revision ID: d4a81f6c2e90
Revises: f27d93bc1a64
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "d4a81f6c2e90"
down_revision: str | Sequence[str] | None = "f27d93bc1a64"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the persisted node and coarse progress shown by each document flowchart."""
    op.add_column(
        "documents",
        sa.Column(
            "ingest_stage",
            sa.String(length=32),
            nullable=False,
            server_default="upload_received",
        ),
    )
    op.add_column(
        "documents",
        sa.Column(
            "ingest_progress",
            sa.Integer(),
            nullable=False,
            server_default="5",
        ),
    )

    # Existing records have no step history. Map their known terminal/running status
    # to the nearest honest diagram node; old failures use "unknown" rather than
    # pretending we know which ingestion step failed.
    op.execute(
        """
        UPDATE documents
        SET ingest_stage = CASE
            WHEN status = 'ready' THEN 'ready'
            WHEN status = 'processing' THEN 'worker_start'
            WHEN status = 'pending' THEN 'queued'
            ELSE 'unknown'
        END,
        ingest_progress = CASE
            WHEN status = 'ready' THEN 100
            WHEN status = 'processing' THEN 20
            WHEN status = 'pending' THEN 10
            ELSE 0
        END
        """,
    )


def downgrade() -> None:
    """Remove the persisted document-ingestion diagram state."""
    op.drop_column("documents", "ingest_progress")
    op.drop_column("documents", "ingest_stage")
