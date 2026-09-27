"""restore chunks hnsw index

Revision ID: a0869ae5ca15
Revises: 2b756bcfb96e

"""
from typing import Sequence, Union

from alembic import op



# revision identifiers, used by Alembic.
revision: str = 'a0869ae5ca15'
down_revision: Union[str, Sequence[str], None] = '2b756bcfb96e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "ix_chunks_embedding_hnsw",
        "chunks",
        ["embedding"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )



def downgrade() -> None:
    op.drop_index("ix_chunks_embedding_hnsw", table_name="chunks")
