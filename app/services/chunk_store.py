"""chunks 表的数据存取。"""

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.db.models.chunk import Chunk


def replace_chunks(
    db: Session,
    document_id: str,
    contents: list[str],
    vectors: list[list[float]],
) -> int:
    """先清掉该文档旧块，再写入新块（幂等，支持重新入库）。"""
    db.execute(delete(Chunk).where(Chunk.document_id == document_id))
    now = datetime.now(UTC)
    for index, (content, vector) in enumerate(zip(contents, vectors, strict=True)):
        db.add(
            Chunk(
                id=str(uuid4()),
                document_id=document_id,
                chunk_index=index,
                content=content,
                embedding=vector,
                created_at=now,
            )
        )
    db.commit()
    return len(contents)
