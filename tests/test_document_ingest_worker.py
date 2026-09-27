import io

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.db.models.chunk import Chunk
from app.db.models.document import Document
from app.services import document_store
from app.services.embedding import FakeEmbedder
from app.workers.document_ingest import _process_document


@pytest.fixture
def pending_document(
    client: TestClient,
    authenticated_headers: dict[str, str],
    db_session,
) -> Document:
    """通过真实上传接口创建一个 pending 文档。"""
    response = client.post(
        "/documents/upload",
        headers=authenticated_headers,
        files={
            "file": (
                "worker-success.txt",
                io.BytesIO("这是 Worker 成功处理的测试文档。".encode()),
                "text/plain",
            ),
        },
    )

    assert response.status_code == 202

    document_id = response.json()["id"]
    document = db_session.get(Document, document_id)

    assert document is not None
    assert document.status == "pending"

    return document


@pytest.fixture
def empty_pending_document(
    client: TestClient,
    authenticated_headers: dict[str, str],
    db_session,
) -> Document:
    """创建一个 Worker 应处理失败的空文档。"""
    response = client.post(
        "/documents/upload",
        headers=authenticated_headers,
        files={
            "file": (
                "worker-empty.txt",
                io.BytesIO(b"   \n"),
                "text/plain",
            ),
        },
    )

    assert response.status_code == 202

    document_id = response.json()["id"]
    document = db_session.get(Document, document_id)

    assert document is not None
    assert document.status == "pending"

    return document


def test_worker_ingests_pending_document(
    db_session,
    pending_document: Document,
) -> None:
    _process_document(
        db_session,
        pending_document.id,
        FakeEmbedder(),
    )

    document = db_session.get(
        Document,
        pending_document.id,
    )

    assert document is not None
    assert document.status == "ready"
    assert document.ingest_stage == "ready"
    assert document.ingest_progress == 100
    assert document.ingest_error is None

    chunks = db_session.scalars(
        select(Chunk).where(Chunk.document_id == pending_document.id).order_by(Chunk.chunk_index),
    ).all()

    assert chunks
    assert [chunk.chunk_index for chunk in chunks] == list(
        range(len(chunks)),
    )


def test_queue_ack_does_not_overwrite_a_worker_that_started_early(
    db_session,
    pending_document: Document,
) -> None:
    claimed = document_store.claim_for_ingest(db_session, pending_document.id)
    assert claimed is not None
    assert claimed.ingest_stage == "worker_start"

    # The queue callback can race with a fast Worker; the API must not move the
    # persistent flow diagram backwards from worker_start to queued.
    assert document_store.mark_queued(db_session, pending_document.id) is False

    refreshed = db_session.get(Document, pending_document.id)
    assert refreshed is not None
    assert refreshed.status == "processing"
    assert refreshed.ingest_stage == "worker_start"
    assert refreshed.ingest_progress == 20


def test_worker_marks_empty_document_failed(
    db_session,
    empty_pending_document: Document,
) -> None:
    _process_document(
        db_session,
        empty_pending_document.id,
        FakeEmbedder(),
    )

    document = db_session.get(
        Document,
        empty_pending_document.id,
    )

    assert document is not None
    assert document.status == "failed"
    assert document.ingest_stage == "chunking"
    assert document.ingest_progress == 45
    assert document.ingest_error is not None
    assert "文本" in document.ingest_error

    chunks = db_session.scalars(
        select(Chunk).where(
            Chunk.document_id == empty_pending_document.id,
        ),
    ).all()

    assert chunks == []


def test_worker_does_not_process_ready_document_twice(
    db_session,
    pending_document: Document,
) -> None:
    _process_document(
        db_session,
        pending_document.id,
        FakeEmbedder(),
    )

    first_chunks = db_session.scalars(
        select(Chunk).where(
            Chunk.document_id == pending_document.id,
        ),
    ).all()

    _process_document(
        db_session,
        pending_document.id,
        FakeEmbedder(),
    )

    second_chunks = db_session.scalars(
        select(Chunk).where(
            Chunk.document_id == pending_document.id,
        ),
    ).all()

    document = db_session.get(
        Document,
        pending_document.id,
    )

    assert document is not None
    assert document.status == "ready"
    assert len(second_chunks) == len(first_chunks)
