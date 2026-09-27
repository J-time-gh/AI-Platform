import io
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings


@pytest.fixture
def auth_headers(client: TestClient) -> dict[str, str]:
    password = "correct-horse-battery-staple"
    register_response = client.post(
        "/auth/register",
        json={
            "email": "document-user@example.com",
            "password": password,
        },
    )
    assert register_response.status_code == 201

    login_response = client.post(
        "/auth/login",
        json={
            "email": "document-user@example.com",
            "password": password,
        },
    )
    assert login_response.status_code == 200

    return {
        "Authorization": f"Bearer {login_response.json()['access_token']}",
    }


def _upload(
    client: TestClient,
    headers: dict[str, str],
    filename: str = "notes.txt",
    content: bytes = b"hello world",
) -> httpx.Response:
    return client.post(
        "/documents/upload",
        headers=headers,
        files={"file": (filename, io.BytesIO(content), "text/plain")},
    )


def test_documents_require_authentication(client: TestClient) -> None:
    response = client.get("/documents")

    assert response.status_code == 401


def test_upload_returns_pending_metadata(
    client: TestClient,
    auth_headers: dict[str, str],
    fake_ingest_queue,
) -> None:
    response = _upload(client, auth_headers)

    assert response.status_code == 202

    body = response.json()

    assert body["filename"] == "notes.txt"
    assert body["size"] == 11
    assert body["id"]
    assert body["uploaded_at"]
    assert body["status"] == "pending"
    assert body["ingest_stage"] == "queued"
    assert body["ingest_progress"] == 10
    assert body["ingest_error"] is None

    assert fake_ingest_queue.document_ids == [body["id"]]


def test_upload_rejects_unsupported_type(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    response = _upload(client, auth_headers, filename="virus.exe")

    assert response.status_code == 415


def test_upload_rejects_oversized_file(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    big = b"x" * (20 * 1024 * 1024 + 1)
    response = _upload(client, auth_headers, filename="big.txt", content=big)

    assert response.status_code == 413


def test_list_documents(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    _upload(client, auth_headers, "a.txt")
    _upload(client, auth_headers, "b.md")

    response = client.get("/documents", headers=auth_headers)

    assert response.status_code == 200

    documents = response.json()
    assert len(documents) == 2
    assert all(document["status"] == "pending" for document in documents)


def test_get_document_by_id(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    doc_id = _upload(client, auth_headers).json()["id"]

    response = client.get(f"/documents/{doc_id}", headers=auth_headers)

    assert response.status_code == 200
    assert response.json()["id"] == doc_id
    assert response.json()["ingest_stage"] == "queued"
    assert response.json()["ingest_progress"] == 10


def test_get_missing_document_returns_404(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    response = client.get("/documents/not-exist", headers=auth_headers)

    assert response.status_code == 404


def test_delete_document(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    doc_id = _upload(client, auth_headers).json()["id"]

    assert client.delete(f"/documents/{doc_id}", headers=auth_headers).status_code == 204
    assert client.get(f"/documents/{doc_id}", headers=auth_headers).status_code == 404


def test_delete_missing_document_returns_404(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    response = client.delete("/documents/nope", headers=auth_headers)

    assert response.status_code == 404


def test_upload_saves_file_to_disk(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    doc_id = _upload(client, auth_headers).json()["id"]

    files = list(Path(settings.upload_dir).glob(f"{doc_id}*"))

    assert len(files) == 1


def test_delete_removes_file_from_disk(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    doc_id = _upload(client, auth_headers).json()["id"]
    path = next(Path(settings.upload_dir).glob(f"{doc_id}*"))

    assert path.exists()
    assert client.delete(f"/documents/{doc_id}", headers=auth_headers).status_code == 204
    assert not path.exists()


def test_delete_pending_document_removes_file_and_record(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    doc_id = _upload(client, auth_headers).json()["id"]
    path = next(Path(settings.upload_dir).glob(f"{doc_id}*"))

    response = client.delete(
        f"/documents/{doc_id}",
        headers=auth_headers,
    )

    assert response.status_code == 204
    assert not path.exists()

    get_response = client.get(
        f"/documents/{doc_id}",
        headers=auth_headers,
    )
    assert get_response.status_code == 404


def test_pending_document_is_not_searchable(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    text = "这是尚未完成入库的机密资料"

    response = _upload(
        client,
        auth_headers,
        filename="pending.txt",
        content=text.encode(),
    )

    assert response.status_code == 202

    search_response = client.post(
        "/search",
        headers=auth_headers,
        json={
            "query": text,
            "top_k": 5,
        },
    )

    assert search_response.status_code == 200
    assert search_response.json() == []
