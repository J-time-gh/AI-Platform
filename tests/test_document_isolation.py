import io

from fastapi.testclient import TestClient


def _auth_headers(client: TestClient, email: str) -> dict[str, str]:
    password = "correct-horse-battery-staple"

    register_response = client.post(
        "/auth/register",
        json={
            "email": email,
            "password": password,
        },
    )
    assert register_response.status_code == 201

    login_response = client.post(
        "/auth/login",
        json={
            "email": email,
            "password": password,
        },
    )
    assert login_response.status_code == 200

    return {
        "Authorization": f"Bearer {login_response.json()['access_token']}",
    }


def _upload_document(
    client: TestClient,
    headers: dict[str, str],
) -> str:
    response = client.post(
        "/documents/upload",
        headers=headers,
        files={
            "file": (
                "private.txt",
                io.BytesIO("用户私有资料".encode()),
                "text/plain",
            ),
        },
    )
    assert response.status_code == 202

    return response.json()["id"]


def test_user_cannot_access_another_users_document(
    client: TestClient,
) -> None:
    user_a_headers = _auth_headers(client, "user-a@example.com")
    user_b_headers = _auth_headers(client, "user-b@example.com")

    document_id = _upload_document(client, user_a_headers)

    user_b_documents = client.get(
        "/documents",
        headers=user_b_headers,
    )
    assert user_b_documents.status_code == 200
    assert user_b_documents.json() == []

    assert (
        client.get(
            f"/documents/{document_id}",
            headers=user_b_headers,
        ).status_code
        == 404
    )

    assert (
        client.delete(
            f"/documents/{document_id}",
            headers=user_b_headers,
        ).status_code
        == 404
    )

    user_a_documents = client.get(
        "/documents",
        headers=user_a_headers,
    )
    assert user_a_documents.status_code == 200
    assert [document["id"] for document in user_a_documents.json()] == [document_id]
