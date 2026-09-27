from fastapi.testclient import TestClient

from app.core.security import get_user_id_from_access_token


def _register(
    client: TestClient,
    email: str = "alice@example.com",
    password: str = "correct-horse-battery-staple",
):
    return client.post(
        "/auth/register",
        json={
            "email": email,
            "password": password,
        },
    )


def test_register_creates_user(client: TestClient) -> None:
    response = _register(client)

    assert response.status_code == 201
    body = response.json()
    assert body["id"]
    assert body["email"] == "alice@example.com"
    assert "password" not in body
    assert "password_hash" not in body


def test_register_rejects_duplicate_email(client: TestClient) -> None:
    _register(client)

    response = _register(client)

    assert response.status_code == 409
    assert response.json()["detail"] == "该邮箱已注册"


def test_register_rejects_short_password(client: TestClient) -> None:
    response = _register(client, password="short")

    assert response.status_code == 422


def test_login_returns_access_token(client: TestClient) -> None:
    response = _register(client)
    user_id = response.json()["id"]

    login_response = client.post(
        "/auth/login",
        json={
            "email": "alice@example.com",
            "password": "correct-horse-battery-staple",
        },
    )

    assert login_response.status_code == 200
    body = login_response.json()
    assert body["token_type"] == "bearer"
    assert get_user_id_from_access_token(body["access_token"]) == user_id


def test_login_rejects_wrong_password(client: TestClient) -> None:
    _register(client)

    response = client.post(
        "/auth/login",
        json={
            "email": "alice@example.com",
            "password": "wrong-password",
        },
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "邮箱或密码错误"


def test_login_rejects_unknown_email(client: TestClient) -> None:
    response = client.post(
        "/auth/login",
        json={
            "email": "unknown@example.com",
            "password": "correct-horse-battery-staple",
        },
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "邮箱或密码错误"


def _authorization_headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
    }


def test_get_me_returns_current_user(client: TestClient) -> None:
    register_response = _register(client)
    expected_user = register_response.json()

    login_response = client.post(
        "/auth/login",
        json={
            "email": "alice@example.com",
            "password": "correct-horse-battery-staple",
        },
    )
    token = login_response.json()["access_token"]

    response = client.get(
        "/auth/me",
        headers=_authorization_headers(token),
    )

    assert response.status_code == 200
    assert response.json() == expected_user


def test_get_me_rejects_missing_token(client: TestClient) -> None:
    response = client.get("/auth/me")

    assert response.status_code == 401


def test_get_me_rejects_invalid_token(client: TestClient) -> None:
    response = client.get(
        "/auth/me",
        headers=_authorization_headers("not-a-valid-token"),
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "无效或已过期的访问令牌"
