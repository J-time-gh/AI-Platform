import pytest

from app.core.security import (
    InvalidAccessTokenError,
    create_access_token,
    get_user_id_from_access_token,
    hash_password,
    verify_password,
)


def test_hash_password_does_not_return_plaintext() -> None:
    password = "correct-horse-battery-staple"

    password_hash = hash_password(password)

    assert password_hash != password
    assert verify_password(password, password_hash) is True


def test_verify_password_rejects_wrong_password() -> None:
    password_hash = hash_password("correct-password")

    assert verify_password("wrong-password", password_hash) is False


def test_access_token_returns_original_user_id() -> None:
    token = create_access_token("user-123")

    assert get_user_id_from_access_token(token) == "user-123"


def test_invalid_access_token_is_rejected() -> None:
    with pytest.raises(InvalidAccessTokenError):
        get_user_id_from_access_token("this-is-not-a-valid-token")
