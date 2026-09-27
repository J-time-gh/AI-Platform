from app.core.security import hash_password
from app.services import user_store


def test_create_user_and_find_by_email(db_session) -> None:
    user = user_store.create(
        db_session,
        email="Alice@Example.com",
        password_hash=hash_password("correct-horse-battery-staple"),
    )

    found = user_store.get_by_email(db_session, "alice@example.com")

    assert found is not None
    assert found.id == user.id
    assert found.email == "alice@example.com"
    assert found.password_hash != "correct-horse-battery-staple"
    assert found.is_active is True


def test_get_by_email_returns_none_for_missing_user(db_session) -> None:
    user = user_store.get_by_email(db_session, "missing@example.com")

    assert user is None


def test_get_by_id_returns_created_user(db_session) -> None:
    created = user_store.create(
        db_session,
        email="user@example.com",
        password_hash=hash_password("correct-horse-battery-staple"),
    )

    found = user_store.get_by_id(db_session, created.id)

    assert found is not None
    assert found.email == "user@example.com"
