"""密码哈希与 JWT 访问令牌工具。"""

from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
from jwt import InvalidTokenError
from pwdlib import PasswordHash

from app.core.config import settings

password_hasher = PasswordHash.recommended()


class InvalidAccessTokenError(Exception):
    """访问令牌缺失、无效或已过期。"""


def hash_password(password: str) -> str:
    """使用 Argon2 哈希密码，禁止存储明文密码。"""
    return password_hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    """验证输入密码是否与保存的密码哈希匹配。"""
    return password_hasher.verify(password, password_hash)


def create_access_token(user_id: str) -> str:
    """为指定用户签发短期 Bearer Token。"""
    now = datetime.now(UTC)
    expires_at = now + timedelta(
        minutes=settings.jwt_access_token_expire_minutes,
    )

    payload: dict[str, Any] = {
        "sub": user_id,
        "type": "access",
        "iat": now,
        "exp": expires_at,
    }

    return jwt.encode(
        payload,
        settings.jwt_secret_key.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )


def get_user_id_from_access_token(token: str) -> str:
    """验证访问令牌，并返回其中的用户 ID。"""
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key.get_secret_value(),
            algorithms=[settings.jwt_algorithm],
        )
    except InvalidTokenError as exc:
        raise InvalidAccessTokenError("访问令牌无效或已过期") from exc

    user_id = payload.get("sub")
    token_type = payload.get("type")

    if token_type != "access" or not isinstance(user_id, str) or not user_id:
        raise InvalidAccessTokenError("访问令牌无效")

    return user_id
