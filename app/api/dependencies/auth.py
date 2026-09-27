"""FastAPI 当前登录用户认证依赖。"""

from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from app.core.security import (
    InvalidAccessTokenError,
    get_user_id_from_access_token,
)
from app.db.models.user import User
from app.db.session import DbSession
from app.services import user_store

# 从请求头中提取 Bearer Token
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def _credentials_error() -> HTTPException:
    """统一返回认证失败响应。"""
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无效或已过期的访问令牌",
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_current_user(
    # 四层检查
    token: Annotated[str, Depends(oauth2_scheme)],
    db: DbSession,
) -> User:
    """验证 Bearer Token，并返回当前启用状态的用户。"""
    try:
        # 验证 JWT
        user_id = get_user_id_from_access_token(token)
    except InvalidAccessTokenError as exc:
        raise _credentials_error() from exc
    # 数据库确认用户
    user = user_store.get_by_id(db, user_id)

    if user is None or not user.is_active:
        raise _credentials_error()

    return user


# 用户隔离的核心
# 简化后续受保护接口
# 以后任何需要登录的接口，都不必重复写 Token 验证逻辑，只需写：
CurrentUserDep = Annotated[User, Depends(get_current_user)]
