"""用户注册与登录接口。创建认证路由"""

from fastapi import APIRouter, HTTPException, status
from sqlalchemy.exc import IntegrityError

from app.api.dependencies.auth import CurrentUserDep
from app.core.security import (
    create_access_token,
    hash_password,
    verify_password,
)
from app.db.session import DbSession
from app.schemas.auth import (
    CurrentUserResponse,
    LoginRequest,
    RegisterRequest,
    TokenResponse,
)
from app.services import user_store

router = APIRouter(prefix="/auth", tags=["auth"])


def _invalid_credentials_error() -> HTTPException:
    """统一返回登录失败，避免泄漏邮箱是否存在。"""
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="邮箱或密码错误",
        headers={"WWW-Authenticate": "Bearer"},
    )


@router.post(
    "/register",
    response_model=CurrentUserResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register(
    request: RegisterRequest,
    db: DbSession,
) -> CurrentUserResponse:
    """注册用户并保存密码哈希。"""
    existing_user = user_store.get_by_email(db, request.email)

    if existing_user is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="该邮箱已注册",
        )

    try:
        user = user_store.create(
            db,
            email=str(request.email),
            password_hash=hash_password(request.password),
        )
    except IntegrityError as exc:
        # 防止两个相同邮箱的注册请求并发到达时绕过前面的查询。
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="该邮箱已注册",
        ) from exc

    return CurrentUserResponse(
        id=user.id,
        email=user.email,
    )


@router.post("/login", response_model=TokenResponse)
async def login(
    request: LoginRequest,
    db: DbSession,
) -> TokenResponse:
    """验证邮箱和密码，成功后签发短期 JWT Access Token。"""
    user = user_store.get_by_email(db, request.email)

    if (
        user is None
        or not user.is_active
        or not verify_password(request.password, user.password_hash)
    ):
        raise _invalid_credentials_error()

    return TokenResponse(
        access_token=create_access_token(user.id),
    )


# response_model=CurrentUserResponse 是一层额外保护
# 即使未来 User ORM 对象新增敏感字段，接口也只会序列化 id 和 email。
@router.get("/me", response_model=CurrentUserResponse)
async def get_me(
    current_user: CurrentUserDep,
) -> CurrentUserResponse:
    """返回当前登录用户的公开信息。"""
    return CurrentUserResponse(
        id=current_user.id,
        email=current_user.email,
    )
