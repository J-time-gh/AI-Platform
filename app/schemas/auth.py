"""认证接口的请求与响应模型。

是 Pydantic Schema，即 API 的输入和输出合同
它定义客户端能提交什么、后端会返回什么"""

from typing import Literal

from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    """注册请求。"""

    email: EmailStr
    password: str = Field(min_length=12, max_length=128)


class LoginRequest(BaseModel):
    """登录请求。"""

    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class TokenResponse(BaseModel):
    """登录成功后的访问令牌。"""

    access_token: str
    token_type: Literal["bearer"] = "bearer"


class CurrentUserResponse(BaseModel):
    """返回给客户端的当前用户信息。"""

    id: str
    email: EmailStr
