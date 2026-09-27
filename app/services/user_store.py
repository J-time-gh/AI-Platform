"""users 表的数据访问层。(只负责读写用户表)
 users 表
     ↕
Python User ORM 对象"""

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models.user import User


def normalize_email(email: str) -> str:
    """统一邮箱格式，避免大小写或空格导致重复账户。"""
    return email.strip().lower()


def get_by_id(db: Session, user_id: str) -> User | None:
    """根据用户 ID 查询用户。"""
    # JWT 验证完成后会得到user_id = JWT payload 中的 sub
    return db.get(User, user_id)


def get_by_email(db: Session, email: str) -> User | None:
    """根据邮箱查询用户。"""
    # 统一邮箱格式，避免大小写或空格导致重复账户。
    normalized_email = normalize_email(email)

    return db.scalar(
        select(User).where(User.email == normalized_email),
    )


def create(
    # 创建用户记录
    db: Session,
    *,
    email: str,
    password_hash: str,
) -> User:
    """创建用户并保存密码哈希。"""
    user = User(
        id=str(uuid4()),
        email=normalize_email(email),
        password_hash=password_hash,
        is_active=True,
        created_at=datetime.now(UTC),
    )
    # 加入当前数据库事务
    db.add(user)
    #  真正写入 PostgreSQL
    db.commit()
    # 从数据库重新读取完整对象
    db.refresh(user)

    return user
