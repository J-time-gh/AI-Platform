from fastapi import APIRouter, HTTPException, status

from app.api.dependencies.auth import CurrentUserDep
from app.db.session import DbSession
from app.schemas.chat_session import (
    ChatSessionResponse,
    CreateChatSessionRequest,
    PersistedChatMessageResponse,
    UpdateChatSessionSkillRequest,
)
from app.services import chat_session_store
from app.services.skill_registry import SkillNotFound, skill_registry

router = APIRouter(
    prefix="/chat/sessions",
    tags=["chat-sessions"],
)


@router.post(
    "",
    response_model=ChatSessionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_chat_session(
    request: CreateChatSessionRequest,
    db: DbSession,
    current_user: CurrentUserDep,
) -> ChatSessionResponse:
    return chat_session_store.create(
        db,
        user_id=current_user.id,
        title=request.title,
    )


@router.get(
    "",
    response_model=list[ChatSessionResponse],
)
async def list_chat_sessions(
    db: DbSession,
    current_user: CurrentUserDep,
) -> list[ChatSessionResponse]:
    return chat_session_store.list_all(
        db,
        current_user.id,
    )


@router.patch(
    "/{session_id}/skill",
    response_model=ChatSessionResponse,
)
async def update_chat_session_skill(
    session_id: str,
    request: UpdateChatSessionSkillRequest,
    db: DbSession,
    current_user: CurrentUserDep,
) -> ChatSessionResponse:
    # 先确认会话属于当前用户，避免对其他用户的会话做任何判断或更新。
    session = chat_session_store.get(
        db,
        session_id,
        current_user.id,
    )
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="会话不存在",
        )

    if request.active_skill_id is not None:
        try:
            skill_registry.get(request.active_skill_id)
        except SkillNotFound as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(exc),
            ) from exc

    updated = chat_session_store.set_active_skill(
        db,
        session_id=session_id,
        user_id=current_user.id,
        active_skill_id=request.active_skill_id,
    )
    if updated is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="会话不存在",
        )

    return updated


@router.get(
    "/{session_id}/messages",
    response_model=list[PersistedChatMessageResponse],
)
async def list_chat_messages(
    session_id: str,
    db: DbSession,
    current_user: CurrentUserDep,
) -> list[PersistedChatMessageResponse]:
    messages = chat_session_store.list_messages(
        db,
        session_id,
        current_user.id,
    )

    if messages is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="会话不存在",
        )

    return messages


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_chat_session(
    session_id: str,
    db: DbSession,
    current_user: CurrentUserDep,
) -> None:
    deleted = chat_session_store.delete(
        db,
        session_id,
        current_user.id,
    )

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="会话不存在",
        )
