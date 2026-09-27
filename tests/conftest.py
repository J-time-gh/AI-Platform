import shutil
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

import app.db.models  # noqa: F401  导入以注册所有模型
from app.core.config import settings
from app.db.base import Base
from app.db.session import get_db
from app.main import app
from app.services.embedding import FakeEmbedder, get_embedder
from app.services.ingest_queue import get_ingest_queue
from app.services.llm import LLMToolResponse, LLMUnavailable, get_llm_client
from app.services.reranker import FakeReranker, get_reranker
from app.workers.document_ingest import _process_document

# 测试的文件落盘目录（与生产目录隔离）
TEST_UPLOAD_DIR = Path("data/test_uploads")
settings.upload_dir = str(TEST_UPLOAD_DIR)
# FakeEmbedder 是哈希向量、没有语义，分数只有"完全相同 ≈1.0 / 其它 ≈0.1~0.4"两种，
# 生产用的 MIN_SCORE 阈值会误杀测试数据，所以测试里把它关掉。
settings.min_vector_score = 0.0
# 精排同理：FakeReranker 只看"字面重合"，生产阈值也会误杀测试数据。
settings.min_rerank_score = 0.0


class FakeLLMClient:
    def __init__(
        self,
        reply: str = "这是模拟回答，依据资料 [1]。",
        replies: list[str] | None = None,
        fail: bool = False,
    ) -> None:
        self.reply = reply
        self.replies = list(replies or [])
        self.tool_replies: list[LLMToolResponse] = []
        self.fail = fail
        self.call_count = 0
        self.last_messages: list[dict[str, Any]] = []
        self.models: list[str | None] = []

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        response_format: dict[str, object] | None = None,
    ) -> str:
        self.call_count += 1
        self.last_messages = messages
        self.models.append(model)

        if self.fail:
            raise LLMUnavailable("模拟模型服务不可用")

        if self.replies:
            return self.replies.pop(0)

        return self.reply

    async def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str | None = None,
        tool_choice: str = "auto",
    ) -> LLMToolResponse:
        self.call_count += 1
        self.last_messages = messages
        self.models.append(model)

        if self.fail:
            raise LLMUnavailable("模拟模型服务不可用")

        if self.tool_replies:
            return self.tool_replies.pop(0)

        assistant_message = {
            "role": "assistant",
            "content": self.reply,
        }
        return LLMToolResponse(
            assistant_message=assistant_message,
            content=self.reply,
            tool_calls=(),
            finish_reason="stop",
        )


class FakeIngestQueue:
    def __init__(self) -> None:
        self.document_ids: list[str] = []

    def enqueue(self, document_id: str) -> str:
        self.document_ids.append(document_id)
        return f"fake-job-{len(self.document_ids)}"


@pytest.fixture
def fake_ingest_queue() -> FakeIngestQueue:
    return FakeIngestQueue()


@pytest.fixture
def upload_ready_document(
    client: TestClient,
    db_session: Session,
) -> Callable[[dict[str, str], str, str], str]:
    """上传文档后同步执行 Worker，供依赖 ready 文档的测试使用。"""

    def upload(
        headers: dict[str, str],
        filename: str,
        content: str,
    ) -> str:
        response = client.post(
            "/documents/upload",
            headers=headers,
            files={
                "file": (
                    filename,
                    content.encode(),
                    "text/plain",
                ),
            },
        )
        assert response.status_code == 202

        document_id = response.json()["id"]
        _process_document(
            db_session,
            document_id,
            FakeEmbedder(),
        )

        return document_id

    return upload


@pytest.fixture
def fake_llm() -> FakeLLMClient:
    """测试里同时要 client 和 fake_llm 时，拿到的是同一个实例（见下面的依赖注入）。"""
    return FakeLLMClient()


@pytest.fixture
def fake_reranker() -> FakeReranker:
    """精排测试替身：不加载 1GB 模型，测试才能跑得快。"""
    return FakeReranker()


@pytest.fixture(scope="session")
def engine() -> Generator[Engine, None, None]:
    test_engine = create_engine(settings.test_database_url)
    with test_engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(test_engine)
    yield test_engine
    Base.metadata.drop_all(test_engine)
    test_engine.dispose()


@pytest.fixture
def db_session(engine: Engine) -> Generator[Session, None, None]:
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection, join_transaction_mode="create_savepoint")
    yield session
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def client(
    db_session: Session,
    fake_llm: FakeLLMClient,
    fake_reranker: FakeReranker,
    fake_ingest_queue: FakeIngestQueue,
) -> Generator[TestClient, None, None]:
    def override_get_db() -> Generator[Session, None, None]:
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_embedder] = lambda: FakeEmbedder()
    app.dependency_overrides[get_llm_client] = lambda: fake_llm
    app.dependency_overrides[get_reranker] = lambda: fake_reranker
    app.dependency_overrides[get_ingest_queue] = lambda: fake_ingest_queue

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture(scope="session", autouse=True)
def _clean_test_uploads() -> Generator[None, None, None]:
    shutil.rmtree(TEST_UPLOAD_DIR, ignore_errors=True)
    yield
    shutil.rmtree(TEST_UPLOAD_DIR, ignore_errors=True)


@pytest.fixture
def authenticated_headers(client: TestClient) -> dict[str, str]:
    password = "correct-horse-battery-staple"

    register_response = client.post(
        "/auth/register",
        json={
            "email": "retrieval-user@example.com",
            "password": password,
        },
    )
    assert register_response.status_code == 201

    login_response = client.post(
        "/auth/login",
        json={
            "email": "retrieval-user@example.com",
            "password": password,
        },
    )
    assert login_response.status_code == 200

    return {
        "Authorization": f"Bearer {login_response.json()['access_token']}",
    }
