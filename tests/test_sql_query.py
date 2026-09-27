import pytest

from app.services.tools.base import ToolError
from app.services.tools.sql_query import SqlQueryTool


class FakeResult:
    """模拟 SQLAlchemy execute() 返回的结果对象。"""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def mappings(self) -> "FakeResult":
        return self

    def all(self) -> list[dict]:
        return self._rows


class FakeSession:
    """模拟数据库 Session，并记录 SQL 和参数。"""

    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.calls: list[dict] = []

    def execute(
        self,
        statement,
        params: dict | None = None,
    ) -> FakeResult:
        sql = str(statement)

        self.calls.append(
            {
                "sql": sql,
                "params": params,
            },
        )

        if "SET LOCAL statement_timeout" in sql:
            return FakeResult([])

        return FakeResult(self.rows)


def _query_call(db: FakeSession) -> dict:
    """从所有数据库调用中找到真正的业务查询，跳过超时设置 SQL。"""
    return [call for call in db.calls if "SET LOCAL statement_timeout" not in call["sql"]][0]


def test_document_count_is_scoped_to_current_user() -> None:
    """统计文档时，SQL 必须按当前工具绑定的用户 ID 过滤。"""
    db = FakeSession([{"count": 5}])
    tool = SqlQueryTool(
        db,
        user_id="user-a",
    )

    result = tool.run(
        {
            "query_name": "document_count",
        },
    )

    assert result == {
        "rows": [{"count": 5}],
        "count": 1,
    }

    query_call = _query_call(db)

    assert "SELECT count(*)" in query_call["sql"]
    assert "WHERE user_id = :user_id" in query_call["sql"]
    assert query_call["params"] == {
        "user_id": "user-a",
    }


def test_chunks_for_document_requires_document_id() -> None:
    db = FakeSession([])
    tool = SqlQueryTool(
        db,
        user_id="user-a",
    )

    with pytest.raises(ToolError, match="document_id"):
        tool.run(
            {
                "query_name": "chunks_for_document",
            },
        )


def test_invalid_query_name_is_rejected() -> None:
    db = FakeSession([])
    tool = SqlQueryTool(
        db,
        user_id="user-a",
    )

    with pytest.raises(ToolError, match="参数错误"):
        tool.run(
            {
                "query_name": "delete_all",
            },
        )


def test_chunks_for_document_is_scoped_to_current_user() -> None:
    """读取 Chunk 时，必须 Join documents 并验证文档归属用户。"""
    db = FakeSession(
        [
            {
                "id": "chunk-1",
                "document_id": "doc-1",
                "chunk_index": 0,
                "content": "测试内容",
            },
        ],
    )
    tool = SqlQueryTool(
        db,
        user_id="user-a",
    )

    result = tool.run(
        {
            "query_name": "chunks_for_document",
            "document_id": "doc-1",
            "limit": 10,
        },
    )

    assert result["count"] == 1
    assert result["rows"][0]["document_id"] == "doc-1"

    query_call = _query_call(db)

    assert "FROM chunks" in query_call["sql"]
    assert "JOIN documents" in query_call["sql"]
    assert "chunks.document_id = documents.id" in query_call["sql"]
    assert "chunks.document_id = :document_id" in query_call["sql"]
    assert "documents.user_id = :user_id" in query_call["sql"]

    assert query_call["params"] == {
        "document_id": "doc-1",
        "user_id": "user-a",
        "limit": 10,
    }
