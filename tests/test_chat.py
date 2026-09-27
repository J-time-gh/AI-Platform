"""聊天接口测试：LLM 走 FakeLLMClient，全程不发真实网络请求。"""

import json
from typing import Any

from fastapi.testclient import TestClient

from app.core.config import settings
from app.services.llm import LLMToolCall, LLMToolResponse


def _tool_reply(
    call_id: str,
    name: str,
    arguments: dict[str, Any],
) -> LLMToolResponse:
    arguments_json = json.dumps(arguments, ensure_ascii=False)
    assistant_message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments_json,
                },
            },
        ],
    }
    return LLMToolResponse(
        assistant_message=assistant_message,
        content=None,
        tool_calls=(
            LLMToolCall(
                id=call_id,
                name=name,
                arguments_json=arguments_json,
            ),
        ),
        finish_reason="tool_calls",
    )


def _final_reply(content: str) -> LLMToolResponse:
    assistant_message = {
        "role": "assistant",
        "content": content,
    }
    return LLMToolResponse(
        assistant_message=assistant_message,
        content=content,
        tool_calls=(),
        finish_reason="stop",
    )


def _queue_rag_turn(fake_llm, query: str, answer: str | None = None) -> None:
    fake_llm.tool_replies = [
        _tool_reply(
            "call_chat_rag_1",
            "rag_search",
            {"query": query},
        ),
        _final_reply(answer or fake_llm.reply),
    ]


def _create_rag_session(
    client: TestClient,
    headers: dict[str, str],
) -> str:
    create_response = client.post(
        "/chat/sessions",
        headers=headers,
        json={"title": "RAG 测试会话"},
    )
    assert create_response.status_code == 201, create_response.text
    session_id = create_response.json()["id"]

    skill_response = client.patch(
        f"/chat/sessions/{session_id}/skill",
        headers=headers,
        json={"active_skill_id": "knowledge_base_qa"},
    )
    assert skill_response.status_code == 200, skill_response.text
    return session_id


def _stream_events(response) -> list[dict[str, Any]]:
    return [
        json.loads(block.removeprefix("data:").strip())
        for block in response.text.split("\n\n")
        if block.startswith("data:")
    ]


# fake_llm 故意不写类型注解：它由 conftest 提供，跨文件导入 conftest 更脆
def test_chat_returns_llm_reply_with_sources(
    client: TestClient,
    fake_llm,
    authenticated_headers: dict[str, str],
    upload_ready_document,
) -> None:
    text = "机器学习是人工智能的一个分支。"
    upload_ready_document(authenticated_headers, "ml.txt", text)
    session_id = _create_rag_session(client, authenticated_headers)
    _queue_rag_turn(fake_llm, text)

    # 问句必须与入库文本完全一致：FakeEmbedder 是哈希向量，无语义。
    response = client.post(
        "/chat",
        headers=authenticated_headers,
        json={
            "message": text,
            "session_id": session_id,
            "mode": "vector",
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["reply"] == fake_llm.reply
    assert body["model"] == settings.llm_model
    assert body["sources"][0]["filename"] == "ml.txt"
    assert body["sources"][0]["score"] > 0.99  # 同一文本 → 距离≈0
    # ChatResponse 需要保留 SearchResult 的展示高亮元数据，供前端来源抽屉使用。
    assert isinstance(body["sources"][0]["highlight_terms"], list)
    assert body["sources"][0]["highlight_terms"]

    history_response = client.get(
        f"/chat/sessions/{session_id}/messages",
        headers=authenticated_headers,
    )
    assert history_response.status_code == 200, history_response.text
    persisted_messages = history_response.json()
    assert persisted_messages[0]["sources"] == []
    assert persisted_messages[1]["sources"] == body["sources"]
    assert persisted_messages[1]["sources"][0]["highlight_terms"]


def test_chat_prompt_contains_retrieved_chunk(
    client: TestClient,
    fake_llm,
    authenticated_headers: dict[str, str],
    upload_ready_document,
) -> None:
    text = "机器学习是人工智能的一个分支。"
    upload_ready_document(
        authenticated_headers,
        "ml.txt",
        text,
    )
    session_id = _create_rag_session(client, authenticated_headers)
    _queue_rag_turn(fake_llm, text)

    response = client.post(
        "/chat",
        headers=authenticated_headers,
        json={
            "message": "机器学习是什么？",
            "session_id": session_id,
        },
    )
    assert response.status_code == 200, response.text

    messages = fake_llm.last_messages

    assert messages[0]["role"] == "system"
    assert "Skill 指令" in messages[0]["content"]
    assert "rag_search" in messages[0]["content"]
    assert messages[-1]["role"] == "tool"
    tool_result = json.loads(messages[-1]["content"])
    assert tool_result["output"]["results"][0]["content"] == text


def test_chat_inserts_history_between_system_and_question(
    client: TestClient,
    fake_llm,
    authenticated_headers: dict[str, str],
) -> None:
    response = client.post(
        "/chat",
        headers=authenticated_headers,
        json={
            "message": "继续说说",
            "history": [{"role": "user", "content": "上一个问题"}],
        },
    )
    assert response.status_code == 200

    messages = fake_llm.last_messages

    assert [message["role"] for message in messages] == ["system", "user", "user"]
    assert "不要声称检索过" in messages[0]["content"]
    assert messages[1]["content"] == "上一个问题"


def test_chat_without_skill_uses_plain_llm_without_retrieval(
    client: TestClient,
    fake_llm,
    authenticated_headers: dict[str, str],
) -> None:
    response = client.post(
        "/chat",
        headers=authenticated_headers,
        json={"message": "知识库里有什么？"},
    )
    assert response.status_code == 200
    body = response.json()

    assert body["reply"] == fake_llm.reply
    assert body["sources"] == []
    assert fake_llm.call_count == 1
    assert [message["role"] for message in fake_llm.last_messages] == [
        "system",
        "user",
    ]


def test_rag_chat_uses_requested_retrieval_strategy_and_top_k(
    client: TestClient,
    fake_llm,
    authenticated_headers: dict[str, str],
    monkeypatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_search_with_mode(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(
        "app.services.tools.rag_search.retrieval.search_with_mode",
        fake_search_with_mode,
    )
    session_id = _create_rag_session(client, authenticated_headers)
    _queue_rag_turn(fake_llm, "无结果测试")

    response = client.post(
        "/chat",
        headers=authenticated_headers,
        json={
            "message": "请检索",
            "session_id": session_id,
            "mode": "bm25",
            "top_k": 2,
        },
    )

    assert response.status_code == 200, response.text
    assert captured["mode"] == "bm25"
    assert captured["top_k"] == 2


def test_rag_skill_returns_empty_sources_when_no_document_matches(
    client: TestClient,
    fake_llm,
    authenticated_headers: dict[str, str],
) -> None:
    session_id = _create_rag_session(client, authenticated_headers)
    _queue_rag_turn(fake_llm, "没有对应文档的搜索词", "没有找到相关资料。")

    response = client.post(
        "/chat",
        headers=authenticated_headers,
        json={
            "message": "请检索知识库",
            "session_id": session_id,
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["reply"] == "没有找到相关资料。"
    assert body["sources"] == []
    assert fake_llm.call_count == 2
    tool_result = json.loads(fake_llm.last_messages[-1]["content"])
    assert tool_result["output"]["count"] == 0


def test_rag_chat_uses_persisted_session_history(
    client: TestClient,
    fake_llm,
    authenticated_headers: dict[str, str],
) -> None:
    create_response = client.post(
        "/chat/sessions",
        headers=authenticated_headers,
        json={"title": "历史记录测试"},
    )
    assert create_response.status_code == 201
    session_id = create_response.json()["id"]

    first_response = client.post(
        "/chat",
        headers=authenticated_headers,
        json={
            "message": "第一轮问题",
            "session_id": session_id,
        },
    )
    assert first_response.status_code == 200

    skill_response = client.patch(
        f"/chat/sessions/{session_id}/skill",
        headers=authenticated_headers,
        json={"active_skill_id": "knowledge_base_qa"},
    )
    assert skill_response.status_code == 200
    _queue_rag_turn(fake_llm, "检索词", "第二轮回答")

    second_response = client.post(
        "/chat",
        headers=authenticated_headers,
        json={
            "message": "第二轮问题",
            "session_id": session_id,
        },
    )
    assert second_response.status_code == 200

    messages = fake_llm.last_messages
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "第一轮问题"
    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] == fake_llm.reply
    assert messages[3]["role"] == "user"
    assert messages[3]["content"] == "第二轮问题"


def test_chat_returns_503_when_llm_unavailable(
    client: TestClient,
    fake_llm,
    authenticated_headers: dict[str, str],
    upload_ready_document,
) -> None:
    upload_ready_document(
        authenticated_headers,
        "ml.txt",
        "机器学习是人工智能的一个分支。",
    )
    fake_llm.fail = True

    response = client.post(
        "/chat",
        headers=authenticated_headers,
        json={"message": "机器学习是什么？"},
    )

    assert response.status_code == 503


def test_chat_accepts_model_override(
    client: TestClient,
    fake_llm,
    authenticated_headers: dict[str, str],
    upload_ready_document,
) -> None:
    upload_ready_document(
        authenticated_headers,
        "ml.txt",
        "机器学习是人工智能的一个分支。",
    )

    body = client.post(
        "/chat",
        headers=authenticated_headers,
        json={"message": "机器学习是什么？", "model": "qwen2.5-14b-instruct"},
    ).json()

    assert fake_llm.models[-1] == "qwen2.5-14b-instruct"
    assert body["model"] == "qwen2.5-14b-instruct"


def test_chat_rejects_empty_message(
    client: TestClient,
    authenticated_headers: dict[str, str],
) -> None:
    assert (
        client.post("/chat", headers=authenticated_headers, json={"message": ""}).status_code == 422
    )


def test_chat_rejects_invalid_role(
    client: TestClient,
    authenticated_headers: dict[str, str],
) -> None:
    payload = {"message": "hi", "history": [{"role": "boss", "content": "hello"}]}
    assert client.post("/chat", headers=authenticated_headers, json=payload).status_code == 422


def test_chat_rejects_top_k_out_of_range(
    client: TestClient,
    authenticated_headers: dict[str, str],
) -> None:
    assert (
        client.post(
            "/chat",
            headers=authenticated_headers,
            json={"message": "hi", "top_k": 0},
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/chat",
            headers=authenticated_headers,
            json={"message": "hi", "top_k": 100},
        ).status_code
        == 422
    )


def test_chat_stream_emits_real_stages_and_final_response(
    client: TestClient,
    fake_llm,
    authenticated_headers: dict[str, str],
) -> None:
    response = client.post(
        "/chat/stream",
        headers=authenticated_headers,
        json={"message": "用一句话介绍你自己"},
    )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _stream_events(response)
    flow_events = [event for event in events if event["kind"] == "flow"]
    stages = {(event["id"], event["status"]) for event in flow_events}

    assert ("chat-request", "complete") in stages
    assert ("chat-llm-final", "active") in stages
    assert ("chat-llm-final", "complete") in stages
    final_event = next(event for event in events if event["kind"] == "complete")
    assert final_event["response"]["reply"] == fake_llm.reply


def test_chat_stream_reports_agent_plan_and_tool_nodes(
    client: TestClient,
    fake_llm,
    authenticated_headers: dict[str, str],
    upload_ready_document,
) -> None:
    text = "知识库问答使用 RAG 检索相关资料。"
    upload_ready_document(authenticated_headers, "flow.txt", text)
    session_id = _create_rag_session(client, authenticated_headers)
    _queue_rag_turn(fake_llm, text)

    response = client.post(
        "/chat/stream",
        headers=authenticated_headers,
        json={"message": text, "session_id": session_id, "mode": "vector"},
    )

    assert response.status_code == 200, response.text
    events = _stream_events(response)
    flow_events = [event for event in events if event["kind"] == "flow"]
    stages = {(event["id"], event["status"]) for event in flow_events}

    assert ("agent-plan-1", "active") in stages
    assert ("agent-plan-1", "complete") in stages
    assert ("agent-tool-1", "active") in stages
    assert ("agent-tool-1", "complete") in stages
    assert ("agent-plan-2", "complete") in stages
    assert any(event["id"] == "chat-answer" for event in flow_events)
