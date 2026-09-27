"""RAG 提示词组装：把检索到的知识块编号成【资料】，约束模型只依据资料回答。"""

from app.schemas.search import SearchResult

SYSTEM_PROMPT = (
    "你是企业知识库助手。只能依据【资料】回答，资料中没有的信息不要编造，"
    "直接回答“根据现有资料无法回答”。引用资料时在句末标注 [编号]。"
)

NO_CONTEXT_REPLY = "根据现有资料无法回答。"


def build_messages(question: str, results: list[SearchResult]) -> list[dict[str, str]]:
    """组装 OpenAI 兼容的 messages：system 约束 + user（资料 + 问题）。"""
    blocks = "\n\n".join(
        f"[{index}] (来源: {result.filename} 第{result.chunk_index}块)\n{result.content}"
        for index, result in enumerate(results, start=1)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"【资料】\n{blocks}\n\n【问题】{question}"},
    ]
