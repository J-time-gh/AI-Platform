"""文档入库编排：解析 → 切分 → 向量化 → 写库。"""

from collections.abc import Callable
from pathlib import Path

from sqlalchemy.orm import Session

from app.schemas.documents import IngestStage
from app.services import chunk_store
from app.services.chunking import split_text
from app.services.embedding import Embedder
from app.services.parsers import parse_document


class IngestError(Exception):
    """用户内容无法处理（解析失败/提取不到文本等）。"""


# Worker 用这个回调把实际执行阶段写回 documents 表，供每份文档的流程图读取。
ProgressReporter = Callable[[IngestStage, int], None]


def ingest_document(
    db: Session,
    document_id: str,
    stored_path: str,
    embedder: Embedder,
    progress_reporter: ProgressReporter | None = None,
) -> int:
    """处理文档并写入向量块；可选回调会在每个可视化节点开始时报告进度。"""
    report = progress_reporter or (lambda _stage, _progress: None)

    # 节点记录：解析可能最耗时，先写入阶段，前端即可显示 Worker 正在解析。
    report("parsing", 30)
    try:
        text = parse_document(Path(stored_path))
    except Exception as exc:
        raise IngestError(f"文档解析失败：{exc}") from exc

    # 节点记录：文本切块完成后，流程图会把解析节点标为完成。
    report("chunking", 45)
    chunks = split_text(text)
    if not chunks:
        raise IngestError("未能从文档中提取到文本内容（可能是扫描件或空文件）")

    # 节点记录：Embedding 可能需要加载模型或批量计算，单独显示此阶段。
    report("embedding", 65)
    vectors = embedder.embed_texts(chunks)

    # 节点记录：向量与文本块写入数据库；完成后由 Worker 标记 ready / 100%。
    report("saving", 85)
    return chunk_store.replace_chunks(db, document_id, chunks, vectors)
