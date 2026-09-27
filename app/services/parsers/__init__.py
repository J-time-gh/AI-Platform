"""文档解析器：按扩展名分发到具体实现，统一返回纯文本。"""

from collections.abc import Callable
from pathlib import Path

from app.services.parsers import doc, docx, excel, pdf, text

Parser = Callable[[Path], str]

_PARSERS: dict[str, Parser] = {
    ".txt": text.parse,
    ".md": text.parse,
    ".pdf": pdf.parse,
    ".docx": docx.parse,
    ".doc": doc.parse,
    ".xlsx": excel.parse,
}


def parse_document(path: Path) -> str:
    """按文件后缀选择解析器，提取纯文本。"""
    suffix = path.suffix.lower()
    parser = _PARSERS.get(suffix)
    if parser is None:
        raise ValueError(f"不支持的解析格式：{suffix or '无扩展名'}")
    return parser(path)
