from pathlib import Path

import pymupdf


def parse(path: Path) -> str:
    """逐页提取 PDF 文本，页间用换行分隔。"""
    pages: list[str] = []
    with pymupdf.open(str(path)) as document:
        for page in document:
            pages.append(page.get_text())
    return "\n".join(pages)
