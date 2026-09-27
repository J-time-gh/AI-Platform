from pathlib import Path


def parse(path: Path) -> str:
    """读取纯文本文件（txt/md）：UTF-8 优先，失败退回 GBK。"""
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")
