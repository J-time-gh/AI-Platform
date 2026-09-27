from pathlib import Path

from app.core.config import settings


def _upload_dir() -> Path:
    directory = Path(settings.upload_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_file(document_id: str, suffix: str, content: bytes) -> str:
    """把文件内容写入磁盘，返回保存路径。"""
    path = _upload_dir() / f"{document_id}{suffix}"
    path.write_bytes(content)
    return str(path)


def delete_file(stored_path: str) -> None:
    """删除磁盘文件（不存在时静默跳过）。"""
    path = Path(stored_path)
    if path.exists():
        path.unlink()
