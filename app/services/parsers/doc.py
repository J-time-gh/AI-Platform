import shutil
import subprocess
import tempfile
from pathlib import Path

from app.services.parsers.docx import parse as parse_docx

_SOFFICE_CANDIDATES = (
    r"C:\Program Files\LibreOffice\program\soffice.exe",
    r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
)


def _find_soffice() -> str:
    """定位 soffice：先查 PATH，再查默认安装目录；找不到给出明确报错。"""
    found = shutil.which("soffice")
    if found:
        return found
    for candidate in _SOFFICE_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    raise RuntimeError(
        "未找到 LibreOffice（soffice），无法解析 .doc 文件。"
        "请安装 LibreOffice 或把文件另存为 .docx。"
    )


def parse(path: Path) -> str:
    """把 .doc 无头转换为 .docx，再复用 docx 解析器。"""
    soffice = _find_soffice()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        profile = tmp_path / "lo-profile"
        result = subprocess.run(
            [
                soffice,
                f"-env:UserInstallation={profile.as_uri()}",
                "--headless",
                "--convert-to",
                "docx",
                "--outdir",
                tmp,
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "未知错误"
            raise RuntimeError(f".doc 转换失败：{detail}")
        converted = tmp_path / f"{path.stem}.docx"
        if not converted.exists():
            raise RuntimeError(".doc 转换失败：LibreOffice 未生成输出文件")
        return parse_docx(converted)
