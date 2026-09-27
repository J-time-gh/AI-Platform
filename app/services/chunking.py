"""递归文本切分：把长文本切成带重叠的块，优先在中文标点处断开。"""

_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", "、", " "]


def split_text(text: str, chunk_size: int = 500, overlap: int = 80) -> list[str]:
    """把长文本切分成带重叠的块。

    每块不超过 chunk_size 个字符，相邻块之间保留 overlap 个字符的重叠，
    优先在段落、换行、中文句读处断开。
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap 必须满足 0 <= overlap < chunk_size")

    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    atoms = _split_atoms(text, chunk_size, _SEPARATORS)
    return _merge(atoms, chunk_size, overlap)


def _split_atoms(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    """把文本拆成不超过 chunk_size 的原子片段（递归降低分隔符优先级）。"""
    if len(text) <= chunk_size:
        return [text]

    for index, separator in enumerate(separators):
        if separator not in text:
            continue
        parts = text.split(separator)
        atoms: list[str] = []
        for position, part in enumerate(parts):
            piece = part + separator if position < len(parts) - 1 else part
            if not piece.strip():
                continue
            if len(piece) <= chunk_size:
                atoms.append(piece)
            else:
                atoms.extend(_split_atoms(piece, chunk_size, separators[index + 1 :]))
        return atoms

    # 没有任何可用分隔符：硬切兜底
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def _merge(atoms: list[str], chunk_size: int, overlap: int) -> list[str]:
    """贪心合并原子片段成块；新块开头带上上一块结尾的重叠片段。"""
    chunks: list[str] = []
    current = ""
    for atom in atoms:
        if current and len(current) + len(atom) > chunk_size:
            chunks.append(current.strip())
            tail = current[-overlap:] if overlap > 0 else ""
            if len(tail) + len(atom) > chunk_size:
                tail = ""
            current = tail + atom
        else:
            current += atom
    if current.strip():
        chunks.append(current.strip())
    return chunks
