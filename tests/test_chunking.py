import pytest

from app.services.chunking import split_text


def test_empty_text_returns_empty_list() -> None:
    assert split_text("") == []
    assert split_text("   \n\n  ") == []


def test_short_text_returns_single_chunk() -> None:
    text = "很短的一句话。"
    assert split_text(text) == [text]


def test_long_text_splits_into_multiple_chunks() -> None:
    text = "这是一个测试句子。" * 200
    chunks = split_text(text, chunk_size=500, overlap=80)
    assert len(chunks) > 1
    assert all(len(chunk) <= 500 for chunk in chunks)


def test_adjacent_chunks_overlap() -> None:
    text = "这是一句用来测试重叠的句子。" * 100
    chunks = split_text(text, chunk_size=300, overlap=50)
    assert len(chunks) > 1
    assert chunks[0][-30:] in chunks[1]


def test_prefers_sentence_boundary() -> None:
    text = "第一句话。" * 120
    chunks = split_text(text, chunk_size=500, overlap=50)
    assert all(chunk.endswith("。") for chunk in chunks)


def test_long_text_without_separators_is_hard_cut() -> None:
    text = "a" * 1300
    chunks = split_text(text, chunk_size=500, overlap=80)
    assert len(chunks) >= 3
    assert all(len(chunk) <= 500 for chunk in chunks)
    assert all(set(chunk) == {"a"} for chunk in chunks)


def test_invalid_overlap_raises() -> None:
    with pytest.raises(ValueError):
        split_text("内容内容", chunk_size=100, overlap=100)
    with pytest.raises(ValueError):
        split_text("内容内容", chunk_size=100, overlap=-1)


def test_chunks_contain_no_blank_whitespace() -> None:
    text = "段落一。\n\n" * 50
    chunks = split_text(text, chunk_size=100, overlap=20)
    assert chunks
    assert all(chunk == chunk.strip() and chunk for chunk in chunks)
