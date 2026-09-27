"""搜索结果的字面命中词提取。

检索负责找出语义相关的 chunk；本模块只从查询和最终结果正文中找出
完全一致的 jieba 词元，供前端做展示层高亮。它不会影响召回、排序或门控。
"""

from app.services.bm25 import tokenize


def matched_query_terms(query: str, content: str) -> list[str]:
    """返回同时存在于查询和正文的有效词元。

    使用和 BM25 相同的 tokenize()，因此会过滤“什么”“多少”等停用词，
    并能将“什么是资本论”正确识别为“资本论”，而不是由前端任意切成两字片段。
    返回查询中的原始词形，长度优先排序，便于前端在重叠词中优先高亮长词。
    """
    content_terms = {term.casefold() for term in tokenize(content)}
    matched: dict[str, str] = {}

    for term in tokenize(query):
        normalized = term.casefold()
        if len(term) >= 2 and normalized in content_terms:
            matched.setdefault(normalized, term)

    return sorted(
        matched.values(),
        key=lambda term: (-len(term), term),
    )
